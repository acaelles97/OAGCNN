from oagcnn.models.feature_extractor.feature_extractor_factory import FeatureExtractorFactory
from oagcnn.models.graph.gcnn_temporal import GCNNTemporal
import torch.nn as nn
import torch
from oagcnn.models.objectives.loss_objective_factory import LossFunctionFactory
from oagcnn.solver.optimizer_factory import OptimizerFactory
from .feature_extractor.rvos_feature_extractor.rvos_feature_extractor_module import RVOSFeatureExtractor
from .feature_extractor.deeplab.custom_deeplab import DeepLabV3Plus
from oagcnn.solver.lr_scheduler_factory import LrSchedulerFactory
from GPUtil import showUtilization as gpu_usage


class ActiveObjectsTracker:
    def __init__(self, device, mask_backpropagation):
        self.device = device
        self.active_objs_masks = None
        self.active_valid_targets = None
        self.active_objs_features = None
        self.mask_backpropagation = mask_backpropagation

    def init_object_tracker(self, init_gt_masks, init_valid_masks):
        if self.mask_backpropagation:
            self.active_objs_masks = init_gt_masks.to(self.device)
            self.active_valid_targets = init_valid_masks.to(self.device)
        else:
            self.active_objs_masks = init_gt_masks.to(self.device)
            self.active_valid_targets = init_valid_masks.to(self.device)

    def update_active_objects(self, new_gt_masks, new_valid_targets):
        objs_changes = torch.ne(self.active_valid_targets, new_valid_targets)
        if objs_changes.any():
            # We just care about objects that are new (1st appearance) on the clip -> batched_valid_target == True and valid_masks_record
            # == False on the positions where there are changes
            new_appearance_ids = torch.bitwise_and(torch.bitwise_and(objs_changes, new_valid_targets),
                                                   torch.bitwise_and(torch.logical_not(self.active_valid_targets), objs_changes))
            # Check if there is any appearance
            if new_appearance_ids.any():
                self.active_valid_targets = torch.bitwise_or(self.active_valid_targets, new_appearance_ids)
                mask_to_op = torch.zeros_like(self.active_objs_masks)
                mask_to_op[new_appearance_ids, :, :] = 1
                self.active_objs_masks = mask_to_op * new_gt_masks + self.active_objs_masks

    def update_masks(self, new_masks):
        if self.mask_backpropagation:
            self.gt_masks = new_masks
        else:
            self.gt_masks = new_masks.detach()


class OAGCNN(nn.Module):

    def __init__(self, cfg, device):
        super(OAGCNN, self).__init__()
        self.device = device

        self.use_previous_inference_mask = cfg.OAGCNN.USE_PREVIOUS_INFERENCE_MASK
        self.mask_backpropagation = cfg.OAGCNN.BACKPROPAGATE_PREDICTED_MASKS
        self.use_temporal_features = cfg.OAGCNN.USE_TEMPORAL_FEATURES
        self.features_backpropagation = cfg.OAGCNN.BACKPROPAGATE_FEATURES

        image_spatial_res = cfg.DATA.IMAGE_SIZE
        self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(cfg.OAGCNN.ARCH.FEATURE_EXTRACTOR, cfg, image_spatial_res).to(
            self.device)

        out_feats_dims = self.feature_extractor.out_channels
        self.gcnn = GCNNTemporal(cfg, self.use_temporal_features, out_feats_dims).to(self.device)

        self.loss_objective = LossFunctionFactory.create_feature_extractor(cfg.OAGCNN.LOSS_FUNCTION).to(self.device)

        self._global_optim = OptimizerFactory.create_optimizer(self.get_parameters_custom(), cfg.SOLVER.OPTIMIZER, cfg.SOLVER.LR,
                                                               cfg.SOLVER.WEIGHT_DECAY)

        self._global_lr_scheduler = None
        if cfg.SOLVER.USE_SCHEDULER:
            self._global_lr_scheduler = LrSchedulerFactory.create_lr_scheduler(self._global_optim, cfg.SOLVER.LR_SCHEDULER)

        self.runtime_actions = {int(epoch): action for action, epoch in cfg.MODEL.RUNTIME_CONFIG.items() if epoch >= 0}

        self.active_obj_masks = None
        self.active_valid_masks = None
        self.active_node_features = None

        # self.active_objects_tracker = ActiveObjectsTracker(self.device, cfg.GCNN.BACKPROPAGATE_PREDICTED_MASKS)

    def init_object_tracker(self, init_gt_masks, init_valid_masks):
        self.active_obj_masks = init_gt_masks.to(self.device)
        self.active_valid_masks = init_valid_masks.to(self.device)
        self.active_node_features = None

    def update_active_objects(self, new_gt_masks, new_valid_targets):
        objs_changes = torch.ne(self.active_valid_masks, new_valid_targets)
        if objs_changes.any():
            # We just care about objects that are new (1st appearance) on the clip -> batched_valid_target == True and valid_masks_record
            # == False on the positions where there are changes
            new_appearance_ids = torch.bitwise_and(torch.bitwise_and(objs_changes, new_valid_targets),
                                                   torch.bitwise_and(torch.logical_not(self.active_valid_masks), objs_changes))
            # Check if there is any appearance
            if new_appearance_ids.any():
                # Set valid mask to True as from now one we will track this objects
                self.active_valid_masks = torch.bitwise_or(self.active_valid_masks, new_appearance_ids)
                mask_to_op = torch.zeros_like(self.active_obj_masks)
                mask_to_op[new_appearance_ids, :, :] = 1
                self.active_obj_masks = mask_to_op * new_gt_masks + self.active_obj_masks

    def get_parameters_custom(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def init_clip(self, init_gt_masks, init_valid_masks):
        self.active_objects_tracker.init_object_tracker(init_gt_masks, init_valid_masks)

    def inference(self, batched_image, batched_gt_mask, batched_valid_target):
        batched_image = batched_image.to(self.device)
        batched_gt_mask = batched_gt_mask.to(self.device)
        batched_valid_target = batched_valid_target.to(self.device)

        self.update_active_objects(batched_gt_mask, batched_valid_target)

        feats = self.feature_extractor(batched_image)
        # (BATCH_SIZE, NUM_OBJ, H, W) -> Same order as given in objs masks so the relation is element to element
        out_mask_logits, out_feats = self.gcnn(feats, self.active_obj_masks, self.active_valid_masks, self.active_node_features)

        self.active_obj_masks = out_mask_logits
        self.active_node_features = out_feats

        out_masks = torch.where(out_mask_logits > 0.5, 1.0, 0.0)
        return out_masks

    def forward(self, batched_image, batched_gt_mask, batched_valid_target):
        batched_image = batched_image.to(self.device)
        batched_gt_mask = batched_gt_mask.to(self.device)
        batched_valid_target = batched_valid_target.to(self.device)

        self.update_active_objects(batched_gt_mask, batched_valid_target)

        feats = self.feature_extractor(batched_image)
        # (BATCH_SIZE, NUM_OBJ, H, W) -> Same order as given in objs masks so the relation is element to element
        out_mask_logits, out_feats = self.gcnn(feats, self.active_obj_masks, self.active_valid_masks, self.active_node_features)

        loss = self.compute_loss(batched_gt_mask, out_mask_logits, batched_valid_target)

        if self.use_previous_inference_mask:
            if self.mask_backpropagation:
                # out_masks = torch.where(out_mask_logits > 0.5, 1.0, 0.0)
                # masking = (out_mask_logits > 0.5).float()
                # out_masks = masking * torch.nn.functional.threshold(out_mask_logits, 0.5, 1.0) + torch.bitwise_not(masking) * out_mask_logits
                self.active_obj_masks = out_mask_logits
            else:
                self.active_obj_masks = out_mask_logits.detach()

        # We need to reset what contains only interested in data
        else:
            self.active_obj_masks = batched_gt_mask

        if self.use_temporal_features:
            if self.features_backpropagation:
                self.active_node_features = out_feats
            else:
                self.active_node_features = out_feats.detach()

        return loss

    def optimizer_step(self, clip_loss):
        self._global_optim.zero_grad()
        clip_loss.backward()
        self._global_optim.step()

    def compute_loss(self, gt_masks, out_masks, valid_targets):
        # (BATCH_SIZE, NUM_OBJ, H*W)
        out_masks = out_masks.view(out_masks.shape[0], out_masks.shape[1], -1)

        gt_masks = gt_masks.view(gt_masks.shape[0], gt_masks.shape[1], -1)
        valid_targets = valid_targets.to(device=out_masks.device).squeeze()

        loss_mask_iou = self.loss_objective(gt_masks, out_masks, valid_targets)

        return loss_mask_iou

    def get_state_dict(self):
        return {
            "feature_extractor_state_dict": self.feature_extractor.state_dict(),
            "gcnn_state_dict": self.gcnn.state_dict(),
            "opt_state_dict": self._global_optim.state_dict(),
        }

    def custom_load_state_dict(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        self.gcnn.load_state_dict(checkpoint["gcnn_state_dict"])

    def resume_training(self, state_dict, resume_epoch):
        self.feature_extractor.load_state_dict(state_dict["feature_extractor_state_dict"])
        self.gcnn.load_state_dict(state_dict["gcnn_state_dict"])
        self._global_optim.load_state_dict(state_dict["opt_state_dict"])
        actions = []
        for epoch in sorted(self.runtime_actions.keys()):
            if epoch > resume_epoch:
                break
            else:
                actions.append(self._change_runtime_parameters(epoch))
        return actions

    def _perform_lr_scheduler_step(self, epoch):
        if self._global_lr_scheduler is not None:
            self._global_lr_scheduler.step()
            if epoch in self._global_lr_scheduler.milestones.keys():
                active_lr = self._global_lr_scheduler.get_lr()
                return "Reducing LR: {}".format(active_lr)

    def _change_runtime_parameters(self, epoch):
        if epoch not in self.runtime_actions:
            return

        action = self.runtime_actions[epoch]

        if action == "FreezeRVOSEncoder":
            if not isinstance(self.feature_extractor, RVOSFeatureExtractor):
                raise ValueError("Current feature extractor is not RVOSFeatureExtractor and action is {}".format(action))
            self.feature_extractor.freeze_rvos_encoder()

        elif action == "UnfreezeRVOSEncoder":
            if not isinstance(self.feature_extractor, RVOSFeatureExtractor):
                raise ValueError("Current feature extractor is not RVOSFeatureExtractor and action is {}".format(action))
            self.feature_extractor.unfreeze_rvos_encoder()

        elif action == "UsePreviousInferenceMask":
            self.use_previous_inference_mask = True

        elif action == "FreezeDeepLabV3Plus":
            if not isinstance(self.feature_extractor, DeepLabV3Plus):
                raise ValueError("Current feature extractor is not DeepLabV3Plus and action is {}".format(action))
            self.feature_extractor.freeze_deeplab()

        return action

    def actions_after_epoch(self, epoch):
        # When an action is performed, we want to reset the patience from the validator
        action_done = []
        action_done.append(self._perform_lr_scheduler_step(epoch))
        return action_done

    def actions_before_epoch(self, epoch):
        action_done = []
        action_done.append(self._change_runtime_parameters(epoch=epoch))
        return action_done

