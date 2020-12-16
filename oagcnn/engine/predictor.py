from oagcnn.models.oagcnn import OAGCNN
import torch
from oagcnn.utils.save_results import SaveResults
from oagcnn.data.dataset_creator import DatasetFactory
import numpy as np


class OAGCNNPredictor:
    def __init__(self, cfg, device, model: OAGCNN):
        self.model = model
        self.model.custom_load_state_dict(cfg.MODEL.LOAD_CHECKPOINT)
        self.device = device

        self.test_dataloaders = DatasetFactory.create_data_loader(cfg.DATASETS.TEST, cfg.DATASETS.TEST_SPLIT, cfg)

        self.overlay_masks = cfg.PREDICTOR.OVERLAY_MASKS
        self.generate_submission = cfg.PREDICTOR.PREPARE_SUBMISSION_RESULTS
        self.instance_results = cfg.PREDICTOR.INSTANCE_RESULTS
        self.use_metadata_gt_info = cfg.PREDICTOR.USE_METADATA_GT_INFO
        self._init_model_name(cfg.MODEL.LOAD_CHECKPOINT)
        self.results_saver = SaveResults(cfg.PATH.OUTPUT_INFERENCE, self.model_name)

    def _init_model_name(self, checkpoint_path):
        if "/" in checkpoint_path:
            checkpoint_path = checkpoint_path.split("/")[-1]
        self.model_name = ".".join(checkpoint_path.split(".")[:-1])

    def run_predictor(self):
        self.model.eval()
        self.model.use_previous_inference_mask = True
        print("Starting to run predictor")
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        with torch.no_grad():
            for test_loader in self.test_dataloaders:

                # Inits objects to track
                sequence_name = test_loader.dataset.get_sequence_name()
                num_instances_clip = test_loader.dataset.get_max_num_instances()
                empty_frames = test_loader.dataset.get_omitted_frames()
                print("Predicting sequence {}".format(sequence_name), flush=True)
                for idx, sample in enumerate(test_loader):
                    image, target, valid_target, sequence_name, frame_name = sample["image"], sample["objs_masks"], sample["valid_masks"], sample[
                        "sequence"], sample["frame_name"]

                    if idx == 0:
                        self.model.init_object_tracker(target.clone(), valid_target.clone())
                        if empty_frames and self.generate_submission:
                            height, width = image.shape[-2:]
                            self.results_saver.empty_mask_for_submission(height, width, empty_frames, sequence_name)

                    # Images & Targets shape: LIST CLIP_LENGTH ELEMENTS EACH ONE SHAPE (BATCH_SIZE, CHANNELS, WIDTH, HEIGHT)
                    out_masks = self.model.inference(image, target, valid_target)

                    out_masks = out_masks.data.cpu()[self.model.active_valid_masks, ...].numpy()
                    instances_id = torch.nonzero(self.model.active_valid_masks.squeeze(), as_tuple=True)[0].cpu().numpy()

                    if self.overlay_masks:
                        original_image = image.data.squeeze(0).cpu().numpy()
                        original_image = np.rollaxis(original_image, 0, 3)
                        original_image = std * original_image + mean
                        original_image = np.clip(original_image, 0, 1)
                        self.results_saver.save_result_overlay(original_image, out_masks, frame_name, sequence_name, num_instances_clip)

                    if self.instance_results:
                        height, width = image.shape[-2:]
                        self.results_saver.save_result(height, width, out_masks, frame_name, sequence_name, num_instances_clip, instances_id)

                    if self.generate_submission:
                        height, width = image.shape[-2:]
                        self.results_saver.results_for_submission(height, width, out_masks, frame_name, sequence_name, instances_id)

                    else:
                        raise ValueError("Predictor does not have any task active")