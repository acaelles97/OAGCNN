import os
import torch
from oagcnn.data.dataset_creator import DatasetFactory
from oagcnn.utils.logger import setup_logger
from oagcnn.utils.loss_container import LossContainer
from .evaluation import Evaluator
from oagcnn.models.oagcnn import OAGCNN
from .checkpointer import CheckPointer
import shutil

class Trainer:
    def __init__(self, cfg, model: OAGCNN, device):
        self.train_loader = DatasetFactory.create_data_loader(cfg.DATASETS.TRAIN, cfg.DATASETS.TRAIN_SPLIT, cfg)
        self.epoch = 0
        self.total_epochs = cfg.SOLVER.NUM_EPOCHS
        self.device = device
        self.model = model
        self.iter = 0
        self.print_every = cfg.GENERAL_CONFIG.PRINT_EVERY

        self.loss_container = LossContainer([epoch for epoch in range(self.total_epochs)], self.print_every, cfg.DATA.CLIP_LENGTH)
        self.training_name = cfg.GENERAL_CONFIG.TRAIN_NAME

        self.evaluator = Evaluator(cfg, model, device)

        self._init_output_paths(cfg.PATH.OUTPUT_DIR, cfg.PATH.CONFIG_FILE)
        self.checkpointer = CheckPointer(cfg.SOLVER.CHECKPOINTER, self.training_name, self.output_path)

        self.logger = setup_logger("OAGCNN", os.path.join(self.output_path, "log.txt"))

    def _init_output_paths(self, out_dir, config_file_path):
        output_path = os.path.join(out_dir, self.training_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        shutil.copy(config_file_path, self.output_path)


    def save_checkpoint(self, output_path):
        self.logger.info("Saving model: {}".format(output_path))
        model_state_dict = self.model.get_state_dict()
        model_state_dict.update({"epoch": self.epoch})
        model_state_dict.update({"train_loss": self.loss_container.get_loss_container()})
        model_state_dict.update({"val_loss": self.evaluator.get_loss()})
        torch.save(model_state_dict, output_path)

    def train_epoch(self):
        self.iter = 0
        self.logger.info("Starting epoch {}/{}\n".format(self.epoch, self.total_epochs))
        self.model.train()
        for idx, sample in enumerate(self.train_loader):
            # Images & Targets shape: LIST CLIP_LENGTH ELEMENTS EACH ONE SHAPE (BATCH_SIZE, CHANNELS, WIDTH, HEIGHT)
            inputs, targets, valid_targets, sequence_name = sample["images"], sample["objs_masks"], sample["valid_masks"],  sample["sequence"]

            clip_loss = torch.tensor(0.).to(self.device)

            active_objs_masks = targets[0].clone().to(self.device)
            active_valid_masks = valid_targets[0].clone().to(self.device)

            # Inits objects to track
            # self.model.init_clip(init_gt_masks, init_valid_masks)

            # Iterate through each frame in the clip
            for batched_image, batched_gt_mask, batched_valid_target in zip(inputs, targets, valid_targets):
                # Note we just care abut GT objects once they appear 1 time
                # Check if there is any change on this frame
                _, loss, active_objs_masks, active_valid_masks = self.model(batched_image, batched_gt_mask, batched_valid_target, active_objs_masks, active_valid_masks)
                clip_loss = loss + clip_loss

            self.model.optimizer_step(clip_loss)
            self.loss_container.update(clip_loss.cpu().item(), self.epoch)

            if self.iter >= 10:
                break

            self.iter += 1

            if self.iter % self.print_every == 0:
                self.logger.info("Iter {}/{} \t Loss: {}".format(self.iter, len(self.train_loader), self.loss_container.get_mean_last_iter(self.epoch)))

        self.logger.info("Finished epoch {} \t Mean Total Loss: {}".format(self.epoch, self.loss_container.get_mean_last_iter(self.epoch)))

    def train(self):
        self.logger.info("Starting training, using device: {}".format(self.device))

        while self.epoch < self.total_epochs and self.checkpointer.keep_training:
            # Looks for actions to be done before starting the epoch
            actions_done = self.model.actions_before_epoch(epoch=self.epoch)
            # Actions that take make model change, so it is fair to reset the patience
            if actions_done:
                self.checkpointer.reset_patience_counter()
                for action in actions_done:
                    self.logger.info("Action done before epoch {}: Action: {}".format(self.epoch, action))

            # Train the epoch
            self.train_epoch()

            # Validate epoch
            epoch_loss = self.evaluator.validate(self.epoch)

            # Use validation loss to decide based on the constructed checkpointer
            output_path = self.checkpointer.update_val(self.epoch, epoch_loss)
            if output_path:
                self.save_checkpoint(output_path)

            actions_done = self.model.actions_after_epoch(epoch=self.epoch)
            if actions_done:
                # Only LrScheduler step implemented, so no need to reset patience right now
                for action in actions_done:
                    self.logger.info("Action done after epoch {}: Action: {}".format(self.epoch, action))

            self.epoch += 1
