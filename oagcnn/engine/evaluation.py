import torch
import logging
from oagcnn.data.dataset_creator import DatasetFactory
from oagcnn.utils.loss_container import LossContainer
import copy
import numpy as np

class Evaluator:

    def __init__(self, cfg, model, device):
        self.device = device
        self.test_loader = DatasetFactory.create_data_loader(cfg.DATASETS.TEST, cfg.DATASETS.TEST_SPLIT)
        self.model = model
        # Evaluation period each epoch hardcoded
        self.validation_epochs = [epoch for epoch in range(cfg.SOLVER.NUM_EPOCHS) if epoch % 1 == 0]
        self.loss_container = LossContainer(self.validation_epochs, 0, 1)
        self.logger = logging.getLogger("OAGCNN")

    def get_loss(self):
        return self.loss_container.get_loss_container()

    def validate(self, epoch):
        if epoch not in self.validation_epochs:
            return

        self.logger.info("Starting validation epoch {}".format(epoch))

        model_usage = copy.deepcopy(self.model.use_previous_inference_mask)
        self.model.eval()
        self.model.use_previous_inference_mask = True
        current_sequence = None
        sequence_loss = []

        with torch.no_grad():

            for idx, sample in enumerate(self.test_loader):
                # Images & Targets shape: LIST CLIP_LENGTH ELEMENTS EACH ONE SHAPE (BATCH_SIZE, CHANNELS, WIDTH, HEIGHT)
                input, target, valid_target, sequence_name = sample["images"], sample["objs_masks"], sample["valid_masks"], sample["sequence"]

                if current_sequence is None or sequence_name != current_sequence:
                    if idx != 0:
                        average_sequence_loss = np.mean(sequence_loss)
                        self.loss_container.update(average_sequence_loss, epoch)
                        sequence_loss = []

                    current_sequence = sequence_name

                    # Inits objects to track
                    init_gt_masks = target
                    init_valid_masks = valid_target
                    self.model.init_clip(init_gt_masks, init_valid_masks)

                _, loss = self.model(input, target, valid_target)
                sequence_loss.append(loss.cpu().item())

        total_val_loss = self.loss_container.get_total_epoch_loss(epoch)

        self.logger.info("Epoch: {}:\tTotal mean validation loss : {}".format(epoch, total_val_loss))
        # Important: restore model usage
        self.model.use_previous_inference_mask = model_usage
        self.model.train()

        return total_val_loss
