import torch
import logging
from oagcnn.data.dataset_creator import DatasetFactory
from oagcnn.utils.loss_container import LossContainer
import copy
import numpy as np


class Evaluator:

    def __init__(self, cfg, model, device):
        self.device = device
        self.test_loaders = DatasetFactory.create_data_loader(cfg.DATASETS.VAL, cfg.DATASETS.VAL_SPLIT, cfg)
        self.model = model
        # Evaluation period each epoch hardcoded
        self.validation_epochs = [epoch for epoch in range(cfg.SOLVER.NUM_EPOCHS) if epoch % 1 == 0]
        self.loss_container = LossContainer(self.validation_epochs, 0, 1)
        self.logger = logging.getLogger("OAGCNN")

    def get_loss(self):
        return self.loss_container.get_loss_container()

    def load_state_dict(self, state_dict):
        self.loss_container.load_loss_container(state_dict["val_loss"])

    def validate(self, epoch):
        if epoch not in self.validation_epochs:
            return

        self.logger.info("Starting validation epoch {}".format(epoch))

        model_usage = copy.deepcopy(self.model.use_previous_inference_mask)
        self.model.eval()
        self.model.use_previous_inference_mask = True

        with torch.no_grad():
            for test_loader in self.test_loaders:
                sequence_loss = []
                for idx, sample in enumerate(test_loader):
                    image, target, valid_target, sequence_name = sample["image"], sample["objs_masks"], sample["valid_masks"], sample[
                        "sequence"]
                    if idx == 0:
                        self.model.init_object_tracker(target.clone(), valid_target.clone())

                    loss = self.model(image, target, valid_target)
                    sequence_loss.append(loss.cpu().item())

                average_sequence_loss = np.mean(sequence_loss)
                self.loss_container.update(average_sequence_loss, epoch)

        total_val_loss = self.loss_container.get_total_epoch_loss(epoch)

        self.logger.info("Epoch: {}:\tTotal mean validation loss : {}".format(epoch, total_val_loss))
        # Important: restore model usage

        self.model.use_previous_inference_mask = model_usage
        self.model.train()

        return total_val_loss
