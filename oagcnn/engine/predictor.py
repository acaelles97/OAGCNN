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

        self.selected_sequences = cfg.PREDICTOR.FOLDER_TO_INFER
        self.overlay_masks = cfg.PREDICTOR.OVERLAY_MASKS
        self._init_model_name(cfg.MODEL.LOAD_CHECKPOINT)

        self.results_saver = SaveResults(cfg.PATH.OUTPUT_DIR, self.model_name)

    def _init_model_name(self, checkpoint_path):
        if "/" in checkpoint_path:
            checkpoint_path = checkpoint_path.split("/")[-1]
        self.model_name = checkpoint_path.split(".")[-2]

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
                print("Predicting sequence {}".format(sequence_name))
                for idx, sample in enumerate(test_loader):
                    image, target, valid_target, sequence_name, frame_name = sample["image"], sample["objs_masks"], sample["valid_masks"], sample[
                        "sequence"], sample["frame_name"]
                    if idx == 0:
                        active_objs_masks = target.clone().to(self.device)
                        active_valid_masks = valid_target.clone().to(self.device)

                    # Images & Targets shape: LIST CLIP_LENGTH ELEMENTS EACH ONE SHAPE (BATCH_SIZE, CHANNELS, WIDTH, HEIGHT)

                    out_masks, active_objs_masks, active_valid_masks = self.model.test_forward(image, target, valid_target, active_objs_masks, active_valid_masks)

                    out_masks = out_masks.data.cpu()[active_valid_masks, ...]

                    original_image = image.data.squeeze(0).cpu().numpy()
                    original_image = np.rollaxis(original_image, 0, 3)
                    original_image = std * original_image + mean
                    original_image = np.clip(original_image, 0, 1)

                    if self.device == "cuda":
                        out_masks = out_masks.data.cpu()

                    if self.overlay_masks:
                        self.results_saver.save_result_overlay(original_image, out_masks, frame_name, sequence_name)
                    else:
                        self.results_saver.save_result(original_image, out_masks, frame_name, sequence_name)