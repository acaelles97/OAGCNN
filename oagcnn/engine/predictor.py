from oagcnn.models.oagcnn import OAGCNN
import torch
from oagcnn.data.test_data import InferenceDataLoader
from oagcnn.utils.save_results import SaveResults


class OAGCNNPredictor:
    def __init__(self, cfg, device, model: OAGCNN):
        self.model = model
        self.model.custom_load_state_dict(cfg.MODEL.LOAD_CHECKPOINT)
        self.device = device

        self.test_dataloaders = InferenceDataLoader.create_dataloader(cfg)
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

        with torch.no_grad():
            for test_loader in self.test_dataloaders:

                # Inits objects to track
                init_gt_masks, init_valid_masks = test_loader.dataset.get_initial_masks()
                sequence_name = test_loader.dataset.get_sequence_name()

                self.model.init_clip(init_gt_masks, init_valid_masks)
                print("Predicting sequence {}".format(sequence_name))
                for idx, sample in enumerate(test_loader):
                    # Images & Targets shape: LIST CLIP_LENGTH ELEMENTS EACH ONE SHAPE (BATCH_SIZE, CHANNELS, WIDTH, HEIGHT)
                    image, original_image, frame_name = sample["image"], sample["original_image"], sample["frame_name"]
                    out_masks = self.model.test_forward(image)
                    out_masks = out_masks.data.cpu()[init_valid_masks, ...]
                    original_image = original_image.squeeze(0).numpy()

                    if self.device == "cuda":
                        out_masks = out_masks.data.cpu()

                    if self.overlay_masks:
                        self.results_saver.save_result_overlay(original_image, out_masks, frame_name, sequence_name)
                    else:
                        self.results_saver.save_result(original_image, out_masks, frame_name, sequence_name)