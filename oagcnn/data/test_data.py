import numpy as np
from .transforms.transforms import CustomComposeTransform
from PIL import Image
import copy
import torch.utils.data as data
import os
from torch.utils.data import DataLoader


class TestDataset(data.Dataset):
    def __init__(self, input_folder, input_folder_init_masks, max_num_obj):
        self._init_frames(input_folder)
        self.max_num_gt_objects = max_num_obj
        self.input_init_masks = input_folder_init_masks
        self.transform = CustomComposeTransform(False)
        self._load_init_annot()


    def _init_frames(self, input_folder):
        self.sequence_name = input_folder.split("/")[-1]
        self.list_frames = [os.path.join(input_folder, frame_name) for frame_name in sorted(os.listdir(input_folder))]

    def _load_init_annot(self):
        """
        Reads segmentation masks and outputs sequence of binary masks and labels
        """
        if os.path.isdir(self.input_init_masks):
            annot_file = os.path.join(self.input_init_masks, sorted(os.listdir(self.input_init_masks))[0])
        else:
            annot_file = self.input_init_masks

        annotation = Image.open(annot_file)
        annotation = self.transform.initial_annot_test_transform(annotation)

        annotation = np.array(annotation, dtype=np.float32)
        annotation[annotation == 255] = 0

        h, w = annotation.shape[0], annotation.shape[1]
        instance_ids = sorted(np.unique(annotation)[1:])
        total_num_instances = len(instance_ids)

        max_instance_id = 0
        if total_num_instances > 0:
            max_instance_id = int(np.max(instance_ids))

        num_instances = max(self.max_num_gt_objects, max_instance_id)

        gt_objs_masks = np.zeros((h, w, num_instances), dtype=np.float32)
        valid_masks = np.zeros((num_instances, 1), dtype=np.bool_)
        # for sorting by size: no used in rvos
        # size_masks = np.zeros((num_instances,))

        for i in range(total_num_instances):
            id_instance = int(instance_ids[i])
            aux_mask = np.zeros((h, w), dtype=np.float32)
            aux_mask[annotation == id_instance] = 1
            gt_objs_masks[:, :, id_instance - 1] = aux_mask
            # Not used on rvos
            # size_masks[id_instance - 1] = np.sum(gt_seg[id_instance - 1, :])
            valid_masks[id_instance - 1] = True

        # Ensures num max object is 10 shape = (this, w*h)
        gt_objs_masks = gt_objs_masks[..., :self.max_num_gt_objects]
        valid_masks = valid_masks[..., :self.max_num_gt_objects].squeeze()
        valid_masks = np.expand_dims(valid_masks, 0)
        gt_objs_masks, valid_masks = self.transform.final_annot_test_transform(gt_objs_masks, valid_masks)
        valid_masks = valid_masks.squeeze(0)
        gt_objs_masks = gt_objs_masks.unsqueeze(0)
        self.gt_objs_masks = gt_objs_masks
        self.valid_masks = valid_masks

    def get_sequence_name(self):
        return self.sequence_name

    def load_frame(self, image_file):
        image = Image.open(image_file)
        image = self.transform.initial_image_test_transform(image)
        original_image = np.array(copy.deepcopy(image), dtype=np.float32)
        transformed_image = self.transform.final_image_test_transform(image)
        return transformed_image, original_image

    def get_initial_masks(self):
        return self.gt_objs_masks, self.valid_masks

    def __len__(self):
        return len(self.list_frames)

    def __getitem__(self, idx):
        # In test mode we go frame by frame
        frame_name = self.list_frames[idx].split("/")[-1]
        image, original_image = self.load_frame(self.list_frames[idx])

        return {"image": image, "original_image": original_image, "frame_name": frame_name}


class InferenceDataLoader:

    @staticmethod
    def create_dataloader(cfg):
        dataloaders = []

        max_num_gt_obj = cfg.DATA.MAX_NUM_OBJ
        base_images_dir = cfg.PREDICTOR.BASE_PATH_IMAGES
        base_annots_dir = cfg.PREDICTOR.BASE_PATH_ANNOTATIONS

        folders_to_infer = cfg.PREDICTOR.FOLDER_TO_INFER

        for folder_to_infer in folders_to_infer:
            image_folder = os.path.join(base_images_dir, folder_to_infer)
            annot_folder = os.path.join(base_annots_dir, folder_to_infer)
            test_dataset = TestDataset(image_folder, annot_folder, max_num_gt_obj)

            dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfg.DATA.NUM_WORKERS,
                drop_last=False,
            )
            dataloaders.append(dataloader)

        return dataloaders
