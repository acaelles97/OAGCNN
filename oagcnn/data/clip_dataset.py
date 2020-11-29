import torch.utils.data as data
from PIL import Image
import numpy as np
from oagcnn.config.defaults import cfg


class ClipDataset(data.Dataset):

    def __init__(self, clip_images_files, clip_annot_files, transform, instances_id, sequence_name):
        self.clip_images_files = clip_images_files
        self.clip_annot_files = clip_annot_files
        self.instances_id = instances_id
        self.transform = transform
        self.max_num_gt_objects = cfg.DATA.MAX_NUM_OBJ
        self.sequence_name = sequence_name

        self.prepare_filenames()

    def prepare_filenames(self):
        clip_images_files = []
        clip_annot_files = []
        for image_file, annot_file in zip(self.clip_images_files, self.clip_annot_files):
            if type(annot_file) != str:
                annot_file = str(annot_file.decode())
            if type(image_file) != str:
                image_file = str(image_file.decode())
            clip_images_files.append(image_file)
            clip_annot_files.append(annot_file)
        self.clip_images_files = clip_images_files
        self.clip_annot_files = clip_annot_files

    def load_frame(self, image_file, annot_file):
        image = Image.open(image_file)
        annotation = Image.open(annot_file)

        image, annotation = self.transform.initial_transform(image, annotation)

        annotation = np.array(annotation, dtype=np.float32)
        if annotation.shape[0] == 1:
            annotation = np.squeeze(annotation, axis=0)
            # We need to un-normalize
            annotation = annotation * 255

        annotation[annotation == 255] = 0

        gt_objs_masks, valid_masks = self.sequence_from_masks(annotation)
        gt_objs_masks = self.transform.final_transform(gt_objs_masks)

        return image, gt_objs_masks, valid_masks

    def get_data(self):
        # In test mode we go frame by frame
        if len(self.clip_images_files) == 1:
            image, gt_objs_masks, valid_masks = self.load_frame(self.clip_images_files[0], self.clip_annot_files[0])
            return {"images": image, "objs_masks": gt_objs_masks, "valid_masks": valid_masks, "sequence": self.sequence_name, "frame_name": self.clip_images_files[0]}

        else:
            clip_images = []
            clip_gts_objs_masks = []
            clip_valid_masks = []

            for image_file, annot_file in zip(self.clip_images_files, self.clip_annot_files):
                image, gt_objs_masks, valid_masks = self.load_frame(image_file, annot_file)

                clip_images.append(image)
                clip_gts_objs_masks.append(gt_objs_masks)
                clip_valid_masks.append(valid_masks)

            return {"images": clip_images, "objs_masks": clip_gts_objs_masks, "valid_masks": clip_valid_masks, "sequence": self.sequence_name}

    def sequence_from_masks(self, annot):
        """
        Reads segmentation masks and outputs sequence of binary masks and labels
        """

        h, w = annot.shape[0], annot.shape[1]

        total_num_instances = len(self.instances_id)
        max_instance_id = 0
        if total_num_instances > 0:
            max_instance_id = int(np.max(self.instances_id))

        num_instances = max(self.max_num_gt_objects, max_instance_id)

        gt_objs_masks = np.zeros((h, w, num_instances), dtype=np.float32)
        valid_masks = np.zeros((num_instances, 1), dtype=np.bool_)
        # for sorting by size: no used in rvos
        # size_masks = np.zeros((num_instances,))

        for i in range(total_num_instances):
            id_instance = int(self.instances_id[i])
            mask_positions = annot == id_instance
            # Note: we need to check that in that clip the instance_id appears. Note that instance_ids refers to all sequence, and here we
            # are working with a small clip.
            # if not np.any(mask_positions):
            #     continue
            aux_mask = np.zeros((h, w), dtype=np.float32)
            aux_mask[mask_positions] = 1
            gt_objs_masks[:, :, id_instance - 1] = aux_mask
            # Not used on rvos
            # size_masks[id_instance - 1] = np.sum(gt_seg[id_instance - 1, :])
            valid_masks[id_instance - 1] = True

        gt_objs_masks = gt_objs_masks[..., :self.max_num_gt_objects]
        valid_masks = valid_masks[..., :self.max_num_gt_objects].squeeze()

        # Goes from (MAX_NUM_GT_OBJ, W*H) -> (MAX_NUM_GT_OBJ, W*H+1) where we add this last vector shape (MAX_NUM_GT_OBJ, 1) that indicates the masks that really contain objects)

        return gt_objs_masks, valid_masks

    # Used when working with Test, as we have a dataloader for each sequence, which will be represented with a clip
    def __getitem__(self, idx):
        image_file, annot_file = self.clip_images_files[idx], self.clip_annot_files[idx]
        image, gt_objs_masks, valid_masks = self.load_frame(image_file, annot_file)
        return {"images": image, "objs_masks": gt_objs_masks, "valid_masks": valid_masks, "sequence": self.sequence_name}


