import torch.utils.data as data
from PIL import Image
import numpy as np


class ClipDataset(data.Dataset):
    def __init__(self, clip_images_files, clip_annot_files, clip_metadata, sequence_name, transform, max_num_gt_objects):
        self.clip_images_files = clip_images_files
        self.clip_annot_files = clip_annot_files
        self.clip_metadata = clip_metadata
        self.transform = transform
        self.max_num_gt_objects = max_num_gt_objects
        self.sequence_name = sequence_name
        self.omitted_frames = None
        self._init_max_num_instances()
        self.correct_clip()

    def get_omitted_frames(self):
        return self.omitted_frames

    def get_sequence_name(self):
        return self.sequence_name

    def get_frame_name(self, filename):
        return filename.split("/")[-1].split(".")[0]

    def get_max_num_instances(self):
        return self.max_num_instances

    def _init_max_num_instances(self):
        max_num_instances = 0
        for instances_in_frame in list(self.clip_metadata.values()):
            max_num_instances = max(max_num_instances, len(instances_in_frame))
        self.max_num_instances = max_num_instances

    def correct_clip(self):

        # First make sure both start at the first frame instances appears
        existing_frames = [self.get_frame_name(image_file) for image_file in self.clip_images_files]
        existing_masks = [self.get_frame_name(annot_file) for annot_file in self.clip_annot_files]

        idx_clip_start_image = existing_frames.index(sorted(list(self.clip_metadata.keys()))[0])
        idx_clip_start_mask = existing_masks.index(sorted(list(self.clip_metadata.keys()))[0])

        # if idx_clip_start_image != 0:
        #     self.omitted_frames = existing_frames[:idx_clip_start_image]

        self.clip_images_files = self.clip_images_files[idx_clip_start_image:]
        self.clip_annot_files = self.clip_annot_files[idx_clip_start_mask:]

        # If len different we might have empty frames without any instance (should only occur at test time) and we want both to be same length
        # and control when a new GT mask enters the scene, so we need to redo the list filling with None.
        if len(self.clip_images_files) != len(self.clip_annot_files):
            existing_frames = [self.get_frame_name(image_file) for image_file in self.clip_images_files]
            existing_masks = [self.get_frame_name(annot_file) for annot_file in self.clip_annot_files]

            new_clip_annot_files = []
            starting_frame_idx = existing_frames.index(existing_masks[0])

            for idx in range(starting_frame_idx, len(existing_frames)):
                frame_name = existing_frames[idx]
                if frame_name in existing_masks:
                    idx = existing_masks.index(frame_name)
                    new_clip_annot_files.append(self.clip_annot_files[idx])
                else:
                    new_clip_annot_files.append(None)

            self.clip_annot_files = new_clip_annot_files


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

        id_frame = self.get_frame_name(image_file)
        gt_objs_masks, valid_masks, categories = self.sequence_from_masks(id_frame, annotation)
        gt_objs_masks = self.transform.final_transform(gt_objs_masks)

        return image, gt_objs_masks, valid_masks, categories

    def get_data(self):
        clip_images = []
        clip_gts_objs_masks = []
        clip_valid_masks = []
        clip_categories = []

        for image_file, annot_file in zip(self.clip_images_files, self.clip_annot_files):
            image, gt_objs_masks, valid_masks, categories = self.load_frame(image_file, annot_file)

            clip_images.append(image)
            clip_gts_objs_masks.append(gt_objs_masks)
            clip_valid_masks.append(valid_masks)
            clip_categories.append(categories)

        return {"images": clip_images, "objs_masks": clip_gts_objs_masks, "valid_masks": clip_valid_masks, "sequence": self.sequence_name}

    def generate_empty_masks(self, height, width):
        gt_objs_masks = np.zeros((height, width, self.max_num_gt_objects), dtype=np.float32)
        valid_masks = np.zeros((self.max_num_gt_objects, 1), dtype=np.bool_)
        categories = np.zeros((self.max_num_gt_objects, 1), dtype="<U25")

        return gt_objs_masks, valid_masks, categories

    def sequence_from_masks(self, id_frame, annot):
        """
        Reads segmentation masks and outputs sequence of binary masks and labels
        """

        h, w = annot.shape[0], annot.shape[1]
        gt_objs_masks, valid_masks, categories = self.generate_empty_masks(h, w)

        frame_instances = self.clip_metadata.get(id_frame)
        if frame_instances is None:
            return gt_objs_masks, valid_masks.squeeze(), categories

        total_num_instances = min(self.max_num_gt_objects, len(frame_instances))
        for instance_idx in range(total_num_instances):
            instance = frame_instances[instance_idx]
            category = instance["category"]
            instance_id = int(instance["instance_id"])
            instance_mask = annot == instance_id

            aux_mask = np.zeros((h, w), dtype=np.float32)
            aux_mask[instance_mask] = 1
            gt_objs_masks[:, :, instance_id - 1] = aux_mask
            valid_masks[instance_id - 1] = True
            categories[instance_id - 1] = category

            # size_masks[id_instance - 1] = np.sum(gt_seg[id_instance - 1, :])
        valid_masks = valid_masks.squeeze()

        return gt_objs_masks, valid_masks, categories

    def load_image_for_test(self, image_file):
        image = Image.open(image_file)
        image = self.transform.test_image_transform(image)
        return image

    def load_annotations_for_test(self, frame_name, annot_file, height, width):
        if annot_file is None:
            gt_objs_masks, valid_masks, categories = self.generate_empty_masks(height, width)
            valid_masks = valid_masks.squeeze()

        else:
            annotation = Image.open(annot_file)
            annotation = self.transform.initial_annot_test_transform(annotation)

            annotation = np.array(annotation, dtype=np.float32)
            annotation[annotation == 255] = 0

            gt_objs_masks, valid_masks, categories = self.sequence_from_masks(frame_name, annotation)

        gt_objs_masks = self.transform.final_transform(gt_objs_masks)

        return gt_objs_masks, valid_masks, categories

    def __len__(self):
        return len(self.clip_images_files)

    def __getitem__(self, idx):
        image_file = self.clip_images_files[idx]
        annot_file = self.clip_annot_files[idx]
        frame_name = self.get_frame_name(image_file)
        image = self.load_image_for_test(image_file)
        height, width = image.shape[1], image.shape[2]

        gt_objs_masks, valid_masks, categories = self.load_annotations_for_test(frame_name, annot_file, height, width)

        return {"image": image, "objs_masks": gt_objs_masks, "valid_masks": valid_masks, "sequence": self.sequence_name,
                "frame_name": frame_name}
