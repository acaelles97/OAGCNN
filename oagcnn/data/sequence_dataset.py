import torch.utils.data as data
from .clip_dataset import ClipDataset

# GIN configurable -> clip_length
class SequenceDataset(data.Dataset):

    def __init__(self, images_files, annotations_files, clip_transform, instances_id, sequence_name, clip_length):

        self.images_files = images_files
        self.annotations_files = annotations_files
        self.instances_id = instances_id
        self.transform = clip_transform
        self.name = sequence_name
        self._data = []
        self._clip_length = clip_length
        self._create_clips_train()

    def _create_test_sequence(self):
        assert len(self.images_files) == len(self.annotations_files)
        for image_file, annotation_file in zip(self.images_files, self.annotations_files):
            clip_dataset = ClipDataset([image_file], [annotation_file], self.transform, self.instances_id, self.name)
            self._data.append(clip_dataset)


    def _create_clips_train(self):
        # assert that number of annotations correspond with number of images
        assert len(self.images_files) == len(self.annotations_files)

        starting_frame_idx = 0
        num_frames = len(self.images_files)
        num_clips = int(num_frames / self._clip_length)

        for idx in range(num_clips):
            # We need to get a slice of
            clip_images_files = self.images_files[starting_frame_idx:(starting_frame_idx + self._clip_length)]
            clip_annot_files = self.annotations_files[starting_frame_idx:(starting_frame_idx + self._clip_length)]
            starting_frame_idx += self._clip_length
            clip_dataset = ClipDataset(clip_images_files, clip_annot_files, self.transform, self.instances_id, self.name)
            self._data.append(clip_dataset)


    def get_data(self):
        return self._data
