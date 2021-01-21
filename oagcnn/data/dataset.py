import os.path as osp
import torch.utils.data as data
from .transforms.transforms import CustomComposeTransform
import lmdb
from .clip_dataset import ClipDataset


class MVOSDataset(data.Dataset):
    def __init__(self,
                 images_dir,
                 annotations_dir,
                 lmdb_env_images_dir,
                 lmdb_env_annotations_dir,
                 metadata,
                 is_train,
                 cfg):
        # TODO Configurable parameters

        # TODO Parameters to be set by the child
        self.images_dir = None

        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.is_train = is_train
        self._metadata = metadata
        self.lmdb_env_images = None
        self.lmdb_env_annotations = None
        self.data = []

        self._init_dataset_config(cfg)
        self.load_lmdb_env(lmdb_env_images_dir, lmdb_env_annotations_dir)
        self.create_data()

    def _init_dataset_config(self, cfg):
        self.max_num_gt_objects = cfg.DATA.MAX_NUM_OBJ
        if self.is_train:
            self.clip_length = cfg.DATA.CLIP_LENGTH
            self.augment = cfg.DATA.AUGMENTATION
        else:
            self.clip_length = 1
            self.augment = False

    def load_lmdb_env(self, lmdb_env_images_dir, lmdb_env_annotations_dir):
        # First check lmdb existence.
        lmdb_env_images = None
        lmdb_env_annotations = None
        if osp.exists(lmdb_env_images_dir) and osp.exists(lmdb_env_annotations_dir):
            lmdb_env_images = lmdb.open(lmdb_env_images_dir)
            lmdb_env_annotations = lmdb.open(lmdb_env_annotations_dir)

        self.lmdb_env_images = lmdb_env_images
        self.lmdb_env_annotations = lmdb_env_annotations

    # LMDB expected to be encoded as SequencePath -> Files in the sequence
    def _get_files_sequence(self, base_path, sequence_name, lmdb_env):
        # Get encoded files for that sequence
        sequence_path = osp.join(base_path, sequence_name)
        key_db = osp.basename(sequence_path)
        with lmdb_env.begin() as txn:
            _files_vec = txn.get(key_db.encode()).decode().split('|')
            files = [bytes(osp.join(sequence_path, f).encode()) for f in _files_vec]
        return files

    @staticmethod
    def decode_filenames(list_filenames):
        decoded_filenames = []
        for filename in list_filenames:
            if type(filename) != str:
                decoded_filenames.append(str(filename.decode()))

        return decoded_filenames

    def get_sequence_data(self, sequence):
        images_files = self._get_files_sequence(self.images_dir, sequence, self.lmdb_env_images)
        annotations_files = self._get_files_sequence(self.annotations_dir, sequence, self.lmdb_env_annotations)

        images_files = self.decode_filenames(images_files)
        annotations_files = self.decode_filenames(annotations_files)

        sequence_metadata = self._metadata[sequence]
        return images_files, annotations_files, sequence_metadata

    def get_clip_metadata(self, clip_images_files, sequence_metadata):
        clip_metadata = {}
        for image_name in clip_images_files:
            image_id = image_name.split("/")[-1].split(".")[0]
            clip_metadata[image_id] = sequence_metadata.get(image_id)
        return clip_metadata

    def create_clips_from_sequence(self, images_files, annotations_files, sequence_metadata, sequence_name, transform):
        assert len(images_files) == len(annotations_files)

        starting_frame_idx = 0
        num_frames = len(images_files)
        num_clips = int(num_frames / self.clip_length)

        for idx in range(num_clips):
            # We need to get a slice of
            clip_images_files = images_files[starting_frame_idx:(starting_frame_idx + self.clip_length)]
            clip_annot_files = annotations_files[starting_frame_idx:(starting_frame_idx + self.clip_length)]
            clip_metadata = self.get_clip_metadata(clip_images_files, sequence_metadata)

            if list(clip_metadata.values())[0] is None or len(list(clip_metadata.values())[0]) == 0:
                starting_frame_idx += self.clip_length
                continue
            starting_frame_idx += self.clip_length
            clip_dataset = ClipDataset(clip_images_files, clip_annot_files, clip_metadata, sequence_name, transform, self.max_num_gt_objects)
            self.data.append(clip_dataset)

    def create_data(self):
        for sequence_name in self._metadata.keys():
            # Get encoded files for that sequence
            images_files, annotations_files, sequence_metadata = self.get_sequence_data(sequence_name)
            transform = CustomComposeTransform(self.augment)
            if self.is_train:
                self.create_clips_from_sequence(images_files, annotations_files, sequence_metadata, sequence_name, transform)
            else:
                sequence = ClipDataset(images_files, annotations_files, sequence_metadata, sequence_name, transform, self.max_num_gt_objects)
                self.data.append(sequence)

    def get_sequences_for_test(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].get_data()




# Note we expect a pre-generated file from davis with all the info
# class Davis(MVOSDataset):
#
#     def __init__(self, year, split, lmdb_dir, config_yaml, images_dir, annotations_dir):
#         self._AVAILABLE_SETS = {"train", "val", "train_val", "test"}
#         self.split = split
#         self.year = year
#         self.config_yaml = config_yaml
#         self.db_sequences_path = db_sequences
#         super().__init__(images_dir, annotations_dir, lmdb_dir)
#
#
#     # Read all sequences from the generated all_db_info.yaml file
#     def _db_read_sequences(self):
#         """ Read list of sequences. """
#
#         with open(, 'r') as f:
#             all_db_sequences_info = yaml.load(f)
#
#         sequences = all_db_sequences_info["sequences"]
#
#         if self.year is not None:
#             sequences = filter(
#                 lambda s: int(sequences["year"]) <= int(self.year), sequences)
#
#         if self.split == "train_val":
#             sequences = filter(
#                 lambda s: ((s["set"] == "val") or (s["set"] == "train")), sequences)
#         else:
#             sequences = filter(
#                 lambda s: s["set"] == self.split, sequences)
#         return sequences
#
#
#     # For each video -> Annotations, Sequences and metadata as dict of key sequence  / value info
#     # We needs to check starting frame of each sequence this is shit
#     def _read_metadata(self):
#         metadata_dict = {}
#         # Note: Usally we read the sequences from ImageSets, but RVOS uses the generated .yaml file with all the info so we will copy the approach
#         # subset_path = os.path.join(self.base_dir, "ImageSets", self.name, self.split + '.txt')
#         # with open(subset_path, 'r') as f:
#         #     sequences_in_split = f.readlines()
#         # sequences = [x.strip() for x in sequences_in_split]
#
#         sequences = self._db_read_sequences()
#         # Sequences on the desired split
#         for sequence in sequences:
#             if self.name == '2017':
#                 mask = np.array(Image.open(os.path.join(self.base_dir, self.annotations_dir, sequence, '00000.png')))
#                 mask[mask == 255] = 0
#                 unique_objs = set(np.unique(mask)) - {0}
#             else:
#                 unique_objs = {1}
#
#             metadata_dict[sequence] = list(unique_objs)
#
#         return sequences
