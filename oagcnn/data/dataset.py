import os.path as osp
import torch.utils.data as data
import json
from .transforms.transforms import CustomComposeTransform
import lmdb
from .sequence_dataset import SequenceDataset
from .clip_dataset import ClipDataset
from abc import abstractmethod
from oagcnn.config.defaults import cfg


# import gin.config


# @gin.configurable
# To configure Augment
class MVOSDataset(data.Dataset):
    def __init__(self,
                 images_dir,
                 annotations_dir,
                 lmdb_env_images_dir,
                 lmdb_env_annotations_dir,
                 is_train=False,
                 ):

        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.is_train = is_train

        self.lmdb_env_images = None
        self.lmdb_env_annotations = None
        self._metadata = None
        self.idx_to_sequence_name = {}
        self._data = []

        self._init_dataset_type()
        self.load_metadata()
        self.load_lmdb_env(lmdb_env_images_dir, lmdb_env_annotations_dir)
        self.create_data()

    def _init_dataset_type(self):
        if self.is_train:
            self.clip_length = cfg.DATA.CLIP_LENGTH
            self.augment = cfg.DATA.AUGMENTATION
        else:
            self.clip_length = 1
            self.augment = False

    def load_metadata(self):
        metadata = self._read_metadata()
        self._metadata = metadata

    @abstractmethod
    def _read_metadata(self):
        pass

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

    def get_sequence_data(self, sequence):
        images_files = self._get_files_sequence(self.images_dir, sequence, self.lmdb_env_images)
        annotations_files = self._get_files_sequence(self.annotations_dir, sequence, self.lmdb_env_annotations)
        instances_id = self._metadata[sequence]
        return images_files, annotations_files, instances_id

    def create_data_for_test(self):
        sequences = []
        for sequence in self._metadata.keys():
            images_files, annotations_files, instances_id = self.get_sequence_data(sequence)
            sequence_transform = CustomComposeTransform(self.augment)
            sequence = ClipDataset(images_files, annotations_files, sequence_transform, instances_id, sequence)

            sequences.append(sequence)

        return sequences


    def create_data(self):
        counter = 0
        for sequence in self._metadata.keys():
            # Get encoded files for that sequence
            images_files, annotations_files, instances_id = self.get_sequence_data(sequence)
            sequence_transform = CustomComposeTransform(self.augment)
            sequence_dataset = SequenceDataset(images_files, annotations_files, sequence_transform, instances_id, sequence, self.clip_length)
            new_data = sequence_dataset.get_data()
            self._data.extend(sequence_dataset.get_data())

            num_clips = len(new_data)
            self.idx_to_sequence_name.update({idx: sequence for idx in range(counter, counter + num_clips)})
            counter += num_clips

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index].get_data()




# Necessito dir on mirar sequences / dir on mirar annotations / metadata amb key sequence
# Info basifca de treure de youtube -> Com llegir el metadata i la informació que trec de ell
class YouTubeVOS(MVOSDataset):
    def __init__(self, images_dir, annotations_dir, metadata_path, lmdb_images_file, lmdb_annotations_file, is_train):
        self._metadata_path = metadata_path
        super().__init__(images_dir, annotations_dir, lmdb_images_file, lmdb_annotations_file, is_train)



    # For each video -> Annotations, Sequences and metadata as dict of key sequence  / value info
    # We needs to check starting frame of each sequence this is shit
    def _read_metadata(self):
        metadata_dict = {}

        with open(self._metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check if any video has no annotations on the first frame
        for sequence in metadata["videos"].keys():
            instance_ids_str = metadata['videos'][sequence]['objects'].keys()
            instance_ids = [int(instance_id) for instance_id in instance_ids_str]
            # Inits instance_ids that we need to be aware of for each sequence
            metadata_dict[sequence] = instance_ids

        return metadata_dict

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
