import os.path as osp
import json
from .lmdb_generator import LMDBGenerator
from collections import OrderedDict
from .dataset import MVOSDataset


# Responsible of creating Davis Dataset
class YouTubeVOSDataset(MVOSDataset):
    _AVAILABLE_SETS = {"train", "val", "train_val", "test"}

    def __init__(self, split, cfg):
        if split not in YouTubeVOSDataset._AVAILABLE_SETS:
            raise ValueError("Split selected {} is not available for YouTube".format(split))

        self._base_dir = cfg.YouTubeVOS.BASE_PATH
        self._split = split
        self._init_split_to_folder_data()
        additional_partitions_folder = cfg.YouTubeVOS.ADDITIONAL_PARTITIONS_FOLDER
        self._init_split_to_folder_metadata(additional_partitions_folder)
        self._init_metadata_path()

        name_folder_images = cfg["YouTubeVOS"]["IMAGES_FOLDER"]
        name_folder_annots = cfg["YouTubeVOS"]["ANNOTATIONS_FOLDER"]
        images_dir = self._get_images_dir(name_folder_images)
        annotations_dir = self._get_annots_dir(name_folder_annots)

        lmdb_dir = cfg["PATH"]["LMDB"]
        lmdb_images_file = self._get_lmdb_images_dir("YouTubeVOS", lmdb_dir)
        lmdb_annotations_file = self._get_lmdb_annotations_dir("YouTubeVOS", lmdb_dir)
        metadata = self._get_metadata()
        is_train = split in ["train", "train_val"]

        super().__init__(images_dir, annotations_dir, lmdb_images_file, lmdb_annotations_file, metadata, is_train, cfg)

    def _init_split_to_folder_data(self):
        if self._split == 'train':
            split_folder = 'train'
        elif self._split == 'val':
            split_folder = 'train'
        elif self._split == 'train_val':
            split_folder = 'train'
        elif self._split == 'test':
            split_folder = 'valid'
        else:
            raise ValueError('Set not available use {}'.format(YouTubeVOSDataset._AVAILABLE_SETS))
        self._folder_split_data = split_folder

    def _init_split_to_folder_metadata(self, additional_partitions_folder):
        if self._split == 'train':
            split_folder = additional_partitions_folder
        elif self._split == 'val':
            split_folder = additional_partitions_folder
        elif self._split == 'train_val':
            split_folder = 'train'
        elif self._split == 'test':
            split_folder = 'valid'
        else:
            raise ValueError('Set not available use {}'.format(YouTubeVOSDataset._AVAILABLE_SETS))
        self._folder_split_metadata = split_folder

    def _init_metadata_path(self):
        if self._split == "train":
            meta_info_path = osp.abspath(osp.join(self._base_dir, self._folder_split_metadata, "train-train-meta.json"))
        elif self._split == "val":
            meta_info_path = osp.abspath(osp.join(self._base_dir, self._folder_split_metadata, "train-val-meta.json"))
        elif self._split == "train_val":
            meta_info_path = osp.abspath(osp.join(self._base_dir, self._folder_split_metadata, "meta.json"))
        elif self._split == "test":
            meta_info_path = osp.abspath(osp.join(self._base_dir, self._folder_split_metadata, "meta.json"))
        else:
            raise ValueError("Split selected not in allowed: {}".format(self._split))
        if not osp.exists(meta_info_path):
            raise FileNotFoundError("Selected meta info file does not exist: {}".format(meta_info_path))

        self._metadata_path = meta_info_path

    def _get_images_dir(self, name_folder_images):
        return osp.join(self._base_dir, self._folder_split_data, name_folder_images)

    def _get_annots_dir(self, name_folder_annots):
        return osp.join(self._base_dir, self._folder_split_data, name_folder_annots)

    @staticmethod
    def get_lmdb_path(dataset, lmdb_dir, lmdb_type):
        filename = LMDBGenerator.get_lmdb_read_path(dataset, lmdb_type)
        lmdb_path = osp.join(lmdb_dir, filename)
        if not osp.exists(lmdb_path):
            raise FileNotFoundError("LMDB not found: {}".format(lmdb_path))

        return lmdb_path

    def _get_lmdb_images_dir(self, dataset, lmdb_dir):
        if self._split != "test":
            lmdb_type = "train_sequences"
        else:
            lmdb_type = "test_sequences"

        return self.get_lmdb_path(dataset, lmdb_dir, lmdb_type)

    def _get_lmdb_annotations_dir(self, dataset, lmdb_dir):
        if self._split != "test":
            lmdb_type = "train_annotations"
        else:
            lmdb_type = "test_annotations"
        return self.get_lmdb_path(dataset, lmdb_dir, lmdb_type)

    def _get_metadata(self):
        metadata_dict = {}

        with open(self._metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check if any video has no annotations on the first frame
        for sequence in metadata["videos"].keys():
            metadata_dict[sequence] = OrderedDict()

            for instance_id, instance_info in metadata['videos'][sequence]['objects'].items():
                list_frames = sorted(instance_info["frames"])
                category = instance_info["category"]

                for frame in sorted(list_frames):
                    frame_info = {"instance_id": instance_id, "category": category}
                    if frame not in metadata_dict[sequence].keys():
                        metadata_dict[sequence][frame] = []
                    metadata_dict[sequence][frame].append(frame_info)

        return metadata_dict