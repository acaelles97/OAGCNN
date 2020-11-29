import os.path as osp
from oagcnn.data.dataset import YouTubeVOS
from oagcnn.data.lmdb_generator import LMDBGenerator
from oagcnn.config.defaults import cfg
from torch.utils.data import DataLoader

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def _create_dataset(name, split):
        if name == "Davis":
            year = cfg[name]["YEAR"]
            dataset = DavisFactory.create_dataset("", split, year)

        elif name == "YouTubeVOS":
            dataset = YouTubeVOSFactory.create_dataset(split)

        else:
            raise ValueError("Not supported Dataset")

        return dataset

    @staticmethod
    def create_data_loader(name, split):
        dataset = DatasetFactory._create_dataset(name, split)

        batch_size = cfg.DATA.BATCH_SIZE if dataset.is_train else cfg.DATA.VAL_BATCH_SIZE
        shuffle = dataset.is_train
        drop_last = cfg.DATA.DROP_LAST if dataset.is_train else False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA.NUM_WORKERS,
            drop_last=drop_last,
        )
        return dataloader

class DavisFactory:

    @staticmethod
    def _get_metadata_path(config_yaml):
        return config_yaml["Davis"]["DB_INFO"]

    @staticmethod
    def get_images_dir(base_dir, config_yaml):
        return osp.join(base_dir, config_yaml["Davis"]["IMAGES_FOLDER"])

    @staticmethod
    def get_annotations_dir(base_dir, config_yaml):
        return osp.join(base_dir, config_yaml["Davis"]["ANNOTATIONS_FOLDER"])

    @staticmethod
    def create_dataset(config_yaml, split, year):
        base_dir = config_yaml["YouTubeVOS"]["BASE_PATH"]
        metadata_path = DavisFactory._get_metadata_path(config_yaml)
        images_dir = DavisFactory.get_images_dir(base_dir, config_yaml)
        annotations_dir = DavisFactory.get_annotations_dir(base_dir, config_yaml)


# Responsible of creating Davis Dataset
class YouTubeVOSFactory:
    _AVAILABLE_SETS = {"train", "val", "train_val", "test"}

    @staticmethod
    def get_images_dir(base_dir, split):
        folder_split = YouTubeVOSFactory._get_split_to_folder(split)
        return osp.join(base_dir, folder_split, cfg["YouTubeVOS"]["IMAGES_FOLDER"])

    @staticmethod
    def get_annotations_dir(base_dir, split):
        folder_split = YouTubeVOSFactory._get_split_to_folder(split)
        return osp.join(base_dir, folder_split, cfg["YouTubeVOS"]["ANNOTATIONS_FOLDER"])

    @staticmethod
    def get_additional_partitions_folder():
        return cfg.YouTubeVOS.ADDITIONAL_PARTITIONS_FOLDER

    @staticmethod
    def _get_split_to_folder(split):
        if split == 'train':
            split_file = 'train'
        elif split == 'val':
            split_file = 'train'
        elif split == 'train_val':
            split_file = 'train'
        elif split == 'test':
            split_file = 'valid'
        else:
            raise ValueError('Set not available use {}'.format(YouTubeVOSFactory._AVAILABLE_SETS))
        return split_file

    @staticmethod
    def _get_metadata_path(split, base_dir):
        folder_split = YouTubeVOSFactory._get_split_to_folder(split)

        if split == "train":
            folder_split = YouTubeVOSFactory.get_additional_partitions_folder()
            meta_info_path = osp.abspath(osp.join(base_dir, folder_split, "train-train-meta.json"))
        elif split == "val":
            folder_split = YouTubeVOSFactory.get_additional_partitions_folder()
            meta_info_path = osp.abspath(osp.join(base_dir, folder_split, "train-val-meta.json"))

        elif split == "train_val":
            meta_info_path = osp.abspath(osp.join(base_dir, folder_split, "meta.json"))
        elif split == "test":
            meta_info_path = osp.abspath(osp.join(base_dir, folder_split, "meta.json"))
        else:
            raise ValueError("Split selected not in allowed: {}".format(split))

        if not osp.exists(meta_info_path):
            raise FileNotFoundError("Selected meta info file does not exist: {}".format(meta_info_path))

        return meta_info_path

    @staticmethod
    def get_lmdb_path(dataset, lmdb_type):
        filename = LMDBGenerator.get_lmdb_read_path(dataset, lmdb_type)
        lmdb_dir = cfg["PATH"]["LMDB"]
        lmdb_path = osp.join(lmdb_dir, filename)
        if not osp.exists(lmdb_path):
            raise FileNotFoundError("LMDB not found: {}".format(lmdb_path))

        return lmdb_path

    @staticmethod
    def get_lmdb_images_dir(dataset, split):
        if split != "test":
            lmdb_type = "train_sequences"
        else:
            lmdb_type = "test_sequences"

        return YouTubeVOSFactory.get_lmdb_path(dataset, lmdb_type)

    @staticmethod
    def get_lmdb_annotations_dir(dataset, split):
        if split != "test":
            lmdb_type = "train_annotations"
        else:
            lmdb_type = "test_annotations"

        return YouTubeVOSFactory.get_lmdb_path(dataset, lmdb_type)

    @staticmethod
    def create_dataset(split):
        if split not in YouTubeVOSFactory._AVAILABLE_SETS:
            raise ValueError("Split selected {} is not available for YouTube".format(split))

        is_train = split in ["train", "train_val"]
        base_dir = cfg["YouTubeVOS"]["BASE_PATH"]
        meta_info_path = YouTubeVOSFactory._get_metadata_path(split, base_dir)
        images_dir = YouTubeVOSFactory.get_images_dir(base_dir, split)
        annotations_dir = YouTubeVOSFactory.get_annotations_dir(base_dir, split)
        lmdb_images_file = YouTubeVOSFactory.get_lmdb_images_dir("YouTubeVOS", split)
        lmdb_annotations_file = YouTubeVOSFactory.get_lmdb_annotations_dir("YouTubeVOS", split)

        youtube_dataset = YouTubeVOS(images_dir=images_dir, annotations_dir=annotations_dir, metadata_path=meta_info_path,
                                     lmdb_images_file=lmdb_images_file, lmdb_annotations_file=lmdb_annotations_file, is_train=is_train)
        return youtube_dataset
