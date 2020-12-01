import os.path as osp
from .youtube_vos_dataset import YouTubeVOSDataset
from oagcnn.data.lmdb_generator import LMDBGenerator
from torch.utils.data import DataLoader


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def _create_dataset(name, split, cfg):
        if name == "Davis":
            year = cfg[name]["YEAR"]
            dataset = DavisFactory.create_dataset("", split, year)

        elif name == "YouTubeVOS":
            dataset = YouTubeVOSDataset(split, cfg)

        else:
            raise ValueError("Not supported Dataset")

        return dataset

    @staticmethod
    def create_data_loader(name, split, cfg):
        dataset = DatasetFactory._create_dataset(name, split, cfg)

        if not dataset.is_train:
            dataloaders = []
            sequences = dataset.get_sequences_for_test()
            for sequence in sequences:
                sequence_dataloader = DataLoader(
                    sequence,
                    batch_size=cfg.DATA.VAL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=cfg.DATA.NUM_WORKERS,
                    drop_last=False,
                )
                dataloaders.append(sequence_dataloader)

            return dataloaders
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.DATA.BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.DATA.NUM_WORKERS,
                drop_last=cfg.DATA.DROP_LAST,
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