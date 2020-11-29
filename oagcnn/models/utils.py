import torch
import os
import pickle

def get_rvos_pretrained_path(cfg):
    if (cfg.DATASETS.TRAIN or cfg.DATASETS.TEST) == "YouTubeVOS":
        return os.path.join(cfg.PATH.MODELS_RVOS_YOUTUBE)
    elif (cfg.DATASETS.TRAIN or cfg.DATASETS.TEST) == "Davis":
        return os.path.join(cfg.PATH.MODELS_RVOS_YOUTUBE)
    else:
        ValueError("Please select a valid Train/Test Dataset. Current train: {} Current test: {}".format(cfg.DATASETS.TRAIN, cfg.DATASETS.TEST))

def load_rvos_pretrained(cfg):
    base_path = get_rvos_pretrained_path(cfg)
    encoder_dict = torch.load(os.path.join(base_path, 'encoder.pt'))

    return encoder_dict
