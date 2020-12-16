import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskCatEncoder:
    def __init__(self, input_channels, params=None):
        self.out_channels = input_channels + 1

    def __call__(self, feats, masks_to_concat, num_obj):
        masks_to_concat = F.interpolate(masks_to_concat.unsqueeze(0), size=feats.shape[-2:])
        return [torch.cat((feats, masks_to_concat[:, obj_idx, ...]), dim=0).unsqueeze(0) for obj_idx in range(num_obj)]
