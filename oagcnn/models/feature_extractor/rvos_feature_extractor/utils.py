import torch.nn.functional as F


class InterpolateLayer(object):

    def __init__(self, out_spatial_res):
        self.out_spatial_res = out_spatial_res

    def __call__(self, x):
        return F.interpolate(x, size=self.out_spatial_res, mode='bilinear', align_corners=True)
