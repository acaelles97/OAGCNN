import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def extract_bboxes(mask: torch.Tensor, dilate=0):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """

    horizontal_indicies = torch.where(torch.any(mask, dim=0))[0]
    vertical_indicies = torch.where(torch.any(mask, dim=1))[0]

    if horizontal_indicies.shape[0]:

        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]

        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1

    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0

    out_masks = torch.tensor([[x1-dilate, y1-dilate, x2+dilate, y2+dilate]])
    return out_masks


class RoIAlignEncoder:
    def __init__(self, input_channels, cfg):
        self.out_channels = input_channels

        self.dilate = cfg.RoIAlignEncoder.DILATE
        self.spatial_size = cfg.RoIAlignEncoder.SPATIAL_SIZE

    def __call__(self, feats, masks_to_concat, num_obj):
        out_instace_features = []
        masks_to_concat = F.interpolate(masks_to_concat.unsqueeze(0), size=feats.shape[-2:])
        for obj_idx in range(num_obj):
            obj_mask = masks_to_concat[:, obj_idx, ...]
            bbx = extract_bboxes(obj_mask, self.dilate)
            instance_features = roi_align(feats, bbx, self.spatial_size)
            out_instace_features.append(instance_features)

        return [torch.cat((feats, masks_to_concat[:, obj_idx, ...]), dim=0).unsqueeze(0) for obj_idx in range(num_obj)]


