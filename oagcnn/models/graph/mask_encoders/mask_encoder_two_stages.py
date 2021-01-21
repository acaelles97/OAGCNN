import torch.nn as nn
import torch.nn.functional as F
import torch


class EncodingBlock(nn.Module):
    def __init__(self, input_ch, out_ch):
        super(EncodingBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        # self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class MaskEncoderOneStage(nn.Module):
    def __init__(self, input_ch, cfg):
        super(MaskEncoderOneStage, self).__init__()

        self.out_channels = cfg.MaskEncoderOneStage.OUT_CHANNELS

        self.encoding_block1 = EncodingBlock(input_ch+1, self.out_channels)

    def forward(self, feats, masks_to_concat, num_obj):
        out_node_states = []
        masks_to_concat = F.interpolate(masks_to_concat.unsqueeze(0), size=feats.shape[-2:])
        for obj_idx in range(num_obj):
            x = torch.cat((feats, masks_to_concat[:, obj_idx, ...]), dim=0).unsqueeze(0)
            out_node_states.append(self.encoding_block1(x))

        return out_node_states


class MaskEncoderTwoStages(nn.Module):
    def __init__(self, input_ch, cfg):
        super(MaskEncoderTwoStages, self).__init__()

        self.out_channels = cfg.MaskEncoderTwoStagesOUT_CHANNELS
        self.use_skip_mask_connection = cfg.MaskEncoderTwoStages.USE_SKIP_MASK_CONNECTION
        self.int_channels = cfg.MaskEncoderTwoStages.INT_CHANNELS
        dropout_prob = cfg.MaskEncoderTwoStages.DROPOUT

        self.encoding_block1 = EncodingBlock(input_ch + 1, self.int_channels)
        self.dropout = nn.Dropout(dropout_prob)

        if self.use_skip_mask_connection:
            self.encoding_block2 = EncodingBlock(self.int_channels + 1, self.out_channels)
        else:
            self.encoding_block2 = EncodingBlock(self.int_channels, self.out_channels)

        # self.dropout = nn.Dropout(config["DROPOUT"])
        # self.encoding_block3 = EncodingBlock(config["INT_CHANNELS"][1], self.out_channels)

    def forward(self, feats, masks_to_concat, num_obj):
        out_node_states = []
        masks_spatial1 = F.interpolate(masks_to_concat.unsqueeze(0), size=feats.shape[-2:])
        for obj_idx in range(num_obj):
            obj_mask = masks_spatial1[:, obj_idx, ...]
            x = torch.cat((feats, obj_mask), dim=0).unsqueeze(0)
            x = self.dropout(self.encoding_block1(x))
            if self.use_skip_mask_connection:
                obj_mask_spatial2 = F.interpolate(obj_mask.unsqueeze(0), size=x.shape[-2:])
                x = torch.cat((x, obj_mask_spatial2), dim=1)
            x = self.encoding_block2(x)
            out_node_states.append(x)

        return out_node_states
