import torch.nn as nn
from oagcnn.models.feature_extractor.rvos_feature_extractor.backbone_factory import BackboneFactory
import torch
from oagcnn.models.utils import load_rvos_pretrained
from GPUtil import showUtilization as gpu_usage
from .utils import InterpolateLayer


class RVOSEncoder(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''

    def __init__(self, cfg):
        super(RVOSEncoder, self).__init__()

        backbone_name = cfg.RVOSFeatureExtractor.RVOS_ENCODER.BACKBONE
        self.kernel_size = cfg.RVOSFeatureExtractor.RVOS_ENCODER.KERNEL_SIZE
        hidden_size = int(cfg.RVOSFeatureExtractor.RVOS_ENCODER.HIDDEN_SIZE)
        load_rvos_encoder = cfg.RVOSFeatureExtractor.RVOS_ENCODER.LOAD_PRETRAINED

        self.base = BackboneFactory.create_backbone(backbone_name)

        skip_dims_in = self.base.skip_dims_in
        self.hidden_size_dims = [hidden_size, hidden_size, int(hidden_size / 2), int(hidden_size / 4)]
        self.padding = 0 if self.kernel_size == 1 else 1

        self.sk5 = nn.Conv2d(skip_dims_in[0], self.hidden_size_dims[0], self.kernel_size, padding=self.padding)
        self.sk4 = nn.Conv2d(skip_dims_in[1], self.hidden_size_dims[1], self.kernel_size, padding=self.padding)
        self.sk3 = nn.Conv2d(skip_dims_in[2], self.hidden_size_dims[2], self.kernel_size, padding=self.padding)
        self.sk2 = nn.Conv2d(skip_dims_in[3], self.hidden_size_dims[3], self.kernel_size, padding=self.padding)

        self.bn5 = nn.BatchNorm2d(self.hidden_size_dims[0])
        self.bn4 = nn.BatchNorm2d(self.hidden_size_dims[1])
        self.bn3 = nn.BatchNorm2d(self.hidden_size_dims[2])
        self.bn2 = nn.BatchNorm2d(self.hidden_size_dims[3])

        if load_rvos_encoder:
            assert backbone_name == "ResNet101" and self.hidden_size_dims[0] == 128, "Pretrained RVOS parameters are from ResNet101, check " \
                                                                                     "your backbone! "
            self.load_from_rvos(cfg)

    def forward(self, x):
        x5, x4, x3, x2, x1 = self.base(x)

        # X1: (BATCH_SIZE, 64, 128, 224)
        # X2: (BATCH_SIZE, 64, 64, 112)
        # X3: (BATCH_SIZE, 128, 32, 56)
        # X4: (BATCH_SIZE, 256, 16, 28)
        # X5: (BATCH_SIZE, 512, 8, 14)

        # _C.DATA.TRANSFORMS.IMAGE_SIZE = (256, 448)
        # (128, 8, 14) ->
        x5_skip = self.bn5(self.sk5(x5))
        # (128, 16, 28)
        x4_skip = self.bn4(self.sk4(x4))
        # (64, 32, 56)
        x3_skip = self.bn3(self.sk3(x3))
        # (32, 64, 112)
        x2_skip = self.bn2(self.sk2(x2))

        return x5_skip, x4_skip, x3_skip, x2_skip

    def load_from_rvos(self, cfg):
        print("Loading pretrained encoder weights from RVOS")
        encoder_dict = load_rvos_pretrained(cfg)
        encoder_dict.pop("base.fc.weight")
        encoder_dict.pop("base.fc.bias")
        self.load_state_dict(encoder_dict)


# Reduces channels dims if needed and converts all to single tensor
class RVOSEncoderAdapter(nn.Module):

    def __init__(self, cfg, input_channels, out_spatial_size):
        super(RVOSEncoderAdapter, self).__init__()

        self.used_connections = cfg.RVOSFeatureExtractor.RVOS_ADAPTER.USED_FEATURES
        self.channel_factor = cfg.RVOSFeatureExtractor.RVOS_ADAPTER.CHANNELS_FACTOR_REDUCTION

        self.in_connections = ["x5", "x4", "x3", "x2"]
        self.input_channels = input_channels
        self.interpolate_layer = InterpolateLayer(out_spatial_size)

        self._init_channels_dims()

        if self.channel_factor != 0:
            reduced_channels = [int(channel_dim / self.channel_factor) for channel_dim in self.intermediate_channel_dims]
            self.intermediate_channel_dims = reduced_channels
            self._init_conv_layers()

    def _init_conv_layers(self):
        for connection in self.used_connections:
            idx = self.in_connections.index(connection)
            in_channels = self.input_channels[idx]
            out_channels = self.intermediate_channel_dims[idx]
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False)
            batch_norm = nn.BatchNorm2d(out_channels)
            setattr(self, connection + "_conv", conv_layer)
            setattr(self, connection + "_bn", batch_norm)

    def _init_channels_dims(self):
        self.intermediate_channel_dims = [0 for _ in self.input_channels]
        for connection in self.used_connections:
            idx = self.in_connections.index(connection)
            self.intermediate_channel_dims[idx] = self.input_channels[idx]

    def forward(self, x):
        # x5_skip, x4_skip, x3_skip, x2_skip
        out_tensor = None
        for connection in self.used_connections:
            idx = self.in_connections.index(connection)

            if self.channel_factor != 1:
                conv_layer = getattr(self, connection + "_conv")
                bn_layer = getattr(self, connection + "_bn")
                out_skip = bn_layer(conv_layer(x[idx]))

            else:
                out_skip = x[idx]

            if out_tensor is None:
                out_tensor = self.interpolate_layer(out_skip)
            else:
                out_tensor = torch.cat((out_tensor, self.interpolate_layer(out_skip)), dim=1)

        return out_tensor


class RVOSEncoderHead(nn.Module):

    def __init__(self, cfg, input_channels, out_channels):
        super(RVOSEncoderHead, self).__init__()

        hidden_size = cfg.RVOSFeatureExtractor.RVOS_HEAD.HIDDEN_SIZE
        dropout = cfg.RVOSFeatureExtractor.RVOS_HEAD.DROPOUT

        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(hidden_size, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return self.conv3(x)


class RVOSFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(RVOSFeatureExtractor, self).__init__()

        # Configuration attributes
        input_dims = cfg.DATA.IMAGE_SIZE
        out_spatial_ratio = cfg.RVOSFeatureExtractor.RVOS_ADAPTER.SPATIAL_SCALE_FACTOR
        self.out_spatial_res = (int(input_dims[0] / out_spatial_ratio), int(input_dims[1] / out_spatial_ratio))
        self.out_channels = cfg.RVOSFeatureExtractor.RVOS_HEAD.OUT_CHANNELS

        # Learnable parameters
        self.encoder = RVOSEncoder(cfg)
        self.adapter = RVOSEncoderAdapter(cfg, self.encoder.hidden_size_dims, self.out_spatial_res)
        head_input_channels = sum(self.adapter.intermediate_channel_dims)
        self.out_head = RVOSEncoderHead(cfg, head_input_channels, self.out_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.adapter(x)
        x = self.out_head(x)
        return x

    def freeze_rvos_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_rvos_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def get_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())
