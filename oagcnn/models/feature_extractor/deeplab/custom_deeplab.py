import torch
from torch import nn
from torch.nn import functional as F
from .backbone import resnet
from .utils import IntermediateLayerGetter



__all__ = ["DeepLabV3Plus"]


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class CustomASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=None):
        super(CustomASPP, self).__init__()
        intermediate_ch = 256
        if out_channels is None:
            out_channels = intermediate_ch

        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, intermediate_ch, 1, bias=False),
            nn.BatchNorm2d(intermediate_ch),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, intermediate_ch, rate1))
        modules.append(ASPPConv(in_channels, intermediate_ch, rate2))
        modules.append(ASPPConv(in_channels, intermediate_ch, rate3))
        modules.append(ASPPPooling(in_channels, intermediate_ch))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * intermediate_ch, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg, image_size):
        super(DeepLabV3Plus, self).__init__()

        backbone_name = cfg.DeepLabV3PlusFeatExtract.BACKBONE_NAME
        output_stride = cfg.DeepLabV3PlusFeatExtract.OUTPUT_STRIDE
        self.out_channels = cfg.DeepLabV3PlusFeatExtract.OUT_CHANNELS

        self.inplanes = 2048
        self.low_level_planes = 256

        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]

        backbone = resnet.__dict__[backbone_name](
            pretrained=True,
            replace_stride_with_dilation=replace_stride_with_dilation)
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}

        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.project = nn.Sequential(
            nn.Conv2d(self.low_level_planes, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = CustomASPP(self.inplanes, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, self.out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

        if cfg.DeepLabV3PlusFeatExtract.LOAD_PRETRAINED:
            self.load_pretrained(cfg.DeepLabV3PlusFeatExtract.WEIGHTS_PATH)
        else:
            self._init_weight()

    def _init_weight(self):
        for m in self.project.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.aspp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def adapt_state_dict(self, weights_path):
        state_dict = torch.load(weights_path)["model_state"]
        new_dict = {}
        for key, value in state_dict.items():
            if "classifier" in key:
                new_key = ".".join(key.split(".")[1:])
                if "classifier" in new_key:
                    continue
                new_dict[new_key] = value
            else:
                new_dict[key] = value
        torch.save(new_dict, "custom_weights_deeplabv3plus_resnet50_voc_os16.pth")

    def load_pretrained(self, weights_path):
        print("Loading pre-trained DeepLabV3+ Resnet50 weights")
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict, strict=False)


    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def freeze_deeplab(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)

        low_level_feature = self.project(features['low_level'])
        output_feature = self.aspp(features['out'])

        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        x = self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

        return x
