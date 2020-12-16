from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet
from .backbone import mobilenetv2

from .custom_deeplab import CustomDeepLabHeadV3Plus

def build_deeplabv3plus(backbone_name, output_stride, out_channels):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=True,
        replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}

    classifier = CustomDeepLabHeadV3Plus(inplanes, low_level_planes, aspp_dilate, out_channels)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)

    return model


# def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
#     if output_stride == 8:
#         aspp_dilate = [12, 24, 36]
#     else:
#         aspp_dilate = [6, 12, 18]
#
#     backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
#
#     # rename layers
#     backbone.low_level_features = backbone.features[0:4]
#     backbone.high_level_features = backbone.features[4:-1]
#     backbone.features = None
#     backbone.classifier = None
#
#     inplanes = 320
#     low_level_planes = 24
#
#     if name == 'deeplabv3plus':
#         return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
#         classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
#     elif name == 'deeplabv3':
#         return_layers = {'high_level_features': 'out'}
#         classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
#     backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
#
#     model = DeepLabV3(backbone, classifier)
#     return model



# Deeplab v3

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return build_deeplabv3plus('resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)