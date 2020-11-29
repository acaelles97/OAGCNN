from .backbones import ResNet50, ResNet34, ResNet101
from .backbones import VGG16


class BackboneFactory:
    # Maybe we could check __all__ from the __init__ file?
    _REGISTERED_BACKBONES = ["ResNet34", "ResNet50", "ResNet101", "VGG16"]

    @staticmethod
    def create_backbone(backbone):
        assert backbone in BackboneFactory._REGISTERED_BACKBONES, "Backbone selected is not available: {} Available: {}". \
            format(BackboneFactory._REGISTERED_BACKBONES, backbone)

        model = globals()[backbone]()

        return model
