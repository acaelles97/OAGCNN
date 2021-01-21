from .readout_modules import *
from .gcnn_modules import *
from .mask_encoders import *


class ReadOutModuleFactory:
    _REGISTERED_MODULES = ["ReadOutSimple", "ReadOutWithRefinement"]

    @staticmethod
    def create_by_name(name, input_channels_feats, input_channels_graph, cfg):
        assert name in ReadOutModuleFactory._REGISTERED_MODULES, "ReadOut Module selected is not available: {} Available: {}". \
            format(ReadOutModuleFactory._REGISTERED_MODULES, name)

        module = globals()[name](input_channels_feats, input_channels_graph, cfg)

        return module


class GCNNModuleFactory:
    _REGISTERED_MODULES = ["GCNNSimple", "GCNNConvBlocks", "GRUGCNN", "AttentionGRUGCNN", "GRUGCNNNoAggregation"]

    @staticmethod
    def create_by_name(name, input_channels):
        assert name in GCNNModuleFactory._REGISTERED_MODULES, "GCNN Module selected is not available: {} Available: {}". \
            format(GCNNModuleFactory._REGISTERED_MODULES, name)

        module = globals()[name](input_channels)

        return module


class MaskEncoderModuleFactory:
    _REGISTERED_MODULES = ["MaskCatEncoder", "MaskEncoderOneStage", "MaskEncoderTwoStages"]

    @staticmethod
    def create_by_name(name, input_channels, cfg):
        assert name in MaskEncoderModuleFactory._REGISTERED_MODULES, "Mask Encoder Module selected is not available: {} Available: {}". \
            format(MaskEncoderModuleFactory._REGISTERED_MODULES, name)

        module = globals()[name](input_channels, cfg)

        return module