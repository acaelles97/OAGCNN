from .gcnn_modules import *


class GCNNModuleFactory:
    _REGISTERED_GCNN_MODULES = ["GCNNSimple", "GCNNConvBlocks", "GRUGCNN"]

    @staticmethod
    def create_by_name(name, input_channels):
        assert name in GCNNModuleFactory._REGISTERED_GCNN_MODULES, "GCNN Module selected is not available: {} Available: {}". \
            format(GCNNModuleFactory._REGISTERED_GCNN_MODULES, name)

        gcnn_module = globals()[name](input_channels)

        return gcnn_module
