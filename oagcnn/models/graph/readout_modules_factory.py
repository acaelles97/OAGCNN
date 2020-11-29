from .readout_modules import ReadOutSimple


class ReadOutModuleFactory:
    _REGISTERED_MODULES = ["ReadOutSimple", ]

    @staticmethod
    def create_by_name(name, input_channels, original_img_size, num_classes):
        assert name in ReadOutModuleFactory._REGISTERED_MODULES, "ReadOut Module selected is not available: {} Available: {}". \
            format(ReadOutModuleFactory._REGISTERED_MODULES, name)

        readout_module = globals()[name](input_channels, original_img_size, num_classes)

        return readout_module
