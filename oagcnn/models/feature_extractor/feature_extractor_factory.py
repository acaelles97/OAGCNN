from .rvos_feature_extractor import RVOSFeatureExtractor
from .deeplab.custom_deeplab import DeepLabV3Plus

class FeatureExtractorFactory:
    _REGISTERED_FEATURE_EXTRACTOR = ["RVOSFeatureExtractor", "DeepLabV3Plus"]

    @staticmethod
    def create_feature_extractor(feature_extractor, cfg, image_size):
        assert feature_extractor in FeatureExtractorFactory._REGISTERED_FEATURE_EXTRACTOR, "Feature Extractor selected is not available: {} " \
                                                                                           "Availables: {}". \
            format(FeatureExtractorFactory._REGISTERED_FEATURE_EXTRACTOR, feature_extractor)

        model = globals()[feature_extractor](cfg, image_size)

        return model
