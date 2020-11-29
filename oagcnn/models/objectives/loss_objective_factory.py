from .objectives import SoftIoULoss, MaskedBCELoss, MaskedNLLLoss


class LossFunctionFactory:
    _REGISTERED_LOSS_FUNCTIONS = ["MaskedNLLLoss", "MaskedBCELoss", "SoftIoULoss"]

    @staticmethod
    def create_feature_extractor(loss_func, *args, **kwargs):
        assert loss_func in LossFunctionFactory._REGISTERED_LOSS_FUNCTIONS, "Loss objective function selected is not available: {} " \
                                                                                           "Availables: {}". \
            format(LossFunctionFactory._REGISTERED_LOSS_FUNCTIONS, loss_func)

        loss_obj = globals()[loss_func]()
        return loss_obj
