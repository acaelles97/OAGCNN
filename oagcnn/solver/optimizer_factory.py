from torch.optim import Adam, SGD

class OptimizerFactory:
    _REGISTERED_OPTIMIZERS = ["Adam", "SGD"]

    @staticmethod
    def create_optimizer(name, parameters, cfg):
        assert name in OptimizerFactory._REGISTERED_OPTIMIZERS, "Selected optimizer not available!"
        lr = cfg.SOLVER.LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        optimizer = globals()[name](parameters, lr=lr, weight_decay=weight_decay)
        return optimizer
