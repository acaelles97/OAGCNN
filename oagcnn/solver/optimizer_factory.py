from torch.optim import Adam, SGD

class OptimizerFactory:

    @staticmethod
    def create_optimizer(parameters, name, lr, weight_decay):
        if name == "Adam":
            return Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif name == " SGD":
            return SGD(parameters, lr=lr, weight_decay=weight_decay)

        else:
            raise ValueError("Selected optimizer not available!")
