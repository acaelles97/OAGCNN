from torch.optim.lr_scheduler import MultiStepLR


class LrSchedulerFactory:

    @staticmethod
    def create_lr_scheduler(optimizer, params):
        if params.NAME == "MultiStepLR":
            return MultiStepLR(optimizer, milestones=params.STEPS, gamma=params.STEP_SIZE)

        else:
            raise ValueError("Selected optimizer not available!")
