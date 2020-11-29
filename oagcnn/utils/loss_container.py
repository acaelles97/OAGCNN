import numpy as np


class LossContainer:
    def __init__(self, epochs, print_every, batch_size):
        self.loss_container = {i: [] for i in epochs}
        self.print_every = print_every
        self.batch_size = batch_size

    def get_loss_container(self):
        return self.loss_container

    def load_loss_container(self, loss_container):
        self.loss_container = loss_container

    def update(self, value, epoch):
        self.loss_container[epoch].append(value/self.batch_size)

    def get_mean_last_iter(self, epoch, indices=None):
        if indices is None:
            indices = self.print_every
        return np.mean(self.loss_container[epoch][-indices:])

    def get_total_epoch_loss(self, epoch):
        return np.mean(self.loss_container[epoch])

