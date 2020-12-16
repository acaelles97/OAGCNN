import torch.nn as nn
import torch


# Receives old states and computes new states
class GCNNSimple(nn.Module):

    def __init__(self, input_channels):
        super(GCNNSimple, self).__init__()
        self.aggregation_op = self._average_op
        self.aggregation_func = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.self_loop_func = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.update_op = self._sum_op
        self.update_func = nn.ReLU()

    def _average_op(self, neighbour_states):
        # Neighbour states: LIST (NUM_NODES-1) EACH: (NUM_BATCHES, NUM_CHANNELS, H, W)
        # Stack: (NUM_BATCHES, NUM_NODES -1, NUM_CHANNELS, H, W)
        # Mean: (NUM_BATCHES, 1, NUM_CHANNELS, H, W)
        return torch.mean(torch.stack(neighbour_states, dim=1), dim=1)

    def _sum_op(self, neighbour_states, self_state):
        return neighbour_states + self_state


    def forward(self, neighbour_states, self_state, previous_state=None):

        if not neighbour_states:
            intra = self.self_loop_func(self_state)
            return self.update_func(intra)

        inter = self.aggregation_op(neighbour_states)
        inter = self.aggregation_func(inter)

        intra = self.self_loop_func(self_state)
        new = self.update_op(inter, intra)
        new = self.update_func(new)
        return new

        # return self.update_func(self.update_op(self.aggregation_func(self.aggregation_op(neighbour_states)), self.self_loop_func(self_state)))