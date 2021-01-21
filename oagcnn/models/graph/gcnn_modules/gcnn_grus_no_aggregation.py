from .convGRU_cell import ConvGRUCell
import torch.nn as nn
import torch


class GRUGCNNNoAggregation(nn.Module):

    def __init__(self, input_channels):
        super(GRUGCNNNoAggregation, self).__init__()
        self.aggregation_op = self._max_op
        self.aggregation_func = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.self_loop_func = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.update_op = self._cat_op
        self.update_func = nn.Conv2d(input_channels*2, input_channels, kernel_size=3, padding=1, bias=False)

        self.state_update = ConvGRUCell(input_channels, input_channels, 0, 3)

    def _max_op(self, neighbour_states):
        # Neighbour states: LIST (NUM_NODES-1) EACH: (NUM_CHANNELS, H, W)
        # Stack: (NUM_NODES -1, NUM_CHANNELS, H, W)
        # Mean: (1, NUM_CHANNELS, H, W)
        # Last channels summarizes all masks from all the other
        return torch.amax(torch.stack(neighbour_states, dim=0), dim=0)

    def _cat_op(self, neighbour_states, self_state):
        return torch.cat((neighbour_states, self_state), dim=1)

    def forward(self, neighbour_states, self_state, previous_state=None):
        # No state to summit from previous frame
        if previous_state is None:
            previous_state = self_state
        # We want to pass state but its first frame, so we put it None to be 0's by GRU module
        elif not isinstance(previous_state, torch.Tensor):
            previous_state = None

        intra = self.self_loop_func(self_state)
        return self.state_update(intra, previous_state)