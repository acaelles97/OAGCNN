import torch.nn as nn
import torch

from .convGRU_cell import ConvGRUCell

# Embedded gaussian similarity -> ha de ser per propagació temporal
from .non_local_modules import NONLocalDotProductBlock2D
from .non_local_modules.attention_module import DotProductAttention, DotProductAttention

# Receives old states and computes new states
class AttentionGRUGCNN(nn.Module):
    def __init__(self, input_channels):
        super(AttentionGRUGCNN, self).__init__()
        # TOCheck: Residual connection might not be necessary, but we need to keep it as weights from the attention mechanism are init with 0
        self.neighbour_attention = DotProductAttention(in_channels=input_channels, inter_channels=None, sub_sample=False, bn_layer=False)
        # inter_channels: Channels that work the module works with (if None input/2)
        # sub_sample: Apply max_pooling once finished: False, we need to preserve spatial dimensions
        # bn_layer: Lets first omit it NOTE: Actual implementation works with BATCH_SIZE=1 at this point so YES
        self.intra_attention = NONLocalDotProductBlock2D(in_channels=input_channels, inter_channels=None, sub_sample=False, bn_layer=False)

        self.temporal_attention = DotProductAttention(in_channels=input_channels, inter_channels=None, sub_sample=False, bn_layer=False)

        self.aggregation_message = nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=3, bias=True, dilation=3)
        self.state_update = ConvGRUCell(input_channels, input_channels, 0, 3)

    def forward(self, neighbour_states, self_state, previous_state=None):
        # K != 0, so we want to just use existing state
        if previous_state is None:
            previous_state = self_state
        # We want to pass state but its first frame, so we put it None to be 0's by GRU module
        elif not isinstance(previous_state, torch.Tensor):
            previous_state = None
        # Compute state with attention between frames
        else:
            previous_state = self.temporal_attention(self_state, previous_state)


        self_message = self.intra_attention(self_state)
        if not neighbour_states:
            return self.state_update(self_message, previous_state)

        neighbour_messages = []
        for neighbour_state in neighbour_states:
            neighbour_messages.append(self.neighbour_attention(self_state, neighbour_state))

        aggregated_message = torch.mean(torch.stack(neighbour_states, dim=1), dim=1)
        message = torch.cat((self_message, aggregated_message), dim=1)
        message = self.aggregation_message(message)

        new_state = self.state_update(message, previous_state)

        return new_state
