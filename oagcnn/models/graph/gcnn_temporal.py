import torch.nn as nn
import torch
import torch.nn.functional as F
from .graph_modules_factory import MaskEncoderModuleFactory, ReadOutModuleFactory, GCNNModuleFactory

class GCNNTemporal(nn.Module):

    def __init__(self, cfg, input_channels):
        super(GCNNTemporal, self).__init__()

        self.message_passing_steps = cfg.OAGCNN.MESSAGE_PASSING_STEPS
        self.use_temporal_features = cfg.OAGCNN.USE_TEMPORAL_FEATURES

        # HEAD_OUT_CHANNELS + 1 from concatenated mask
        self.input_channels = input_channels

        self.mask_encoder = MaskEncoderModuleFactory.create_by_name(cfg.OAGCNN.ARCH.GRAPH.MASK_ENCODING_MODULE, self.input_channels, cfg)

        # Graph k-message passing step arch
        self.gcnn_module = GCNNModuleFactory.create_by_name(cfg.OAGCNN.ARCH.GRAPH.GCNN_MODULE, self.mask_encoder.out_channels)

        # Arch to decode a mask for each of the individual nodes we have
        self.read_out_module = ReadOutModuleFactory.create_by_name(cfg.OAGCNN.ARCH.GRAPH.READ_OUT, self.input_channels, self.mask_encoder.out_channels)

    def get_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(self, batch_node_feats, batch_previous_masks, batch_valid_indices, batch_previous_states):
        # feats (BATCH_SIZE, CH, H, W) -> frame from a clip
        # obj_masks (BATCH_SIZE, NUM_OBJ, H, W)

        # graph data structure (BATCH_SIZE, NUM_OBJ, CH, H, W)
        # nodes_state List NUM_OBJ elements each: (BATCH_SIZE, CH, H, W) //
        # batch_node_feats, batch_previous_masks, batch_previous_states, batch_valid_indices = args

        out_masks = torch.zeros_like(batch_previous_masks)
        out_states = None
        batch_size = batch_node_feats.shape[0]


        for idx in range(batch_size):
            # (CH, H, W)
            node_feats = batch_node_feats[idx, ...]
            # (OBJ_IN_FRAME,)
            valid_masks_id = batch_valid_indices[idx, ...]

            previous_masks = batch_previous_masks[idx, ...][valid_masks_id, ...]

            # Num obj tensors
            num_obj = previous_masks.shape[0]

            # Note: If does not exists will be 0's as we will work with fixed size tensors filled with 0's
            previous_state = None
            if self.use_temporal_features:
                if batch_previous_states is None:
                    previous_state = [False for _ in range(num_obj)]
                else:
                    previous_state = batch_previous_states[idx, ...][valid_masks_id, ...]
                    previous_state = [previous_state[obj_idx, ...].unsqueeze(0) for obj_idx in range(num_obj)]

            # mask encoding:
            node_states = self.mask_encoder(node_feats, previous_masks, num_obj)

            for k in range(self.message_passing_steps):
                new_states = []

                for i in range(num_obj):
                    if k == 0 and self.use_temporal_features:
                        new_state = self.gcnn_module(node_states[:i] + node_states[i + 1:], node_states[i], previous_state[i])
                    else:
                        new_state = self.gcnn_module(node_states[:i] + node_states[i + 1:], node_states[i])

                    new_states.append(new_state)

                # Update previous state
                node_states = new_states

            if self.use_temporal_features:
                if out_states is None:
                    out_size = batch_valid_indices.shape + node_states[0].shape[-3:]
                    out_states = torch.zeros(out_size)
                    out_states = out_states.to(batch_valid_indices.device)
                out_states[idx, valid_masks_id, :, :] = torch.cat(node_states, dim=0)

            # Masks from the batch
            mask_batch = torch.cat([self.read_out_module(node_feats, node_state) for node_state in node_states], dim=1)
            out_masks[idx, valid_masks_id, :, :] = mask_batch

        return out_masks, out_states
