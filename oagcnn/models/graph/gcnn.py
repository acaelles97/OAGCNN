import torch.nn as nn
import torch
from .gcnn_modules_factory import GCNNModuleFactory
from .readout_modules_factory import ReadOutModuleFactory
import torch.nn.functional as F


class GCNN(nn.Module):

    def __init__(self, cfg, input_channels):
        super(GCNN, self).__init__()
        self.num_nodes = cfg.DATA.MAX_NUM_OBJ
        self.message_passing_steps = cfg.GCNN.MESSAGE_PASSING_STEPS

        # HEAD_OUT_CHANNELS + 1 from concatenated mask
        self.input_channels = input_channels + 1

        # Graph k-message passing step arch
        self.gcnn_module = GCNNModuleFactory.create_by_name(cfg.GCNN.MODULE, self.input_channels)

        # Arch to decode a mask for each of the individual nodes we have
        self.read_out_module = ReadOutModuleFactory.create_by_name(cfg.GCNN.READ_OUT, self.input_channels,  cfg.DATA.IMAGE_SIZE, cfg.GCNN.NUM_CLASSES)


    # def forward(self, feats, objs_masks):
    #     # feats (BATCH_SIZE, CH, H, W) -> frame from a clip
    #     # obj_masks (BATCH_SIZE, NUM_OBJ, H, W)
    #
    #     # graph data structure (BATCH_SIZE, NUM_OBJ, CH, H, W)
    #     # nodes_state List NUM_OBJ elements each: (BATCH_SIZE, CH, H, W) //
    #     nodes_state = [torch.cat((feats, objs_masks[:, i, ...].unsqueeze(1)), dim=1) for i in range(self.num_nodes)]
    #
    #     for k in range(self.message_passing_steps):
    #         new_states = [None for _ in range(self.num_nodes)]
    #
    #         for idx in range(self.num_nodes):
    #             new_state = self.gcnn_module(nodes_state[:idx] + nodes_state[idx + 1:], nodes_state[idx])
    #             new_states[idx] = new_state
    #
    #         nodes_state = new_states
    #
    #     out_masks = torch.cat([self.read_out_module(node_state) for node_state in nodes_state], dim=1)
    #
    #     return out_masks

    def get_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(self, feats, objs_masks, valid_indices):
        # feats (BATCH_SIZE, CH, H, W) -> frame from a clip
        # obj_masks (BATCH_SIZE, NUM_OBJ, H, W)

        # graph data structure (BATCH_SIZE, NUM_OBJ, CH, H, W)
        # nodes_state List NUM_OBJ elements each: (BATCH_SIZE, CH, H, W) //

        out_masks = torch.zeros_like(objs_masks)
        batch_size = feats.shape[0]

        if feats.shape[-2:] != objs_masks.shape[-2:]:
            objs_masks = F.interpolate(objs_masks, size=feats.shape[-2:])

        for idx in range(batch_size):
            # (CH, H, W)
            image_feats = feats[idx, ...]
            # (OBJ_IN_FRAME,)
            valid_masks_id = valid_indices[idx, ...]
            # (OBJ_IN_FRAME, H, W)
            obj_masks = objs_masks[idx, ...][valid_masks_id, ...]

            num_obj = obj_masks.shape[0]

            node_states = [torch.cat((image_feats, obj_masks[obj_idx, ...].unsqueeze(0)), dim=0).unsqueeze(0) for obj_idx in range(num_obj)]

            for k in range(self.message_passing_steps):
                new_states = []

                for i in range(num_obj):
                    new_state = self.gcnn_module(node_states[:i] + node_states[i + 1:], node_states[i])
                    new_states.append(new_state)

                node_states = new_states

            # # Masks from the batch
            mask_batch = torch.cat([self.read_out_module(node_state) for node_state in node_states], dim=1)

            out_masks[idx, valid_masks_id, :, :] = mask_batch



        return out_masks
