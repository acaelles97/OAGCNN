from torch import nn
import torch
from torch.nn import functional as F


# Returns a transformed version of the positions that we need to attend to x2 when computing x1 and wondering how we want to pich the other state for our new update from x1.
class _AttentionModule(nn.Module):
    available_pairwise_funcs = ["softmax", "dot_product"]

    def __init__(self, in_channels, pairwise_func, inter_channels=None, sub_sample=True, bn_layer=True):
        super(_AttentionModule, self).__init__()

        assert pairwise_func in _AttentionModule.available_pairwise_funcs

        self.pairwise_func = pairwise_func
        # self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))

        # bn = nn.BatchNorm2d(self.in_channels)
        # if bn_layer:
        #     self.W = nn.Sequential(
        #         conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
        #                 kernel_size=1, stride=1, padding=0),
        #         bn(self.in_channels)
        #     )
        #
        #     nn.init.constant_(self.W[1].weight, 0)
        #     nn.init.constant_(self.W[1].bias, 0)
        # else:

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        # Inits with 0: First models omits and just uses slef-state with 1x1 conv but as it can gain capacity to learn as trainings flows
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, base, to_transform, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = base.size(0)
        # x1: (1, CH, H, W)
        # x2: (1, CH, H, W)

        g_to_transform = self.g(to_transform).view(batch_size, self.inter_channels, -1)
        g_to_transform = g_to_transform.permute(0, 2, 1)

        # theta_x1 (BATCH_SIZE, INTER_CH, (CH-INTER_CH) * H * W)
        theta_base = self.theta(base).view(batch_size, self.inter_channels, -1)

        # theta_x1 (BATCH_SIZE, (CH-INTER_CH) * H * W, INTER_CH)
        theta_base = theta_base.permute(0, 2, 1)

        # phi_x2 (BATCH_SIZE, INTER_CH, (CH-INTER_CH) * H * W)
        phi_to_transform = self.phi(to_transform).view(batch_size, self.inter_channels, -1)

        # Dot product of both Tensor 1x1 conv transformations
        # f (BATCH_SIZE, (CH-INTER_CH) * H * W, (CH-INTER_CH) * H * W)
        f = torch.matmul(theta_base, phi_to_transform)

        # Normalization as input can have any size
        # N number positions in f
        if self.pairwise_func == "dot_product":
            num_elements = f.shape[-1]
            attention_score_map = f / num_elements
        elif self.pairwise_func == "softmax":
            attention_score_map = F.softmax(f, dim=-1)
        else:
            raise ValueError("Error pairwise function: No available pairwise function")

        transformed_neighbour = torch.matmul(attention_score_map, g_to_transform)
        transformed_neighbour = transformed_neighbour.permute(0, 2, 1).contiguous()
        y = transformed_neighbour.view(batch_size, self.inter_channels, *base.size()[2:])
        w_y = self.W(y)

        z = w_y + to_transform
        if return_nl_map:
            return z, attention_score_map

        return z


class SoftmaxAttention(_AttentionModule):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels, "softmax", inter_channels, sub_sample, bn_layer)


class DotProductAttention(_AttentionModule):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels, "dot_product", inter_channels, sub_sample, bn_layer)