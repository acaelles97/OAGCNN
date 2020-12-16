import torch.nn as nn
import torch.nn.functional as F
import torch

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim is None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        m = self.ResMM(m)
        return m


class ReadOutWithRefinement(nn.Module):
    def __init__(self, input_channels_feats, input_channels_graph, original_img_size, config):
        super(ReadOutWithRefinement, self).__init__()
        self.original_img_size = original_img_size
        self.prediction_resolution = (original_img_size[0]/2, original_img_size[1]/2)
        self.refinement_bloc1 = ResBlock(input_channels_feats + input_channels_graph, config["INTERMEDIATE_CHANNELS"])

        self.main_classifier = nn.Conv2d(config["INTERMEDIATE_CHANNELS"], 1, kernel_size=1, padding=(1, 1), stride=1)
        self.softmax = nn.Sigmoid()


    def forward(self, image_feats, node_out):
        upsampled_node_out = F.interpolate(node_out, size=feats.shape[-2:], mode='bilinear', align_corners=False)
        node_feats = torch.cat((image_feats, upsampled_node_out), dim=1)
        node_feats = self.refinement_bloc(node_feats)
        node_feats = F.interpolate(node_feats, size=self.prediction_resolution, mode='bilinear', align_corners=False)
        out_feats = self.main_classifier(node_feats)
        out_logits = F.interpolate(out_feats, size=self.original_img_size, mode='bilinear', align_corners=False)
        out_probabilities = self.softmax(out_logits)

        return out_probabilities


