import torch.nn as nn
import torch.nn.functional as F
import torch


class ReadOutSimple(nn.Module):
    def __init__(self, input_channels, original_img_size, num_classes=1):
        super(ReadOutSimple, self).__init__()
        # Final nodes features + Input image features
        # input_channels = input_channels * 2
        self.original_img_size = original_img_size
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()
        self.main_classifier1 = nn.Conv2d(input_channels, num_classes, kernel_size=1, bias=True)
        self.softmax = nn.Sigmoid()

    def forward(self, node_features):

        # feature_map = torch.cat([node_features, image_features], 1)
        feature_map = self.conv1(node_features)
        feature_map = self.bn1(feature_map)
        feature_map = self.relu(feature_map)
        x1 = self.main_classifier1(feature_map)
        if x1.shape[-2:] != self.original_img_size:
            x1 = F.upsample(x1, self.original_img_size, mode='bilinear')
        x1 = self.softmax(x1)

        return x1
