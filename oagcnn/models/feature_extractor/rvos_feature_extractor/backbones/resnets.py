# from torchvision.models.resnet import BasicBlock, Bottleneck, ResNetreplace_stride_with_dilation
from .old_torchvision_resnet import BasicBlock, Bottleneck, ResNet
from torchvision import models


class ResNet101(ResNet):
    """Returns intermediate features from ResNet-50"""

    def __init__(self):
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3], 1000)
        self.skip_dims_in = [2048, 1024, 512, 256, 64]
        self.load_state_dict(models.resnet101(pretrained=True).state_dict())
        del self.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        # Omits average pool from the original implementation as well as the FC layer

        return x5, x4, x3, x2, x1


class ResNet50(ResNet):
    """Returns intermediate features from ResNet-50"""

    def __init__(self):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], 1000)
        self.skip_dims_in = [2048, 1024, 512, 256, 64]
        self.load_state_dict(models.resnet50(pretrained=True).state_dict())
        del self.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x5, x4, x3, x2, x1


class ResNet34(ResNet):
    """Returns intermediate features from ResNet-34"""

    def __init__(self):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], 1000)
        self.skip_dims_in = [512, 256, 128, 64, 64]
        self.load_state_dict(models.resnet34(pretrained=True).state_dict())
        del self.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x5, x4, x3, x2, x1
