import torch
from torch import nn
import torchvision.models as models


class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)

    def forward(self, x):
        x = self.resnet(x)
        return x
