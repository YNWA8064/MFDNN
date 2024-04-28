import torch
from torch import nn
import numpy as np
from torch.nn.modules.module import T


class MyConvLayer(nn.Module):
    def __init__(self):
        super(MyConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(256,128, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 1)

    def forward(self, x):
        out = nn.functional.relu(self.conv1(x))
        out = nn.functional.relu(self.conv2(out))
        out = nn.functional.relu(self.conv3(out))
        return out

class STMConvLayer(nn.Module):
    def __init__(self):
        super(STMConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(256,4,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4,4,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4,256,1),
            nn.ReLU(inplace=True)
        )
        self.conv_layer = nn.ModuleList([self.conv for i in range(32)])

    def forward(self, x):
        out = self.conv_layer[0](x)
        for i in range(1, 32):
            out += self.conv_layer[i](x)
        return out