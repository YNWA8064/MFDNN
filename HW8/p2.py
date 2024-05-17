import torch
from torch import nn

r=3
layer1 = nn.Upsample(scale_factor=r, mode='nearest')

layer2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=r, stride=r, bias=False)
layer2.weight.data = torch.ones_like(layer2.weight.data)

a = torch.randn((1, 1, 3, 4))
r1 = layer1(a)
r2 = layer2(a)
print(f'diff = {torch.norm(r1-r2)}')