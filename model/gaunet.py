from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModel
from model.modules import CAB, CBR, CR
from model.utils import pad2same

class GA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(out_channel, out_channel, 1)
        self.transconv1 = nn.ConvTranspose2d(in_channel, out_channel, 3, 2, bias=False)
        self.conv2 = nn.Conv2d(out_channel, 1, 1)
        self.transconv2 = nn.ConvTranspose2d(in_channel, out_channel, 3, 2)
    
    def forward(self, x, g):
        g2 = self.transconv2(g)
        g = self.transconv1(g)
        x = pad2same(x, g)
        ga = torch.sigmoid(self.conv2(F.relu(self.conv1(g + x))))
        return ga * g2

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_first=False):
        super().__init__()
        if is_first:
            self.conv = nn.Sequential(
                CBR(in_channel // 2, in_channel, 3, 1, 1),
                CBR(in_channel, in_channel, 3, 1, 1)
            )
        else:
            self.conv = nn.Sequential(
                CBR(in_channel, in_channel, 3, 1, 1),
                CBR(in_channel, in_channel, 3, 1, 1)
            )
        self.convtrans = nn.ConvTranspose2d(in_channel, out_channel, 3, 2)
        self.up = nn.ConvTranspose2d(in_channel, out_channel, 3, 2)
        self.ga = GA(out_channel, out_channel)
    
    def forward(self, x, g):
        x = self.conv(x)
        x1 = self.convtrans(x)
        ga = self.ga(x1, g)
        x = self.up(x)
        x = pad2same(x, ga)
        return x + ga

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.down = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        h = x + self.bn(F.relu(self.conv3(F.relu(self.conv2(x)))))
        return self.down(h), h 

class UBlock(nn.Module):
    def __init__(self, downblock, upblock, subnet=None):
        super().__init__()
        self.downsample = downblock
        self.upsample = upblock
        self.subnet = subnet
    
    def forward(self, x):
        x, h = self.downsample(x)
        sub = self.subnet(x)
        return self.upsample(sub, x)

class GaUnet(BaseModel):
    def __init__(self):
        super().__init__()
        bottleneck = nn.Identity()
        sub1 = UBlock(DownBlock(128, 256), UpBlock(512, 256, True), bottleneck)
        sub2 = UBlock(DownBlock(64, 128), UpBlock(256, 128), sub1)
        sub3 = UBlock(DownBlock(32, 64), UpBlock(128, 64), sub2)
        self.net = UBlock(DownBlock(3, 32), UpBlock(64, 32), sub3)
        self.out = nn.Sequential(
            CBR(32, 32, 3, 1, 1),
            CBR(32, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        y = self.net(x)
        y = pad2same(y, x)
        y = self.out(y)
        self.pre = {
            'mask': y
        }
        return y
    

    