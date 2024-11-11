import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModel, wrap_bce
from model.modules import CR, SE, ModuleStack
from model.utils import pad2same

class DFE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(DFE, self).__init__()
        self.conv1 = nn.Sequential(
            CR(in_channels, in_channels, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            CR(in_channels, in_channels, 3, 1, 1),
            CR(in_channels, in_channels, 3, 1, 1)
        )
        self.conv3 = nn.Sequential(
            CR(in_channels, in_channels, 3, 1, 1),
            CR(in_channels, in_channels, 3, 1, 1),
            CR(in_channels, in_channels, 3, 1, 1)
        )
        self.se = SE(in_channels * 3, reduction_ratio)
        self.conv = nn.Conv2d(in_channels * 3, in_channels, 1)

    def forward(self, x):
        cat = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        cat = self.se(cat)
        return self.conv(cat)
    
class RCR(nn.Module):
    def __init__(self, in_channels, step=2):
        super(RCR, self).__init__()
        self.cr = CR(in_channels, in_channels, 3, 1, 1)
        self.step = step

    def forward(self, x):
        ox = x
        for _ in range(self.step):
            x = ox + self.cr(x)
        return x
    
class CFE(nn.Module):
    def __init__(self, in_channels):
        super(CFE, self).__init__()
        self.rcrs = nn.Sequential(
            RCR(in_channels),
            RCR(in_channels),
            RCR(in_channels)
        )

    def forward(self, x):
        return self.rcrs(x) + x
    
class HP(nn.Module):
    def __init__(self, in_channels):
        super(HP, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 2)

    def forward(self, x):
        pool = self.pool(x)
        conv = pad2same(self.conv(x), pool)
        return torch.cat([pool, conv], dim=1)
    
class IF(nn.Module):
    def __init__(self, in_channels):
        super(IF, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x1, x2):
        x = x1 + x2
        cat = torch.cat([x1 * x, x2 * x], dim=1)
        return self.conv(cat)
    
class CLF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CLF, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 3, 2)
        self.se = nn.Sequential(
            nn.Conv2d(out_channels * 2,out_channels, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, e, d):
        up = pad2same(self.up(d), e)
        cat = torch.cat([up, e], dim=1)
        return self.se(cat) * e + up
    
class SFF(nn.Module):
    def __init__(self, in_channels, out_channels=16):
        super(SFF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = nn.Conv2d(in_channels * 2, out_channels, 1)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv3 = nn.Conv2d(in_channels * 4, out_channels, 1)
        self.fc = nn.Conv2d(out_channels * 3, out_channels * 3, 1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, d3, d2, d1):
        x1 = self.conv1(d1)
        x2 = self.conv2(self.up2(d2))
        x3 = self.conv3(self.up3(d3))
        cat = torch.cat([x1, x2, x3], dim=1)
        h = self.fc(self.maxpool(cat)) + self.fc(self.avgpool(cat))
        return torch.sigmoid(h) * cat
    
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.DFE = DFE(in_channels)
        self.CFE = CFE(in_channels)
        self.IF = IF(in_channels)
        self.DHP = HP(in_channels)
        self.CHP = HP(in_channels)
    
    def forward(self, x):
       dfe = self.DFE(x[0] if isinstance(x, list) else x)
       cfe = self.CFE(x[1] if isinstance(x, list) else x)
       xif = self.IF(dfe, cfe)
       return [self.DHP(dfe), self.CHP(cfe), xif]
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.CLF = CLF(out_channels * 2, out_channels)
        self.DFE = DFE(out_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 3, 2)

    def forward(self, e, d):
        clf = self.CLF(e[2], d[0])
        up = pad2same(self.up(d[1][-1]), clf)
        dfe = torch.cat([clf, self.DFE(up)], dim=1)
        return [e[2], d[1] + [dfe]]

class BottleNeck(nn.Module):
    def __init__(self, in_channels):
        super(BottleNeck, self).__init__()
        self.encoder = Encoder(in_channels)
        
    def forward(self, x):
        x = self.encoder(x)
        return [x[2], [x[2]]]
    
class DPFNet(BaseModel):
    def __init__(self, in_channels=3, out_channels=1, mid_channels=32):
        super(DPFNet, self).__init__()
        self.trans = nn.Conv2d(in_channels, mid_channels, 1)
        self.net = ModuleStack(
            Encoder(mid_channels),
            ModuleStack(
                Encoder(mid_channels * 2),
                ModuleStack(
                    Encoder(mid_channels * 4),
                    BottleNeck(mid_channels * 8),
                    Decoder(mid_channels * 8, mid_channels * 4),
                ),
                Decoder(mid_channels * 8, mid_channels * 2),
            ),
            Decoder(mid_channels * 4, mid_channels),
        )
        self.SFF = SFF(mid_channels * 2)
        self.out = nn.Conv2d(16 * 3, out_channels, 1)

    def forward(self, x):
        x = self.trans(x)
        x = self.net(x)
        x = self.SFF(*x[1][-3:])
        out = self.out(x)
        self.pre = {
            'mask': out
        }
        return out
    
    def backward(self, x, optimer, closure=wrap_bce, clear_stored=True):
        return super().backward(x, optimer, closure, clear_stored)
