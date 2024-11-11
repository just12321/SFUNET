"""
This experiment aims to explore which part of the skip connections in Unet is more important.\n
Using ASPP
"""
from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModel, wrap_bce
from model.modules import CA, CAB, CBA, CBR, CR, SE, CloseGate, DepthWiseConv2d, GroupConv2d, OpenCloseGate, OpenGate, PNNorm, PixelNorm, QKPlus, SEPlus, ShuffleUp
from model.utils import interlaced_cat, pad2same
import einops as eop

from utils.losses import contain_loss, diff_loss, keep_loss


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, groups=1, with_bn=True, activate=nn.ReLU()):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels * groups
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels) if with_bn else nn.Identity(),
            activate,
            GroupConv2d(mid_channels, out_channels, 3, 1, 1, bias=False, out_groups=groups),
            nn.BatchNorm2d(out_channels * groups) if with_bn else nn.Identity(),
            activate
        )

    def forward(self, x):
        return self.double_conv(x)

class MDown(nn.Module):
    """Downscaling with maxpool and conv then double conv"""

    def __init__(self, in_channels, out_channels, with_bn=True, groups=1):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2),
            CA(in_channels, in_channels, 1)
        )
        self.down_conv = nn.Sequential(
            CA(in_channels, in_channels, 4, 2, 1),
            CA(in_channels, in_channels, 1)
        )
        self.fuse_conv = DoubleConv(in_channels*2, out_channels, with_bn=with_bn, groups=groups)

    def forward(self, x):
        return self.fuse_conv(torch.cat([self.maxpool(x), self.down_conv(x)], dim=1))
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, with_bn=True, groups=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, with_bn=with_bn, groups=groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, mode='cat', with_bn=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, with_bn=with_bn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // (2 if mode=='cat' else 1), kernel_size=3, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, with_bn=with_bn)
        self.mode = mode

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1) if self.mode == 'cat' else x1 + x2
        return self.conv(x)

class GateUpv_0(nn.Module):
    """Upscaling then double conv with gate"""

    def __init__(self, in_channels, out_channels, bilinear=True, with_bn=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels*2, in_channels*2, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                CR(in_channels*2, in_channels, 3, 1, 1)
            )
            # self.up = ShuffleUp(in_channels*2, in_channels)
        self.conv = DoubleConv(in_channels, out_channels // 2, with_bn=with_bn, groups=2)
        self.gate = Gate(out_channels)

    def forward(self, x1, x1_, x2_, x2=None):
        if x2 is None:
            x1 = self.up(x1)
        else:
            x1, x2 = torch.chunk(self.up(torch.cat([x1, x2], dim=1)), 2, 1)
        # gate = self.gate(x1, x1_)
        # input is CHW
        diffY = x2_.size()[2] - x1.size()[2]
        diffX = x2_.size()[3] - x1.size()[3]

        gate = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        
        if x2 is not None:
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        gate_ = torch.sigmoid(gate)
        if x2 is not None:
            x = gate_ * x2 + (1-gate_) * x2_
        else:
            x = gate_ * x2_
        return torch.chunk(self.conv(torch.cat([gate, x], dim=1)), 2, 1)

class GateUpv_1(nn.Module):
    """Upscaling then double conv with gate. Using Gatev_0, namely $$F(x,y)=f(x,y)*\sigma(g(x,y))$$"""

    def __init__(self, in_channels, out_channels, bilinear=True, with_bn=True):
        super().__init__()
        self.gate = Gatev_0(in_channels)
        self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                DoubleConv(in_channels, in_channels//2, with_bn=with_bn)
            )
        self.conv = DoubleConv(in_channels, out_channels, with_bn=with_bn)

    def forward(self, x, y):
        x = self.up(x)
        return self.conv(self.gate(x, y))
    
class GateUpv_2(GateUpv_1):
    """Upscaling then double conv with gate. Using Gatev_3, namely $$F(x,y)=x*\sigma(g(y))+h(y)$$"""

    def __init__(self, in_channels, out_channels, bilinear=True, with_bn=True):
        super().__init__(in_channels, out_channels, bilinear, with_bn)
        self.gate = Gatev_3(in_channels//2)
        self.conv = DoubleConv(in_channels//2, out_channels, with_bn=with_bn)

class GateUpv_3(GateUpv_1):
    """Using Gatev_5."""
    
    def __init__(self, in_channels, out_channels, bilinear=True, with_bn=True, k_factor=4):
        super().__init__(in_channels, out_channels, bilinear, with_bn)
        self.gate = Gatev_5(in_channels//2, k_factor=k_factor)
        self.conv = DoubleConv(in_channels//2, out_channels, with_bn=with_bn)

class GateUpv_4(nn.Module):
    """Using Gatev_4."""
    
    def __init__(self, in_channels, out_channels, bilinear=True, with_bn=True, k_factor=4):
        super().__init__()
        self.gate = Gatev_4(in_channels//2, k_factor=k_factor)
        self.conv = DoubleConv(in_channels, out_channels, with_bn=with_bn)
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels*2, in_channels*2, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                CR(in_channels*2, in_channels, 3, 1, 1)
            )

    def forward(self, x1, x1_, x2_, x2=None): # q, _, v, k
        if x2 is None:
            x1 = self.up(x1)
        else:
            x1, x2 = torch.chunk(self.up(torch.cat([x1, x2], dim=1)), 2, 1)
        # input is CHW
        diffY = x2_.size()[2] - x1.size()[2]
        diffX = x2_.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        if x2 is not None:
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        gate = self.gate(x2, x1, x2_) if x2 is not None else self.gate(x1, x1, x2_)
        return torch.chunk(self.conv(torch.cat([gate, x2 if x2 is not None else x1], dim=1)), 2, 1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DOutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        return self.conv1(x), self.conv2(x)

class SumUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024 // factor, 512 // factor, bilinear, mode='sum'))
        self.up2 = (Up(512 // factor, 256 // factor, bilinear, mode='sum'))
        self.up3 = (Up(256 // factor, 128 // factor, bilinear, mode='sum'))
        self.up4 = (Up(128 // factor, 64, bilinear, mode='sum'))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x) 
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4)
        x = self.up1(x5, x4) 
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class GUNet(BaseModel):
    def __init__(self, n_channels, n_classes, mid=[32, 64, 128, 256],bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GUNet, self).__init__(**kwargs)
        down = [DoubleConv(n_channels, mid[0], groups=2)]
        for i in range(len(mid)-1):
            down.append(Down(mid[i], mid[i+1], groups=2))
        down.append(Down(mid[-1], mid[-1]*2))
        self.down = nn.ModuleList(down)
        up = []
        for i in range(len(mid)-1, 0, -1):
            up.append(Up(mid[i]*2, mid[i], bilinear))
        up.append(Up(mid[0]*2, mid[0]*2, bilinear))
        self.up = nn.ModuleList(up)
        self.outc = OutConv(mid[0]*2, n_classes)
        self.return_feature = return_feature
        self.bottle = bottle

    def forward(self, x):
        res_features = []
        for layer in self.down:
            if len(res_features) > 0:
                x, h = torch.chunk(res_features.pop(), 2, 1)
                res_features.append(h)
            x = layer(x)
            res_features.append(x)
        x = self.bottle(res_features.pop())
        h = x
        for layer in self.up:
            x = layer(x, res_features.pop())
        y = self.outc(x)
        pre = {
            'mask': y,
            'feature': h,
        }
        if not self.as_layer:
            self.pre = pre
        return y if not self.return_feature else (y, h)
    
    def backward(self, x, optimer, closure:Callable[[Dict], Dict]=None, clear_stored=True):
        def closure_(pre):
            with torch.enable_grad():
                if closure is not None:
                    return closure(pre)
                else:
                    loss = {}
                    mask = pre['mask']
                    optimer.zero_grad()
                    loss_bce = F.binary_cross_entropy_with_logits(mask, x.float())
                    loss['bce'] = loss_bce.item()
                    loss_bce.backward()
                    optimer.step()
                    return loss
        return super().backward(x, optimer, closure_, clear_stored)
    
    def memo(self):
        return """
            When downsample, the output of the layer is halved, one is the input of the next layer, and the other is the residual connection.
        """

class GUNet_n(GUNet):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GUNet_n, self).__init__(mid=[32,64,128,256], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear, 
                                    return_feature=return_feature, bottle=bottle, **kwargs)

class GUNet_s(GUNet):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GUNet_s, self).__init__(mid=[16,32,64,128], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                    return_feature=return_feature, bottle=bottle, **kwargs
        )

class GUNet_tiny(GUNet):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GUNet_tiny, self).__init__(mid=[8,16,32,64], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                    return_feature=return_feature, bottle=bottle, **kwargs
        )

class UNet(BaseModel):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity()):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))
        self.return_feature = return_feature
        self.bottle = bottle

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottle(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        self.pre = {
            'mask': logits,
        }
        return logits if not self.return_feature else (logits, x5)

class Gate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # self.trans = DoubleConv(channels, channels//2, activate=nn.LeakyReLU(inplace=True))
        self.gate = CA(channels, channels//2, 3, padding=1, groups=channels//2, activation=nn.Sigmoid())
        self.trans = CR(channels, channels//2, 1, groups=channels//2)
    
    def forward(self, x, y):
        h = interlaced_cat(x, y)
        return self.gate(h) * self.trans(h)
    
class GlobalGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.trans = DoubleConv(channels, channels, activate=nn.LeakyReLU(), groups=channels)
        self.channels = channels
    
    def forward(self, x):
        b, c, h, w = x.shape
        assert self.channels%c == 0, f"The channels of the input should be divisible by the channels of the global gate."
        factor = self.channels//c
        x = x.view(b, 1, c, h, w).expand(-1, factor, -1, -1, -1).reshape(b, -1 ,h, w)
        h = torch.chunk(self.trans(x), factor, dim=1)
        return h[0]
    
class Gatev_0(nn.Module):
    """
    $$ y(x,y)=f(x,y)*\sigma(g(x,y))$$
    """
    def __init__(self, in_channels):
        super().__init__()
        self.trans = DoubleConv(in_channels, in_channels)
        self.gate = nn.Sequential(
            CR(in_channels, in_channels*2, 3, 1, 1),
            CA(in_channels*2, in_channels, 3, 1, 1, activation=nn.Sigmoid())
        )
    
    def forward(self, x, y):
        h = torch.cat([x, y], dim=1)
        return self.gate(h) * self.trans(h)

class Gatev_1(nn.Module):
    """
        $$ g(x)=\sigma(Conv_{depth-wise}(Avg(x))) $$
        $$ f(x)=Conv_{point-wise}(x*g(x)) $$
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            CA(channels, channels, 1, groups=channels, activation=nn.Sigmoid())
        )
        self.trans = CR(channels, channels, 1, groups=channels)
        self.proj = CBR(channels, channels, 1)
    
    def forward(self, x):
        return self.proj(self.trans(x) * self.gate(x))

class Gatev_2(nn.Module):
    def __init__(self, channels, step=1):
        super(Gatev_2, self).__init__()
        self.open = OpenGate(channels, step)
        self.close = CloseGate(channels, step)

    def forward(self, x):
        return self.open(x) + self.close(x)

class Gatev_3(nn.Module):
    """
    $$ F(x,y)=x*\sigma(g(y))+h(y)$$
    """
    def __init__(self, in_channels):
        super(Gatev_3, self).__init__()
        self.trans = DoubleConv(in_channels, in_channels)
        self.gate = nn.Sequential(
            CR(in_channels, in_channels*2, 3, 1, 1),
            CA(in_channels*2, in_channels, 3, 1, 1, activation=nn.Sigmoid())
        )
    
    def forward(self, x, y):
        return y * self.gate(x) + self.trans(x)
    
class Gatev_4(nn.Module):
    """
    Q: B, 1 x C, H, W 
    K: B, C, H, W
    V: B, C, H, W
    As is seen from the hidden map, the information of $$\mathbb{I}^Q_{i,j}$$ only w.r.t. its neighbors, hence local attention is enough.
    """
    def __init__(self, in_channels, kernel_size=3, k_factor=4):
        super(Gatev_4, self).__init__()
        k_dim = in_channels // k_factor
        self.Q = CR(in_channels, in_channels, kernel_size, 1, padding='same', groups=in_channels)
        self.K = CR(in_channels, k_dim, kernel_size, 1, padding='same')
        self.V = CR(1, k_dim, 1)
        self.k_dim = k_dim
    
    def forward(self, x, y, z):
        b, c, h, w = y.shape
        Q = eop.rearrange(self.Q(y), 'b c h w -> (b c h w) () ()')
        K = eop.rearrange(self.K(x).view(b, 1, -1, h, w).expand(-1, c, -1, -1, -1), 'b c k h w -> (b c h w) k ()')
        V = eop.rearrange(self.V(eop.rearrange(z, 'b c h w -> (b c) () h w')), 'l k h w -> (l h w) k ()') # (b c) c h w
        QKV = F.scaled_dot_product_attention(Q, K, V)
        return eop.rearrange(QKV, '(b c h w) () () -> b c h w', b=b, c=c, h=h, w=w)
    
class Gatev_5(Gatev_4):
    """
    Q: B, 1 x C, H, W 
    K: B, C, H, W
    V: B, C, H, W
    As is seen from the hidden map, the information of $$\mathbb{I}^Q_{i,j}$$ only w.r.t. its neighbors, hence local attention is enough.
    """
    def __init__(self, in_channels, kernel_size=3, k_factor=4):
        super(Gatev_5, self).__init__(in_channels, kernel_size=3, k_factor=4)
    
    def forward(self, x, y):
        super().forward(x, y, y)

class GGUNet(GUNet):
    """
        Down:pre->f1,f2;
        Up:fuse(pre->pre^, f1),*f2->pre+,f2+
    """
    def __init__(self, n_channels, n_classes, mid=[32, 64, 128, 256],bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGUNet, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                    return_feature=return_feature, bottle=bottle, **kwargs)
        up = []
        for i in range(len(mid)-1, 0, -1):
            up.append(GateUpv_0(mid[i]*2, mid[i]*2, bilinear))
        up.append(GateUpv_0(mid[0]*2, mid[0]*2, bilinear))
        up[0].up = nn.ConvTranspose2d(mid[-1]*2, mid[-1], 3, 2)
        up[0].gate = Gate(mid[-1]*2)
        self.up = nn.ModuleList(up)
        self.outc = OutConv(mid[0], n_classes)

    def forward(self, x):
        res_features = []
        for layer in self.down:
            if len(res_features) > 0:
                x, h = torch.chunk(res_features.pop(), 2, 1)
                res_features.append((x, h))
            x = layer(x)
            res_features.append(x)
        x = self.bottle(res_features.pop())
        h = x
        gate, x = x, None
        for layer in self.up:
            x_, h_ = res_features.pop()
            gate, x = layer(gate, x_, h_, x)
        y = self.outc(x)
        pre = {
            'mask': y,
            'feature': h,
        }
        if not self.as_layer:
            self.pre = pre
        return y if not self.return_feature else (y, h)
    
    def memo(self):
        return f"Based on {super().__class__.__name__}(:{super().memo()}), the residual connection is replaced by a gate, to filter the information from the counterpart. Using Gatev_2 in the bottleneck."
    
class GGUNet_s(GGUNet):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGUNet_s, self).__init__(mid=[16,32,64,128], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                    return_feature=return_feature, bottle=bottle, **kwargs
        )

class GGUNet_tiny(GGUNet):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGUNet_tiny, self).__init__(mid=[8,16,32,64], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                    return_feature=return_feature, bottle=bottle, **kwargs
        )

class GGGUNet(GGUNet):
    def __init__(self, n_channels, n_classes, mid=[32, 64, 128, 256], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGGUNet, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                return_feature=return_feature, bottle=bottle, **kwargs  
        )
        self.global_gate = GlobalGate(mid[-1]*2)

    def forward(self, x):
        res_features = []
        for layer in self.down:
            if len(res_features) > 0:
                x, h = torch.chunk(res_features.pop(), 2, 1)
                res_features.append((x * (1 + torch.sigmoid(self.global_gate(h))), h))
            x = layer(x)
            res_features.append(x)
        neck = res_features.pop()
        x = self.bottle(neck * (1 + torch.sigmoid(self.global_gate(neck))))
        h = x
        gate, x = x, None
        for layer in self.up:
            x_, h_ = res_features.pop()
            gate, x = layer(gate, x_, h_, x)
        y = self.outc(x)
        pre = {
            'mask': y,
            'feature': h,
        }
        if not self.as_layer:
            self.pre = pre
        return y if not self.return_feature else (y, h)

    def memo(self):
        return f"Based on {super().__class__.__name__}(:{super().memo()}), with global gate to filter common information cross various scales."
  
class GGGUNet_s(GGGUNet):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGGUNet_s, self).__init__(mid=[16,32,64,128], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                return_feature=return_feature, bottle=bottle, **kwargs
        )

class GGGUNet_tiny(GGGUNet):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGGUNet_tiny, self).__init__(mid=[8,16,32,64], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                            return_feature=return_feature, bottle=bottle, **kwargs
        )

class GateUnet(BaseModel):
    def __init__(self, n_channels, n_classes, mid=[32, 64, 128, 256],bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GateUnet, self).__init__(**kwargs)
        down = [DoubleConv(n_channels, mid[0])]
        for i in range(len(mid)-1):
            down.append(Down(mid[i], mid[i+1]))
        down.append(Down(mid[-1], mid[-1]*2))
        self.down = nn.ModuleList(down)
        up = []
        for i in range(len(mid)-1, 0, -1):
            up.append(GateUpv_2(mid[i]*2, mid[i], bilinear))
        up.append(GateUpv_2(mid[0]*2, mid[0]*2, bilinear))
        self.up = nn.ModuleList(up)
        self.outc = OutConv(mid[0]*2, n_classes)
        self.return_feature = return_feature
        self.bottle = bottle

    def forward(self, x):
        res_features = []
        for layer in self.down:
            x = layer(x)
            res_features.append(x)
        x = self.bottle(res_features.pop())
        h = x
        for layer in self.up:
            x = layer(x, res_features.pop())
        y = self.outc(x)
        pre = {
            'mask': y,
            'feature': h,
        }
        if not self.as_layer:
            self.pre = pre
        return y if not self.return_feature else (y, h)
    
    def backward(self, x, optimer, closure:Callable[[Dict], Dict]=None, clear_stored=True):
        def closure_(pre):
            with torch.enable_grad():
                if closure is not None:
                    return closure(pre)
                else:
                    loss = {}
                    mask = pre['mask']
                    optimer.zero_grad()
                    loss_bce = F.binary_cross_entropy_with_logits(mask, x.float())
                    loss['bce'] = loss_bce.item()
                    loss_bce.backward()
                    optimer.step()
                    return loss
        return super().backward(x, optimer, closure_, clear_stored)
    
    def memo(self):
        return """
            Vanilla gated Unet. Using GateUpv_2.
        """

class GateUnet_s(GateUnet):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GateUnet_s, self).__init__(mid=[16,32,64,128], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                    return_feature=return_feature, bottle=bottle, **kwargs
        )

class GateUnet_tiny(GateUnet):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GateUnet_tiny, self).__init__(mid=[8,16,32,64], n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                            return_feature=return_feature, bottle=bottle, **kwargs
        )

class Emm(BaseModel):
    def __init__(self, n_channels, n_classes, bottle=nn.Identity(), **kwargs):
        super(Emm, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down = Down(64, 64)
        self.up = GateUpv_2(128, 64)
        self.bottom = DoubleConv(64, 64)
        self.bottle = bottle
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        x = self.inc(x)
        features = []
        while x.shape[-1] >= 16 and x.shape[-2] >= 16:
            x = self.down(x)
            features.append(x)
        x = self.bottle(self.bottom(features.pop()))
        while len(features) > 0:
            x = self.up(x, features.pop())
        y = self.outc(x)
        pre = {
            'mask': y,
            'img': x
        }
        self.pre = pre
        return y

    def backward(self, x, optimer, closure:Callable[[Dict], Dict]=None, clear_stored=True):
        def closure_(pre):
            with torch.enable_grad():
                if closure is not None:
                    return closure(pre)
                else:
                    loss = {}
                    mask = pre['mask']
                    optimer.zero_grad()
                    loss_bce = F.binary_cross_entropy_with_logits(mask, x.float())
                    loss['bce'] = loss_bce.item()
                    loss_bce.backward()
                    optimer.step()
                    return loss
        return super().backward(x, optimer, closure_, clear_stored)
    
    def memo(self):
        return """
            Emm...
        """

class GLAUnet(GateUnet):
    """"
    Using Local Attention instead of Gate.
    """
    def __init__(self, n_channels, n_classes, mid=[32, 64, 128, 256], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GLAUnet, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                    return_feature=return_feature, bottle=bottle, **kwargs
        )
        up = []
        for i in range(len(mid)-1, 0, -1):
            up.append(GateUpv_3(mid[i]*2, mid[i], bilinear))
        up.append(GateUpv_3(mid[0]*2, mid[0]*2, bilinear))
        self.up = nn.ModuleList(up)

    def memo(self):
        return """
            GLAUnet. Using GateUpv_4(Local Attention) instead of Gate.
        """

class GLAUnet_s(GLAUnet):
    def __init__(self, n_channels, n_classes, mid=[16,32,64,128], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GLAUnet_s, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                            return_feature=return_feature, bottle=bottle, **kwargs
        )

class GLAUnet_tiny(GLAUnet):
    def __init__(self, n_channels, n_classes, mid=[8,16,32,64], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GLAUnet_tiny, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                                return_feature=return_feature, bottle=bottle, **kwargs
        )

class GGLAUnet(GGUNet):
    """"
    Using Global Attention instead of Gate.
    """
    def __init__(self, n_channels, n_classes, mid=[32, 64, 128, 256], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGLAUnet, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                        return_feature=return_feature, bottle=bottle, **kwargs
        )
        up = []
        for i in range(len(mid)-1, 0, -1):
            up.append(GateUpv_4(mid[i]*2, mid[i]*2, bilinear))
        up.append(GateUpv_4(mid[0]*2, mid[0]*2, bilinear))
        up[0].up = nn.Sequential(
            nn.ConvTranspose2d(mid[-1]*2, mid[-1], 3, 2),
            DoubleConv(mid[-1], mid[-1])
        )
        up[0].gate = Gatev_4(mid[-1])
        self.up = nn.ModuleList(up)
        self.outc = OutConv(mid[0], n_classes)
        

    def memo(self):
        return f"{self.__doc__} Based on {super().__class__.__name__}(:{super().memo()}), the residual connection is replaced by an attention, to filter the information from the counterpart."
  
class GGLAUnet_s(GGLAUnet):
    def __init__(self, n_channels, n_classes, mid=[16,32,64,128], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGLAUnet_s, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                            return_feature=return_feature, bottle=bottle, **kwargs
        )

class GGLAUnet_tiny(GGLAUnet):
    def __init__(self, n_channels, n_classes, mid=[8,16,32,64], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGLAUnet_tiny, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                                return_feature=return_feature, bottle=bottle, **kwargs
        )

class FGDown(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=True):
        super(FGDown, self).__init__()
        self.f_down = nn.Sequential(
            nn.MaxPool2d(2),
            CR(in_channels, out_channels, 3, 1, 1),
            CAB(out_channels, out_channels, 3, 1, 1)
        )

    def forward(self, fx):
        return self.f_down(fx)
    
class FGDownv_1(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=True):
        super(FGDownv_1, self).__init__()
        self.f_down = nn.Sequential(
            nn.MaxPool2d(2),
        )
        self.g_down = nn.Sequential(
            CR(in_channels, in_channels, 3, 2),
        )
        self.fuse = DoubleConv(in_channels*2, out_channels, out_channels, with_bn=with_bn, groups=2)

    def forward(self, fx, gx):
        fx, gx = self.f_down(fx), self.g_down(gx)
        fx = pad2same(fx, gx)
        return torch.chunk(self.fuse(interlaced_cat(fx, gx)), 2, 1)

class FGDown_(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=True):
        super(FGDown_, self).__init__()
        self.down = DoubleConv(in_channels, out_channels, with_bn=with_bn, groups=1)
    
    def forward(self, fx):
        return self.down(fx)
    
class CatConv(nn.Module):
    def __init__(self, x_channels, y_channels, out_channels, with_bn=False, activation=nn.ReLU()):
        super(CatConv, self).__init__()
        self.conv = CA(x_channels+y_channels, out_channels, kernel_size=1, activation=activation)
    
    def forward(self, x, y):
        return self.conv(torch.cat([x, y], dim=1))
class Sim(nn.Module):
    def __init__(self, in_channels, k_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, k_dim, 3, 1, 1),
            nn.ReLU(),
            DepthWiseConv2d(k_dim, 1),
        )

    def forward(self, x, y):
        return self.conv(torch.einsum('bchw,bdhw->bchw', x, y))
    
class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, *args, **kwargs):
        return self.lambd(*args, **kwargs)

class FGGate(nn.Module):
    def __init__(self, in_channels, use_expansion=True, k_factor=2, min_k=2, max_k=16):
        super(FGGate, self).__init__()
        k_dim = min(max(min_k, in_channels // k_factor), max_k)
        # self.V1 = nn.Sequential(CR(in_channels, in_channels, 1))
        # self.V2 = nn.Sequential(CR(in_channels, in_channels, 3, 1, 1), PixelNorm())
        # self.Sim = Sim(in_channels, in_channels)
        # self.Q = nn.Sequential(CR(in_channels, in_channels, 3, 1, 1)) # rough region
        self.K = nn.Sequential(CR(in_channels, in_channels, 1)) # focus on detail
        # self.QK = SEPlus(k_dim, k_dim, in_channels) if use_expansion else CatConv(k_dim, k_dim, in_channels)
        # self.QK = QKPlus(in_channels, in_channels, in_channels)
        # self.multi1 = Lambda(lambda x, y:x * y)
        self.fuse = nn.Sequential(
            CR(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, in_channels, 1),
        )

    def forward(self, x, y, z, x_):
        # x1 = self.V1(x)
        x1 = x
        # x1_ = self.V2(x_)
        # qk = self.QK(self.Q(y), self.K(z))
        qk = self.K(z)
        # sim = torch.sigmoid(self.Sim(x1, x1_))
        return self.fuse(x1 + qk)

class FGUp(nn.Module):
    def __init__(self, in_channels, qk_channels=None, v_channels=None, out_channels=None):
        super(FGUp, self).__init__()
        out_channels = in_channels // 2 if out_channels is None else out_channels
        self.f_up = nn.Sequential(
            nn.ConvTranspose2d(v_channels, out_channels, 3, 2)
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # CBR(v_channels, in_channels, 3, 1, 1),
        ) if v_channels is not None else nn.Identity()
        self.k_up = nn.Sequential(
            nn.ConvTranspose2d(qk_channels, out_channels, 3, 2),
            # nn.BatchNorm2d(qk_channels),
            # nn.ReLU(),
            # nn.Conv2d(qk_channels, in_channels, 3, 1, 1),
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(),
        )if qk_channels is not None else nn.Identity()
        # self.q_up = nn.Sequential(
        #     nn.ConvTranspose2d(qk_channels, in_channels, 3, 2),
        #     # nn.Upsample(scale_factor=2),
        #     # nn.Conv2d(qk_channels, in_channels, 3, 1 ,1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(),
        #     CBR(in_channels, in_channels, 3, 1, 1),
        # )
        self.q_up = nn.Identity()
        self.gate = Gatev_4(in_channels, k_factor=2)
        self.fuse = DoubleConv(in_channels*2, in_channels)
    
    def forward(self, fx, gx, vx, g_):
        g_ = self.f_up(g_)
        f, g = torch.chunk(self.qk_up(torch.cat([fx, gx], dim=1)), 2, 1)
        f = pad2same(f, vx)
        g = pad2same(g, vx)
        g_ = pad2same(g_, vx)
        filtered = self.gate(f, g, vx)
        g = self.fuse(torch.cat([g_, filtered], dim=1))
        return f, g
    
class FGUpv_1(FGUp):
    def __init__(self, in_channels, qk_channels=None, v_channels=None, out_channels=None, use_expansion=True):
        super(FGUpv_1, self).__init__(in_channels, qk_channels, v_channels, out_channels)
        out_channels = in_channels // 2 if out_channels is None else out_channels
        self.gate = FGGate(out_channels, use_expansion=use_expansion)
        del self.fuse 
    
    def forward(self, fx, gx, vx, g_):
        g_ = self.f_up(g_)
        f, g = self.q_up(fx), self.k_up(gx)
        f = pad2same(f, vx)
        g = pad2same(g, vx)
        g_ = pad2same(g_, vx)
        return f, self.gate(vx, f, g, g_)

class FGFuse(nn.Module):
    def __init__(self, in_channels):
        super(FGFuse, self).__init__()
        self.se = SE(in_channels)
        self.fuse = nn.Sequential(
            CR(in_channels, in_channels, 3, 1, 1),
            CAB(in_channels, in_channels, 3, 1, 1)
        )
        self.resample = MultiHeadResample(in_channels, 4*2)
        self.dv = CAB(in_channels, in_channels * 2, 1, groups=2)
    
    def forward(self, f):
        h = self.resample(self.se(f))
        return torch.chunk(self.dv(h), 2, 1)

class Similar(nn.Module):
    """
    Efficient Attention
    """
    def __init__(self, in_channels):
        super(Similar, self).__init__()
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q_ = torch.softmax(q, dim=1) * c
        k_ = k * torch.rsqrt(h * w)
        kv = torch.einsum('bchw,bChw->bcC', k_, v)
        return torch.einsum('bchw,bcC->bChw', q_, kv)
    

class MultiHeadResample(nn.Module):
    """Non-local with multi-head attention"""
    def __init__(self, in_channels, num_heads=1):
        super(MultiHeadResample, self).__init__()
        assert in_channels % num_heads == 0, "in_channels should be divisible by num_heads"
        self.num_heads = num_heads
        self.inner_channels = in_channels // num_heads

        self.Q = nn.Conv2d(in_channels, in_channels, 1)
        self.K = nn.Conv2d(in_channels, in_channels, 1)
        self.V = nn.Conv2d(in_channels, in_channels, 1)
        self.G = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.InstanceNorm2d(in_channels, affine=True)
        )
        self.W = nn.Conv2d(in_channels, in_channels, 1)

        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        _, _, h, w = x.shape
        g = torch.sigmoid(self.G(x))
        q = eop.rearrange(self.Q(x) * (1 - g), 'b (head c) h w -> b head (h w) c', head=self.num_heads, c=self.inner_channels)
        k = eop.rearrange(self.K(x) * g, 'b (head c) h w -> b head (h w) c', head=self.num_heads, c=self.inner_channels)
        v = eop.rearrange(self.V(x), 'b (head c) h w -> b head (h w) c', head=self.num_heads, c=self.inner_channels)
        
        y = eop.rearrange(F.scaled_dot_product_attention(q, k, v), 'b head (h w) c -> b (head c) h w', h=h, w=w)
        
        return self.W(y) + x
    
class ScaledMultiHeadResample(MultiHeadResample):
    def __init__(self, in_channels, num_heads=1):
        super(ScaledMultiHeadResample, self).__init__(in_channels, num_heads)
        self.R = nn.Parameter(torch.tensor(1.))
    
    def forward(self, x):
        y = super().forward(x)
        return self.R / (torch.std(y) + 1e-8) * y

class SimpleFilter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resample = CBR(in_channels, in_channels, 3, 1, 'same', dilation=2)
    
    def forward(self, x):
        return self.resample(x)
    
class Divide(nn.Module):
    def __init__(self, in_channels=64, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or in_channels
        self.value1 = nn.Sequential(
            CBR(in_channels, mid_channels, 3, 1, 'same'),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(1, 1, 1)
        )
        self.value2 = nn.Sequential(
            CBR(in_channels, mid_channels, 3, 1, 'same'),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(1, 1, 1)
        )
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(1, 1, 1)
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(1, 1, 1),
        )
        self.trans3 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(1, 1, 1),
        )

    def forward(self, inc):
        value1, value2 = self.value1(inc), self.value2(inc)
        seg = self.trans2(self.trans1(inc) - value1)
        return self.trans3(seg - value2)

class FGUnet(BaseModel):
    """
    Two flows: Gate and Feature.
    """
    def __init__(self, n_channels=4, n_classes=1, base_channel=32, depth=4, bottle=nn.Identity(), **kwargs):
        super(FGUnet, self).__init__(**kwargs)
        self.proj = nn.Sequential(
            CR(n_channels,base_channel, 3, 1, 1),
            CAB(base_channel, base_channel, 3, 1, 1)
        )
        # self.inc = nn.Sequential(
        #     nn.Conv2d(n_channels, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        # )
        # self.trans = Divide(64, 128)
        down = [
            FGDown(base_channel << i, base_channel << (i + 1)) for i in range(depth)
        ]
        self.down = nn.ModuleList(down)
        up = [
            *[FGUp(*[base_channel << (i + 1)] * 2, None ) for i in range(depth, 0, -1)]
        ]
        self.up = nn.ModuleList(up)
        self.out = OutConv(base_channel, n_classes)
        self.bottle = bottle
        self.neck_fuse = FGFuse(base_channel << depth)

    def forward(self, x):
        proj = self.proj(x)
        hiddens = [proj]
        # inc = self.trans(self.inc(x))
        f, g = proj, proj
        for layer in self.down:
            f = layer(f)
            hiddens.append(f)
        g_ = hiddens.pop()
        f, g = self.neck_fuse(self.bottle(f))
        for layer in self.up:
            v = hiddens.pop()
            f, g = layer(f, g, v, g_)
            g_ = v
        y = self.out(g)
        pre = {
            # 'inc': inc,
            'mask': y,
            'hidden': f
        }
        self.pre = pre
        return y

    def memo(self):
        return """Divide the model into two parts: the feature flow and the gate flow."""

class FGUnet_s(FGUnet):
    def __init__(self, n_channels, n_classes, bottle=nn.Identity(), **kwargs):
        super(FGUnet_s, self).__init__(base_channel=16, depth=4, n_channels=n_channels, n_classes=n_classes, bottle=bottle, **kwargs)

class FGUnet_tiny(FGUnet):
    def __init__(self, n_channels, n_classes, bottle=nn.Identity(), **kwargs):
        super(FGUnet_tiny, self).__init__(base_channel=8, depth=4, n_channels=n_channels, n_classes=n_classes, bottle=bottle, **kwargs)

class FGUnetv_1(FGUnet):
    def __init__(self, n_channels, n_classes, base_channel=32, depth=4, bottle=nn.Identity(), **kwargs):
        super(FGUnetv_1, self).__init__(base_channel=base_channel, depth=depth, n_channels=n_channels, n_classes=n_classes, bottle=bottle, **kwargs)
        up = [
            FGUpv_1(*[base_channel << i] * 2, None) for i in range(depth, 0, -1)
        ]
        self.up = nn.ModuleList(up)
        # self.out = DOutConv(mid[0], n_classes)
        # self.trans = nn.Conv2d(1, 1, 1)
    
    # def forward(self, x):
    #     self.clear_pre()
    #     mask, latent = super().forward(x)
    #     # mask = mask * torch.sigmoid(latent)
    #     pre = {
    #         'img': x,
    #         'mask': mask,
    #         'latent': latent
    #     }
    #     self.pre |= pre
    #     return mask
    
    def backward(self, x, optimer, closure: Callable[[Dict], Dict] = None, clear_stored=True):
        def closure_(pre):
            with torch.enable_grad():
                if closure is not None:
                    return closure(pre)
                else:
                    loss = {}
                    mask = pre['mask']
                    # inc = pre['inc']
                    # latent = pre['latent']
                    optimer.zero_grad()
                    # loss_diff = diff_loss(mask, x.float(), inc)
                    loss_diff = F.binary_cross_entropy_with_logits(mask, x.float())
                    # loss['diff'] = loss_diff.item()
                    # loss_latent = keep_loss(latent, x.float(), None) 
                    # loss['latent'] = loss_latent.item() 
                    loss_total = loss_diff
                    loss['total'] = loss_total.item()
                    loss_total.backward()
                    optimer.step()
                    return loss
        return super().backward(x, optimer, closure_, clear_stored)

    def memo(self):
        return f"{super().memo()}. Using FGGate"
    
class FGUnetv_1_s(FGUnetv_1):
    def __init__(self, n_channels, n_classes, bottle=nn.Identity(), **kwargs):
            super(FGUnetv_1_s, self).__init__(base_channel=16, depth=4, n_channels=n_channels, n_classes=n_classes, bottle=bottle, **kwargs)
            
class FGUnetv_1_tiny(FGUnetv_1):
    def __init__(self, n_channels, n_classes, bottle=nn.Identity(), **kwargs):
            super(FGUnetv_1_tiny, self).__init__(base_channel=8, depth=4, n_channels=n_channels, n_classes=n_classes, bottle=bottle, **kwargs)

class GGDGUnet(GUNet):
    """"
    Using Double Gate instead of Gate.
    """
    def __init__(self, n_channels, n_classes, mid=[32, 64, 128, 256], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGDGUnet, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                        return_feature=return_feature, bottle=bottle, **kwargs
        )
        up = []
        for i in range(len(mid)-1, -1, -1):
            up.append(FGUpv_1(mid[i]))
        self.up = nn.ModuleList(up)
        self.outc = OutConv(mid[0], n_classes)
    def forward(self, x):
        res_features = []
        for layer in self.down:
            if len(res_features) > 0:
                x, h = torch.chunk(res_features.pop(), 2, 1)
                res_features.append((x, h))
            x = layer(x)
            res_features.append(x)
        x = self.bottle(res_features.pop())
        h = x
        gate, x = x, x
        for layer in self.up:
            x_, h_ = res_features.pop()
            gate, x = layer(gate, x, h_, x_)
        y = self.outc(x)
        pre = {
            'mask': y,
            'feature': h,
        }
        if not self.as_layer:
            self.pre = pre
        return y if not self.return_feature else (y, h)

    def memo(self):
        return f"{self.__doc__} Based on {super().__class__.__name__}(:{super().memo()}), the residual connection is replaced by an doubled gate, to filter the information from the counterpart."
  
class GGDGUnet_s(GGDGUnet):
    def __init__(self, n_channels, n_classes, mid=[16,32,64,128], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGDGUnet_s, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                        return_feature=return_feature, bottle=bottle, **kwargs
        )
        
class GGDGUnet_tiny(GGDGUnet):
    def __init__(self, n_channels, n_classes, mid=[8,16,32,64], bilinear=False, return_feature=False, bottle=nn.Identity(), **kwargs):
        super(GGDGUnet_tiny, self).__init__(mid=mid, n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                                        return_feature=return_feature, bottle=bottle, **kwargs
        )






