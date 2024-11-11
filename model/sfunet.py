from functools import partial
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModel, wrap_iou
from model.modules import CA, CBR, CR, ModuleStack, PixelNorm, Sum, pixelnorm, _pair
from model.utils import pad2same
from einops import rearrange
from torchvision.utils import save_image
from utils.utils import plot_surface

from utils.losses import IOU_loss, dice_loss, focal_loss

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.trans = nn.Sequential(
            CBR(in_channels, out_channels, 3, 1, 1),
            CR(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        h = self.trans(x[0])
        return self.down(h), F.relu(h)
    
class ScaleIt(nn.Module):
    def __init__(self, in_channels):
        super(ScaleIt, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1, bias=False, groups=in_channels)
    
    def forward(self, x):
        return self.conv(self.conv(x))

def wrapped_kl(x, y, reduction='none'):
    # return F.kl_div(x.log(), (y >= 0.5).float(), reduction=reduction)
    return F.kl_div(F.log_softmax(x, dim=1), F.softmax(y, dim=1), reduction=reduction)

class DecoderBlock(nn.Module):
    losses = {
        'b': F.binary_cross_entropy,
        'l': F.l1_loss,
        'm': F.mse_loss,
        'f': partial(focal_loss, from_logits=True),
        'k': wrapped_kl,
        'd': partial(dice_loss, from_logits=True),
        'i': partial(IOU_loss, from_logits=True),
    }
    def __init__(self, in_channels, out_channels, use_gate=True, gcm_prob=0.5, refine_gate=True):
        super(DecoderBlock, self).__init__()
        self.gcm_prob = gcm_prob
        self.refine_gate = refine_gate
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.convert = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels // 2),
            nn.Conv2d(out_channels // 2, out_channels, 3, 1, 1),
        )
        self.glob = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            ScaleIt(out_channels),
            Sum(1, True),
            nn.Sigmoid()
        ) if use_gate else None
 
        self.local = nn.Sequential(
            CBR(out_channels, out_channels // 2, 3, 1, 1),
            CBR(out_channels // 2, out_channels // 2, 3, 1, 1),
            nn.Conv2d(out_channels // 2, out_channels, 1),
        )

        self.attention = nn.Sequential(
            CBR(out_channels, out_channels // 2, 3, 1, 1), 
            CR(out_channels // 2, out_channels // 2, 3, 1, 1),
            nn.InstanceNorm2d(out_channels // 2),
            CA(out_channels // 2, 1, 1, activation=nn.Sigmoid()),
        )
        self.filtered_x_log = nn.Identity()
        self.pre_log = nn.Identity()
        self.post_log = nn.Identity()

    def l1_compensation(self, x, gate, tar=None, with_eq=True, threshold=0.5, p=0., margin=0.):
        tar = gate if tar is None else tar
        tar = tar if threshold is None else (tar <= threshold if with_eq else tar < threshold).float()
        return F.relu(F.dropout(gate * tar * self.randn(x), p) - margin)
    
    def contrastive_compensation(self, x, gate, tar=None, with_eq=True, threshold=0.5, p=0., margin=0.):
        tar = gate if tar is None else tar
        tar = tar if threshold is None else (tar <= threshold if with_eq else tar < threshold).float()
        return F.relu(F.dropout(F.binary_cross_entropy(gate, tar, reduction='none') * self.randn(x), p) - margin)
    
    def compensation(self, x, gate, tar=None, threshold=0.5, p=0., type='bl', margin=0.):
        tar = gate if tar is None else tar
        target = tar if threshold is None else (tar > threshold).float()
        loss = torch.where(target < 1, 
                        self.losses[type[0]](gate, target, reduction='none'), 
                        self.losses[type[1]](gate, target, reduction='none'))
        return F.relu(F.dropout(loss * self.randn(x), p) - margin)

    def align_compensation(self, x, e_f, d_f):
        mean_e = F.adaptive_avg_pool2d(e_f.relu(), (1, 1))
        mean_d = F.adaptive_avg_pool2d(d_f.relu(), (1, 1))
        kl = F.kl_div(F.log_softmax(mean_e, dim=1), F.softmax(mean_d, dim=1))
        return  kl * self.randn(x)
    
    def randn(self, x):
        return (torch.randn_like(x) * x.std() + x.mean()).clone().detach()

    def forward(self, x, h):
        h_up = self.upsample(h[0])  
        h_g = self.convert(h[1])
        h_up = pad2same(h_up, x[1])
        h_g = pad2same(h_g, x[1])
        gate = self.glob(h_g) if self.glob else 1.
        # However, `self.attention(self.local(h_up + gate * x[1]))`, namely filter encoder feature first sometimes offer a better performance. 
        # It might because the lack of coherence in spatial features caused by filtering precisely improves generalization ability through regularization.
        # Here, we choose to filter after fusing mainly based on the AG itself is just designed to break the continuity by GCM, hence release more parameters focus on fusing. 
        x = self.local(self.pre_log(h_up + x[1]))
        att = self.attention(self.filtered_x_log(x * gate))
        out = x * (att + gate)
        if self.training and self.glob:
            out = out\
                    + self.compensation(out, gate, p=self.gcm_prob, type='kk')\
                    + self.l1_compensation(out, att, (gate + att) / 2, p=self.gcm_prob)
        return out, h_g * (att + 0.1) if self.refine_gate else h_g
    
class CatDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CatDecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 3, 2)
        self.block = nn.Sequential(
            CBR(in_channels, out_channels, 3, 1, 1),
            CBR(out_channels, out_channels, 3, 1, 1),
        )
    
    def forward(self, x, h):
        f = self.up(h[0])
        f = pad2same(f, x[1])
        return self.block(torch.cat([f, x[1]], dim=1)),

class FuseHeads(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fuse = CBR(in_channels, in_channels, 1)
    
    def forward(self, x):
        return self.fuse(x)

def same_conv(conv:nn.Conv2d, x, dilation):
    kernel_size = conv.kernel_size
    dilation = _pair(dilation)
    _reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
    for d, k, i in zip(dilation, kernel_size,
                    range(len(kernel_size) - 1, -1, -1)):
        total_padding = d * (k - 1)
        left_pad = total_padding // 2
        _reversed_padding_repeated_twice[2 * i] = left_pad
        _reversed_padding_repeated_twice[2 * i + 1] = (
            total_padding - left_pad)
    return F.conv2d(F.pad(x, _reversed_padding_repeated_twice), 
                    conv.weight, conv.bias, 1, _pair(0), dilation)

class MultiScale(nn.Module):
    def __init__(self, in_channels, out_channels=None, strides=[1,3,5,7]):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels
        self.strides = strides
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1)
        self.BRs = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU()) for _ in range(len(strides))]
        )

    def forward(self, x):
        y = x
        for stride, BR in zip(self.strides, self.BRs):
            y = y + BR(same_conv(self.conv, x, stride))
        return y

class GSG(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_heads=1, drop_out=0.):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.hidden_dim = hidden_dim if hidden_dim is not None else out_channels
        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = max(num_heads, 1)

        self.Q = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), PixelNorm())
        self.K = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), PixelNorm())
        self.V = nn.Sequential(nn.Conv2d(in_channels, out_channels + self.hidden_dim, 1))
        self.fuse = FuseHeads(out_channels)
        self.dropout = nn.Dropout(drop_out)
        self.qkgv_log = nn.Identity()
        self.qkv_log = nn.Identity()
        self.gate_log = nn.Identity()
    
    def forward(self, x):
        pattern = 'b (g c) h w -> (b g) c h w'
        rvpattern = '(b g) c h w -> b (g c) h w'
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        q = rearrange(q, pattern, g=self.num_heads)
        k = rearrange(k, pattern, g=self.num_heads)
        v = rearrange(v, pattern, g=self.num_heads)

        hidden_dim = self.hidden_dim // self.num_heads

        kv = torch.einsum('bchw,bChw->bcC', k, v)
        qkv = torch.einsum('bchw,bcC->bChw', q, pixelnorm(kv))
        qkv, g = torch.split(qkv, [qkv.size(1) - hidden_dim, hidden_dim], dim=1)

        qkv = self.qkv_log(qkv)
        H, W = g.shape[2:]
        hidden_gate = g.mean(dim=1, keepdim=True) 
        T = sqrt(H * W)
        gate = (hidden_gate / T).flatten(1).softmax(dim=-1).view(-1, 1, H, W)
        self.gate_log(gate)
        qkgv = gate * qkv
        qkgv = self.qkgv_log(qkgv)
        qkgv = rearrange(qkgv, rvpattern, g=self.num_heads)
        fuse = self.dropout(self.fuse(qkgv))
        return fuse

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_heads=1, use_gsg=True):
        super(Bottleneck, self).__init__()

        self.com = nn.Sequential(
            CBR(in_channels, out_channels, 1),
        )
        self.att = nn.Sequential(
            MultiScale(in_channels),
            GSG(in_channels, out_channels, hidden_dim, num_heads) if use_nlg else CBR(in_channels, out_channels, 3, 1, 1)
        )
        
    def forward(self, x): 
        h = self.com(x[0])
        g = self.att(x[0])
        return h, g

class SFUnet(BaseModel):
    def __init__(self, in_channels, num_classes, mid_channels=32, depth=4, hidden_dim=None, num_heads=1, use_sfg=True, use_gsg=True):
        super(SFUnet, self).__init__()
        self.num_heads = num_heads
        self.use_sfg = use_sfg
        self.use_gsg = use_gsg
        self.net = self.build_stack(in_channels, mid_channels, depth, hidden_dim)
        self.out = nn.Sequential(
            CBR(mid_channels, mid_channels, 3, 1, 1),
            nn.Conv2d(mid_channels, num_classes, 1)
        )

    def build_stack(self, in_channels, mid_channels, depth, hidden_dim):
        if depth <= 0:
            return Bottleneck(in_channels, mid_channels, hidden_dim, self.num_heads, self.use_gsg)
        
        return ModuleStack(
            EncoderBlock(in_channels, mid_channels),
            self.build_stack(mid_channels, mid_channels * 2, depth - 1, hidden_dim),
            DecoderBlock(mid_channels * 2, mid_channels) if self.use_sfg else CatDecoderBlock(mid_channels * 2, mid_channels)
        )

    def forward(self, x):  
        y = self.out(self.net((x, ))[0])
        return y

    default_closure = wrap_iou

