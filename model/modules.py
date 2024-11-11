import collections
from itertools import repeat
import math
from turtle import forward
from typing import Iterator, List, Optional, Tuple, TypeVar, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
import einops as eop
from model.utils import adaptive_avgmax_pool2d, adaptive_catavgmax_pool2d, adaptive_pool_feat_mult, conv2d_same, drop_path, generate_gaussian_distance_map, get_channel_dim, get_padding_value, get_spatial_dim


_int_tuple_2_t = Union[int, Tuple[int, int]]
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")

def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()

def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

class CBA(nn.Sequential):
    """
    Conv2d + BatchNorm2d + Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, activation=nn.ReLU(), **kwargs):
        super(CBA, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs),
            nn.BatchNorm2d(out_channels),
            activation
        )

class CA(nn.Sequential):
    """
    Conv2d + Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, activation=nn.ReLU(), **kwargs):
        super(CA, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs),
            activation
        )

class CIA(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, affine=False, activation=nn.ReLU(), **kwargs):
        super(CIA, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs),
            nn.InstanceNorm2d(out_channels, affine=affine),
            activation
        )

class CBR(CBA):
    """
    Conv2d + BatchNorm2d + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, activation=nn.ReLU(), **kwargs)
    
class CR(CA):
    """
    Conv2d + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, activation=nn.ReLU(), **kwargs)
    
class CAB(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, activation=nn.ReLU(), **kwargs):
        super(CAB, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs),
            activation,
            nn.BatchNorm2d(out_channels)    
        )

class Bias(nn.Module):
    """
    Bias layer
    """
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        return x + self.bias

class SPP(nn.Module):
    """
    Spatial Pyramid Pooling\n
    Back-propagation through the SPP layer is highly inefficient when each training sample (i.e. RoI) comes from a different image.(see `https://www.semanticscholar.org/reader/7ffdbc358b63378f07311e883dddacc9faeeaf4b`)
    """
    def __init__(self, levels):
        super(SPP, self).__init__()
        self.levels = levels

    def forward(self, x):
        N, C, H, W = x.size()
        pooled_features = []
        for level in self.levels:
            kh, kw = int(H / level), int(W / level)
            for i in range(level):
                for j in range(level):
                    sub_feature = x[:, :, i * kh:(i + 1) * kh, j * kw:(j + 1) * kw]
                    pooled_feature = torch.max(sub_feature.view(N, C, -1), dim=2)[0]
                    pooled_features.append(pooled_feature)
        output = torch.cat(pooled_features, dim=1)
        return output

class ASPP_C(nn.Sequential):
    """
    Convolution module of `ASPP`
    """
    def __init__(self, in_channels, out_channels, rate, with_bn=True):
        super().__init__(
            CBR(in_channels, out_channels, 3, padding=rate, dilation=rate) if with_bn else CR(in_channels, out_channels, 3, padding=rate, dilation=rate)
        )

class ASPP_P(nn.Sequential):
    """Pooling module of `ASPP`"""
    def __init__(self , in_channels, out_channels, with_bn=True):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            CBR(in_channels, out_channels, 1) if with_bn else CR(in_channels, out_channels, 1),
        )
    
    def forward(self, x):
        shape = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, shape, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    """
    def __init__(self, in_channels, rates, out_channels=256, with_bn=True, project_dropout=0):
        super().__init__()
        self.convs = nn.ModuleList([
            CBR(in_channels, out_channels, 1) if with_bn else CR(in_channels, out_channels, 1),
            *[ASPP_C(in_channels, out_channels, rate, with_bn) for rate in rates],
            ASPP_P(in_channels, out_channels, with_bn)
        ])
        self.proj = nn.Sequential(
            CBR(len(self.convs) * out_channels, out_channels, 1) if with_bn else CR(len(self.convs) * out_channels, out_channels, 1),
            nn.Dropout2d(project_dropout) if project_dropout > 0 else nn.Identity()
        )
    
    def forward(self, x):
        return self.proj(torch.cat([mod(x) for mod in self.convs], dim=1))
    
class GroupConv2d(nn.Conv2d):
    """
    Out channels grouped convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_groups=1, bias=True, **kwargs):
        super(GroupConv2d, self).__init__(in_channels, out_channels * out_groups, kernel_size, stride, padding, dilation, bias=bias, **kwargs)

class Embed2d(nn.Module):
    """
    Embedding convolution(same as conv2d(in, out, 1))
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(Embed2d, self).__init__()
        self.embed = nn.Linear(in_channels, out_channels, bias=bias)
    
    def forward(self, x):
        b, _, h, w = x.shape
        x = eop.rearrange(x, 'b c h w -> (b h w) c')
        x = self.embed(x)
        x = eop.rearrange(x, '(b h w) c -> b c h w', b=b, h=h)
        return x

class DepthWiseConv2d(nn.Conv2d):
    "Depth-wise convolution operation"
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__(channels, channels, kernel_size, stride=stride, padding='same', groups=channels)

class PointWiseConv2d(nn.Conv2d):
    "Point-wise (1x1) convolution operation"
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1, **kwargs)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class DepthwiseSeperableConv2d(nn.Module):
    """
    Depthwise separable convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, act=nn.ReLU(inplace=True)):
        super(DepthwiseSeperableConv2d, self).__init__()
        self.dc = CBA(in_channels, in_channels, kernel_size, stride, padding, activation=act)
        self.pc = CBA(in_channels, out_channels, 1, 1, 0, activation=act)
    
    def forward(self, x):
        x = self.dc(x)
        return self.pc(x)

class DepthwiseSeparableConvPlus(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            se_layer=None, drop_path_rate=0.):
        super(DepthwiseSeparableConvPlus, self).__init__()
        assert in_chs% group_size == 0, 'in_chs must be divisible by group_size'
        groups = in_chs // group_size
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv

        self.conv_dw = create_conv2d_pad(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, groups=groups)
        self.bn1 = nn.Sequential(
                norm_layer(in_chs),
                act_layer(inplace=True)
            )

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d_pad(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = nn.Sequential(
                norm_layer(out_chs),
                act_layer(inplace=True) if self.has_pw_act else nn.Identity()
            )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

class SE(nn.Module):
    """
    Squeeze-and-Excitation Networks, as proposed in https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            CR(in_channels, in_channels // reduction_ratio, 1),
            CA(in_channels // reduction_ratio, in_channels, 1, activation=nn.Sigmoid()),
        )
    
    def forward(self ,x):
        return x * self.se(x)

class SCSEModule(nn.Module):
    """
    Concurrent spatial and channel squeeze & excitation attention module, as proposed in https://arxiv.org/pdf/1803.02579.pdf.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = SE(in_channels, reduction)
        self.sSE = CA(in_channels, 1, 1, activation=nn.Sigmoid())

    def forward(self, x):
        return self.cSE(x) + x * self.sSE(x)

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions. Now padding = 'same' can replace it. 
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, 0, dilation, groups, bias,
        )

    def forward(self, x):
        return conv2d_same(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )
    
class FastAdaptiveAvgPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: F = 'NCHW'):
        super(FastAdaptiveAvgPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.mean(self.dim, keepdim=not self.flatten)

class FastAdaptiveMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.amax(self.dim, keepdim=not self.flatten)

class FastAdaptiveAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim, keepdim=not self.flatten)
        x_max = x.amax(self.dim, keepdim=not self.flatten)
        return 0.5 * x_avg + 0.5 * x_max

class FastAdaptiveCatAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = 'NCHW'):
        super(FastAdaptiveCatAvgMaxPool, self).__init__()
        self.flatten = flatten
        self.dim_reduce = get_spatial_dim(input_fmt)
        if flatten:
            self.dim_cat = 1
        else:
            self.dim_cat = get_channel_dim(input_fmt)

    def forward(self, x):
        x_avg = x.mean(self.dim_reduce, keepdim=not self.flatten)
        x_max = x.amax(self.dim_reduce, keepdim=not self.flatten)
        return torch.cat((x_avg, x_max), self.dim_cat)

class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)

class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)

class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)

class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(
            self,
            output_size: _int_tuple_2_t = 1,
            pool_type: str = 'fast',
            flatten: bool = False,
            input_fmt: str = 'NCHW',
    ):
        super(SelectAdaptivePool2d, self).__init__()
        assert input_fmt in ('NCHW', 'NHWC')
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        if not pool_type:
            self.pool = nn.Identity()  # pass through
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        elif pool_type.startswith('fast') or input_fmt != 'NCHW':
            assert output_size == 1, 'Fast pooling and non NCHW input formats require output_size == 1.'
            if pool_type.endswith('catavgmax'):
                self.pool = FastAdaptiveCatAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('avgmax'):
                self.pool = FastAdaptiveAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('max'):
                self.pool = FastAdaptiveMaxPool(flatten, input_fmt=input_fmt)
            else:
                self.pool = FastAdaptiveAvgPool(flatten, input_fmt=input_fmt)
            self.flatten = nn.Identity()
        else:
            assert input_fmt == 'NCHW'
            if pool_type == 'avgmax':
                self.pool = AdaptiveAvgMaxPool2d(output_size)
            elif pool_type == 'catavgmax':
                self.pool = AdaptiveCatAvgMaxPool2d(output_size)
            elif pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool2d(output_size)
            else:
                self.pool = nn.AdaptiveAvgPool2d(output_size)
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'
    
class DistanceMap(nn.Module):
    def __init__(self, sigma = 1.):
        super().__init__()
        self.sigma = sigma

    def forward(self, clickmap):
        return generate_gaussian_distance_map(clickmap, self.sigma)

class ClickMap(nn.Module):
    '''Points: [b, n, 2]'''
    def __init__(self):
        super().__init__()

    def forward(self, x, points):
        _, _, h, w = x.shape
        b = points.size(0)
        clickmap = torch.zeros((b, h, w), device = x.device, dtype = torch.float32).view(b, -1)
        flat_indices:torch.Tensor = points[:, :, 0] * w + points[:, :, 1]
        for i in range(b):
            item = flat_indices[i]
            clickmap[i] = clickmap[i].scatter(0, item[item >= 0], 1)
        return clickmap.reshape(b,1,h,w)

class Mean(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        
    def forward(self, x):
        return torch.mean(x, dim = self.dim, keepdim=self.keepdim)

class Sum(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        
    def forward(self, x):
        return torch.sum(x, dim = self.dim, keepdim=self.keepdim)

class Residual(nn.Module):
    def __init__(self, fn, do_residual=True):
        super().__init__()
        self.fn = fn
        self.do_residual = do_residual

    def forward(self, x, **kwargs):
        if self.do_residual:
            return self.fn(x, **kwargs) + x
        else:
            return self.fn(x, **kwargs)

class BRBlock(nn.Sequential):
    """
    Bottleneck Residual Block, as proposed in MobileNetV2: Inverted Residuals and Linear Bottlenecks(https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)
    """
    def __init__(self, in_channels, out_channels, expansion=2, act=nn.ReLU(inplace=True), noskip=False):
        expanded = in_channels * expansion
        if in_channels != out_channels:
            noskip = True
        super(BRBlock, self).__init__(
            Residual(
                nn.Sequential(
                    CBA(in_channels, expanded, 1, activation=act),
                    CBA(expanded, expanded, 3, groups = expanded, activation=act),
                    CBA(expanded, out_channels, 1, activation=nn.Identity())
                ), not noskip
            ),
            act
        )

class SEBRBlock(nn.Sequential):
    """
    Bottleneck ResidualBlock with SE
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=2, expansion_se=2, act=nn.ReLU6(inplace=True), noskip=False):
        expanded = in_channels * expansion
        if in_channels != out_channels or stride != 1:
            noskip = True
        super(SEBRBlock, self).__init__(
            Residual(
                nn.Sequential(
                    CBA(in_channels, expanded, 1, activation=act),
                    CBA(expanded, expanded, 3, stride=stride, padding=1 if noskip else"same", groups = expanded, activation=act),
                    CBA(expanded, out_channels, 1, activation=nn.Identity()),
                    SE(out_channels, expansion_se),
                    CBA(out_channels, out_channels, 1, activation=nn.Identity())
                ), not noskip
            ),
            act
        )

class PAModulev0(nn.Module):
    """
    Patch attention module, as proposed in https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2022.892418/full#h11
    """
    def __init__(self, channel, patch_size:int=2, upsampler=None):
        super(PAModulev0, self).__init__()
        self.patch_size = patch_size
        self.Q = nn.Conv1d(channel, channel, 1)
        self.K = nn.Conv1d(channel, channel, 1)
        self.V = nn.Conv1d(channel, channel, 1)
        self.upsampler = upsampler

    def forward(self, x):
        b, c, H, W = x.size()
        n = self.patch_size
        patch_feature = eop.rearrange(F.unfold(x, kernel_size=n, stride = n), "b (c k) (h w) -> b c h w k", k=4, h=H // n, w= W //n).mean(-1)
        patch_feature = patch_feature.view(b, c, -1) 
        Q = self.Q(patch_feature).transpose(-1, -2)
        K = self.K(patch_feature).transpose(-1, -2)
        V = self.V(patch_feature).transpose(-1, -2)
        att = F.scaled_dot_product_attention(Q, K, V).transpose(-1, -2)
        feature = patch_feature + att
        feature = feature.view(b, c, H // n, W // n)
        if self.upsampler is not None:
            y = self.upsampler(feature)
        else:
            y = F.interpolate(feature, size=(H, W), mode="bilinear")
        return x + y
    
class ShuffleUp(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, factor=2):
        super(ShuffleUp, self).__init__(
            nn.Conv2d(in_channels, out_channels * factor ** 2, kernel_size=kernel_size),
            nn.PixelShuffle(factor)
        )

class OpenGate(nn.Module):
    def __init__(self, in_channels, step=1):
        super(OpenGate, self).__init__()
        self.step = step
        self.pool = nn.MaxPool2d(3, 1, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels)

    def forward(self, x):
        x_ = x
        for _ in range(self.step):
            x_ = -self.pool(-x_)
        for _ in range(self.step):
            x_ = self.pool(x_)
        return x * (1 - torch.sigmoid(F.adaptive_avg_pool2d(self.proj(x_-x), 1)))
    
class CloseGate(OpenGate):
    def __init__(self, in_channels, step=1):
        super(CloseGate, self).__init__(in_channels, step)
    
    def forward(self, x):
        x_ = x
        for _ in range(self.step):
            x_ = self.pool(x_)
        for _ in range(self.step):
            x_ = -self.pool(-x_)
        return x * (1-torch.sigmoid(F.adaptive_avg_pool2d(self.proj(x_-x), 1)))
    
class OpenCloseGate(nn.Module):
    def __init__(self, in_channels, step=1):
        super(OpenCloseGate, self).__init__()
        self.open = OpenGate(in_channels, step)
        self.close = CloseGate(in_channels, step)
        self.fuse = CBR(in_channels*2, in_channels, 1)

    def forward(self, x):
        return self.fuse(torch.cat([self.open(x), self.close(x)], dim=1))
    
class SameHW(nn.Module):
    def __init__(self, layer, mode='bilinear'):
        super(SameHW, self).__init__()
        self.mode = mode
        assert isinstance(layer, nn.Module), "layer must be nn.Module"
        self.layer = layer

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.layer(x)
        return F.interpolate(x, size=(h, w), mode=self.mode)

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

def pixelnorm(x:torch.Tensor, eps=1e-8):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)

class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = eps
        
    def forward(self, x):
        return pixelnorm(x, self.eps)
    
class GroupPixelNorm(nn.Module):
    def __init__(self, num_groups: int, eps=1e-8):
        super(GroupPixelNorm, self).__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.shape
        assert c % self.num_groups == 0, "The number of channels must be divisible by the number of groups"

        group_size = c // self.num_groups
        x = x.view(b, self.num_groups, group_size, h, w)
        x = pixelnorm(x, self.eps)
        x = x.view(b, c, h, w)

        return x
    
class ContrastNorm(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(ContrastNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        positive_mask = x > 0
        negative_mask = x < 0

        positive_values = x * positive_mask
        negative_values = x * negative_mask  

        positive_mean = positive_values.sum(dim=[2, 3], keepdim=True) / positive_mask.sum(dim=[2, 3], keepdim=True).clamp(min=self.epsilon)
        positive_variance = ((positive_values - positive_mean) ** 2 * positive_mask).sum(dim=[2, 3], keepdim=True) / positive_mask.sum(dim=[2, 3], keepdim=True).clamp(min=self.epsilon)
        
        negative_mean = negative_values.sum(dim=[2, 3], keepdim=True) / negative_mask.sum(dim=[2, 3], keepdim=True).clamp(min=self.epsilon)
        negative_variance = ((negative_values - negative_mean) ** 2 * negative_mask).sum(dim=[2, 3], keepdim=True) / negative_mask.sum(dim=[2, 3], keepdim=True).clamp(min=self.epsilon)

        positive_std = torch.sqrt(positive_variance + self.epsilon)
        negative_std = torch.sqrt(negative_variance + self.epsilon)
        
        normalized_positive_values = (positive_values - positive_mean) / positive_std
        normalized_negative_values = (negative_values - negative_mean) / negative_std

        result = normalized_positive_values * positive_mask + normalized_negative_values * negative_mask

        return F.relu(-result)
    
class PN_NCA(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.CA = CA(in_channels, in_channels, kernel_size=kernel_size, padding='same', activation=nn.LeakyReLU()) if in_channels else None
    
    def forward(self, x):
        x = x.relu()
        x_norm = pixelnorm(x)
        return self.CA(x_norm) if self.CA else x_norm

class PNNorm(nn.Module):
    def __init__(self, in_channels=None):
        super(PNNorm, self).__init__()
        self.P = PN_NCA(in_channels)
        self.N = PN_NCA(in_channels)
    
    def forward(self, x):
        return self.P(x) - self.N(-x)
    
class SEPlus(nn.Module):
    def __init__(self, q_channels, k_channels, out_channels):
        super(SEPlus, self).__init__()
        self.qk = nn.Sequential(
            nn.Conv2d(q_channels*k_channels, out_channels, 3, 1, 1),
            nn.ReLU()
            # nn.LeakyReLU(),
            # nn.Conv2d(out_channels, out_channels, 1, groups=out_channels)
        )

    def forward(self, q, k):
        b, _, h, w = q.shape
        return self.qk(torch.einsum('bchw,bdhw->bcdhw', q, k).reshape(b, -1, h, w))

class ChannelAtt(nn.Module):
    def __init__(self, q_channels, k_channels, out_channels):
        super().__init__()
        self.Q = nn.Conv2d(q_channels, out_channels, 1)
        self.K = nn.Conv2d(k_channels, out_channels, 1)
        self.V = nn.Conv2d(k_channels, out_channels, 1)

    def forward(self, q, k):
        return torch.softmax(self.Q(q) * self.K(k), 1) * self.V(k)

class DropRand(nn.Module):
    def __init__(self, in_channels, p=0.5):
        super(DropRand, self).__init__()
        self.proj = CR(in_channels, in_channels, 3, 1, 1)
        self.p = p
        
    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
            return x * mask + self.proj(x) * (1-mask)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pe', self._get_pe(max_len, d_model))

    def forward(self, x):
        assert x.dim() == 4, "Input tensor must be 4-dimensional (batch_size, channels, height, width)"

        pe = self.pe[:, :, :x.size(2), :x.size(3)]  # Trim to match input spatial dimensions
        return x + pe

    def _get_pe(self, max_len, d_model):
        pe = torch.zeros(1, d_model, max_len, max_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2, :, :] = torch.sin(position * div_term)
        pe[:, 1::2, :, :] = torch.cos(position * div_term)
        return pe
    
class QKPlus(nn.Module):
    def __init__(self, q_channels, k_channels, out_channels):
        super(QKPlus, self).__init__()
        self.q_proj = nn.Conv2d(q_channels, out_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(k_channels, out_channels, kernel_size=1)
        
        self.fusion = nn.Sequential(
            CR(out_channels * 2, out_channels, 3, 1, 1),
            CR(out_channels, out_channels, 3, 1, 1)
        )
        
        self.attention = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, q, k):
        q_proj = self.q_proj(q)  
        k_proj = self.k_proj(k)  
        
        combined = torch.cat([q_proj, k_proj], dim=1)  
        fused = self.fusion(combined)  
        
        attention_map = torch.sigmoid(self.attention(fused)) 
        
        enhanced_features = k * attention_map  
        
        return enhanced_features

class ModuleStack(nn.Module):
    def __init__(self, pre:nn.Module, sub:nn.Module=None, post:nn.Module=None):
        super().__init__()
        self.pre = pre
        self.sub = sub
        self.post = post

    def forward(self, *args, **kwargs):
        pre = self.pre(*args, **kwargs)
        sub = self.sub(pre) if self.sub else pre
        post = self.post(pre, sub) if self.post else sub
        return post
