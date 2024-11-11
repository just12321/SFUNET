from enum import Enum
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import einops as eop

class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'
    
FormatT = Union[str, Format]
_int_tuple_2_t = Union[int, Tuple[int, int]]

def generate_gaussian_distance_map(clickmap:torch.Tensor, sigma = 1.)->torch.Tensor:
    '''
    To generate a Gaussian Distance Map from the binary user `clickmap`, 1 represents clicked points whereas 0 means unclicked points.\n
    If a point belongs to multiple clicked region, the maximum is taken. \n
    :param clickmap: [b,1,h,w]
    :param sigma:  
    :return:  [b,h,w]
    '''
    _,_,h,w = clickmap.shape
    xx,yy = torch.meshgrid(torch.arange(0,w,1, device = clickmap.device),torch.arange(0,h,1, device = clickmap.device), indexing = "xy")
    if isinstance(sigma, (tuple, list)):
        dist_maps = [torch.zeros_like(clickmap).squeeze(1) for _ in range(len(sigma))] 
        clicks = clickmap.nonzero()
        for i in range(clicks.size(0)):
            idx, x, y = clicks[i, 0], clicks[i, 3], clicks[i, 2]
            dist = (xx - x) ** 2 + (yy - y) ** 2
            for j, s in enumerate(sigma):
                s = 2 * s ** 2
                dist_maps[j][idx] = torch.where(dist_maps[j][idx] < (dist_new := torch.exp(-dist / s)), dist_new, dist_maps[j][idx])
        return dist_maps
    else:
        dist_map = torch.zeros_like(clickmap).squeeze(1)
        sigma = 2 * sigma ** 2
        clicks = clickmap.nonzero()
        for i in range(clicks.size(0)):
            idx, x, y = clicks[i, 0], clicks[i, 2], clicks[i, 1]
            dist = (xx - x) ** 2 + (yy - y) ** 2
            dist_map[idx] = torch.where(dist_map[idx] < (dist_new := torch.exp(-dist / sigma)), dist_new, dist_map[idx])
        return dist_map.unsqueeze(1)

def _init_weight_goog(m, n='', fix_group_fanout=True):
    """ Weight initialization as per Tensorflow official implementations.

    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs

    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)

def init_weights(model):
    if hasattr(model, 'init_weights'):
        model.init_weights()
    elif hasattr(model, '_init_weights'):
        # model._init_weights(model)
        pass
    elif isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)

# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
    if isinstance(x, torch.Tensor):
        return torch.clamp(((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x, min=0)
    else:
        return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)

# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic

def pad_same(
        x,
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int] = (1, 1),
        value: float = 0,
):
    ih, iw = x.size()[-2:]
    pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    return x

def pad2same(x, y):
        # input is BCHW
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]

        return F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

def conv2d_same(
    x,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

def get_spatial_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim

def get_channel_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x, output_size: _int_tuple_2_t = 1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size: _int_tuple_2_t = 1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool2d(x, pool_type='avg', output_size: _int_tuple_2_t = 1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

def interlaced_cat(x: torch.Tensor, y: torch.Tensor):
    """
    Cat x and y interlaced.
    """
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")
    return eop.rearrange(torch.stack([x,y],dim=1),'b n c h w -> b (c n) h w', n=2)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(
        [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    x = windows.reshape(
        [-1, H // window_size, W // window_size, window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, H, W, C])
    return x
