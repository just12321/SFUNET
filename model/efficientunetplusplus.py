from copy import deepcopy
import functools
from itertools import chain
import math
import re
import types
from typing import Callable, Dict, List, Optional, Tuple, Union
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import BaseModel, LossWrap
from model.modules import Activation, Conv2dSame, DepthWiseConv2d, DepthwiseSeparableConvPlus, DropPath, PointWiseConv2d, SCSEModule, SelectAdaptivePool2d, create_conv2d_pad, patch_first_conv
from model.utils import _init_weight_goog, pad2same
from torch.utils.checkpoint import checkpoint

from utils.losses import dice_loss, focal_loss

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None, round_limit=0.9):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    return make_divisible(channels * multiplier, divisor, channel_min, round_limit=round_limit)

def _scale_stage_depth(stack_args, repeats, depth_multiplier=1.0, depth_trunc='ceil'):
    """ Per-stage depth scaling
    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    num_repeat = sum(repeats)
    if depth_trunc == 'round':
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via 'ceil'.
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled

def make_ds(c, k, s, se):
    return dict(
        block_type="ds",
        out_chs=c,
        stride=s,
        act_layer=None,
        dw_kernel_size=k,
        pw_kernel_size=1,
        se_ratio=se,
        pw_act=False,
        noskip=False,
    )

def make_ir(c, k, s, e, se):
    return dict(
        block_type="ir",
        out_chs=c,
        stride=s,
        act_layer=None,
        dw_kernel_size=k,
        exp_kernel_size=1,
        pw_kernel_size=1,
        exp_ratio=e,
        se_ratio=se,
        noskip=False,
    )

def decode_arch_def(
        depth_multiplier=1.0,
):
    """ Decode block architecture definition strings -> block kwargs

    Args:
        arch_def: architecture definition strings, list of list of strings
        depth_multiplier: network depth multiplier
        depth_trunc: networ depth truncation mode when applying multiplier
        experts_multiplier: CondConv experts multiplier
        fix_first_last: fix first and last block depths when multiplier is applied
        group_size: group size override for all blocks that weren't explicitly set in arch string

    Returns:
        list of list of block kwargs
    """
    arch_args = []
    archs = [
        [[make_ds(16, 3, 1, 0.25), 1]],
        [[make_ir(24, 3, 2, 6, 0.25), 2]],
        [[make_ir(40, 5, 2, 6, 0.25), 2]],
        [[make_ir(80, 3, 2, 6, 0.25), 2]],
        [[make_ir(112, 5, 1, 6, 0.25), 1]],
        [[make_ir(192, 5, 2, 6, 0.25), 2]],
        [[make_ir(320, 3, 1, 6, 0.25), 1]],
    ]
    depth_multiplier = (depth_multiplier,) * len(archs)
    for stack_idx, (block, multiplier) in enumerate(zip(archs, depth_multiplier)):
        stack_args = []
        repeats = []
        for block_pair in block:
            ba, rep = block_pair
            stack_args.append(ba)
            repeats.append(rep)
        arch_args.append(_scale_stage_depth(stack_args, repeats, multiplier))
    return arch_args

def _create_pool(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features

def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc

def create_classifier(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: str = 'NCHW',
        drop_rate: Optional[float] = None,
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    if drop_rate is not None:
        dropout = nn.Dropout(drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc

def efficientnet_init_weights(model: nn.Module, init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n)

def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()

def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x

class EfficientNetBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """
    def __init__(self, output_stride=32, pad_type='', round_chs_fn=round_channels, se_from_exp=False,
                 act_layer=None, norm_layer=None, se_layer=None, drop_path_rate=0., feature_location=''):
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.round_chs_fn = round_chs_fn
        self.se_from_exp = se_from_exp  # calculate se channel reduction from expanded (mid) chs
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.se_layer = se_layer
        try:
            self.se_layer(8, rd_ratio=1.0)  # test if attn layer accepts rd_ratio arg
            self.se_has_ratio = True
        except TypeError:
            self.se_has_ratio = False
        self.drop_path_rate = drop_path_rate
        if feature_location == 'depthwise':
            # old 'depthwise' mode renamed 'expansion' to match TF impl, old expansion mode didn't make sense
            feature_location = 'expansion'
        self.feature_location = feature_location
        assert feature_location in ('bottleneck', 'expansion', '')

        # state updated during build, consumed by model
        self.in_chs = None
        self.features = []

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self.round_chs_fn(ba['out_chs'])
        if 'force_in_chs' in ba and ba['force_in_chs']:
            # NOTE this is a hack to work around mismatch in TF EdgeEffNet impl
            ba['force_in_chs'] = self.round_chs_fn(ba['force_in_chs'])
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        ba['norm_layer'] = self.norm_layer
        ba['drop_path_rate'] = drop_path_rate
        if bt != 'cn':
            se_ratio = ba.pop('se_ratio')
            if se_ratio and self.se_layer is not None:
                if not self.se_from_exp:
                    # adjust se_ratio by expansion ratio if calculating se channels from block input
                    se_ratio /= ba.get('exp_ratio', 1.0)
                if self.se_has_ratio:
                    ba['se_layer'] = functools.partial(self.se_layer, rd_ratio=se_ratio)
                else:
                    ba['se_layer'] = self.se_layer

        if bt == 'ir':
            block =  InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            block = DepthwiseSeparableConvPlus(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt

        self.in_chs = ba['out_chs']  # update in_chs for arg of next block
        return block

    def __call__(self, in_chs, model_block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            model_block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        stages = []
        if model_block_args[0][0]['stride'] > 1:
            # if the first block starts with a stride, we need to extract first level feat from stem
            feature_info = dict(module='bn1', num_chs=in_chs, stage=0, reduction=current_stride)
            self.features.append(feature_info)

        # outer list of block_args defines the stacks
        for stack_idx, stack_args in enumerate(model_block_args):
            last_stack = stack_idx + 1 == len(model_block_args)
            assert isinstance(stack_args, list)

            blocks = []
            # each stack (stage of blocks) contains a list of block arguments
            for block_idx, block_args in enumerate(stack_args):
                last_block = block_idx + 1 == len(stack_args)

                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:   # only the first block in any stack can have a stride > 1
                    block_args['stride'] = 1

                extract_features = False
                if last_block:
                    next_stack_idx = stack_idx + 1
                    extract_features = next_stack_idx >= len(model_block_args) or \
                        model_block_args[next_stack_idx][0]['stride'] > 1

                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                # create the block
                block = self._make_block(block_args, total_block_idx, total_block_count)
                blocks.append(block)

                # stash feature module name and channel info for model feature extraction
                if extract_features:
                    feature_info = dict(
                        stage=stack_idx + 1,
                        reduction=current_stride,
                        **block.feature_info(self.feature_location),
                    )
                    leaf_name = feature_info.get('module', '')
                    if leaf_name:
                        feature_info['module'] = '.'.join([f'blocks.{stack_idx}.{block_idx}', leaf_name])
                    else:
                        assert last_block
                        feature_info['module'] = f'blocks.{stack_idx}'
                    self.features.append(feature_info)

                total_block_idx += 1  # incr global block idx (across all stacks)
            stages.append(nn.Sequential(*blocks))
        return stages

class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.ReLU,
            gate_layer=nn.Sigmoid, force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class EfficientNet(nn.Module):
    """ EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large, XL & B0-B3
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * MobileNet-V2
      * FBNet C
      * Single-Path NAS Pixel1
      * TinyNet
    """

    def __init__(
            self,
            block_args,
            num_classes=1000,
            num_features=1280,
            in_chans=3,
            stem_size=32,
            fix_stem=False,
            output_stride=32,
            pad_type='',
            round_chs_fn=round_channels,
            act_layer=None,
            norm_layer=nn.BatchNorm2d,
            se_layer=None,
            drop_rate=0.,
            drop_path_rate=0.,
            global_pool='avg'
    ):
        super(EfficientNet, self).__init__()
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d_pad(in_chans, stem_size, 3, stride=2, groups=1, padding=pad_type)
        self.bn1 = nn.Sequential(
                    nn.BatchNorm2d(stem_size),
                    nn.SiLU(inplace=True)
                )

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            act_layer=act_layer,
            norm_layer=norm_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d_pad(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = nn.Sequential(
                nn.BatchNorm2d(self.num_features),
                nn.SiLU(inplace=True)
            )
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
                (r'conv_head|bn2', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        assert mid_chs % group_size == 0, "mid_chs % group_size != 0"
        groups = mid_chs //group_size
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        # Point-wise expansion
        self.conv_pw = create_conv2d_pad(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = nn.Sequential(
                norm_layer(mid_chs),
                act_layer(inplace=True)
            )

        # Depth-wise convolution
        self.conv_dw = create_conv2d_pad(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            groups=groups, padding=pad_type, **conv_kwargs)
        self.bn2 = nn.Sequential(
                norm_layer(mid_chs),
                act_layer(inplace=True)
            )

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d_pad(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = nn.Sequential(
                norm_layer(out_chs),
                act_layer(inplace=True)
            )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

class InvertedResidualForDecoder(nn.Module):
    """
    Inverted bottleneck residual block with an scSE block embedded into the residual layer, after the 
    depthwise convolution. By default, uses batch normalization and Hardswish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, expansion_ratio = 1, squeeze_ratio = 1, \
        activation = nn.Hardswish(True), normalization = nn.BatchNorm2d):
        super().__init__()
        self.same_shape = in_channels == out_channels
        self.mid_channels = expansion_ratio*in_channels
        self.block = nn.Sequential(
            PointWiseConv2d(in_channels, self.mid_channels),
            normalization(self.mid_channels),
            activation,
            DepthWiseConv2d(self.mid_channels, kernel_size=kernel_size, stride=stride),
            normalization(self.mid_channels),
            activation,
            #md.sSEModule(self.mid_channels),
            SCSEModule(self.mid_channels, reduction = squeeze_ratio),
            #md.SEModule(self.mid_channels, reduction = squeeze_ratio),
            PointWiseConv2d(self.mid_channels, out_channels),
            normalization(out_channels)
        )
        
        if not self.same_shape:
            # 1x1 convolution used to match the number of channels in the skip feature maps with that 
            # of the residual feature maps
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                normalization(out_channels)
            )
          
    def forward(self, x):
        residual = self.block(x)
        
        if not self.same_shape:
            x = self.skip_conv(x)
        return x + residual

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            squeeze_ratio=1,
            expansion_ratio=1
    ):
        super().__init__()

        # Inverted Residual block convolutions
        self.conv1 = InvertedResidualForDecoder(
            in_channels=in_channels+skip_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )
        self.conv2 = InvertedResidualForDecoder(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            x = pad2same(x, skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class EfficientUnetPlusPlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            squeeze_ratio=1,
            expansion_ratio=1
    ):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels

        # combine decoder keyword arguments
        kwargs = dict(squeeze_ratio=squeeze_ratio, expansion_ratio=expansion_ratio)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx+1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx+1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx+1-depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f'x_{depth_idx}_{layer_idx}'] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f'x_{0}_{len(self.in_channels)-1}'] =\
            DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], **kwargs)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels)-1):
            for depth_idx in range(self.depth-layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f'x_{depth_idx}_{depth_idx}'](features[depth_idx], features[depth_idx+1])
                    dense_x[f'x_{depth_idx}_{depth_idx}'] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f'x_{idx}_{dense_l_i}'] for idx in range(depth_idx+1, dense_l_i+1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i+1]], dim=1)
                    dense_x[f'x_{depth_idx}_{dense_l_i}'] =\
                        self.blocks[f'x_{depth_idx}_{dense_l_i}'](dense_x[f'x_{depth_idx}_{dense_l_i-1}'], cat_features)
        dense_x[f'x_{0}_{self.depth}'] = self.blocks[f'x_{0}_{self.depth}'](dense_x[f'x_{0}_{self.depth-1}'])
        return dense_x[f'x_{0}_{self.depth}']

class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError

    def make_dilated(self, stage_list, dilation_list):
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

class EfficientNetBaseEncoder(EfficientNet, EncoderMixin):

    def __init__(self, stage_idxs, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv_stem, self.bn1),
            self.blocks[:self._stage_idxs[0]],
            self.blocks[self._stage_idxs[0]:self._stage_idxs[1]],
            self.blocks[self._stage_idxs[1]:self._stage_idxs[2]],
            self.blocks[self._stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.bias")
        state_dict.pop("classifier.weight")
        super().load_state_dict(state_dict, **kwargs)

class EfficientNetEncoder(EfficientNetBaseEncoder):

    def __init__(self, stage_idxs, out_channels, depth=5, channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
        cm = 1280 * channel_multiplier
        new_v = max(8, int(cm + 4) // 8 * 8)
        if new_v < 0.9 * cm:
            new_v += 8
        kwargs = {
            "block_args": decode_arch_def(depth_multiplier),
            "num_features": new_v,
            "stem_size": 32,
            # "channel_multiplier": channel_multiplier,
            "act_layer": nn.SiLU,
            # "norm_kwargs": {},
            "drop_rate": drop_rate,
            "drop_path_rate": 0.2,
        }
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)

class EfficientUnetPlusPlus(nn.Module):
    """The EfficientUNet++ is a fully convolutional neural network for ordinary and medical image semantic segmentation. 
    Consists of an *encoder* and a *decoder*, connected by *skip connections*. The encoder extracts features of 
    different spatial resolutions, which are fed to the decoder through skip connections. The decoder combines its 
    own feature maps with the ones from skip connections to produce accurate segmentations masks.  The EfficientUNet++ 
    decoder architecture is based on the UNet++, a model composed of nested U-Net-like decoder sub-networks. To 
    increase performance and computational efficiency, the EfficientUNet++ replaces the UNet++'s blocks with 
    inverted residual blocks with depthwise convolutions and embedded spatial and channel attention mechanisms.
    Synergizes well with EfficientNet encoders. Due to their efficient visual representations (i.e., using few channels
    to represent extracted features), EfficientNet encoders require few computation from the decoder.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone) to extract features
        encoder_depth: Number of stages of the encoder, in range [3 ,5]. Each stage generate features two times smaller, 
            in spatial dimensions, than the previous one (e.g., for depth=0 features will haves shapes [(N, C, H, W)]), 
            for depth 1 features will have shapes [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in the decoder.
            Length of the list should be the same as **encoder_depth**
        in_channels: The number of input channels of the model, default is 3 (RGB images)
        classes: The number of classes of the output mask. Can be thought of as the number of channels of the mask
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is built 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **EfficientUnet++**

    Reference:
        https://arxiv.org/abs/2106.11447
    """

    def __init__(
        self,
        encoder_depth: int = 5,
        encoder_pretrained:bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        squeeze_ratio: int = 1,
        expansion_ratio: int = 1,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.classes = classes
        self.encoder = EfficientNetEncoder(depth=encoder_depth, out_channels=(3,32,24,40,112,320), \
                                           stage_idxs=(2,3,5), channel_multiplier=1., depth_multiplier=1., drop_rate=0.2)
        if encoder_pretrained:
            # weights = {
            #     "mean": (0.485, 0.456, 0.406),
            #     "std": (0.229, 0.224, 0.225),
            #     "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ns-c0e6a31c.pth",
            #     "input_range": (0, 1),
            #     "input_space": "RGB",
            # }
            self.encoder.load_state_dict(torch.load(r"F:\开题报告\Codes\Project\weights\tf_efficientnet_b0.pth", map_location=torch.device('cpu')), strict=False)
        self.encoder.set_in_channels(in_channels)
        self.decoder = EfficientUnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "EfficientUNet++-efficientnet_b0"
        self.initialize()

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            output = self.forward(x)

        if self.classes > 1:
            probs = torch.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(x.size[1]),
                transforms.ToTensor()
            ]
        )
        full_mask = tf(probs.cpu())   

        return full_mask
    
    def initialize(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.segmentation_head.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.classification_head is not None:
            for m in self.classification_head.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
    
class EfficientUnetPP(BaseModel):
    """
    EfficientUNet++ model from https://arxiv.org/pdf/2106.11447
    """
    def __init__(self, in_channels=3, n_classes=1, encoder_depth=5, activation=None, encoder_pretrained=True):
        super().__init__()
        self.net = EfficientUnetPlusPlus(encoder_depth=encoder_depth, encoder_pretrained=encoder_pretrained, 
                                         in_channels=in_channels, classes=n_classes, activation=activation)
        self.pre = None
    
    def forward(self, x):
        mask = self.net(x)
        self.pre = {
            "img": x,
            "mask": mask
        }
        return mask

    def backward(self, x, optimer, closure:Callable[[Dict], Dict]=None, clear_stored=True):
        default = LossWrap(
            {
                'focal':{
                    'loss': focal_loss,
                    'args': {},
                    'weight': 1.0,
                },
                'dice':{
                    'loss': dice_loss,
                    'args': {
                        'use_weights': True,
                        'k': 0.75
                    },
                    'weight': 1.0,
                }
            }
        )
        losses =  super().backward(x, optimer, closure if closure else default, clear_stored)
        nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
        return losses
    
    def memo(self):
        return """
        EfficientUnet++ model from https://arxiv.org/pdf/2106.11447
        """