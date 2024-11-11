from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModel, LossWrap
from model.modules import PAModulev0, SEBRBlock
from model.utils import pad2same
from utils.losses import dice_loss

class BRUnet(BaseModel):
    """
    As proposed in https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2022.892418/full#h11
    """
    def __init__(self, in_channels=3, out_channels=2, **kwargs):
        super(BRUnet, self).__init__(**kwargs)
        self.inc = SEBRBlock(in_channels, 16)
        self.down = nn.ModuleList([
            SEBRBlock(16, 16),
            *[SEBRBlock(16 * 2**i, 16 * 2**(i+1), stride=2) for i in range(4)],
        ])
        self.bottle = PAModulev0(256)
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, 3, 2, output_padding=1),
            nn.Sequential(
                SEBRBlock(256, 128),
                SEBRBlock(128, 128),
                nn.ConvTranspose2d(128, 64, 3, 2, output_padding=1)
            ),
            nn.Sequential(
                SEBRBlock(128, 64),
                nn.ConvTranspose2d(64, 32, 3, 2, output_padding=1)
            ),
            nn.Sequential(
                SEBRBlock(64, 32),
                nn.ConvTranspose2d(32, 16, 3, 2, output_padding=1)
            )
        ])
        self.out = nn.Sequential(
            SEBRBlock(32, 16),
            nn.Conv2d(16, out_channels, 1)
        )
    
    def forward(self, x):
        x = self.inc(x)
        down_outs = []
        for layer in self.down:
            x = layer(x)
            down_outs.append(x)
        x = self.bottle(down_outs.pop())
        for layer in self.up:
            h = down_outs.pop()
            x = pad2same(layer(x), h)
            x = torch.cat([x, h], dim=1)
        y = self.out(x)
        self.pre = {
            'mask': y,
        }
        return y
    
    def backward(self, x, optimer, closure=None, clear_stored=True):
        default = LossWrap(
            {
                'bce':{
                    'loss':F.binary_cross_entropy_with_logits,
                    'args':{}
                },
                'dice':{
                    'loss':dice_loss,
                    'args':{}
                }
            }
        )
        return super().backward(x, optimer, closure if not closure is None else default, clear_stored)

    def memo(self):
        return f"Inserting the patch attention module after the encoder module and inserted the squeeze and excitation block into the bottleneck residual blocks of plain BRU-Net, as proposed in https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2022.892418/full#h11 "
    
