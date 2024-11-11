__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
import einops

from model.base import BaseModel, wrap_dice

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True, 
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class double_conv_block(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, *args, **kwargs):
        super().__init__()
        self.conv1 = conv_block(in_features=in_features, out_features=out_features1, *args, **kwargs)
        self.conv2 = conv_block(in_features=out_features1, out_features=out_features2, *args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class double_conv_block_a(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, norm1, norm2, act1, act2, *args, **kwargs):
        super().__init__()
        self.conv1 = conv_block(in_features=in_features, out_features=out_features1, norm_type=norm1, activation=act1, *args, **kwargs)
        self.conv2 = conv_block(in_features=out_features1, out_features=out_features2, norm_type=norm2, activation=act2, *args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Upconv(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                activation=True,
                norm_type='bn', 
                scale=(2, 2)) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, 
                              mode='bilinear', 
                              align_corners=True)
        self.conv = conv_block(in_features=in_features, 
                                out_features=out_features, 
                                norm_type=norm_type, 
                                activation=activation)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class depthwise_conv_block(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=(3, 3),
                stride=(1, 1), 
                padding=(1, 1), 
                dilation=(1, 1),
                groups=None, 
                norm_type='bn',
                activation=True, 
                use_bias=True,
                pointwise=False, 
                ):
        super().__init__()
        self.pointwise = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation, 
            bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv2d(in_features, 
                                        out_features, 
                                        kernel_size=(1, 1), 
                                        stride=(1, 1), 
                                        padding=(0, 0),
                                        dilation=(1, 1), 
                                        bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x    

class depthwise_projection(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                groups,
                kernel_size=(1, 1), 
                padding=(0, 0), 
                norm_type=None, 
                activation=False, 
                pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features, 
                                        out_features=out_features, 
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        groups=groups,
                                        pointwise=pointwise, 
                                        norm_type=norm_type,
                                        activation=activation)
                            
    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P) 
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')      
        return x

class ScaleDotProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
                                                    
    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        x12 = torch.einsum('bhcw, bhwk -> bhck', x1, x2) * scale
        att = self.softmax(x12)
        x123 = torch.einsum('bhcw, bhwk -> bhck', att, x3) 
        return x123  

class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features, 
                                            out_features=out_features, 
                                            groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features) 

        self.projection = depthwise_projection(in_features=out_features, 
                                    out_features=out_features, 
                                    groups=out_features)
        self.sdp = ScaleDotProduct()        
        

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5                     
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k ,v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att

class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4) -> None:
        super().__init__()
        self.n_heads = n_heads

        self.q_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features)
        self.k_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features)
        self.v_map = depthwise_projection(in_features=out_features, 
                                            out_features=out_features, 
                                            groups=out_features)       

        self.projection = depthwise_projection(in_features=out_features, 
                                    out_features=out_features, 
                                    groups=out_features)                                             
        self.sdp = ScaleDotProduct()        

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)  
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5        
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k ,v, scale).transpose(1, 2).flatten(2)    
        x = self.projection(att)
        return x

class CCSABlock(nn.Module):
    def __init__(self, 
                features, 
                channel_head, 
                spatial_head, 
                spatial_att=True, 
                channel_att=True) -> None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                    eps=1e-6) 
                                                    for in_features in features])   

            self.c_attention = nn.ModuleList([ChannelAttention(
                                                in_features=sum(features),
                                                out_features=feature,
                                                n_heads=head, 
                                        ) for feature, head in zip(features, channel_head)])
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                    eps=1e-6) 
                                                    for in_features in features])          
          
            self.s_attention = nn.ModuleList([SpatialAttention(
                                                    in_features=sum(features),
                                                    out_features=feature,
                                                    n_heads=head, 
                                                    ) 
                                                    for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x_ca = self.channel_attention(x)
            x = self.m_sum(x, x_ca)   
        if self.spatial_att:
            x_sa = self.spatial_attention(x)
            x = self.m_sum(x, x_sa)   
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        x_att = self.m_apply(x_in, self.c_attention)
        return x_att    

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]        
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att 
        

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]    

    def cat(self, *args):
        return torch.cat((args), dim=2)

class PoolEmbedding(nn.Module):
    def __init__(self,
                pooling,
                patch,
                ) -> None:
        super().__init__()
        self.projection = pooling(output_size=(patch, patch))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')        
        return x
    
class UpsampleConv(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=(3, 3),
                padding=(1, 1), 
                norm_type=None, 
                activation=False,
                scale=(2, 2), 
                conv='conv') -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, 
                              mode='bilinear', 
                              align_corners=True)
        if conv == 'conv':
            self.conv = conv_block(in_features=in_features, 
                                    out_features=out_features, 
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    norm_type=norm_type, 
                                    activation=activation)
        elif conv == 'depthwise':
            self.conv = depthwise_conv_block(in_features=in_features, 
                                    out_features=out_features, 
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    norm_type=norm_type, 
                                    activation=activation)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class DCA(nn.Module):
    def __init__(self,
                features,
                strides,
                patch=28,
                channel_att=True,
                spatial_att=True,   
                n=1,              
                channel_head=[1, 1, 1, 1], 
                spatial_head=[4, 4, 4, 4], 
                ):
        super().__init__()
        self.n = n
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(
                                                    pooling = nn.AdaptiveAvgPool2d,            
                                                    patch=patch, 
                                                    )
                                                    for _ in features])                
        self.avg_map = nn.ModuleList([depthwise_projection(in_features=feature,
                                                            out_features=feature, 
                                                            kernel_size=(1, 1),
                                                            padding=(0, 0), 
                                                            groups=feature
                                                            )
                                                    for feature in features])         
                                
        self.attention = nn.ModuleList([
                                        CCSABlock(features=features, 
                                                  channel_head=channel_head, 
                                                  spatial_head=spatial_head, 
                                                  channel_att=channel_att, 
                                                  spatial_att=spatial_att) 
                                                  for _ in range(n)])
                     
        self.upconvs = nn.ModuleList([UpsampleConv(in_features=feature, 
                                                    out_features=feature,
                                                    kernel_size=(1, 1),
                                                    padding=(0, 0),
                                                    norm_type=None,
                                                    activation=False,
                                                    scale=stride, 
                                                    conv='conv')
                                                    for feature, stride in zip(features, strides)])                                                      
        self.bn_relu = nn.ModuleList([nn.Sequential(
                                                    nn.BatchNorm2d(feature), 
                                                    nn.ReLU()
                                                    ) 
                                                    for feature in features])
    
    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg) 
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return (*x_out, )      

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]  
        
    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch) 

class DCAUnet(BaseModel):
    def __init__(self,
                in_features=3, 
                out_features=3, 
                input_size=(512, 512),
                n=1,
                k=0.5,
                patch_size=8,
                spatial_att=True,
                channel_att=True,
                spatial_head_dim=[4, 4, 4, 4],
                channel_head_dim=[1, 1, 1, 1],
                ) -> None:
        super().__init__()

        patch = input_size[0] // patch_size
   
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.relu = nn.ReLU()
        norm2 = None
        self.conv1 = double_conv_block_a(in_features=in_features, 
                                        out_features1=int(64 * k), 
                                        out_features2=int(64 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm1 = nn.BatchNorm2d(int(64 * k))
        self.conv2 = double_conv_block_a(in_features=int(64 * k), 
                                        out_features1=int(128 * k), 
                                        out_features2=int(128 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm2 = nn.BatchNorm2d(int(128 * k))

        self.conv3 = double_conv_block_a(in_features=int(128 * k), 
                                        out_features1=int(256 * k), 
                                        out_features2=int(256 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm3 = nn.BatchNorm2d(int(256 * k))

        self.conv4 = double_conv_block_a(in_features=int(256 * k), 
                                        out_features1=int(512 * k), 
                                        out_features2=int(512 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)  
        self.norm4 = nn.BatchNorm2d(int(512 * k))
    
        self.conv5 = double_conv_block(in_features=int(512 * k), 
                                        out_features1=int(1024 * k), 
                                        out_features2=int(1024 * k), 
                                        norm_type='bn')   


        self.DCA = DCA(n=n,                                            
                            features = [int(64 * k), int(128 * k), int(256 * k), int(512 * k)],                                                                                                              
                            strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                            patch=patch,
                            spatial_att=spatial_att,
                            channel_att=channel_att, 
                            spatial_head=spatial_head_dim,
                            channel_head=channel_head_dim,
                                                        )  
          

        self.up1 = Upconv(in_features=int(1024 * k), 
                            out_features=int(512 * k), 
                            norm_type='bn')

        self.upconv1 = double_conv_block(in_features=int(512 * k + 512 * k), 
                                        out_features1=int(512 * k), 
                                        out_features2=int(512 * k), 
                                        norm_type='bn')

        self.up2 = Upconv(in_features=int(512 * k), 
                            out_features=int(256 * k), 
                            norm_type='bn')


        self.upconv2 = double_conv_block(in_features=int(256 * k + 256 * k), 
                                        out_features1=int(256 * k), 
                                        out_features2=int(256 * k), 
                                        norm_type='bn')

        self.up3 = Upconv(in_features=int(256 * k), 
                            out_features=int(128 * k), 
                            norm_type='bn')


        self.upconv3 = double_conv_block(in_features=int(128 * k + 128 * k), 
                                        out_features1=int(128 * k), 
                                        out_features2=int(128 * k), 
                                        norm_type='bn')

        self.up4 = Upconv(in_features=int(128 * k), 
                            out_features=int(64 * k), 
                            norm_type='bn')

        self.upconv4 = double_conv_block(in_features=int(64 * k + 64 * k), 
                                        out_features1=int(64 * k), 
                                        out_features2=int(64 * k), 
                                        norm_type='bn')    

        self.out = conv_block(in_features=int(64 * k), 
                            out_features=out_features, 
                            norm_type=None,
                            activation=False, 
                            kernel_size=(1, 1), 
                            padding=(0, 0))   

        # self.initialize_weights()                                     

    def forward(self, x):
        x1 = self.conv1(x)
        x1_n = self.norm1(x1)
        x1_a = self.relu(x1_n)
        x2 = self.maxpool(x1_a)
        x2 = self.conv2(x2)
        x2_n = self.norm2(x2)
        x2_a = self.relu(x2_n)
        x3 = self.maxpool(x2_a) 
        x3 = self.conv3(x3)
        x3_n = self.norm3(x3)
        x3_a = self.relu(x3_n)
        x4 = self.maxpool(x3_a)
        x4 = self.conv4(x4)
        x4_n = self.norm4(x4)
        x4_a = self.relu(x4_n)
        x5 = self.maxpool(x4_a)
        x = self.conv5(x5)
        x1, x2, x3, x4 = self.DCA([x1, x2, x3, x4])
        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.upconv1(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.upconv2(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.upconv3(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.upconv4(x)
        x = self.out(x)
        self.pre = {
            'mask': x
        }
        return x
    
    def backward(self, x, optimer, closure=wrap_dice, clear_stored=True):
        return super().backward(x, optimer, closure, clear_stored)

