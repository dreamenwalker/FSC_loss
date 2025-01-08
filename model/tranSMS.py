import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange
from math import log2


class par_cvt_rdnDualNonSq(nn.Module):
    """config = {
    'initialConvFeatures' : 32, #### free param
    'scaleFactor':scale_factor,
    'rdn_nb_of_features' : 24,
    'rdn_nb_of_blocks' : 4,
    'rdn_layer_in_each_block' : 5, #### untitled.png
    'rdn_growth_rate' : 6,
    'img_size1' : 8, #e.g 8 or (6,8) input image size ####
    'img_size2' : 8, #e.g 8 or (6,8) input image size ####
    "cvt_out_channels": 32,#C1 ####
    "cvt_dim" : 32
    "convAfterConcatLayerFeatures" : 32 #C3 ###
    }"""

    def __init__(self, config):
        super(par_cvt_rdnDualNonSq, self).__init__()
        self.initialConv = nn.Conv2d(config['inChannel'], config['initialConvFeatures'], 3, 1, 1)
        self.rdn = rdn1x(input_channels=config['initialConvFeatures'],
                         nb_of_features=config['rdn_nb_of_features'],
                         nb_of_blocks=config['rdn_nb_of_blocks'],
                         layer_in_each_block=config["rdn_layer_in_each_block"],
                         growth_rate=config["rdn_growth_rate"])
        self.transformer = CvTNonSquare(image_size1=config["img_size1"], image_size2=config["img_size2"],
                                        in_channels=config['initialConvFeatures'],
                                        out_channels=config["cvt_out_channels"], dim=config["cvt_dim"])
        self.convAfterConcat = nn.Conv2d(config['rdn_nb_of_features'] + config['cvt_out_channels'],
                                         config["convAfterConcatLayerFeatures"], 3, 1, 1)
        upSamplersList = []
        for _ in range(int(log2(config['scaleFactor']))):
            upSamplersList.append(UpSampleBlock(config["convAfterConcatLayerFeatures"], 2))
        self.upSampler = nn.Sequential(*upSamplersList)
        self.lastConv = nn.Conv2d(config['convAfterConcatLayerFeatures'], config['outChannel'], 3, 1, 1)

    def forward(self, x):
        x = self.initialConv(x)
        rdnSkip = self.rdn(x)
        x = self.transformer(x)
        x = torch.cat([x, rdnSkip], dim=1)
        x = self.convAfterConcat(x)
        x = self.upSampler(x)
        x = self.lastConv(x)
        return x


# Residual Dense Net the operation for multiple input is concatenate
class UpSampleBlock(nn.Module):
    def __init__(self, input_channels, scale_factor=2):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, input_channels * scale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(scale_factor)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffler(x)
        return self.lrelu(x)


class dense_block(nn.Module):
    def __init__(self, in_channels, addition_channels):
        super(dense_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=addition_channels, kernel_size=3, stride=1,
                              padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], dim=1)


class rdb(nn.Module):  # residual dense block used in class rdn1x
    def __init__(self, in_channels, C, growth_at_each_dense):
        super(rdb, self).__init__()
        denses = nn.ModuleList()
        for i in range(0, C):
            denses.append(dense_block(in_channels + i * growth_at_each_dense, growth_at_each_dense))
        self.local_res_block = nn.Sequential(*denses)
        self.last_conv = nn.Conv2d(in_channels=in_channels + C * growth_at_each_dense, out_channels=in_channels,
                                   kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return x + self.last_conv(self.local_res_block(x))


class rdn1x(nn.Module):
    def __init__(self, input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate):
        super(rdn1x, self).__init__()
        self.conv0 = nn.Conv2d(input_channels, nb_of_features, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(nb_of_features, nb_of_features, kernel_size=3, stride=1, padding=1)
        self.rdbs = nn.ModuleList()
        for i in range(0, nb_of_blocks):
            self.rdbs.append(rdb(nb_of_features, layer_in_each_block, growth_rate))
        self.conv2 = nn.Conv2d(in_channels=nb_of_blocks * nb_of_features, out_channels=nb_of_features, kernel_size=1,
                               stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=nb_of_features, out_channels=nb_of_features, kernel_size=3, stride=1,
                               padding=1)

    def forward(self, x):
        x = self.conv0(x)
        residual0 = x
        x = self.conv1(x)
        rdb_outs = list()
        for layer in self.rdbs:
            x = layer(x)
            rdb_outs.append(x)
        x = torch.cat(rdb_outs, dim=1)
        x = self.conv2(x)
        x = self.conv3(x) + residual0
        return x


# Conv_transformer
class CvTNonSquare(nn.Module):  # used in par_cvt_rdnDualNonSq
    def __init__(self, image_size1, image_size2, in_channels, out_channels=32, dim=54, kernels=[3, 3, 3],
                 strides=[1, 1, 2],
                 heads=[4, 4, 4], depth=[1, 1, 1], pool='cls', dropout=0., emb_dropout=0., scale_dim=2, ):
        super().__init__()

        self.dim = dim

        ##### Stage 1 #######
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[0], strides[0], 1),
            Rearrange('b c h w -> b (h w) c', h=image_size1 // 1, w=image_size2 // 1),
            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            TransformerNonSq(dim=dim, img_size1=image_size1 // 1, img_size2=image_size2 // 1, depth=depth[0],
                             heads=heads[0], dim_head=self.dim,
                             mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h=image_size1 // 1, w=image_size2 // 1)
        )

        ##### Stage 2 #######
        in_channels = dim
        scale = heads[1] // heads[0]
        dim = scale * dim
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            Rearrange('b c h w -> b (h w) c', h=image_size1, w=image_size2),
            nn.LayerNorm(dim)
        )
        self.stage2_transformer = nn.Sequential(
            TransformerNonSq(dim=dim, img_size1=image_size1 // 1, img_size2=image_size2 // 1, depth=depth[1],
                             heads=heads[1], dim_head=self.dim,
                             mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h=image_size1 // 1, w=image_size2 // 1)
        )

        ##### Stage 3 #######
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim
        self.stage3_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[2], strides[2], 1),
            Rearrange('b c h w -> b (h w) c', h=image_size1 // 2, w=image_size2 // 2),
            nn.LayerNorm(dim)
        )
        self.stage3_transformer = nn.Sequential(
            TransformerNonSq(dim=dim, img_size1=image_size1 // 2, img_size2=image_size2 // 2, depth=depth[2],
                             heads=heads[2], dim_head=self.dim,
                             mlp_dim=dim * scale_dim, dropout=dropout, last_stage=False),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.lastConv = nn.Conv2d(dim, out_channels, 3, 1, 1)
        self.upsampler = UpSampleBlock(out_channels)

    def forward(self, img):
        xs = self.stage1_conv_embed(img)
        xs = self.stage1_transformer(xs)
        xs = self.stage2_conv_embed(xs)
        xs = self.stage2_transformer(xs)
        xs = self.stage3_conv_embed(xs)
        xs = self.stage3_transformer(xs)
        xs = xs.permute(0, 2, 1).contiguous()
        xs = xs.contiguous().view(xs.shape[0], xs.shape[1], img.shape[2] // 2, img.shape[3] // 2)
        xs = self.lastConv(xs)
        xs = self.upsampler(xs)
        return xs


class TransformerNonSq(nn.Module):
    def __init__(self, dim, img_size1, img_size2, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        ConvAttentionNonSq(dim, img_size1, img_size2, heads=heads, dim_head=dim_head, dropout=dropout,
                                           last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, ):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConvAttention(nn.Module):  # 这个是和后边加nonsq区别，在函数定义中未使用
    def __init__(self, dim, img_size, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1,
                 dropout=0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h=h).contiguous()

        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size).contiguous()
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h).contiguous()

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h).contiguous()

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h).contiguous()

        if self.last_stage:
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)

        #         print("Q shape: ",q.shape)
        #         print("K shape: ",k.shape)
        dots = torch.matmul(q, k.transpose(-1, -2).contiguous()) * self.scale
        attn = torch.nn.Softmax(dim=-1)(dots)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
        out = self.to_out(out)
        return out


class ConvAttentionNonSq(nn.Module):  # used for TransformerNonSq
    def __init__(self, dim, img_size1, img_size2, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1,
                 v_stride=1, dropout=0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size1 = img_size1
        self.img_size2 = img_size2
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h=h).contiguous()

        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size1, w=self.img_size2).contiguous()
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h).contiguous()

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h).contiguous()

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h).contiguous()

        if self.last_stage:
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)

        #         print("Q shape: ",q.shape)
        #         print("K shape: ",k.shape)
        dots = torch.matmul(q, k.transpose(-1, -2).contiguous()) * self.scale
        attn = torch.nn.Softmax(dim=-1)(dots)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
        out = self.to_out(out)
        return out
