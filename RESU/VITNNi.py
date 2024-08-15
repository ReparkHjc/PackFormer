import torch
import os
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from data_preprocess import date_preprocess, con
import pandas as pd
import numpy as np
import torch.utils.data
import torch.nn.functional as F


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl i created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... i've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=32, patch_size=2, in_c=3, embed_dim=1024, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=16,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # scale根号下dk分之一
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # num_patches + 1是class token，total_embed_dim =1024
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape  # N=HW
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # (q @ k.transpose(-2, -1))->[num_patches + 1,num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # dim = -1 每一行softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]  拼接
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SE_VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_c=3,
                 embed_dim=1024, depth=3, num_heads=16, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None):

        super(SE_VisionTransformer, self).__init__()

        self.se = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=(3,3), padding=(0,1)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(12, 6, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(6, 3, kernel_size=(1,1)),
            nn.Sigmoid()
        )

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # if self.dist_token is not None:
        #     nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):

        resident=x
        x1 = self.se(x)
        x=x1*x
        x=x+resident

        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # return x
        return x[:,0]



# input [batchsize,3,1024]
class Flat_decoder(nn.Module):
    def __init__(self,cell_num=3,embed_dim=1024):
        super(Flat_decoder, self).__init__()
        self.Flatten=nn.Flatten(1,2)
        # self.hidden_layyer=nn.Linear(in_features=cell_num*embed_dim,out_features=96,bias=True)
        # self.output_layyer=nn.Linear(in_features=96,out_features=cell_num,bias=True)
        # self.softmax=nn.Softmax()
        # self.Fc1 = nn.Linear(in_features=embed_dim, out_features=32, bias=True)
        # self.Fc2 = nn.Linear(in_features=32, out_features=1, bias=True)
        self.hidden_layyer=nn.Linear(in_features=cell_num*embed_dim,out_features=96)
        self.output_layyer=nn.Linear(in_features=96,out_features=cell_num)
        self.softmax=nn.Softmax()
        self.Fc1 = nn.Linear(in_features=embed_dim, out_features=96)
        self.Fc2 = nn.Linear(in_features=96, out_features=1)


    def forward(self,x):
        x_weight=self.Flatten(x)
        x_weight=x_weight.unsqueeze(dim=1)
        x_weight=self.hidden_layyer(x_weight)
        x_weight=self.output_layyer(x_weight)
        att_x=self.softmax(x_weight)
        x=torch.matmul(att_x,x)
        x=self.Fc1(x)
        x=self.Fc2(x)
        return x


# flatten_decoder
class flatten_decoder(nn.Module):
    def __init__(self,cell_num=3,embed_dim=1024):
        super(flatten_decoder, self).__init__()
        self.fc=nn.Linear(cell_num*embed_dim,int(cell_num*embed_dim/10))
        self.fc1=nn.Linear(150,cell_num)
        # self.fc2 = nn.Linear(embed_dim, 1)
        self.fc2=nn.Linear(embed_dim,96)
        self.fc3 = nn.Linear(96, 1)
        self.flatten=nn.Flatten(1,2)


    def forward(self,x):
        # 残差结构
        resident=x
        x=self.flatten(x).unsqueeze(dim=1)
        x=self.fc(x)
        x=self.fc1(x)
        x=torch.softmax(x,dim=2)

        x=torch.matmul(x,resident)
        x=self.fc2(x).squeeze()
        return x


# spatial_deocder()
class SpatialAttention(nn.Module):
    def __init__(self,cell_num=3,embed_dim=500, kernel_size=3):
        super(SpatialAttention, self).__init__()
        padding = 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(1,3)
        self.fc=nn.Linear(cell_num*embed_dim,96)
        self.fc1 = nn.Linear(96, 1)
    def forward(self, x):  # x.size() 30,40,50,30
        x=x.unsqueeze(dim=2)    # x.size() 8,3,1,500
        resident=x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 8,1,1,500
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        x=resident.mul(self.sigmoid(x))
        x=self.flatten(x)
        x=self.fc(x)
        x = self.fc1(x)
        return x



# 如果输出是batchsize，Feature_len,[3,1,1024]
class SE_Decoder(nn.Module):
    def __init__(self,cell_num=2,embed_dim=1024,decoderFc_size=96,momentum=0.1,prediction_num=1):
        super(SE_Decoder, self).__init__()

        self.se = nn.Sequential(
            nn.Conv2d(cell_num, 12, kernel_size=(1,3), padding=(0,1)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(12, 6, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(6, cell_num, kernel_size=(1,1)),
            nn.Sigmoid()
        )
        self.Flatten=nn.Flatten(1,3)
        self.Fc=nn.Linear(cell_num*embed_dim,decoderFc_size)
        # self.dropout=nn.Dropout(p=0.35)
        # 96 32
        self.Fc1=nn.Linear(decoderFc_size,prediction_num)

        self.norm=nn.BatchNorm2d(cell_num, eps=1e-05, momentum=momentum, affine=True)

    def forward(self,x):
        # 残差结构

        residual = x.unsqueeze(dim=2)
        x = x.unsqueeze(dim=2)
        saved_x_before = residual.squeeze(dim=2).clone()
        att_x=self.se(x)
        print(att_x[0])
        saved_x_after = att_x.squeeze(dim=2).clone()
        x=x*att_x
        x=residual+x
        # 保存 saved_x_before 和 saved_x_after
        # i=0
        # if i == 0:
        save_dir = 'plot_data'
        os.makedirs(save_dir, exist_ok=True)

        saved_x_before_path = os.path.join(save_dir, 'saved_x_before.pt')
        saved_x_after_path = os.path.join(save_dir, 'saved_x_after.pt')

        torch.save(saved_x_before, saved_x_before_path)
        torch.save(saved_x_after, saved_x_after_path)
            # i+=1

        x=torch.relu(x)
        x = self.norm(x)
        x=self.Flatten(x)

        # x=self.dropout(x)

        x=self.Fc(x)
        x = self.Fc1(x)
        return x

class ablation_Decoder(nn.Module):
    def __init__(self,cell_num=3,embed_dim=1024,decoderFc_size=96,momentum=0.1,prediction_num=1):
        super(ablation_Decoder, self).__init__()

        self.se = nn.Sequential(
            nn.Conv2d(cell_num, 12, kernel_size=(1,3), padding=(0,1)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(12, 6, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(6, cell_num, kernel_size=(1,1)),
            nn.Sigmoid()
        )
        self.Flatten=nn.Flatten(1,3)
        self.Fc=nn.Linear(cell_num*embed_dim,prediction_num)
        # self.dropout=nn.Dropout(p=0.35)
        # 96 32
        self.Fc1=nn.Linear(decoderFc_size,prediction_num)
        self.norm=nn.BatchNorm2d(cell_num, eps=1e-05, momentum=momentum, affine=True)

    def forward(self,x):
        # 残差结构
        x = x.unsqueeze(dim=2)

        x=torch.relu(x)
        x = self.norm(x)
        x=self.Flatten(x)

        # x=self.dropout(x)

        x=self.Fc(x)
        # x = self.Fc1(x)
        return x
# B0005
class SE_VIT_Decoder(nn.Module):
    def __init__(self,type=1,cell_num=8,img_size=60,patch_size=4,in_c=3,embed_dim=1024,
                     depth=3,num_heads=4,mlp_ratio=4,qkv_bias=False,attn_drop_ratio=0,
                     drop_path_ratio=0,norm_layer=None,act_layer=None,momentum=0.1, decoderFc_size=256, prediction_num=1):
        #attn_drop_ratio  0.45 0.35
        """
            Args:
                   type:1:SE_Decoder()
                        2:Flat_decoder()
                        3:spatial_decoder()
                        4:Decoder_Block()
                   patch_size (int, tuple): patch size
                   in_c (int): number of input channels
                   num_classes (int): number of classes for classification head
                   embed_dim (int): embedding dimension
                   depth (int): depth of transformer
                   num_heads (int): number of attention heads
                   mlp_ratio (int): ratio of mlp hidden dim to embedding dim
                   qkv_bias (bool): enable bias for qkv if True
                   qk_scale (float): override default qk scale of head_dim ** -0.5 if set
                   representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
                   distilled (bool): model includes a distillation token and head as in DeiT models
                   drop_ratio (float): dropout rate
                   attn_drop_ratio (float): attention dropout rate
                   drop_path_ratio (float): stochastic depth rate
                   embed_layer (nn.Module): patch embedding layer
                   norm_layer: (nn.Module): normalization layer
        """

        super(SE_VIT_Decoder, self).__init__()
        self.cell_num=cell_num
        self.encode=SE_VisionTransformer(img_size=img_size,patch_size=patch_size,in_c=in_c,embed_dim=embed_dim,
                                         depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                                         attn_drop_ratio=attn_drop_ratio,drop_path_ratio=drop_path_ratio,
                                         norm_layer=norm_layer,act_layer=act_layer)
        self.type=type
        if type == 1:
            self.decoder=SE_Decoder(cell_num=cell_num,embed_dim=embed_dim,decoderFc_size=decoderFc_size,momentum=momentum,prediction_num=prediction_num)
        elif type == 2:
            self.decoder=Flat_decoder(cell_num=cell_num,embed_dim=embed_dim)
        elif type == 3:
            self.decoder = SpatialAttention(cell_num=cell_num, embed_dim=embed_dim,kernel_size=3)
        elif type == 4:
            self.decoder = ablation_Decoder(cell_num=cell_num,embed_dim=embed_dim,decoderFc_size=decoderFc_size,momentum=momentum,prediction_num=prediction_num)


    # x[batchsize，3个电池，channel=3，32，32]
    def forward(self,x):
        cell1 = x[:, 0]
        cell2 = x[:, 1]
        cell3 = x[:, 2]
        cell4 = x[:, 3]
        cell5 = x[:, 4]
        cell6 = x[:, 5]
        cell7 = x[:, 6]
        cell8 = x[:, 7]

        cell1 = self.encode(cell1).unsqueeze(dim=1)
        cell2 = self.encode(cell2).unsqueeze(dim=1)
        cell3 = self.encode(cell3).unsqueeze(dim=1)
        cell4 = self.encode(cell4).unsqueeze(dim=1)
        cell5 = self.encode(cell5).unsqueeze(dim=1)
        cell6 = self.encode(cell6).unsqueeze(dim=1)
        cell7 = self.encode(cell7).unsqueeze(dim=1)
        cell8 = self.encode(cell8).unsqueeze(dim=1)

        cell_all=torch.cat([cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8], dim=1)
        cell_all=self.decoder(cell_all)
        return cell_all


