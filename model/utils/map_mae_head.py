import torch
from torch import nn
from functools import partial
import timm.models.vision_transformer
from .base import CamEncode, BevEncode

class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, has_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.has_relu = has_relu

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        if self.has_relu:
            return self.relu(feat)
        return feat
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x


class MapVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,
                 data_conf=None, 
                 instance_seg=True, 
                 embedded_dim=16, 
                 direction_pred=True, 
                 direction_dim=36, 
                 lidar=None,
                 **kwargs):
        super(MapVisionTransformer, self).__init__(**kwargs)
        self.bev_head = BevEncode(inC=kwargs['embed_dim'], 
                                   outC=data_conf['num_channels'], 
                                   instance_seg=instance_seg, 
                                   embedded_dim=embedded_dim, 
                                   direction_pred=direction_pred, 
                                   direction_dim=direction_dim+1)
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]  # 30.0
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  # 60.0
        self.canvas_h = int(patch_h / data_conf['ybound'][2])           # 200
        self.canvas_w = int(patch_w / data_conf['xbound'][2])           # 400
        self.conv_up = nn.Sequential(
                nn.ConvTranspose2d(kwargs['embed_dim'], kwargs['embed_dim'], kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose2d(kwargs['embed_dim'], kwargs['embed_dim'], kernel_size=4, stride=2, padding=1),
                nn.Upsample(size=(self.canvas_h, self.canvas_w), mode='bilinear', align_corners=False),
                ConvBNReLU(kwargs['embed_dim'], kwargs['embed_dim'], 1, stride=1, padding=0, has_relu=False),
            )
        self.map_patch_embed = PatchEmbed(kwargs['img_size'], kwargs['patch_size'], kwargs['in_chans'], kwargs['embed_dim'])

    def forward_features(self, x):
        B = x.shape[0] # (b,c,h,w)
        # import pdb; pdb.set_trace()
        x = self.map_patch_embed(x) # (b,dim,12,25)

        _, dim, h, w = x.shape
        x = x.flatten(2).transpose(1, 2) # (b,n,dim)
        x = x + self.pos_embed[:, :-1]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        outcome = x.permute(0,2,1).reshape(B, dim, h, w)
        outcome = self.conv_up(outcome)
        return outcome
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.bev_head(x)
        return x


def vit_base_patch8(**kwargs):
    model = MapVisionTransformer(
        patch_size=8, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        in_chans=4,
        **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = MapVisionTransformer(
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        in_chans=4,
        **kwargs)
    return model

def vit_base_patch32(**kwargs):
    model = MapVisionTransformer(
        patch_size=32, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        in_chans=4,
        **kwargs)
    return model