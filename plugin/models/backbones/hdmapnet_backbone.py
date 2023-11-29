import copy
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyquaternion import Quaternion
from mmdet.models import BACKBONES
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from .bevformer.grid_mask import GridMask
from mmdet3d.models import builder

@BACKBONES.register_module()
class HDMapNetBackbone(nn.Module):

    def __init__(self,
                 roi_size,
                 bev_h,
                 bev_w,
                 img_backbone=None, 
                 lidar_backbone=None,
                 img_neck=None,               
                 transformer=None,
                 up_outdim=128,
                 lidar=False,
                 **kwargs):
        super(HDMapNetBackbone, self).__init__()
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if lidar_backbone and lidar:
            self.lidar_backbone = builder.build_backbone(lidar_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
            self.with_img_neck = True
        else:
            self.with_img_neck = False
        self.camC = 64
        self.downsample = 16

        self.patch_h = bev_h
        self.patch_w = bev_w
        self.lidar = lidar
        self.transformer = build_transformer(transformer)
        # self._init_layers()
        self.init_weights()

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        self.img_backbone.init_weights()
        if self.with_img_neck:
            self.img_neck.init_weights()
       
    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.img_backbone(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def forward(self, 
                img, 
                img_metas, 
                points=None, 
                lidar_mask=None,
                **kwargs):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        intrins = []
        rots = []
        trans = []
        post_rots = [] 
        post_trans = []
        car_trans = []
        yaw_pitch_roll = []
        for meta in img_metas:
            ego2imgs = meta['cam_extrinsics'] # list of 7 cameras, 4x4 array
            ego2global_tran = meta['ego2global_translation'] # 3
            ego2global_rot = np.array(meta['ego2global_rotation']) # 3x3
            rots.append(torch.stack([img.new_tensor(ego2img[:3, :3].T) for ego2img in ego2imgs])) # 7x3x3
            trans.append(torch.stack([img.new_tensor((ego2img[:3, :3].T).dot(-ego2img[:3, 3])) for ego2img in ego2imgs])) # 7x3
            post_rots = rots
            post_trans = trans
            intrins.append(torch.stack([img.new_tensor(intri) for intri in meta['cam_intrinsics']]))
            car_trans.append(img.new_tensor(ego2global_tran)) # [3]
            pos_rotation = Quaternion(matrix=ego2global_rot)
            yaw_pitch_roll.append(img.new_tensor(pos_rotation.yaw_pitch_roll))

        img_feats = self.get_cam_feats(img)
        if self.lidar and (points is not None):
            lidar_feats = self.lidar_backbone(points, lidar_mask)
        else:
            lidar_feats = None
        bev_feats = self.transformer(
            img_feats, 
            torch.stack(intrins),
            torch.stack(rots),
            torch.stack(trans), 
            torch.stack(post_rots), 
            torch.stack(post_trans),
            torch.stack(car_trans), 
            torch.stack(yaw_pitch_roll),
            lidar_feats)

        return bev_feats