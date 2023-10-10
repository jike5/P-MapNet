import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from .homography import bilinear_sampler, IPM

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs

@TRANSFORMER.register_module()
class MLPViewTransformer(BaseModule):
    def __init__(self,
                fv_size=None,
                xyzbound=None,
                camera_number=None,
                in_channel=None,
                extrinsic=True,
                **kwargs):
        super().__init__(**kwargs)
        dx, bx, nx = gen_dx_bx(xyzbound['xbound'], xyzbound['ybound'], xyzbound['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)
        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.ipm = IPM(ipm_xbound, 
                       ipm_ybound, 
                       N=camera_number, 
                       C=in_channel,
                       extrinsic=extrinsic)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def get_Ks_RTs_and_post_RTs(self, 
                                intrins, 
                                rots, 
                                trans, 
                                post_rots, 
                                post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts
        post_RTs = None
        return Ks, RTs, post_RTs
    
    def forward(self, 
                img_feats, 
                intrins,
                rots,
                trans, 
                post_rots, 
                post_trans,
                car_trans, 
                yaw_pitch_roll,
                lidar_feats=None):
        x = self.view_fusion(img_feats)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(
            intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        topdown = self.up_sampler(topdown)
        if lidar_feats is not None:
            topdown = torch.cat([topdown, lidar_feats], dim=1)
        return topdown

    def init_weights(self):
        pass
