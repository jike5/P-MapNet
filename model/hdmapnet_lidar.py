import torch
from torch import nn

from .homography import bilinear_sampler, IPM
from .utils import plane_grid_2d, get_rot_2d, cam_to_pixel
from .pointpillar import PointPillarEncoder
from .base import CamEncode, BevEncode
from data_osm.utils import gen_dx_bx


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


class HDMapNet_lidar(nn.Module):
    def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False):
        super(HDMapNet_lidar, self).__init__()

        self.conv_osm = nn.Sequential(
        nn.Conv2d(1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True))

        self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        self.bevencode = BevEncode(inC=128, outC=data_conf['num_channels_bd'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)


    def forward(self, lidar_data, lidar_mask):
        
        # osm_encoding = self.conv_osm(osm_mask)

        lidar_feature = self.pp(lidar_data, lidar_mask)

        # topdown = torch.cat([lidar_feature, osm_encoding], dim=1)

        return self.bevencode(lidar_feature)
