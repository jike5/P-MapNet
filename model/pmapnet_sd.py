import torch
from torch import nn
from .utils.homography import IPM
from .utils.pointpillar import PointPillarEncoder
from .utils.base import CamEncode, BevEncode
from data_osm.utils import gen_dx_bx
from .utils.sdmap_cross_attn import SDMapCrossAttn
from .utils.position_encoding import PositionEmbeddingSine

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


class PMapNet_SD(nn.Module):
    def __init__(self,  data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False):
        super(PMapNet_SD, self).__init__()

        self.lidar = lidar
        self.camC = 64
        self.LiDARC = 128
        self.downsample = 16

        #cross attn params
        hidden_dim = 64
        self.position_embedding = PositionEmbeddingSine(hidden_dim//2, normalize=True)

        if lidar:
            feat_numchannels = self.camC+self.LiDARC
            self.pp = PointPillarEncoder(self.LiDARC, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        else:
            feat_numchannels = self.camC

        self.input_proj = nn.Conv2d(feat_numchannels, hidden_dim, kernel_size=1)

        # sdmap_cross_attn
        self.sdmap_crossattn = SDMapCrossAttn(d_model=hidden_dim, num_decoder_layers=2, dropout=0.1)
        
        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        num_cams = data_conf.get('num_cams', 6)
        # import pdb; pdb.set_trace()
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size, n_views=num_cams)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=num_cams, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pool = nn.AvgPool2d(kernel_size=10, stride=10)
        self.conv_osm = self.nn_Sequential(1, hidden_dim // 2, hidden_dim)
        self.conv_bev = self.nn_Sequential(feat_numchannels, feat_numchannels, feat_numchannels)
        self.conv_up = self.nn_Sequential_Transpose(hidden_dim, feat_numchannels, feat_numchannels)

        self.bevencode = BevEncode(inC=feat_numchannels, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def nn_Sequential(self, in_dim=192, mid_dim=192, out_dim=192):
        return nn.Sequential(
                nn.Conv2d(in_dim, out_channels=mid_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(mid_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                )
    
    def nn_Sequential_Transpose(self, in_dim=192, mid_dim=192, out_dim=192):
        return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_channels=mid_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(mid_dim, out_channels=out_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                )
    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm):
        x = self.get_cam_feats(img)
        # import pdb; pdb.set_trace()
        x = self.view_fusion(x)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        topdown = self.up_sampler(topdown)
        if self.lidar:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)

        bev_small = self.conv_bev(topdown)
        
        conv_osm = self.conv_osm(osm)

        bs,c,h,w = bev_small.shape
        self.mask = torch.zeros([1,h,w],dtype=torch.bool)

        pos = self.position_embedding(bev_small[-1], self.mask.to(bev_small.device)).to(bev_small.dtype)
        bs = bev_small.shape[0]
        pos = pos.repeat(bs, 1, 1, 1)
        bev_out = self.sdmap_crossattn(self.input_proj(bev_small), conv_osm, pos = pos)[0]
        bev_final = self.conv_up(bev_out)
        return self.bevencode(bev_final)
