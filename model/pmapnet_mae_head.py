import torch
from torch import nn

from .homography import bilinear_sampler, IPM
from .utils import plane_grid_2d, get_rot_2d, cam_to_pixel
from .pointpillar import PointPillarEncoder
from .base import CamEncode, BevEncode
from data_osm.utils import gen_dx_bx
from .sdmap_cross_attn_self import SDMapCrossAttn
from .position_encoding import build_position_encoding,PositionEmbeddingSine
from .map_mae_head import vit_base_patch8, vit_base_patch16, vit_base_patch32

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

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


class PMapNet_mae_head(nn.Module):
    def __init__(self,  
                 data_conf,
                 args, 
                 instance_seg=True, 
                 embedded_dim=16, 
                 direction_pred=True, 
                 direction_dim=36, 
                 lidar=False):
        super(PMapNet_mae_head, self).__init__()

        self.camC = 64
        self.downsample = 16

        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]  # 30.0
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  # 60.0
        canvas_h = int(patch_h / data_conf['ybound'][2])           # 200
        canvas_w = int(patch_w / data_conf['xbound'][2]) 

        #cross attn params
        hidden_dim = 64
        self.position_embedding = PositionEmbeddingSine(hidden_dim//2, normalize=True)
        
        feat_numchannels = self.camC+128

        self.input_proj = nn.Conv2d(feat_numchannels, hidden_dim, kernel_size=1)
        # sdmap_cross_attn
        self.sdmap_crossattn = SDMapCrossAttn(d_model=hidden_dim, num_decoder_layers=args.num_decoder_layers)
        
        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_osm = nn.Sequential(
            nn.Conv2d(1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.conv_bev = nn.Sequential(
            nn.Conv2d(192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AvgPool2d(kernel_size=10, stride=10)

        self.lidar = lidar
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            # self.bevencode = BevEncode(inC=self.camC+128+64, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        
        self.mae_head = vit_base_patch8(
            data_conf=data_conf, 
            instance_seg=instance_seg, 
            embedded_dim=embedded_dim, 
            direction_pred=direction_pred, 
            direction_dim=direction_dim, 
            lidar=True,
            img_size=(canvas_h, canvas_w))
        
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

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm):
        x = self.get_cam_feats(img)
        x = self.view_fusion(x)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        topdown = self.up_sampler(topdown)
        if self.lidar:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)

        # BEV size 192*800*200 -> 192*100*25
        bev_small = self.conv_bev(topdown)
        
        # osm size 1*800*200 -> 64*100*25
        conv_osm = self.conv_osm(osm)

        bs,c,h,w = bev_small.shape
        self.mask = torch.zeros([1,h,w],dtype=torch.bool)

        pos = self.position_embedding(bev_small[-1], self.mask.to(bev_small.device)).to(bev_small.dtype)
        bs = bev_small.shape[0]
        pos = pos.repeat(bs, 1, 1, 1)

        # input_proj 1*1 conv : 192->64(hidden_dim)
        bev_out = self.sdmap_crossattn(self.input_proj(bev_small), conv_osm, pos = pos)[0]

        bev_final = self.conv_up(bev_out)
        output = self.bevencode(bev_final)
        if self.mae_head is not None:
            output = self.mae_head(output[0])

        return output

class PMapNet_mae_head16(PMapNet_mae_head):
    def __init__(self,  
                 data_conf,
                 args, 
                 instance_seg=True, 
                 embedded_dim=16, 
                 direction_pred=True, 
                 direction_dim=36, 
                 lidar=False):
        super(PMapNet_mae_head, self).__init__()

        self.camC = 64
        self.downsample = 16

        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]  # 30.0
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  # 60.0
        canvas_h = int(patch_h / data_conf['ybound'][2])           # 200
        canvas_w = int(patch_w / data_conf['xbound'][2]) 

        #cross attn params
        hidden_dim = 64
        self.position_embedding = PositionEmbeddingSine(hidden_dim//2, normalize=True)
        
        feat_numchannels = self.camC+128

        self.input_proj = nn.Conv2d(feat_numchannels, hidden_dim, kernel_size=1)
        # sdmap_cross_attn
        self.sdmap_crossattn = SDMapCrossAttn(d_model=hidden_dim, num_decoder_layers=args.num_decoder_layers)
        
        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_osm = nn.Sequential(
            nn.Conv2d(1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.conv_bev = nn.Sequential(
            nn.Conv2d(192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AvgPool2d(kernel_size=10, stride=10)

        self.lidar = lidar
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            # self.bevencode = BevEncode(inC=self.camC+128+64, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        
        self.mae_head = vit_base_patch16(
            data_conf=data_conf, 
            instance_seg=instance_seg, 
            embedded_dim=embedded_dim, 
            direction_pred=direction_pred, 
            direction_dim=direction_dim, 
            lidar=True,
            img_size=(canvas_h, canvas_w))


class PMapNet_mae_head32(PMapNet_mae_head):
    def __init__(self,  
                 data_conf,
                 args, 
                 instance_seg=True, 
                 embedded_dim=16, 
                 direction_pred=True, 
                 direction_dim=36, 
                 lidar=False):
        super(PMapNet_mae_head, self).__init__()

        self.camC = 64
        self.downsample = 16

        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]  # 30.0
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  # 60.0
        canvas_h = int(patch_h / data_conf['ybound'][2])           # 200
        canvas_w = int(patch_w / data_conf['xbound'][2]) 

        #cross attn params
        hidden_dim = 64
        self.position_embedding = PositionEmbeddingSine(hidden_dim//2, normalize=True)
        
        feat_numchannels = self.camC+128

        self.input_proj = nn.Conv2d(feat_numchannels, hidden_dim, kernel_size=1)
        # sdmap_cross_attn
        self.sdmap_crossattn = SDMapCrossAttn(d_model=hidden_dim, num_decoder_layers=args.num_decoder_layers)
        
        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_osm = nn.Sequential(
            nn.Conv2d(1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.conv_bev = nn.Sequential(
            nn.Conv2d(192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AvgPool2d(kernel_size=10, stride=10)

        self.lidar = lidar
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            # self.bevencode = BevEncode(inC=self.camC+128+64, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        
        self.mae_head = vit_base_patch32(
            data_conf=data_conf, 
            instance_seg=instance_seg, 
            embedded_dim=embedded_dim, 
            direction_pred=direction_pred, 
            direction_dim=direction_dim, 
            lidar=True,
            img_size=(canvas_h, canvas_w))