from .hdmapnet import HDMapNet
from .ipm_net import IPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar
from .hdmapnet_lidar import HDMapNet_lidar
from .pmapnet_sdmap import PMapNet_sdmap

from .pmapnet_sdmap_cam import PMapNet_sdmap_cam
from .pmapnet_mae_head import PMapNet_mae_head, PMapNet_mae_head16, PMapNet_mae_head32 
from .pmapnet_cam_mae_head import PMapNet_cam_mae_head, PMapNet_cam_mae_head16

def get_model(method, args, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    if method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_cam':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)
    elif method == 'HDMapNet_lidar':
        model = HDMapNet_lidar(data_conf, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    
    elif method == 'PMapNet_sdmap':
        model = PMapNet_sdmap(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)          
    elif method == 'PMapNet_sdmap_cam':
        model = PMapNet_sdmap_cam(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)       
    
    
    elif method == 'PMapNet_mae_head':
        model = PMapNet_mae_head(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True) 
    elif method == 'PMapNet_cam_mae_head':
        model = PMapNet_cam_mae_head(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)       
    elif method == 'PMapNet_mae_head16':
        model = PMapNet_mae_head16(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    elif method == 'PMapNet_mae_head32':
        model = PMapNet_mae_head32(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)       
    elif method == 'PMapNet_cam_mae_head16':
        model = PMapNet_cam_mae_head16(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False) 
 
    else:
        raise NotImplementedError

    return model
