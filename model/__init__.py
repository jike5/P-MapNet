from .hdmapnet import HDMapNet
from .lift_splat import LiftSplat
from .pmapnet_sd import PMapNet_SD
from .pmapnet_hd import PMapNet_HD, PMapNet_HD16, PMapNet_HD32 
from .utils.map_mae_head import vit_base_patch8, vit_base_patch16, vit_base_patch32

def get_model(cfg, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    patch_h = data_conf['ybound'][1] - data_conf['ybound'][0] 
    patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  
    canvas_h = int(patch_h / data_conf['ybound'][2])           
    canvas_w = int(patch_w / data_conf['xbound'][2]) 
    
    method = cfg.model
    if "dataset" in cfg:
        if cfg.dataset == 'av2':
            data_conf.update({"num_cams":7})

    if method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim)
    
    # HDMapNet model
    elif method == 'HDMapNet_cam':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)

    # P-MapNet sd prior model
    elif method == 'pmapnet_sd':
        model = PMapNet_SD(data_conf,  instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)       
    elif method == 'pmapnet_sd_cam':
        model = PMapNet_SD(data_conf,  instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)       
    
    # P-MapNet hd prior model
    elif method == 'pmapnet_hd':
        model = PMapNet_HD(data_conf,  instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True) 
    elif method == 'pmapnet_hd16':
        model = PMapNet_HD16(data_conf,  instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    elif method == 'pmapnet_hd32':
        model = PMapNet_HD32(data_conf,  instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)       
    elif method == 'pmapnet_hd_cam':
        model = PMapNet_HD(data_conf,  instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)       
    elif method == 'pmapnet_hd_cam16':
        model = PMapNet_HD16(data_conf,  instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False) 
 
    # P-MapNet hd pretrain model
    elif method == "hdmapnet_pretrain":
        model = vit_base_patch8(data_conf=data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True, img_size=(canvas_h, canvas_w))
    elif method == "hdmapnet_pretrain16":
        model = vit_base_patch16(data_conf=data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True, img_size=(canvas_h, canvas_w))
    elif method == "hdmapnet_pretrain32":
        model = vit_base_patch32(data_conf=data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True, img_size=(canvas_h, canvas_w))
    else:
        raise NotImplementedError

    return model
