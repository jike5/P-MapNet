import sys
sys.path.append('.')
import argparse
import tqdm
import os
import cv2
import torch
from tools.evaluation.iou import get_batch_iou
from tools.config import Config
from data_osm.dataset import semantic_dataset
from data_osm.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tools.evaluation import lpips
from data_osm.image import denormalize_img
import warnings
warnings.filterwarnings("ignore")


Nu_SCENE_CANDIDATE = [
    'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558',
    'scene-1065', 'scene-1066', 'scene-1067', 'scene-1068'
    'scene-0275', 'scene-0276', 'scene-0277', 'scene-0278',
    'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522',
    'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914',
    ]

AV2_SCENE_CANDIDATE = [
    'f1275002-842e-3571-8f7d-05816bc7cf56',
    'ba67827f-6b99-3d2a-96ab-7c829eb999bb',
    'bf360aeb-1bbd-3c1e-b143-09cf83e4f2e4',
    'ded5ef6e-46ea-3a66-9180-18a6fa0a2db4',
    'e8c9fd64-fdd2-422d-a2a2-6f47500d1d12',
    '1f434d15-8745-3fba-9c3e-ccb026688397',
    '6f128f23-ee40-3ea9-8c50-c9cdb9d3e8b6',
]

SCENE_CANDIDATE = None

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def vis(semantic, semantic_gt, sd_map, time, scene_id, save_path, with_gt=False):
    car_img = Image.open('icon/car_gray.png')
    semantic = onehot_encoding(semantic)
    semantic = semantic.clone().cpu().numpy()
    semantic[semantic < 0.1] = np.nan
    semantic_gt_mask = semantic_gt.clone().cpu().numpy()
    semantic_gt_mask[semantic_gt < 0.1] = np.nan
    sd_map = sd_map.cpu().numpy()
    sd_map[sd_map < 0.1] = np.nan

    b, c, h, w = semantic.shape
    alpha = 0.8
    dpi = 600
    divier = 'Blues'
    ped_crossing = 'Greens'
    boundary = 'Purples'
    vmax = 1
    for i in range(semantic.shape[0]):
        if scene_id[i] not in SCENE_CANDIDATE:
            continue
        save_path_seg = os.path.join(save_path, f'{scene_id[i]}', f'{time[i]}')
        if not os.path.exists(save_path_seg):
            os.makedirs(save_path_seg)
        # vis hdmap gt with sd map
        imname = os.path.join(save_path_seg, 'gt_sd_map.png')
        if not os.path.exists(imname):
            plt.figure(figsize=(w*2/100, 4))
            plt.imshow(semantic_gt_mask[i][1]*0.5, vmin=0, cmap= divier, vmax=vmax, alpha=alpha)
            plt.imshow(semantic_gt_mask[i][2]*0.5, vmin=0, cmap= ped_crossing, vmax=vmax, alpha=alpha)
            plt.imshow(semantic_gt_mask[i][3]*0.5, vmin=0, cmap=boundary, vmax=vmax, alpha=alpha)
            plt.imshow(sd_map[i][0]*0.8, vmin=0, cmap='Greys', vmax=1, alpha=0.9)
            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.axis('off')
            plt.tight_layout()
            print('saving', imname)
            plt.savefig(imname, bbox_inches='tight', format='png', dpi=dpi)
            plt.close()

        imname = os.path.join(save_path_seg, 'sd_map.png')
        if not os.path.exists(imname):
            plt.figure(figsize=(w*2/100, 4))
            plt.imshow(sd_map[i][0]*0.8, vmin=0, cmap='Greys', vmax=1, alpha=0.9)
            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.axis('off')
            plt.tight_layout()
            print('saving', imname)
            plt.savefig(imname, bbox_inches='tight', format='png', dpi=dpi)
            plt.close()

        # vis pred hdmap
        imname = os.path.join(save_path_seg, 'pred_map.png')
        if not os.path.exists(imname):
            plt.figure(figsize=(w*2/100, 4))
            plt.imshow(semantic[i][1]*0.5, vmin=0, cmap= divier, vmax=vmax, alpha=alpha)
            plt.imshow(semantic[i][2]*0.5, vmin=0, cmap= ped_crossing, vmax=vmax, alpha=alpha)
            plt.imshow(semantic[i][3]*0.5, vmin=0, cmap=boundary, vmax=vmax, alpha=alpha)
            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.imshow(car_img, extent=[w//2-15, w//2+15, h//2-12, h//2+12])
            plt.axis('off')
            plt.tight_layout()
            print('saving', imname)
            plt.savefig(imname, bbox_inches='tight', format='png', dpi=dpi)
            plt.close()

        if with_gt:
            # vis hdmap gt
            imname = os.path.join(save_path_seg, 'gt_map.png')
            if not os.path.exists(imname):
                plt.figure(figsize=(w*2/100, 4))
                plt.imshow(semantic_gt_mask[i][1]*0.5, vmin=0, cmap=divier, vmax=vmax, alpha=alpha)
                plt.imshow(semantic_gt_mask[i][2]*0.5, vmin=0, cmap=ped_crossing, vmax=vmax, alpha=alpha)
                plt.imshow(semantic_gt_mask[i][3]*0.5, vmin=0, cmap=boundary, vmax=vmax, alpha=alpha)
                plt.xlim(0, w)
                plt.ylim(0, h)
                plt.imshow(car_img, extent=[w//2-15, w//2+15, h//2-12, h//2+12])
                plt.axis('off')
                plt.tight_layout()
                print('saving ', imname)
                plt.savefig(imname, bbox_inches='tight', format='png', dpi=dpi)
                plt.close()


def vis_vec(coords, timestamp, scene_id, save_path, h, w):
    save_path_vec = os.path.join(save_path, 'vec', f'{scene_id}')
    if not os.path.exists(save_path_vec):
        os.makedirs(save_path_vec)

    car_img = Image.open('icon/car_gray.png')
    
    plt.figure(figsize=(w*2/100, 2))
    for coord in coords:
        plt.plot(coord[:, 0], coord[:, 1], linewidth=2)

    plt.xlim((0, w))
    plt.ylim((0, h))
    plt.axis('off')
    plt.grid(False)
    plt.imshow(car_img, extent=[w//2-15, w//2+15, h//2-12, h//2+12])

    img_name = os.path.join(save_path_vec, f'{timestamp}_vecz_.jpg')
    print('saving', img_name)
    plt.savefig(img_name)
    plt.close()


def eval_vis_all(model, save_path, val_loader):
    model.eval()
    total_intersects = 0
    total_union = 0
    i=0
    with torch.no_grad():
        for (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
            yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, osm_masks, osm_vectors, masked_map, timestamps, scene_ids) in tqdm.tqdm(val_loader):
            # import pdb; pdb.set_trace()
            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                            post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                            lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), osm_masks.float().cuda())

            semantic_gt = semantic_gt.cuda().float()
            device = semantic_gt.device
            if semantic.device != device:
                semantic = semantic.to(device)
            intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union
            vis(semantic.cpu().float(), semantic_gt.cpu().float(), osm_masks.float(), timestamps, scene_ids, save_path, with_gt=True)
            i+=1
    return (total_intersects / (total_union + 1e-7))

def main(cfg):
    # import pdb; pdb.set_trace()
    global SCENE_CANDIDATE
    SCENE_CANDIDATE = Nu_SCENE_CANDIDATE
    if 'dataset' in cfg:
        if cfg.dataset == 'av2':
            SCENE_CANDIDATE = AV2_SCENE_CANDIDATE

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': cfg.image_size,
        'xbound': cfg.xbound,
        'ybound': cfg.ybound,
        'zbound': cfg.zbound,
        'dbound': cfg.dbound,
        'thickness': cfg.thickness,
        'angle_class': cfg.angle_class,
        'patch_w': cfg.patch_w, 
        'patch_h': cfg.patch_h, 
        'mask_ratio': cfg.mask_ratio,
        'mask_flag': cfg.mask_flag,
        'sd_map_path': cfg.sd_map_path,
    }

    train_loader, val_loader = semantic_dataset(cfg, cfg.version, cfg.dataroot, data_conf, 
        cfg.batch_size, cfg.nworkers, cfg.dataset)
    model = get_model(cfg,  data_conf, cfg.instance_seg, cfg.embedding_dim, cfg.direction_pred, cfg.angle_class)

    state_dict_model = torch.load(cfg.modelf)
    new_state_dict = OrderedDict()
    for k, v in state_dict_model.items(): 
        name = k[7:] 
        new_state_dict[name] = v
    # import pdb; pdb.set_trace()
    model.load_state_dict(new_state_dict, strict=True)
    model.cuda()
    if "vis_path" not in cfg:
        cfg.vis_path = os.path.join(cfg.logdir, "vis")
    eval_vis_all(model, cfg.vis_path, val_loader)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P-MapNet pre-train HD Prior.')
    parser.add_argument("config", help = 'path to config file', type=str, default=None)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    main(cfg)