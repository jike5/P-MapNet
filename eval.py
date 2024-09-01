import argparse
import tqdm
import os
import torch

# from data_osm.dataset import semantic_dataset

from evaluation.iou import get_batch_iou
# from model import get_model

from data_osm.dataset import semantic_dataset
from data_osm.const import NUM_CLASSES,NUM_CLASSES_BD
from model import get_model
from model.base import BDEncoder, BevEncode
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def eval_iou(model, val_loader):
    model.eval()
    total_intersects = 0
    total_union = 0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt,osm_masks, osm_vectors, masks_bd_osm, mask_bd, timestamp, scene_id in tqdm.tqdm(val_loader):

            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), osm_masks.float().cuda())

            # semantic, embedding, direction = model(lidar_data.cuda(),lidar_mask.cuda())

            semantic_gt = semantic_gt.cuda().float()
            
            device = semantic_gt.device
            if semantic.device != device:
                semantic = semantic.to(device)
                embedding = embedding.to(device)
                direction = direction.to(device)

            
            intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union
    return total_intersects / (total_union + 1e-7)


def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    print(eval_iou(model, val_loader))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='./dataset/nuscenes') 
 
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='Pmapnet_mae')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)
    parser.add_argument('--modelf_mae', type=str, default=None)
    
    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    # parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    # parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-60.0, 60.0, 0.3])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-30.0, 30.0, 0.3])

    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    parser.add_argument('--mask_flag', action='store_true')
    parser.add_argument("--patch_w", type=int, default=2)
    parser.add_argument("--patch_h", type=int, default=2)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--sd_thickness", type=int, default=5)
    parser.add_argument("--convbd", action='store_false', help='conv the bd')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])#
    
    parser.add_argument('--is_onlybd', action='store_true')
    parser.add_argument('--is_newsplit', action='store_false')

    args = parser.parse_args()
    main(args)
