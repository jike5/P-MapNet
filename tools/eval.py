import argparse
import tqdm
import os
import sys
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + '/..')
import torch
from tools.config import Config
from tools.evaluation.iou import get_batch_iou
from tools.evaluation import lpips
from data_osm.dataset import semantic_dataset
from data_osm.const import NUM_CLASSES
from model import get_model
from tools.postprocess.vectorize import vectorize
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

# eval only pre-train mae
def eval_pretrain(bevencode_bd, val_loader):
    bevencode_bd.eval()
    total_intersects = 0
    total_union = 0

    with torch.no_grad():
        total_epe = 0
        index = 0 
        for (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
             yaw_pitch_roll, semantic_gt, instance_gt, direction_gt,osm_masks, 
             osm_vectors, masked_map, timestamp, scene_id) in tqdm.tqdm(val_loader):

            semantic, embedding, direction = bevencode_bd(masked_map.cuda().float())
            semantic_gt = semantic_gt.cuda().float()
            intersects, union = get_batch_iou(onehot_encoding(semantic.cuda()), semantic_gt)
            total_intersects += intersects
            total_union += union
            index = index + 1
    return total_intersects / (total_union + 1e-7) 


def eval_iou(model, val_loader):
    model.eval()
    total_intersects = 0
    total_union = 0
    with torch.no_grad():
        for (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
             yaw_pitch_roll, semantic_gt, instance_gt, direction_gt,osm_masks, 
             osm_vectors, masked_map, timestamp, scene_id) in tqdm.tqdm(val_loader):
            
            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), osm_masks.float().cuda())

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


def eval_all(model, val_loader):
    model.eval()
    total_intersects = 0
    total_union = 0
    i=0
    lpipss1 = []
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt,osm_masks, osm_vectors, masks_bd_osm, mask_bd, timestamp, scene_ids in tqdm.tqdm(val_loader):
            

            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), osm_masks.float().cuda())

            gt = semantic_gt[:,1:4,:,:].clone().cuda()
            pred = semantic[:,1:4,:,:].clone().cuda()

            lpipss1.append(lpips(pred.float(), gt.float(), net_type='alex') / pred.shape[0])

            semantic_gt = semantic_gt.cuda().float()

            device = semantic_gt.device
            if semantic.device != device:
                semantic = semantic.to(device)
            intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union
            i+=1
    print("  LPIPS1: {:>12.7f}".format(torch.tensor(lpipss1).mean(), ".5"))
    print("  IOU: {:>12.7f}".format((total_intersects / (total_union + 1e-7))), ".5")
    # return (total_intersects / (total_union + 1e-7))


def main(args):
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

    train_loader, val_loader = semantic_dataset(args, args.version, args.dataroot, data_conf, 
        args.batch_size, args.nworkers, cfg.dataset)
    model = get_model(args,  data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
 
    state_dict_model = torch.load(args.modelf)
    new_state_dict = OrderedDict()
    for k, v in state_dict_model.items(): 
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()

    if "pretrain" in str(args.config):
        print(eval_pretrain(model, val_loader))
    else:
        print(eval_iou(model, val_loader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate HDMap Construction Results..')
    parser.add_argument("config", help = 'path to config file', type=str, default=None)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    main(cfg)  
