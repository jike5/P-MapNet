import os
import numpy as np
import sys
import logging
import time
from tensorboardX import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tools.config import Config
from torch.optim.lr_scheduler import StepLR
from tools.loss import SimpleLoss, DiscriminativeLoss
from data_osm.dataset import semantic_dataset
from data_osm.const import NUM_CLASSES
from tools.evaluation.iou import get_batch_iou
from tools.evaluation.angle_diff import calc_angle_diff
from tools.eval import onehot_encoding, eval_iou
from model.utils.map_mae_head import vit_base_patch8
import warnings
warnings.filterwarnings("ignore")

import tqdm
import pdb
from PIL import Image
from model import get_model

from collections import OrderedDict
import torch.nn.functional as F
from sklearn import metrics


def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)

def train(cfg):
    if not os.path.exists(cfg.logdir):
        os.makedirs(cfg.logdir)
    logname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    logging.basicConfig(filename=os.path.join(cfg.logdir, logname+'.log'),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

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

    model = get_model(cfg, data_conf, cfg.instance_seg, cfg.embedding_dim, cfg.direction_pred, cfg.angle_class)
    # import pdb; pdb.set_trace()
    if "hd" in cfg.model:
        cfg.modelf_map = cfg.modelf_map if "modelf_map" in cfg else None
        cfg.modelf_mae = cfg.modelf_mae if "modelf_mae" in cfg else None
        if cfg.modelf_map:
            state_dict_model = torch.load(cfg.modelf_map)
            new_state_dict = OrderedDict()
            for k, v in state_dict_model.items(): 
                name = k[7:] 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)

        if cfg.modelf_mae:
            state_dict_model = torch.load(cfg.modelf_mae)
            new_state_dict = OrderedDict()
            for k, v in state_dict_model.items():
                name = k.replace('module', 'mae_head')
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
        
        cfg.freeze_backbone = cfg.freeze_backbone if "freeze_backbone" in cfg else None
        if cfg.freeze_backbone:
            for name, param in model.named_parameters():
                if 'mae_head' not in name:
                    param.requires_grad = False

    if 'resume' in cfg and cfg.resume is not None:
        print("Loading checkpoint from cfg.resume: ", cfg.resume)
        state_dict_model = torch.load(cfg.resume)
        new_state_dict = OrderedDict()
        for k, v in state_dict_model.items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    model = nn.DataParallel(model, device_ids=cfg.gpus)
    model.cuda(device=cfg.gpus[0])
    # import pdb; pdb.set_trace()
    train_loader, val_loader = semantic_dataset(cfg, cfg.version, cfg.dataroot, data_conf, 
        cfg.batch_size, cfg.nworkers, cfg.dataset)
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = StepLR(opt, 3, 0.1)
    writer = SummaryWriter(logdir=cfg.logdir)
    
    loss_fn = SimpleLoss(cfg.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(cfg.embedding_dim, cfg.delta_v, cfg.delta_d).cuda()
    direction_loss_fn = torch.nn.BCELoss(reduction='none')

    counter = 0
    last_idx = len(train_loader) - 1
    for epoch in range(cfg.nepochs):
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
            yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, osm_masks, osm_vectors, masked_map, timestamps, scene_ids) in enumerate(train_loader):
            # import pdb; pdb.set_trace()
            t0 = time.time()
            opt.zero_grad()
            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                   post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                   lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), osm_masks.float().cuda())

            semantic_gt = semantic_gt.cuda().float()
            instance_gt = instance_gt.cuda()

            device = semantic_gt.device
            if semantic.device != device:
                semantic = semantic.to(device)
                embedding = embedding.to(device)
                direction = direction.to(device)
            
            seg_loss = loss_fn(semantic, semantic_gt)
            if cfg.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if cfg.direction_pred:
                direction_gt = direction_gt.cuda()
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(direction, direction_gt, cfg.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            final_loss = seg_loss * cfg.scale_seg + var_loss * cfg.scale_var + dist_loss * cfg.scale_dist + direction_loss * cfg.scale_direction
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            counter += 1
            t1 = time.time()

            if counter % 100 == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                iou = intersects / (union + 1e-7)
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1-t0:>7.4f}    "
                            f"Loss: {final_loss.item():>7.4f}    "
                            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

                write_log(writer, iou, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/direction_loss', direction_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)
                writer.add_scalar('train/angle_diff', angle_diff, counter)
        
        model_name = os.path.join(cfg.logdir, f"model{epoch}.pt")
        torch.save(model.state_dict(), model_name)
        logger.info(f"{model_name} saved")

        iou = eval_iou(model, val_loader)
        logger.info(f"EVAL[{epoch:>2d}]:    "
                    f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")
        write_log(writer, iou, 'eval', counter)
        model.train()
        sched.step()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P-MapNet training with HD Prior.')
    parser.add_argument("config", help = 'path to config file', type=str, default=None)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    if not os.path.exists(cfg.logdir):
        os.makedirs(cfg.logdir)
    with open(os.path.join(cfg.logdir, 'config.txt'), 'w') as f:
        argsDict = cfg.__dict__
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + " : " + str(value) + "\n")
    train(cfg)

