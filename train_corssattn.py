import os

# os.environ['CUDA_VISIBLE_DEVICES']= '3'
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import StepLR
from loss import SimpleLoss, DiscriminativeLoss

from data_osm.dataset import semantic_dataset
from data_osm.const import NUM_CLASSES, NUM_CLASSES_BD
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from model import get_model
from eval import onehot_encoding, eval_iou
import torch.nn as nn
from collections import OrderedDict

def save_fig(img, name):
    plt.matshow(img)
    print('saving', name)
    plt.axis('off')
    plt.savefig(name)
    plt.close()

def save_figs(mask_bd, semantic_gt, timestamp):
    base_path = '/data2/jiangz/mae/runs_lidar/bd_imgs'
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    for si in range(mask_bd.shape[0]):
        plt.figure()
        maskbd = plt.subplot(2,2,1)
        plt.imshow(mask_bd[si][0], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        plt.axis('off')
        maskbd.set_title('mask_bd')


        gtbd = plt.subplot(2,2,2)
        plt.imshow(semantic_gt[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        plt.axis('off')
        gtbd.set_title('gt_bd')

        img_name = f'{timestamp[si]}.png'
        img_name = os.path.join(base_path, img_name)

        plt.savefig(img_name)


def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)


def train(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'num_channels_bd': NUM_CLASSES_BD + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'patch_w': args.patch_w, 
        'patch_h': args.patch_h, 
        'mask_ratio': args.mask_ratio,
        'sd_thickness': args.sd_thickness,
        'mask_flag': args.mask_flag,
        'convbd': args.convbd,
        'is_onlybd': args.is_onlybd,
    }

    train_loader, val_loader = semantic_dataset(args.version,args.data_val, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model,args, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model = nn.DataParallel(model, device_ids=args.gpus)
    if args.modelf is not None:
        model.load_state_dict(torch.load(args.modelf))

    model.cuda(device=args.gpus[0])

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fill.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    sched = StepLR(opt, args.steplr , 0.1)
    writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).cuda()
    direction_loss_fn = torch.nn.BCELoss(reduction='none')

    model.train()
    counter = 0
    last_idx = len(train_loader) - 1
    for epoch in range(args.nepochs):
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
            yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, osm_masks, osm_vectors, masks_bd_osm, mask_bd, timestamp,scene_id) in enumerate(train_loader):
            t0 = time()
            opt.zero_grad()
            # save_figs(mask_bd, semantic_gt, timestamp)

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
            if args.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.cuda()
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0 

            final_loss = seg_loss * args.scale_seg + var_loss * args.scale_var + dist_loss * args.scale_dist + direction_loss * args.scale_direction
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            counter += 1
            # t1 = time()

            if counter % 10 == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                t1 = time()
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
                writer.add_scalar('train/lr', sched.get_last_lr()[0], counter)


        model_name = os.path.join(args.logdir, f"model{epoch}.pt")
        torch.save(model.state_dict(), model_name)
        logger.info(f"{model_name} saved")

        iou = eval_iou(model, val_loader)
        logger.info(f"EVAL[{epoch:>2d}]:    "
                    f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

        write_log(writer, iou, 'eval', counter)
        model.train()

        sched.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config

    parser.add_argument("--logdir", type=str, default='./output/60*30')
    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='./dataset/nuscenes/') 

    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--data_val', type=str, default='trainval', choices=['vis', 'trainval'])

    # model config

    parser.add_argument("--model", type=str, default='PMapNet_sdmap')

    parser.add_argument("--num_decoder_layers", type=int, default=2)
    

    parser.add_argument("--convbd", action='store_true', help='conv the bd')
    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--nworkers", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steplr", type=int, default=10)
    
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)
    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])

    # parser.add_argument("--xbound", nargs=3, type=float, default=[-120.0, 120.0, 0.3])
    # parser.add_argument("--xbound", nargs=3, type=float, default=[-60.0, 60.0, 0.3])
    # parser.add_argument("--ybound", nargs=3, type=float, default=[-30.0, 30.0, 0.3])

    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--sd_thickness", type=int, default=5)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])#
 
    parser.add_argument("--patch_w", type=int, default=2)
    parser.add_argument("--patch_h", type=int, default=2)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument('--mask_flag', action='store_true')
   
    # embedding config
    parser.add_argument('--instance_seg', action='store_false')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_false')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=0.1)
    parser.add_argument("--scale_dist", type=float, default=0.1)
    parser.add_argument("--scale_direction", type=float, default=0.1)

    parser.add_argument('--is_onlybd', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(os.path.join(args.logdir, 'config.txt'), 'w') as f:
        argsDict = args.__dict__
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + " : " + str(value) + "\n")
    train(args)
