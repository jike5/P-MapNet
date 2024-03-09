import torch
import tqdm
import argparse
from config import Config
import sys
import os
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + '/..')
from tools.evaluation.dataset import PMapNetEvalDataset
from tools.evaluation.chamfer_distance import semantic_mask_chamfer_dist_cum
from tools.evaluation.AP import instance_mask_AP
from tools.evaluation.iou import get_batch_iou

SAMPLED_RECALLS = torch.linspace(0.1, 1, 10)
# THRESHOLDS = [0.2, 0.5, 1.0]
THRESHOLDS = [0.5, 1.0, 1.5]

def get_val_info(args):
    data_conf = {
        'xbound': args.xbound,
        'ybound': args.ybound,
        'thickness': args.thickness,
        'sd_map_path': args.sd_map_path
    }

    dataset = PMapNetEvalDataset(
        args.version, args.dataroot, 'val', args.result_path, data_conf)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    total_CD1 = torch.zeros(args.max_channel).cuda()
    total_CD2 = torch.zeros(args.max_channel).cuda()
    total_CD_num1 = torch.zeros(args.max_channel).cuda()
    total_CD_num2 = torch.zeros(args.max_channel).cuda()
    total_intersect = torch.zeros(args.max_channel).cuda()
    total_union = torch.zeros(args.max_channel).cuda()
    AP_matrix = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()
    AP_count_matrix = torch.zeros((args.max_channel, len(THRESHOLDS))).cuda()


    print('running eval...')
    for pred_map, confidence_level, gt_map in tqdm.tqdm(data_loader):
        
        pred_map = pred_map.cuda() 
        confidence_level = confidence_level.cuda()
        gt_map = gt_map.cuda()

     
        intersect, union = get_batch_iou(pred_map, gt_map)
        CD1, CD2, num1, num2 = semantic_mask_chamfer_dist_cum(
            pred_map, gt_map, args.xbound[2], args.ybound[2], threshold=args.CD_threshold)

        instance_mask_AP(AP_matrix, AP_count_matrix, pred_map, gt_map, args.xbound[2], args.ybound[2],
                         confidence_level, THRESHOLDS, sampled_recalls=SAMPLED_RECALLS, bidirectional=args.bidirectional, threshold_iou=args.threshold_iou)

        total_intersect += intersect.cuda()
        total_union += union.cuda()
        total_CD1 += CD1
        total_CD2 += CD2
        total_CD_num1 += num1
        total_CD_num2 += num2


    CD_pred = total_CD1 / total_CD_num1
    CD_label = total_CD2 / total_CD_num2
    CD = (total_CD1 + total_CD2) / (total_CD_num1 +total_CD_num2) 
    AP = AP_matrix / AP_count_matrix

    return {
        'iou': total_intersect / total_union,
        'CD_pred': CD_pred,
        'CD_label': CD_label,
        'CD': CD,
        'AP': AP,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Vectorized HDMap Construction Results.')
    parser.add_argument("config", help = 'path to config file', type=str, default=None)

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    print(get_val_info(cfg))

