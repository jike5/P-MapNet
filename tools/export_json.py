import argparse
import tqdm
import torch
import mmcv
from config import Config
import sys
import os
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + '/..')
from data_osm.dataset import semantic_dataset
from data_osm.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize
from collections import OrderedDict
from tools.evaluation.iou import get_batch_iou
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os
from PIL import Image



def gen_dx_bx(xbound, ybound):
    dx = [row[2] for row in [xbound, ybound]]
    bx = [row[0] + row[2] / 2.0 for row in [xbound, ybound]]
    nx = [(row[1] - row[0]) / row[2] for row in [xbound, ybound]]
    return dx, bx, nx
def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def export_to_json(model, val_loader, angle_class, args):
    submission = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_external": False,
            "vector": True,
        },
        "results": {}
    } # todo: add mode
        
    dx, bx, nx = gen_dx_bx(args.xbound, args.ybound)
    count = 0
    model.eval()
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, 
            yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, osm_masks, osm_vectors, masked_map, timestamp,scene_id) in enumerate(tqdm.tqdm(val_loader)):
                    
                    segmentation, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                        post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                        lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), osm_masks.float().cuda())
                    
                    for si in range(segmentation.shape[0]):
                        coords, confidences, line_types = vectorize(segmentation[si], embedding[si], direction[si], angle_class)
                        count += 1
                        vectors = []
                        for coord, confidence, line_type in zip(coords, confidences, line_types):
                            vector = {'pts': coord * dx + bx, 'pts_num': len(coord), "type": line_type, "confidence_level": confidence}
                            vectors.append(vector)
                        rec = val_loader.dataset.samples[batchi * val_loader.batch_size + si]
                        submission['results'][rec['token']] = vectors
    mmcv.dump(submission, args.result_path)


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
    model = get_model(args,  data_conf,  args.instance_seg, args.embedding_dim, args.direction_pred,  args.angle_class)
    # import pdb; pdb.set_trace()
    state_dict_model_120 = torch.load(args.modelf)
    new_state_dict_120 = OrderedDict()
    for k, v in state_dict_model_120.items(): 
        name = k[7:] 
        new_state_dict_120[name] = v
    model.load_state_dict(new_state_dict_120, strict=True)
    model.cuda()

    export_to_json(model, val_loader, args.angle_class, args)

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export vector results to json.')
    parser.add_argument("config", help = 'path to config file', type=str, default=None)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    print("cfg: ", cfg)
    main(cfg)  
