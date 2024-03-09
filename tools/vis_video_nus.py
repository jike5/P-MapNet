import sys
sys.path.append('.')
import argparse
import tqdm
import os
import cv2
import torch
import imageio # TODO: pip install imageio[ffmpeg] imageio[pyav]
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

CAMS = ['FRONT_LEFT','FRONT','FRONT_RIGHT',
             'BACK_LEFT','BACK','BACK_RIGHT',]

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

def vis_surround(val_loader, save_path):
    for (imgs, trans, rots, intrins, post_trans, post_rots, 
         lidar_data, lidar_mask, car_trans, yaw_pitch_roll, 
         semantic_gt, instance_gt, direction_gt, osm_masks, 
         osm_vectors, masked_map, timestamps, scene_ids
         ) in tqdm.tqdm(val_loader):
        # import pdb; pdb.set_trace()
        for i in range(imgs.shape[0]):
            save_path_seg = os.path.join(save_path, f'{scene_ids[i]}', f'{timestamps[i]}')
            if not os.path.exists(save_path_seg):
                os.makedirs(save_path_seg)
            imname = os.path.join(save_path_seg, 'surround.png')
            img_list = [denormalize_img(img) for img in imgs[i]] # PIL
            row_1_list = []
            for i in range(3):
                cv2_img = cv2.cvtColor(np.asarray(img_list[i]), cv2.COLOR_RGB2BGR)
                lw = 8
                tf = max(lw - 1, 1)
                w, h = cv2.getTextSize(CAMS[i], 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                p1 = (0,0)
                p2 = (w,h+3)
                color=(0, 0, 0)
                txt_color=(255, 255, 255)
                cv2.rectangle(cv2_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(cv2_img,
                            CAMS[i], (p1[0], p1[1] + h + 2),
                            0,
                            lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
                
                row_1_list.append(cv2_img)
            row_2_list = []
            for j in range(3,6):
                if j == 4:
                    cv2_img = cv2.cvtColor(np.asarray(img_list[j].transpose(Image.FLIP_LEFT_RIGHT)), cv2.COLOR_RGB2BGR)     
                else:
                    cv2_img = cv2.cvtColor(np.asarray(img_list[j]), cv2.COLOR_RGB2BGR)
                lw = 8
                tf = max(lw - 1, 1)
                w, h = cv2.getTextSize(CAMS[j], 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                p1 = (0,0)
                p2 = (w,h+3)
                color=(0, 0, 0)
                txt_color=(255, 255, 255)
                cv2.rectangle(cv2_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(cv2_img,
                            CAMS[j], (p1[0], p1[1] + h + 2),
                            0,
                            lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
                row_2_list.append(cv2_img)
            row_1_img=cv2.hconcat(row_1_list)
            row_2_img=cv2.hconcat(row_2_list)
            cams_img = cv2.vconcat([row_1_img,row_2_img])
            print('saving', imname)
            cv2.imwrite(imname, cams_img, [cv2.IMWRITE_JPEG_QUALITY, 70])


def generate_video(surround_imgs, vis_path, cfg):
    parent_dir = os.path.join(vis_path, "..")
    vis_subdir_list = []
    size = (1680, 464)
    video_path = os.path.join(parent_dir, '{}.mp4'.format(cfg.video_name))
    video = imageio.get_writer(video_path, fps=cfg.fps)

    scene_id_files = os.listdir(vis_path)
    scene_id_files = sorted(scene_id_files)
    for scene_id in tqdm.tqdm(scene_id_files):
        if scene_id not in SCENE_CANDIDATE:
            continue
        timestamp_files = os.listdir(os.path.join(vis_path, scene_id))
        timestamp_files = sorted(timestamp_files) 
        for timestamp in timestamp_files:
            surround_img_path = os.path.join(surround_imgs, scene_id, timestamp, "surround.png")
            gt_img_path = os.path.join(vis_path, scene_id, timestamp, "gt_map.png")
            sd_map_path = os.path.join(vis_path, scene_id, timestamp, "sd_map.png")
            pred_map_path = os.path.join(vis_path, scene_id, timestamp, "pred_map.png")

            surround_img = cv2.imread(surround_img_path)
            gt_map_img = cv2.imread(gt_img_path)
            sdmap_img = cv2.imread(sd_map_path)
            pred_map_img = cv2.imread(pred_map_path)
            if surround_img is None or gt_map_img is None or sdmap_img is None or pred_map_img is None:
                continue

            border_value = (0,0,0)
            sdmap_img = np.rot90(sdmap_img)
            gt_map_img = np.rot90(gt_map_img)
            pred_map_img = np.rot90(pred_map_img)
            
            
            sdmap_img = cv2.copyMakeBorder(sdmap_img, 1, 1, 5, 5, cv2.BORDER_CONSTANT, None, value = border_value)
            gt_map_img = cv2.copyMakeBorder(gt_map_img, 1, 1, 5, 5, cv2.BORDER_CONSTANT, None, value = border_value)
            pred_map_img = cv2.copyMakeBorder(pred_map_img, 1, 1, 5, 2, cv2.BORDER_CONSTANT, None, value = border_value)

            cams_h, cam_w,_ = surround_img.shape
            map_h,  map_w,_ = gt_map_img.shape
            resize_ratio = cams_h / map_h
            resized_w = map_w * resize_ratio
            resized_sdmap_img= cv2.resize(sdmap_img, (int(resized_w), int(cams_h)))
            resized_gt_map_img = cv2.resize(gt_map_img,(int(resized_w),int(cams_h)))
            resized_pred_map_img = cv2.resize(pred_map_img,(int(resized_w),int(cams_h)))

            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale
            fontScale = 2
            # Line thickness of 2 px
            thickness = 5
            # org
            org = (20, 50)      
            # Blue color in BGR
            color = (0, 0, 0)
            # Using cv2.putText() method
            resized_pred_map_img = cv2.putText(resized_pred_map_img, 'PRED', org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            resized_gt_map_img = cv2.putText(resized_gt_map_img, 'GT', org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            resized_sdmap_img = cv2.putText(resized_sdmap_img, 'SD MAP', org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            
            # resized_sdmap_img = cv2.putText(resized_sdmap_img, str(scene_id), (20, 200), font, 
            #                 fontScale, color, thickness, cv2.LINE_AA)

            sample_img = cv2.hconcat([surround_img, resized_sdmap_img, resized_gt_map_img, resized_pred_map_img])
            
            print("save ", os.path.join(vis_path, scene_id, timestamp, "sample_img.png"))
            resized_img = cv2.resize(sample_img, size)
            cv2.imwrite(os.path.join(vis_path, scene_id, timestamp, "sample_img.png"), resized_img)
            # video.write(resized_img)
            video.append_data(cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

    video.close()
    print("finished processing video")

def main(cfg):
    global SCENE_CANDIDATE
    SCENE_CANDIDATE = Nu_SCENE_CANDIDATE
    if 'dataset' in cfg:
        if cfg.dataset == 'av2':
            SCENE_CANDIDATE = AV2_SCENE_CANDIDATE

    if cfg.dataset == "nuScenes":
        cfg.image_size = [900, 1600]
    else:
        pass
        # TODO: 

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

    if "surround_imgs" not in cfg:
        cfg.surround_imgs = os.path.join("./Work_dir", "surround_imgs")
    if "fps" not in cfg:
        cfg.fps = 4
    if "video_name" not in cfg:
        cfg.video_name = "demo_test"

    # import pdb; pdb.set_trace()
    if not os.path.exists(cfg.surround_imgs):
        _, val_loader = semantic_dataset(cfg, cfg.version, cfg.dataroot, data_conf, 
            cfg.batch_size, cfg.nworkers, cfg.dataset)
        vis_surround(val_loader, cfg.surround_imgs)
    
    generate_video(cfg.surround_imgs, cfg.vis_path, cfg)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P-MapNet pre-train HD Prior.')
    parser.add_argument("config", help = 'path to config file', type=str, default=None)
    parser.add_argument("vis_path", help = 'path to vis_path', type=str, default=None)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.vis_path = args.vis_path
    main(cfg)