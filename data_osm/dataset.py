import os
import numpy as np

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from torch.utils.data import Dataset
from data_osm.rasterize import preprocess_map, preprocess_map_withbd, preprocess_osm_map, preprocess_map_onlybd
from .const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W, NUM_CLASSES_BD
from .vector_map import VectorizedLocalMap
from .lidar import get_lidar_data
from .image import normalize_img, img_transform
from .utils import label_onehot_encoding
from model.voxel import pad_or_trim_to_np

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class HDMapNetDataset(Dataset):
    def __init__(self, version, data_val, dataroot, data_conf, is_train):
        super(HDMapNetDataset, self).__init__()
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]  # 30.0
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  # 60.0
        canvas_h = int(patch_h / data_conf['ybound'][2])           # 200
        canvas_w = int(patch_w / data_conf['xbound'][2])           # 400
        # patch_h = 60
        # patch_w = 120
        # canvas_h = int(patch_h/0.15)
        # canvas_w = int(patch_w/0.15)


        self.is_only_bd = data_conf['is_onlybd']

        self.is_train = is_train
        self.data_conf = data_conf
        self.patch_size = (patch_h, patch_w)    # 物理世界中bev大小，m制
        self.canvas_size = (canvas_h, canvas_w) # 画布大小，像素
        # self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)
        self.scenes = self.get_scenes(version, is_train) # ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']
        
        self.samples = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
            'v1.0-test': {True: 'test_mini', False: 'test_mini'},
        }[version][is_train]

        return create_splits_scenes()[split]


    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]
        scene_id=[]
        for samp in samples:
            scene_id.append(self.nusc.get('scene', samp['scene_token'])['name'])
            samples = [samp for samp in samples if 
                       self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp'])) # keys: dict_keys(['token', 'timestamp', 'prev', 'next', 'scene_token', 'data', 'anns'])

        return samples

    def get_lidar(self, rec):
        lidar_data = get_lidar_data(self.nusc, rec, nsweeps=3, min_distance=2.2)
        lidar_data = lidar_data.transpose(1, 0)
        num_points = lidar_data.shape[0]
        lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        return lidar_data, lidar_mask

    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        car_trans = ego_pose['translation']
        pos_rotation = Quaternion(ego_pose['rotation'])
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size'] # 128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H) # IMG_ORIGIN_W:900, IMG_ORIGIN_H:1600 图像resize后的大小与原图大小的比例
        resize_dims = (fW, fH)
        return resize, resize_dims


    def get_imgs(self, rec):
        imgs = []        # resize后的图片
        trans = []       # [3] 值为0
        rots = []        # 3*3 diag(resize[0], resize[1], 1)
        intrins = []     # 3*3 K
        post_trans = []  # 3
        post_rots = []   # 3*3

        for cam in CAMS: # 共有六个摄像头
            samp = self.nusc.get('sample_data', rec['data'][cam]) # 获取某个cam的数据token,数据位置等信息, rec['data'] dict_keys(['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])
            imgname = os.path.join(self.nusc.dataroot, samp['filename']) # samp: dict_keys(['token', 'sample_token', 'ego_pose_token', 'calibrated_sensor_token', 'timestamp', 'fileformat', 'is_key_frame', 'height', 'width', 'filename', 'prev', 'next', 'sensor_modality', 'channel'])
            img = Image.open(imgname) # size: 1600 900

            resize, resize_dims = self.sample_augmentation() # resize为resize后的大小与原图的大小比例 resize_dims=(128,352)为main parser中设置的image size
            img, post_rot, post_tran = img_transform(img, resize, resize_dims) # img为resize后的img; post_rot为3*3，前2*2为缩放因子diag(resize[0], resize[1]); post_tran 为[3]，值为0
            # resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            # img, post_rot, post_tran = img_transform(img, resize, resize_dims, crop, flip, rotate)

            img = normalize_img(img)
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            imgs.append(img)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(torch.Tensor(sens['translation']))                         # 3
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)) # 3*3
            intrins.append(torch.Tensor(sens['camera_intrinsic']))                  # 3*3 K
        return torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)

    def get_vectors(self, rec):
        

        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location'] # 'boston-seaport', 'singapore-north' 地图的名称
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token']) # 和get_ego_pose中一样，获取车位姿的token等信息，描述在地图的局部坐标系中
        # time_candi = [1542799345696426, 1537298056450495, 1537298047650680, 1537298038950431, 1535489299547057, 1533151741547700, 1533151616447606, 1533151369947807]
        # if rec['timestamp'] in time_candi:
        #     print('time',rec['timestamp'])
        #     print('location: ', location)
        #     print('ego_pose: ', ego_pose)
        if self.is_only_bd:
            vectors, polygon_geom, osm_vectors = self.vector_map.gen_vectorized_samples_only_bd(location, ego_pose['translation'], ego_pose['rotation'])
        else:
            vectors, polygon_geom, osm_vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        
        return vectors, polygon_geom, osm_vectors # 返回的为列表，每个元素为dict；pts为n*2的numpy矩阵，表示线的点坐标；pts_num为n；type表示标签
    
    def __getitem__(self, idx):
        rec = self.samples[idx]

        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        vectors, _, _ = self.get_vectors(rec)

        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, vectors


class HDMapNetSemanticDataset(HDMapNetDataset):
    def __init__(self, version,data_val, dataroot, data_conf, is_train):
        super(HDMapNetSemanticDataset, self).__init__(version,data_val, dataroot, data_conf, is_train)
        self.thickness = data_conf['thickness']
        self.sd_thickness = data_conf['sd_thickness']
        self.angle_class = data_conf['angle_class']
        self.data_conf = data_conf
    def get_semantic_map(self, rec):
        time = rec['timestamp']
        scene_token = rec['scene_token']
        scene_id = self.nusc.get('scene', scene_token)['name']

        vectors, _, osm_vectors = self.get_vectors(rec) # 返回的为列表，每个元素为dict；pts为n*2的numpy矩阵，表示线的点坐标；pts_num为n；type表示标签
        # import pdb;pdb.set_trace()
        # print(vectors)
        osm_masks, _ = preprocess_osm_map(time,scene_id, osm_vectors, self.patch_size, self.canvas_size, self.sd_thickness)
        # osm_masks = osm_masks != 0
        if self.is_only_bd:
            instance_masks, forward_masks, backward_masks, instance_mask_bd = preprocess_map_onlybd(self.data_conf, time, vectors, self.patch_size, self.canvas_size, NUM_CLASSES_BD, self.thickness, self.angle_class)
        else:
            instance_masks, forward_masks, backward_masks, instance_mask_bd = preprocess_map_withbd(self.data_conf, time, vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class)

        
        # add masked bd and osm
        # masks_bd_osm = osm_masks + instance_mask_bd
        masks_bd_osm = osm_masks
        masks_bd_osm = masks_bd_osm != 0

        mask_bd = instance_mask_bd != 0

        # instance_masks, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class)
        semantic_masks = instance_masks != 0 # instance_masks: 3*200*400;  forward_masks, backward_masks: 200*400
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks]) # 4,200,400
        instance_masks = instance_masks.sum(0)
        forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1) #  将label中不同的值映射到不同层里，例如对于方向0~36，将37个类别，对应37个通道对应位置设置为1
        backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
        direction_masks = forward_oh_masks + backward_oh_masks # 37*200*400
        direction_masks = direction_masks / direction_masks.sum(0)
        return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks, osm_masks, osm_vectors, masks_bd_osm, mask_bd


    def __getitem__(self, idx):
        rec = self.samples[idx] # dict_keys(['token', 'timestamp', 'prev', 'next', 'scene_token', 'data', 'anns']) 某一时刻的token信息
        timestamp = torch.tensor(rec['timestamp'])
        # timestamp = rec['timestamp']
        # import pdb;pdb.set_trace()
        # breakpoint()
        scene_token = rec['scene_token']
        scene_id = self.nusc.get('scene', scene_token)['name']

        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec) # trans rots为相机标定外参
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec) # 这里car_trans和ypr应该为车身的位置和姿态 car_trans: 平移量
        semantic_masks, instance_masks, _, _, direction_masks, osm_masks, osm_vectors, masks_bd_osm, mask_bd = self.get_semantic_map(rec)
        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks, osm_masks, osm_vectors, masks_bd_osm, mask_bd, timestamp, scene_id

def collate_wrapper(batch):
    # imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks, osm_masks, osm_vectors
    imgs, trans, rots, intrins, post_trans, post_rots, lidar_datas, lidar_masks, car_trans, yaw_pitch_rolls, semantic_masks, instance_masks, direction_masks, osm_masks, osm_vectors, masks_bd_osms,mask_bds, timestamps, scene_ids = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for img, tran, rot, intrin, post_tran, post_rot, lidar_data, lidar_mask, car_tran, yaw_pitch_roll, semantic_mask, instance_mask, direction_mask, osm_mask, osm_vector, masks_bd_osm, mask_bd, timestamp, scene_id in batch:
        imgs.append(img)
        trans.append(tran)
        rots.append(rot)
        intrins.append(intrin)
        post_trans.append(post_tran)
        post_rots.append(post_rot)
        lidar_datas.append(torch.tensor(lidar_data))
        lidar_masks.append(torch.tensor(lidar_mask))
        car_trans.append(car_tran)
        yaw_pitch_rolls.append(yaw_pitch_roll)
        semantic_masks.append(semantic_mask)
        instance_masks.append(instance_mask)
        direction_masks.append(direction_mask)
        osm_masks.append(osm_mask)
        osm_vectors.append(osm_vector) # not stack
        masks_bd_osms.append(masks_bd_osm)
        mask_bds.append(mask_bd)
        timestamps.append(timestamp)
        scene_ids.append(scene_id)
    imgs = torch.stack(imgs)
    trans = torch.stack(trans)
    rots = torch.stack(rots)
    intrins = torch.stack(intrins)
    post_trans = torch.stack(post_trans)
    post_rots = torch.stack(post_rots)
    lidar_datas = torch.stack(lidar_datas)
    lidar_masks = torch.stack(lidar_masks)
    car_trans = torch.stack(car_trans)
    yaw_pitch_rolls = torch.stack(yaw_pitch_rolls)
    semantic_masks = torch.stack(semantic_masks)
    instance_masks = torch.stack(instance_masks)
    direction_masks = torch.stack(direction_masks)
    osm_masks = torch.stack(osm_masks)
    masks_bd_osms = torch.stack(masks_bd_osms)
    mask_bds = torch.stack(mask_bds)
    timestamps = torch.stack(timestamps)
    return imgs, trans, rots, intrins, post_trans, post_rots, lidar_datas, lidar_masks, car_trans, yaw_pitch_rolls, semantic_masks, instance_masks, direction_masks, osm_masks, osm_vectors, masks_bd_osms,mask_bds, timestamps, scene_ids

def semantic_dataset(version,data_val, dataroot, data_conf, bsz, nworkers):
    train_dataset = HDMapNetSemanticDataset(version,data_val, dataroot, data_conf, is_train=True)
    val_dataset = HDMapNetSemanticDataset(version,data_val, dataroot, data_conf, is_train=False)
    # imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, sd_map, sd_vector = val_dataset.__getitem__(720)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True, collate_fn=collate_wrapper)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers, collate_fn=collate_wrapper)
    return train_loader, val_loader


if __name__ == '__main__':
    data_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = HDMapNetSemanticDataset(version='v1.0-mini', dataroot='/DATA_EDS/jiangz/HDMapNet_all/dataset_mini/nuscenes', data_conf=data_conf, is_train=False)
    for idx in range(dataset.__len__()):
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_mask = dataset.__getitem__(idx)

