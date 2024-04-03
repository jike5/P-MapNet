import os
import numpy as np

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from torch.utils.data import Dataset
from data_osm.rasterize import preprocess_map, preprocess_osm_map
from .const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W
from .vector_map import VectorizedLocalMap
from .lidar import get_lidar_data
from .image import normalize_img, img_transform
from .utils import label_onehot_encoding
from model.utils.voxel import pad_or_trim_to_np
from .av2_dataset import AV2PMapNetSemanticDataset

class PMapNetDataset(Dataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(PMapNetDataset, self).__init__()
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]  
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  
        canvas_h = int(patch_h / data_conf['ybound'][2])           
        canvas_w = int(patch_w / data_conf['xbound'][2])           

        self.is_train = is_train
        self.data_conf = data_conf
        self.patch_size = (patch_h, patch_w)    
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size, sd_map_path=data_conf['sd_map_path'])
        self.scenes = self.get_scenes(version, is_train)
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

        samples.sort(key=lambda x: (x['scene_token'], x['timestamp'])) 

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
        fH, fW = self.data_conf['image_size'] 
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H) 
        resize_dims = (fW, fH)
        return resize, resize_dims

    def get_imgs(self, rec):
        imgs = []        
        trans = []       
        rots = []        
        intrins = []     
        post_trans = [] 
        post_rots = []  

        for cam in CAMS: 
            samp = self.nusc.get('sample_data', rec['data'][cam]) 
            imgname = os.path.join(self.nusc.dataroot, samp['filename']) 
            img = Image.open(imgname) 

            resize, resize_dims = self.sample_augmentation() 
            img, post_rot, post_tran = img_transform(img, resize, resize_dims) 
            img = normalize_img(img)
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            imgs.append(img)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(torch.Tensor(sens['translation']))                         
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)) 
            intrins.append(torch.Tensor(sens['camera_intrinsic']))                  
        return torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)

    def get_vectors(self, rec):
        
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location'] 
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token']) 
        vectors, polygon_geom, osm_vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        
        return vectors, polygon_geom, osm_vectors
    
    def __getitem__(self, idx):
        rec = self.samples[idx]

        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        vectors, _, _ = self.get_vectors(rec)

        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, vectors


class PMapNetSemanticDataset(PMapNetDataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(PMapNetSemanticDataset, self).__init__(version, dataroot, data_conf, is_train)
        self.thickness = data_conf['thickness']

        self.angle_class = data_conf['angle_class']
        self.data_conf = data_conf
        
    def get_semantic_map(self, rec):
        time = rec['timestamp']
        scene_token = rec['scene_token']
        scene_id = self.nusc.get('scene', scene_token)['name']

        vectors, _, osm_vectors = self.get_vectors(rec) 
        osm_masks, _ = preprocess_osm_map(osm_vectors, self.patch_size, self.canvas_size)

        instance_masks, forward_masks, backward_masks, instance_mask_map = preprocess_map(
            self.data_conf, vectors, self.patch_size, self.canvas_size, 
            NUM_CLASSES, self.thickness, self.angle_class)

        masked_map = instance_mask_map != 0
        masked_map = torch.cat([(~torch.any(masked_map, axis=0)).unsqueeze(0), masked_map])

        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks]) 
        instance_masks = instance_masks.sum(0)
        forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
        backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
        direction_masks = forward_oh_masks + backward_oh_masks 
        direction_masks = direction_masks / direction_masks.sum(0)
        return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks, osm_masks, osm_vectors, masked_map

    def __getitem__(self, idx):
        rec = self.samples[idx] 
        timestamp = torch.tensor(rec['timestamp'])
        scene_token = rec['scene_token']
        scene_id = self.nusc.get('scene', scene_token)['name']

        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec) 
        semantic_masks, instance_masks, _, _, direction_masks, osm_masks, osm_vectors, masked_map = self.get_semantic_map(rec)
        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks, osm_masks, osm_vectors, masked_map, timestamp, scene_id


def collate_wrapper(batch):
    imgs, trans, rots, intrins, post_trans, post_rots, lidar_datas, lidar_masks, car_trans, yaw_pitch_rolls, semantic_masks, instance_masks, direction_masks, osm_masks, osm_vectors, masked_maps, timestamps, scene_ids = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for img, tran, rot, intrin, post_tran, post_rot, lidar_data, lidar_mask, car_tran, yaw_pitch_roll, semantic_mask, instance_mask, direction_mask, osm_mask, osm_vector, masked_map, timestamp, scene_id in batch:
        imgs.append(img)
        trans.append(tran)
        rots.append(rot)
        intrins.append(intrin)
        post_trans.append(post_tran)
        post_rots.append(post_rot)
        if lidar_data is not None:
            lidar_datas.append(torch.tensor(lidar_data))
        if lidar_mask is not None:
            lidar_masks.append(torch.tensor(lidar_mask))
        # TODO: debug here!!!
        car_trans.append(car_tran)
        yaw_pitch_rolls.append(yaw_pitch_roll)
        semantic_masks.append(semantic_mask)
        instance_masks.append(instance_mask)
        direction_masks.append(direction_mask)
        osm_masks.append(osm_mask)
        osm_vectors.append(osm_vector) # not stack
        masked_maps.append(masked_map)
        timestamps.append(timestamp)
        scene_ids.append(scene_id)
    imgs = torch.stack(imgs)
    trans = torch.stack(trans)
    rots = torch.stack(rots)
    intrins = torch.stack(intrins)
    post_trans = torch.stack(post_trans)
    post_rots = torch.stack(post_rots)
    if lidar_datas:
        lidar_datas = torch.stack(lidar_datas)  
    if lidar_masks:
        lidar_masks = torch.stack(lidar_masks)
    car_trans = torch.stack(car_trans)
    yaw_pitch_rolls = torch.stack(yaw_pitch_rolls)
    semantic_masks = torch.stack(semantic_masks)
    instance_masks = torch.stack(instance_masks)
    direction_masks = torch.stack(direction_masks)
    osm_masks = torch.stack(osm_masks)
    masked_maps = torch.stack(masked_maps)
    timestamps = torch.stack(timestamps)
    return imgs, trans, rots, intrins, post_trans, post_rots, lidar_datas, lidar_masks, car_trans, yaw_pitch_rolls, semantic_masks, instance_masks, direction_masks, osm_masks, osm_vectors, masked_maps, timestamps, scene_ids


def semantic_dataset(cfg, version, dataroot, data_conf, batch_size, nworkers, dataset_type='nuScenes'):
    if dataset_type == 'nuScenes':
        train_dataset = PMapNetSemanticDataset(version, dataroot, data_conf, is_train=True)
        val_dataset = PMapNetSemanticDataset(version, dataroot, data_conf, is_train=False)
    elif dataset_type == 'av2':
        if 'ann_path' in cfg:
            data_conf.update({"ann_path":cfg.ann_path})
        train_dataset = AV2PMapNetSemanticDataset(dataroot, data_conf, is_train=True)
        val_dataset = AV2PMapNetSemanticDataset(dataroot, data_conf, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers, drop_last=True, collate_fn=collate_wrapper)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nworkers, collate_fn=collate_wrapper)
    return train_loader, val_loader


if __name__ == '__main__':
    data_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = PMapNetSemanticDataset(version='v1.0-mini', dataroot='', data_conf=data_conf, is_train=False)
    for idx in range(dataset.__len__()):
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_mask = dataset.__getitem__(idx)

