import os
import numpy as np
import os.path as osp
import torch
import mmcv
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from time import time
from functools import partial
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from data_osm.rasterize import preprocess_map, preprocess_osm_map
from .const import NUM_CLASSES
from .image import normalize_tensor_img
from .utils import label_onehot_encoding
from .av2map_extractor import AV2MapExtractor
from .pipelines import VectorizeMap, LoadMultiViewImagesFromFiles, FormatBundleMap
from .pipelines import PhotoMetricDistortionMultiViewImage, ResizeMultiViewImages, PadMultiViewImages


CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    # 'stereo_front_left', 'stereo_front_right',
    ]

FAIL_LOGS = [
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d',
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
]

def get_data_from_logid(log_id, loaders, data_root):
    samples = []
    discarded = 0

    # find corresponding loader
    for i in range(3):
        if log_id in loaders[i]._sdb.get_valid_logs():
            loader = loaders[i]
    
    # use lidar timestamps to query all sensors.
    # the frequency is 10Hz
    cam_timestamps = loader._sdb.per_log_lidar_timestamps_index[log_id]
    prev = -1
    for ts in cam_timestamps:
        cam_ring_fpath = [loader.get_closest_img_fpath(
                log_id, cam_name, ts
            ) for cam_name in CAM_NAMES]
        lidar_fpath = loader.get_closest_lidar_fpath(log_id, ts)

        # if bad sensor synchronization, discard the sample
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue

        cams = {}
        for i, cam_name in enumerate(CAM_NAMES):
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
            cams[cam_name] = dict(
                img_fpath=str(cam_ring_fpath[i]),
                intrinsics=pinhole_cam.intrinsics.K,
                extrinsics=pinhole_cam.extrinsics,
            )
        
        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation
        
        samples.append(dict(
            e2g_translation=e2g_translation,
            e2g_rotation=e2g_rotation,
            cams=cams, 
            lidar_fpath=str(lidar_fpath),
            prev=prev,
            # map_fpath=map_fname,
            token=str(ts),
            timestamp=ts,
            log_id=log_id,
            scene_name=log_id))
        
        prev = str(ts)

    return samples, discarded


class AV2Dataset(Dataset):
    """Argoverse2 map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        meta (dict): meta information
        pipeline (Config): data processing pipeline config,
        interval (int): annotation load interval

    """
    def __init__(self, 
                 ann_file,
                 cat2id,
                 roi_size,
                 meta,
                 interval=1,
                 sd_map=None,
                 data_root=None,):
        super().__init__()
        self.ann_file = ann_file
        self.meta = meta
        
        self.classes = list(cat2id.keys())
        self.num_classes = len(self.classes)
        self.cat2id = cat2id
        self.interval = interval
        self.data_root = data_root

        self.load_annotations(self.ann_file)
        
        self.idx2token = {}
        for i, s in enumerate(self.samples):
            self.idx2token[i] = s['token']
        self.token2idx = {v: k for k, v in self.idx2token.items()}
        self.roi_size = roi_size

        self.map_extractor = AV2MapExtractor(self.roi_size, self.id2map, sd_map)
        self.sd_map = True if sd_map is not None else False
        

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.samples)

    def create_av2_infos_mp(self,
                            log_ids,
                            dest_path=None,
                            num_multithread=64):
        for i in FAIL_LOGS:
            if i in log_ids:
                log_ids.remove(i)
        # dataloader by original split
        # import pdb; pdb.set_trace()
        train_loader = AV2SensorDataLoader(Path(osp.join(self.data_root, 'train')), 
            Path(osp.join(self.data_root, 'train')))
        val_loader = AV2SensorDataLoader(Path(osp.join(self.data_root, 'val')), 
            Path(osp.join(self.data_root, 'val')))
        test_loader = AV2SensorDataLoader(Path(osp.join(self.data_root, 'test')), 
            Path(osp.join(self.data_root, 'test')))
        loaders = [train_loader, val_loader, test_loader]

        print('collecting samples...')
        start_time = time()
        print('num cpu:', multiprocessing.cpu_count())
        print(f'using {num_multithread} threads')

        # ignore warning from av2.utils.synchronization_database
        # sdb_logger = logging.getLogger('av2.utils.synchronization_database')
        # prev_level = sdb_logger.level
        # sdb_logger.setLevel(logging.CRITICAL)

        pool = Pool(num_multithread)
        fn = partial(get_data_from_logid, loaders=loaders, data_root=self.data_root)
        
        rt = pool.map_async(fn, log_ids)
        pool.close()
        pool.join()
        results = rt.get()

        samples = []
        discarded = 0
        sample_idx = 0
        for _samples, _discarded in tqdm(results):
            for i in range(len(_samples)):
                _samples[i]['sample_idx'] = sample_idx
                sample_idx += 1
            samples.extend(_samples)
            discarded += _discarded
        
        # sdb_logger.setLevel(prev_level)
        print(f'{len(samples)} available samples, {discarded} samples discarded')

        id2map = {}
        for log_id in tqdm(log_ids):
            for i in range(3):
                if log_id in loaders[i]._sdb.get_valid_logs():
                    loader = loaders[i]
            
            map_path_dir = osp.join(loader._data_dir, log_id, 'map')
            map_fname = str(list(Path(map_path_dir).glob("log_map_archive_*.json"))[0])
            map_fname = osp.join(map_path_dir, map_fname)
            id2map[log_id] = map_fname

        print('collected in {:.1f}s'.format(time() - start_time))
        infos = dict(samples=samples, id2map=id2map)

        print(f'saving results to {dest_path}')
        dir_path = os.path.dirname(dest_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        mmcv.dump(infos, dest_path)

    def mask_ann_file(self, ann_file):
        if 'train' in ann_file:
            train_split = os.listdir(osp.join(self.data_root, 'train'))
            self.create_av2_infos_mp(
                log_ids=train_split,
                dest_path=ann_file)
        elif 'val' in ann_file:
            val_split = os.listdir(osp.join(self.data_root, 'val'))
            self.create_av2_infos_mp(
                log_ids=val_split,
                dest_path=ann_file)
        return ann_file
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # import pdb; pdb.set_trace()
        start_time = time()
        if os.path.exists(ann_file):
            print("Load an existing file: ", ann_file)
            ann = mmcv.load(ann_file)
        else:
            print("Strat create an ann_file: ", ann_file)
            return_file = self.mask_ann_file(ann_file)
            # import pdb; pdb.set_trace()
            ann = mmcv.load(return_file)

        self.id2map = ann['id2map']
        samples = ann['samples']
        samples = samples[::self.interval]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        log_id = sample['log_id']
        
        map_geoms = self.map_extractor.get_map_geom(log_id, sample['e2g_translation'], 
                sample['e2g_rotation'])

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        if self.sd_map:
            sd_map_data = self.map_extractor.get_osm_geom(log_id, 
                    sample['e2g_translation'], sample['e2g_rotation'])
        
        ego2img_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)
        
        input_dict = {
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, NOTE: **ego2cam**
            # 'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'cam_extrinsics': [np.linalg.inv(c['extrinsics']) for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': sample['e2g_rotation'],
            'scene_name': sample['scene_name'],
            'timestamp': sample['timestamp'],
        }
        if self.sd_map:
            input_dict.update({'sd_vectors':sd_map_data})

        return input_dict


class AV2PMapNetSemanticDataset(AV2Dataset):
    def __init__(self, dataroot, data_conf, is_train):
        # import pdb; pdb.set_trace()
        self.thickness = data_conf['thickness']
        self.sd_thickness = data_conf.get('sd_thickness', 5)
        self.angle_class = data_conf['angle_class']
        self.data_conf = data_conf
        self.is_train = is_train
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]
        canvas_h = int(patch_h / data_conf['ybound'][2])
        canvas_w = int(patch_w / data_conf['xbound'][2])           
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.sd_map = data_conf.get('sd_map_path', None)
        self.ann_path = data_conf.get('ann_path', './Work_dir/av2/dataset')
        if is_train:
            self.ann_file = osp.join(self.ann_path, 'av2_map_infos_train.pkl')
        else:
            self.ann_file = osp.join(self.ann_path, 'av2_map_infos_val.pkl')
        
        img_size = (data_conf['image_size'][0], data_conf['image_size'][1])
        # category configs
        cat2id = {
            'ped_crossing': 1,
            'divider': 0,
            'boundary': 2,
        }
        # bev configs
        roi_size = (data_conf['xbound'][1] * 2, data_conf['ybound'][1] * 2)
        # vectorize params
        coords_dim = 2
        sample_dist = -1
        sample_num = -1
        # meta info for submission pkl
        meta = dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            output_format='vector')

        # model configs
        num_points = -1
        sample_dist = 1
        permute = False
        # data processing pipelines
        self.vectorize_map = VectorizeMap(coords_dim=coords_dim,
                roi_size=roi_size,
                sample_num=num_points,
                sample_dist=sample_dist,
                normalize=False,
                permute=permute,)
        self.load_images = LoadMultiViewImagesFromFiles(to_float32=True)
        self.aug_images = PhotoMetricDistortionMultiViewImage()
        self.resize_images = ResizeMultiViewImages(size=img_size,
                change_intrinsics=False)
        self.pad_images = PadMultiViewImages(size_divisor=32)
        self.format = FormatBundleMap()

        dataset_cfg = dict(
            ann_file=self.ann_file,
            meta=meta,
            roi_size=roi_size,
            cat2id=cat2id,
            interval=1,
            sd_map=self.sd_map,
            data_root=dataroot,
        )
        super(AV2PMapNetSemanticDataset, self).__init__(**dataset_cfg)
    
    def pipeline(self, input_dict):
        # VectorizeMap
        data = self.vectorize_map(input_dict)
        # LoadMultiViewImagesFromFiles
        data = self.load_images(data)
        # PhotoMetricDistortionMultiViewImage
        # only train
        if self.is_train:
            data = self.aug_images(data)
        # ResizeMultiViewImages
        data = self.resize_images(data)
        # PadMultiViewImages
        data = self.pad_images(data)
        # format
        data = self.format(data)
        return data


    def __getitem__(self, idx):
        input_dict = self.get_sample(idx)
        data = self.pipeline(input_dict)

        imgs = data['img'].data # N,3,H,W
        imgs = normalize_tensor_img(imgs/255)
        ego2imgs = data['cam_extrinsics'] # list of 7 cameras, 4x4 array
        ego2global_tran = data['ego2global_translation'] # 3
        ego2global_rot = data['ego2global_rotation'] # 3x3
        timestamp = torch.tensor(data['timestamp'] / 1e9) # TODO: verify ns?
        scene_id = data['scene_name']
        rots = torch.stack([torch.Tensor(ego2img[:3, :3]) for ego2img in ego2imgs]) # 7x3x3
        trans = torch.stack([torch.Tensor(ego2img[:3, 3]) for ego2img in ego2imgs]) # 7x3
        post_rots = rots
        post_trans = trans
        intrins = torch.stack([torch.Tensor(intri) for intri in data['cam_intrinsics']])
        car_trans = torch.tensor(ego2global_tran) # [3]
        pos_rotation = Quaternion(matrix=ego2global_rot)
        yaw_pitch_roll = torch.tensor(pos_rotation.yaw_pitch_roll)
        semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks, osm_masks, osm_vectors, masked_map = self.get_semantic_map(data)
        # for uniform interface
        lidar_data = lidar_mask = torch.zeros((1,5))

        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks, osm_masks, osm_vectors, masked_map, timestamp, scene_id

    def get_semantic_map(self, data):
            raw_vector_data = data['vectors'].data
            vectors = []
            for id, pts in raw_vector_data.items():
                for pt in pts:
                    vectors.append(
                        dict(
                            pts=pt,
                            pts_num=pt.shape[0],
                            type=id
                        )
                    )
            osm_vectors = data['sd_vectors']  
            osm_masks, _ = preprocess_osm_map(osm_vectors, self.patch_size, self.canvas_size, self.sd_thickness)
            instance_masks, forward_masks, backward_masks, map_masked = preprocess_map(
                self.data_conf, vectors, self.patch_size, self.canvas_size, 
                NUM_CLASSES, self.thickness, self.angle_class)

            map_masked = map_masked != 0
            map_masked = torch.cat([(~torch.any(map_masked, axis=0)).unsqueeze(0), map_masked])
            semantic_masks = instance_masks != 0
            semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
            instance_masks = instance_masks.sum(0)
            forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
            backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
            direction_masks = forward_oh_masks + backward_oh_masks
            direction_masks = direction_masks / direction_masks.sum(0)
            return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks, osm_masks, osm_vectors, map_masked

