_base_ = [
    './_base_/default_runtime.py'
]

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'plugin/'

# img configs
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=False)

img_h = 128 #480
img_w = 352 #800
img_size = (img_h, img_w)
fv_downsample = 16
fv_size = (img_h//fv_downsample, img_w // fv_downsample)

num_gpus = 6
batch_size = 5
num_iters_per_epoch = 27846 // (num_gpus * batch_size)
num_epochs = 24
num_epochs_single_frame = num_epochs // 6
total_iters = num_epochs * num_iters_per_epoch
num_queries = 100

# category configs
cat2id = {
    'ped_crossing': 1,
    'divider': 0,
    'boundary': 2,
}
num_class = len(cat2id)

# bev configs
roi_size = (60, 30)
bev_h = 200  # bev grid size
bev_w = 400
pc_range = [-roi_size[0]/2, -roi_size[1]/2, -10, roi_size[0]/2, roi_size[1]/2, 10]
xyzbound = dict(
    xbound = [-roi_size[0]/2, roi_size[0]/2, roi_size[0] / bev_w],
    ybound = [-roi_size[1]/2, roi_size[1]/2, roi_size[1] / bev_h],
    zbound = [pc_range[2], pc_range[5], (pc_range[5] - pc_range[2]) / 1.0]
)
# vectorize params
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True

# meta info for submission pkl
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='raster')

# model configs
img_dims = 64
lidar_dims = 128
embed_dims = (img_dims+lidar_dims) if meta['use_lidar'] else img_dims
num_feat_levels = 3
norm_cfg = dict(type='BN2d')
num_points = -1
sample_dist = 1
permute = False

model = dict(
    type='HDMapNet',
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    backbone_cfg=dict(
        type='HDMapNetBackbone',
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        lidar = meta['use_lidar'],
        img_backbone=dict(
            type='EfficientNetBackbone',
            out_channel=img_dims,),
        lidar_backbone=dict(
            type='PointPillarEncoder',
            out_channel=lidar_dims,
            xyzbound=xyzbound),
        img_neck=None,
        transformer=dict(
            type='MLPViewTransformer',
            fv_size=fv_size,
            xyzbound=xyzbound,
            camera_number=6,
            in_channel=img_dims,
            extrinsic=True,),
    ),
    head_cfg=dict(
        type='MapSegmentHead',
         embed_dims=embed_dims,
         num_calsses=num_class+1,
         instance_seg=True, 
         embedded_dim=16, 
         direction_pred=True, 
         direction_dim=37,
         loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            class_weight=[2.13],
            ),
         loss_ins=dict(
            type='DiscriminativeLoss',
            embed_dim=16, 
            delta_v=0.5, 
            delta_d=3.0,
            scale_var=0.1,
            scale_dist=0.1,
            scale_reg=0.0,
         ),
         loss_dir=dict(
            type='DirectionLoss',
            reduction='none',
            loss_weight=0.1,
         ),
    )
)

# data processing pipelines
train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        sample_dist=sample_dist,
        normalize=False,
        permute=permute,
    ),
    dict(
        type='HDMapNetRasterizeMap',
        roi_size=roi_size, 
        canvas_size=(bev_w, bev_h), 
        thickness=5, 
        coords_dim=2,
        angle_class=36,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='NuscLoadPointsFromFile',
         nsweeps=3,
         min_distance=2.2,
         coord_type='LIDAR'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=False,
         ),
    # dict(type='Normalize3D', **img_norm_cfg),
    dict(type='Normalize3D', **img_norm_cfg, normalize=True),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', 
         keys=['img', 'vectors', 'points', 'lidar_mask',
               'semantic', 'instance', 'direction'], 
         meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name', 'cam_intrinsics',
        'cam_extrinsics'))
]

# data processing pipelines
test_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        sample_dist=sample_dist,
        normalize=False,
        permute=permute,
        ),
    dict(
        type='HDMapNetRasterizeMap',
        roi_size=roi_size, 
        canvas_size=(bev_w, bev_h), 
        thickness=5, 
        coords_dim=2,
        angle_class=36,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='NuscLoadPointsFromFile',
         nsweeps=3,
         min_distance=2.2,
         coord_type='LIDAR'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=False,
         ),
    dict(type='Normalize3D', **img_norm_cfg, normalize=True),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', 
         keys=['img', 'vectors', 'points', 'lidar_mask',
               'semantic'], 
         meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name', 'cam_intrinsics', 'cam_extrinsics'))
]

# configs for evaluation code
# DO NOT CHANGE
eval_config = dict(
    type='NuscDataset',
    data_root='./dataset/nuscenes',
    ann_file='./dataset/nuscenes_map_infos_val.pkl',
    meta=meta,
    nsweeps=0, # for test
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=test_pipeline,
    interval=100,
)

# dataset configs
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type='NuscDataset',
        data_root='./dataset/nuscenes',
        ann_file='./dataset/nuscenes_map_infos_train.pkl',
        nsweeps=0,
        # nsweeps=3,
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-1,
        interval=100,
    ),
    val=dict(
        type='NuscDataset',
        data_root='./dataset/nuscenes',
        ann_file='./dataset/nuscenes_map_infos_val.pkl',
        map_ann_file='./dataset/nuscenes_map_anno_gts.pkl',
        meta=meta,
        nsweeps=0, # for test
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=-1,
        samples_per_gpu=1,
        interval=100,
    ),
    test=dict(
        type='NuscDataset',
        data_root='./dataset/nuscenes',
        ann_file='./dataset/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=-1,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler',),
    nonshuffler_sampler=dict(type='DistributedSampler')
    # shuffler_sampler=None,
    # nonshuffler_sampler=None
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4 * (batch_size / 4),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy & schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=3e-3)

evaluation = dict(interval=num_epochs_single_frame*num_iters_per_epoch)
find_unused_parameters = True #### when use checkpoint, find_unused_parameters must be False
# checkpoint_config = dict(interval=1)
checkpoint_config = dict(interval=num_epochs_single_frame*num_iters_per_epoch)

runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

SyncBN = False