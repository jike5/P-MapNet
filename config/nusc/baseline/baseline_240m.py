# DATA
dataset='nuScenes'
dataroot = './dataset/nuScenes'
version= 'v1.0-trainval'

xbound = [-120.0, 120.0, 0.3] #240m*60m, bev_size:800*200
ybound = [-30.0, 30.0, 0.3]

zbound = [-10.0, 10.0, 20.0]
dbound = [4.0, 45.0, 1.0]
image_size = [128, 352]
thickness = 5
# EXP
logdir = './Work_dir/baseline_240'
sd_map_path='./data_osm/osm'
# TRAIN
model = 'HDMapNet_fusion'
nepochs = 30
batch_size = 8
nworkers = 20
gpus = [0, 1, 2, 3]

# OPT
lr = 5e-4
weight_decay = 1e-7
max_grad_norm = 5.0
pos_weight = 2.13
steplr = 10

# CHECK_POINTS
modelf = None

# LOSS
scale_seg = 1.0
scale_var = 0.1
scale_dist = 0.1
scale_direction = 0.1

direction_pred = True
instance_seg = True
embedding_dim = 16
delta_v = 0.5
delta_d = 3.0
angle_class = 36

# Mask config
mask_flag = False
mask_ratio = -1 # random ratio
patch_h = 20
patch_w = 20




