# DATA
dataset='nuScenes'
dataroot = './dataset/nuScenes'
version= 'v1.0-trainval' #'v1.0-mini'

xbound = [-60.0, 60.0, 0.3] #120m*60m, bev_size:400*200
ybound = [-30.0, 30.0, 0.3]

zbound = [-10.0, 10.0, 20.0]
dbound = [4.0, 45.0, 1.0]
image_size = [128, 352]
thickness = 5
# Path
vis_path = './vis_map'
sd_map_path='./data_osm/osm'

# CHECK_POINTS
modelf = 'ckpt/fusion_120_sd_model23.pt'

# Model
model = 'pmapnet_sd'

# Morphological_process mode in the vectorized post-process 
morpho_mode='MORPH_CLOSE' # 'MORPH_OPEN', 'None'

batch_size = 1
nworkers = 20
gpus = [0]


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




