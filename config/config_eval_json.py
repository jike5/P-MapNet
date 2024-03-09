result_path = './120_sd.json'
dataroot = './dataset/nuScenes'
version= 'v1.0-trainval' #'v1.0-mini'

CD_threshold = 5
threshold_iou = 0.1
xbound = [-60.0, 60.0, 0.3] 
ybound = [-30.0, 30.0, 0.3]
batch_size = 4
eval_set = 'val' #'train', 'val', 'test', 'mini_train', 'mini_val'
thickness = 5
max_channel = 3
bidirectional = False
