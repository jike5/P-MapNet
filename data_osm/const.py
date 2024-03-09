MAP = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
CLASS2LABEL = {
    'road_divider': 0, # 道路分隔线
    'lane_divider': 0, # 车道分隔线
    'ped_crossing': 1, # 人行道
    'contours': 2,     # 轮廓线
    'others': -1
}

NUM_CLASSES = 3
IMG_ORIGIN_H = 900
IMG_ORIGIN_W = 1600
