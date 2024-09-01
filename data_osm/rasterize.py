import cv2
import numpy as np
import pdb
import torch
import os.path as osp
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, GeometryCollection
import random
import matplotlib.pyplot as plt
import os 

def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max) # x:-30~30  y:-15~15
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    # try:
    #     coords = np.asarray(list(lines.geoms[0].coords), np.int32)
    # except:
    #     return mask, idx
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0) # n*2的矩阵，上下颠倒，第一个point变为最后一个

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else: # forward
        for i in range(len(coords) - 1): # 将线段点之间的角度通过agnle_class=36进行近似，用不同的颜色进行区分，取值应该为0~36
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx


def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box) # 应该是一个划分好的grid，二维的，local_box大小为取车道线时的patch大小

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch) # 取相交部分
        if not new_line.is_empty:
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])     # 平移变换
            new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))  # 尺度缩放到canvas
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line.geoms:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx # 不同线的实例用不同颜色和idx进行了区分


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def preprocess_map(vectors, patch_size, canvas_size, num_classes, thickness, angle_class):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes): # num_classes == 3 0表示道路/车道分隔线，1表示人行道，2表示道路轮廓线
        vector_num_list[i] = []

    for vector in vectors:
        # if vector['type'] != # 0 1 2 代表车道线、人行道、边界线
        if vector['pts_num'] >= 2: # 如果线段超过两个点及以上
            vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']])) # 用LineString对点进行了转换

    local_box = (0.0, 0.0, patch_size[0], patch_size[1]) # patch_x, patch_y, patch_h, patch_w

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx) # 返回出来的idx为了保证实例在本帧的地图所有线中id唯一
        instance_masks.append(map_mask) # 划分不同的实例线
        filter_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask) # 用来过滤掉画到原本车道线附近外的点吧
        forward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask) # 对线段的角度进行归类(0~36近似)
        backward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask) # 对线段的起始点和终止点进行调换

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype('int32')

    return torch.tensor(instance_masks), torch.tensor(forward_masks), torch.tensor(backward_masks)

def preprocess_sd_map(polygon_geom, patch_size, canvas_size):
    roads = polygon_geom[0][1] # roda_segment MultiPolygon
    lanes = polygon_geom[1][1] 
    union_roads = ops.unary_union(roads) # Polygon ops.unary_union: 合并重叠的部分
    union_lanes = ops.unary_union(lanes) # Polygon
    union_segments = ops.unary_union([union_roads, union_lanes])
    max_x = patch_size[1] / 2
    max_y = patch_size[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    
    local_box = (0.0, 0.0, patch_size[0], patch_size[1])
    patch_x, patch_y, patch_h, patch_w = local_box
    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w
    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0
    region = union_segments.intersection(local_patch)
    region = affinity.affine_transform(region, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])     # 平移变换
    region = affinity.scale(region, xfact=scale_width, yfact=scale_height, origin=(0, 0))
    coords_list = []
    if isinstance(region, MultiPolygon) and region.geoms is not None:
        for single_region in region.geoms:
            coords_list.extend(single_region.exterior.coords)
    elif isinstance(region, GeometryCollection) and region.geoms is not None:
        for single_geom in region.geoms:
            if isinstance(single_geom, MultiPolygon):
                for single_region in single_geom.geoms:
                    coords_list.extend(single_region.exterior.coords)
            else:
                coords_list = list(region.exterior.coords)
    else:
        coords_list = list(region.exterior.coords)
    coords = np.asarray(coords_list, np.int32)
    coords = coords.reshape((-1, 2))
    map_mask = np.zeros(canvas_size, np.uint8)
    if len(coords) < 2:
        return torch.tensor(map_mask)
    else:
        cv2.fillPoly(map_mask, [coords], True)
    
    return torch.tensor(map_mask)

def preprocess_osm_map(time, scene_id, vectors, patch_size, canvas_size, thickness):
    confidence_levels = [-1]
    
    vector_num_list = []
    for pts, pts_num in vectors:
        if pts_num >= 2: # 如果线段超过两个点及以上
            vector_num_list.append(LineString(pts))
    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    # local_box_osm = (0.0, 0.0, 200, 400)
    # canvas_size_osm = (200,400)

    osm_mask, _ = line_geom_to_mask(vector_num_list, confidence_levels, local_box, canvas_size, thickness, 1)
    filter_mask, _ = line_geom_to_mask(vector_num_list, confidence_levels, local_box, canvas_size, thickness + 4, 1)
    osm_mask = osm_mask[np.newaxis, :, :]
    filter_mask = filter_mask[np.newaxis, :, :]
    # osm_mask = overlap_filter(osm_mask, filter_mask)
    # pdb.set_trace()
    osm_mask = osm_mask.astype(np.int32)

    # 让sdmap变成 -1 到 1
    osm_mask[np.where(osm_mask == 0)] = -1
    osm_mask[np.where(osm_mask != -1)] = 1
    '''
    # osm_mask[np.where(osm_mask != 0)] = 1
    plt.figure(figsize=(4, 4))
    plt.imshow(osm_mask[0])
    plt.axis('off')

    imgpath = '/DATA_EDS/jiangz/AWORK/sdmap_complete/HDMapNet/osm_onservice_imgs'
    # if not osp.exists(imgpath):
    #     os.makedirs(imgpath)
    imgname = osp.join(imgpath, f'{time}_{scene_id}.jpg')
    plt.savefig(imgname)
    plt.close()
    print("saving ",imgname)
    '''
    return torch.tensor(osm_mask), vector_num_list

def random_mask(map_bd, patch_num_h, patch_num_w , mask_ratio):
    H,W = map_bd.shape
    mask = torch.ones(H, W)

    W_patch_size = W//patch_num_w  # 每个patch的宽
    H_patch_size = H//patch_num_h  # 每个patch的高
    patch_num = patch_num_w*patch_num_h  #所有patch数量
    mask_patch_num = int(patch_num * mask_ratio) #需要mask的patch数量
    # 使用torch.rand()函数随机打乱序列
    block_indices = torch.randperm(patch_num)[:mask_patch_num] #随机需要mask的patch idx

    row_indices = ((block_indices // patch_num_w) * H_patch_size)
    col_indices = ((block_indices % patch_num_w ) * W_patch_size)

    start_row = row_indices.int().tolist()
    end_row = (row_indices + H_patch_size).int().tolist()
    start_col = col_indices.int().tolist()
    end_col = (col_indices + W_patch_size).int().tolist()

    for i in range(mask_patch_num):
        mask[start_row[i]:end_row[i], start_col[i]:end_col[i]] = 0

    return map_bd*mask.detach().numpy()


def line_geom_to_mask_bd(
    data_conf,
    time,    
    layer_geom,
    confidence_levels,
    local_box,
    canvas_size,
    thickness,
    idx,
    type='index',
    angle_class=36,
):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)  # 应该是一个划分好的grid，二维的，local_box大小为取车道线时的patch大小

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)
    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)  # 取相交部分？
        if not new_line.is_empty:
            
            new_line = affinity.affine_transform(
                new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])  # 平移变换
            new_line = affinity.scale(new_line,
                                      xfact=scale_width,
                                      yfact=scale_height,
                                      origin=(0, 0))  # 尺度缩放到canvas
            confidence_levels.append(confidence)


            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line:

                    map_mask, idx = mask_for_lines(
                        new_single_line,
                        map_mask,
                        thickness,
                        idx,
                        type,
                        angle_class,
                    )
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask,
                                                  thickness, idx, type,
                                                  angle_class)

        else:
            print("line is empty")
    # base_path = '/DATA_EDS/jiangz/AWORK/sdmap_complete/HDMapNet/masked_bd_img_200_50'
    # img_name = f'{time}_ori.png'
    # img_name = os.path.join(base_path, img_name)
    # save_fig(map_mask, img_name)

    mask_flag = data_conf['mask_flag']
    patch_num_h = data_conf['patch_h']
    patch_num_w = data_conf['patch_w']
    mask_ratio = data_conf['mask_ratio']

    num_candi = [0,1,2,4,5,8,10]
    
    patch_num_h = random.choice(num_candi)
    patch_num_w = random.choice(num_candi)
    mask_ratio = random.random()

    if mask_flag:
        map_mask = random_mask(map_mask, patch_num_h, patch_num_w , mask_ratio)
  
    # img_name = f'{time}_masked.png'
    # img_name = os.path.join(base_path, img_name)
    # save_fig(map_mask_bd, img_name)

    return map_mask, idx

def preprocess_map_withbd(data_conf, time, vectors, patch_size, canvas_size, num_classes, thickness, angle_class):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']]))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    instance_mask_bd = []
    for i in range(num_classes):
        if i == 2:
            map_mask_bd, idx = line_geom_to_mask_bd(data_conf, time, vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
            instance_mask_bd.append(map_mask_bd)

        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)
    instance_mask_bd = np.stack(instance_mask_bd)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    instance_mask_bd = overlap_filter(instance_mask_bd, filter_masks)
    forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype('int32')

    return torch.tensor(instance_masks), torch.tensor(forward_masks), torch.tensor(backward_masks), torch.tensor(instance_mask_bd)

def preprocess_map_onlybd(data_conf, time, vectors, patch_size, canvas_size, num_classes, thickness, angle_class):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']]))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    instance_mask_bd = []
    for i in range(num_classes):
        # if i == 2:
        map_mask_bd, idx = line_geom_to_mask_bd(data_conf, time, vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_mask_bd.append(map_mask_bd)

        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)
    instance_mask_bd = np.stack(instance_mask_bd)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    instance_mask_bd = overlap_filter(instance_mask_bd, filter_masks)
    forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype('int32')

    return torch.tensor(instance_masks), torch.tensor(forward_masks), torch.tensor(backward_masks), torch.tensor(instance_mask_bd)


def rasterize_map(vectors, patch_size, canvas_size, num_classes, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append((LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels
