import cv2
import numpy as np
import torch
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, GeometryCollection
import random

def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max) 
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch

def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg

def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)

    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0) 

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else: # forward
        for i in range(len(coords) - 1): 
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx

def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box
    patch = get_patch_coord(local_box) 
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
        new_line = line.intersection(patch) 
        if not new_line.is_empty:
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])     
            new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))  
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line.geoms:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx 

def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask

def preprocess_osm_map(vectors, patch_size, canvas_size, thickness=5):
    confidence_levels = [-1]
    
    vector_num_list = []
    for pts, pts_num in vectors:
        if pts_num >= 2: 
            vector_num_list.append(LineString(pts))
    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    osm_mask, _ = line_geom_to_mask(vector_num_list, confidence_levels, local_box, canvas_size, thickness, 1)
    filter_mask, _ = line_geom_to_mask(vector_num_list, confidence_levels, local_box, canvas_size, thickness + 4, 1)
    osm_mask = osm_mask[np.newaxis, :, :]
    filter_mask = filter_mask[np.newaxis, :, :]
    osm_mask = osm_mask.astype(np.int32)

    osm_mask[np.where(osm_mask == 0)] = -1
    osm_mask[np.where(osm_mask != -1)] = 1

    return torch.tensor(osm_mask), vector_num_list

def random_mask(map, patch_num_h, patch_num_w , mask_ratio_bd , mask_ratio):
    C,H,W = map.shape
    mask = np.ones((H, W))
    mask_bd = np.ones((H, W))
    map_masked = []
    W_patch_size = W//patch_num_w  
    H_patch_size = H//patch_num_h  
    patch_num = patch_num_w*patch_num_h  
    mask_patch_num_bd = round(patch_num * mask_ratio_bd)
    mask_patch_num = round(patch_num * mask_ratio)

    block_indices_bd = torch.randperm(patch_num)[:mask_patch_num_bd] 
    block_indices = torch.randperm(patch_num)[:mask_patch_num]

    row_indices_bd = ((block_indices_bd // patch_num_w) * H_patch_size)
    col_indices_bd = ((block_indices_bd % patch_num_w ) * W_patch_size)

    row_indices = ((block_indices // patch_num_w) * H_patch_size)
    col_indices = ((block_indices % patch_num_w ) * W_patch_size)

    start_row = row_indices.int().tolist()
    end_row = (row_indices + H_patch_size).int().tolist()
    start_col = col_indices.int().tolist()
    end_col = (col_indices + W_patch_size).int().tolist()
    
    start_row_bd = row_indices_bd.int().tolist()
    end_row_bd = (row_indices_bd + H_patch_size).int().tolist()
    start_col_bd = col_indices_bd.int().tolist()
    end_col_bd = (col_indices_bd + W_patch_size).int().tolist()

    for i in range(mask_patch_num):
        mask[start_row[i]:end_row[i], start_col[i]:end_col[i]] = 0

    for i in range(mask_patch_num_bd):
        mask_bd[start_row_bd[i]:end_row_bd[i], start_col_bd[i]:end_col_bd[i]] = 0

    map[:2] = map[:2] * mask
    map[2] = map[2] * mask_bd
    return map

def grid_mask(map_mask, patch_num_h, patch_num_w, mask_ratio):
    C, H, W = map_mask.shape  
    mask = np.ones((H, W)) 
    # 计算每个patch的大小
    W_patch_size = W // patch_num_w
    H_patch_size = H // patch_num_h
    patch_num = patch_num_w * patch_num_h  # 计算总的patch数量
    # 计算需要mask的patch数量
    mask_patch_num = round(patch_num * mask_ratio)
    # 生成随机的mask patch索引
    block_indices = np.random.choice(patch_num, mask_patch_num, replace=False)
    # 计算mask patch的起始和结束坐标
    for index in block_indices:
        row_start = (index // patch_num_w) * H_patch_size
        col_start = (index % patch_num_w) * W_patch_size
        mask[row_start:row_start + H_patch_size, col_start:col_start + W_patch_size] = 0

    map_masked = map_mask * mask
    return map_masked

def preprocess_map(data_conf, vectors, patch_size, canvas_size, num_classes, thickness, angle_class):
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

    for i in range(num_classes):
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

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype('int32')

    mask_flag = data_conf['mask_flag']
    patch_num_h = data_conf['patch_h']
    patch_num_w = data_conf['patch_w']
    mask_ratio = data_conf['mask_ratio']
    
    # grid based mask
    # if mask_flag:
    #     instance_masks_mae = instance_masks.copy()
    #     instance_masks_mae[np.where(instance_masks_mae != 0)] = 1
    #     map_mask = grid_mask(instance_masks_mae, patch_num_h, patch_num_w, mask_ratio)
    
    # random patch size and random mask ratio
    if mask_flag:
        num_candi = [25, 20, 10, 8]
        patch_num_h = random.choice(num_candi)
        patch_num_w = patch_num_h
        if mask_ratio < 0:
            mask_ratio_bd = random.uniform(0, 0.5)
            mask_ratio  = random.uniform(0, 0.7)
        else:
            mask_ratio_bd = mask_ratio
        instance_masks_mae = instance_masks.copy()
        instance_masks_mae[np.where(instance_masks_mae != 0)] = 1
        map_mask = random_mask(instance_masks_mae, patch_num_h, patch_num_w , mask_ratio_bd, mask_ratio)

    return torch.tensor(instance_masks), torch.tensor(forward_masks), torch.tensor(backward_masks), torch.tensor(map_mask)

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
