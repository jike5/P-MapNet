import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString, Polygon
from shapely import affinity
import cv2
from PIL import Image, ImageDraw
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict
import torch

@PIPELINES.register_module(force=True)
class RasterizeMap(object):
    """Generate rasterized semantic map and put into 
    `semantic_mask` key.

    Args:
        roi_size (tuple or list): bev range
        canvas_size (tuple or list): bev feature size
        thickness (int): thickness of rasterized lines
        coords_dim (int): dimension of point coordinates
    """

    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 canvas_size: Union[Tuple, List], 
                 thickness: int, 
                 coords_dim: int):

        self.roi_size = roi_size
        self.canvas_size = canvas_size
        self.scale_x = self.canvas_size[0] / self.roi_size[0]
        self.scale_y = self.canvas_size[1] / self.roi_size[1]
        self.thickness = thickness
        self.coords_dim = coords_dim
    
    def line_ego_to_mask(self, 
                         line_ego: LineString, 
                         mask: NDArray, 
                         color: int=1, 
                         thickness: int=3) -> None:
        ''' Rasterize a single line to mask.
        
        Args:
            line_ego (LineString): line
            mask (array): semantic mask to paint on
            color (int): positive label, default: 1
            thickness (int): thickness of rasterized lines, default: 3
        '''

        trans_x = self.canvas_size[0] / 2
        trans_y = self.canvas_size[1] / 2
        line_ego = affinity.scale(line_ego, self.scale_x, self.scale_y, origin=(0, 0))
        line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
        
        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        assert len(coords) >= 2
        
        cv2.polylines(mask, np.int32([coords]), False, color=color, thickness=thickness)
    
    def polygons_ego_to_mask(self, 
                             polygons: List[Polygon], 
                             color: int=1) -> NDArray:
        ''' Rasterize a polygon to mask.
        
        Args:
            polygons (list): list of polygons
            color (int): positive label, default: 1
        
        Returns:
            mask (array): mask with rasterize polygons
        '''

        mask = Image.new("L", size=(self.canvas_size[0], self.canvas_size[1]), color=0) 
        # Image lib api expect size as (w, h)
        trans_x = self.canvas_size[0] / 2
        trans_y = self.canvas_size[1] / 2
        for polygon in polygons:
            polygon = affinity.scale(polygon, self.scale_x, self.scale_y, origin=(0, 0))
            polygon = affinity.affine_transform(polygon, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            ext = np.array(polygon.exterior.coords)[:, :2]
            vert_list = [(x, y) for x, y in ext]

            ImageDraw.Draw(mask).polygon(vert_list, outline=1, fill=color)

        return np.array(mask, np.uint8)
    
    def get_semantic_mask(self, map_geoms: Dict) -> NDArray:
        ''' Rasterize all map geometries to semantic mask.
        
        Args:
            map_geoms (dict): map geoms by class
        
        Returns:
            semantic_mask (array): semantic mask
        '''

        num_classes = len(map_geoms)
        semantic_mask = np.zeros((num_classes, self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)

        for label, geom_list in map_geoms.items():
            if len(geom_list) == 0:
                continue
            if geom_list[0].geom_type == 'LineString':
                for line in geom_list:
                    self.line_ego_to_mask(line, semantic_mask[label], color=1,
                        thickness=self.thickness)
            elif geom_list[0].geom_type == 'Polygon':
                # drivable area 
                polygons = []
                for polygon in geom_list:
                    polygons.append(polygon)
                semantic_mask[label] = self.polygons_ego_to_mask(polygons, color=1)
            else:
                raise ValueError('map geoms must be either LineString or Polygon!')
        
        return np.ascontiguousarray(semantic_mask)
    
    def __call__(self, input_dict: Dict) -> Dict:
        map_geoms = input_dict['map_geoms'] # {0: List[ped_crossing: LineString], 1: ...}

        semantic_mask = self.get_semantic_mask(map_geoms)
        input_dict['semantic_mask'] = semantic_mask # (num_class, canvas_size[1], canvas_size[0])
        return input_dict
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(roi_size={self.roi_size}, '
        repr_str += f'canvas_size={self.canvas_size}), '
        repr_str += f'thickness={self.thickness}), ' 
        repr_str += f'coords_dim={self.coords_dim})'

        return repr_str

def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1)
    return onehot

def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg

def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask

@PIPELINES.register_module(force=True)
class HDMapNetRasterizeMap(object):
    def __init__(self, 
                 angle_class,
                 roi_size: Union[Tuple, List], 
                 canvas_size: Union[Tuple, List], 
                 thickness: int, 
                 coords_dim: int):
        super().__init__()
        self.angle_class = angle_class
        self.roi_size = roi_size
        self.canvas_size = canvas_size
        self.scale_x = self.canvas_size[0] / self.roi_size[0]
        self.scale_y = self.canvas_size[1] / self.roi_size[1]
        self.thickness = thickness
        self.coords_dim = coords_dim

    def mask_for_lines(self,
                       lines, 
                       mask, 
                       thickness, 
                       idx, 
                       type='index', 
                       angle_class=36):
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
                cv2.polylines(mask, 
                              [coords[i:]], 
                              False, 
                              color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class), 
                              thickness=thickness)
        return mask, idx

    def line_geom_to_mask(self, 
                          layer_geom, 
                          confidence_levels, 
                          thickness, 
                          idx, 
                          type='index', 
                          angle_class=36):
        # map_mask = Image.new("L", size=(self.canvas_size[0], self.canvas_size[1]), color=0) # (h,w)
        # Image lib api expect size as (w, h)
        map_mask = np.zeros((self.canvas_size[1], self.canvas_size[0]), np.uint8)
        trans_x = self.canvas_size[0] / 2
        trans_y = self.canvas_size[1] / 2
        for line in layer_geom:
            if isinstance(line, tuple):
                line, confidence = line
            else:
                confidence = None
            if not line.is_empty:
                line = affinity.scale(line, self.scale_x, self.scale_y, origin=(0, 0))
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                confidence_levels.append(confidence)
                if line.geom_type == 'MultiLineString':
                    for new_single_line in line.geoms:
                        map_mask, idx = self.mask_for_lines(
                            new_single_line, map_mask, thickness, idx, type, angle_class)
                else:
                    map_mask, idx = self.mask_for_lines(
                        line, map_mask, thickness, idx, type, angle_class)
        return map_mask, idx

    def preprocess_map(self, 
                       map_geoms):
        num_classes = len(map_geoms)
        confidence_levels = [-1]
        idx = 1
        filter_masks = []
        instance_masks = []
        forward_masks = []
        backward_masks = []
        for i in range(num_classes):
            map_mask, idx = self.line_geom_to_mask(map_geoms[i], confidence_levels, self.thickness, idx)
            instance_masks.append(map_mask)
            filter_mask, _ = self.line_geom_to_mask(map_geoms[i], confidence_levels, self.thickness + 4, 1)
            filter_masks.append(filter_mask)
            forward_mask, _ = self.line_geom_to_mask(map_geoms[i], confidence_levels, self.thickness, 1, 
                                                     type='forward', angle_class=self.angle_class)
            forward_masks.append(forward_mask)
            backward_mask, _ = self.line_geom_to_mask(map_geoms[i], confidence_levels, self.thickness, 1, 
                                                      type='backward', angle_class=self.angle_class)
            backward_masks.append(backward_mask)

        filter_masks = np.stack(filter_masks)
        instance_masks = np.stack(instance_masks)
        forward_masks = np.stack(forward_masks)
        backward_masks = np.stack(backward_masks)

        instance_masks = overlap_filter(instance_masks, filter_masks)
        forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype('int32')
        backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype('int32')

        return torch.tensor(instance_masks), torch.tensor(forward_masks), torch.tensor(backward_masks)


    def __call__(self, input_dict: Dict) -> Dict:
        map_geoms = input_dict['map_geoms']
        instance_masks, forward_masks, backward_masks = self.preprocess_map(map_geoms)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
        instance_masks = instance_masks.sum(0)
        forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
        backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
        direction_masks = forward_oh_masks + backward_oh_masks
        direction_masks = direction_masks / direction_masks.sum(0)
        input_dict['semantic'] = semantic_masks
        input_dict['instance'] = instance_masks
        input_dict['direction'] = direction_masks
        return input_dict
    
