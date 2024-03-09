from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from shapely.geometry import LineString, box, Polygon, MultiLineString
from shapely import ops, affinity
import numpy as np
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour, remove_repeated_lines, transform_from, \
        connect_lines, remove_boundary_dividers
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union
import geopandas as gpd
import pandas as pd
import mmcv
import os
from av2.geometry.utm import convert_city_coords_to_utm, CityName
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

CityNames = [
    "ATX",  # Austin, Texas
    "DTW",  # Detroit, Michigan
    "MIA",  # Miami, Florida
    "PAO",  # Palo Alto, California
    "PIT",  # Pittsburgh, PA
    "WDC"]

def get_patch_coord(patch_box: Tuple[float, float, float, float],
                    patch_angle: float = 0.0) -> Polygon:
    """
    Convert patch_box to shapely Polygon coordinates.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :return: Box Polygon for patch_box.
    """
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch

class AV2MapExtractor(object):
    """Argoverse 2 map ground-truth extractor.

    Args:
        roi_size (tuple or list): bev range
        id2map (dict): log id to map json path
    """
    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 id2map: Dict,
                 sd_map: str = None) -> None:
        self.roi_size = roi_size
        self.id2map = {}
        self.id2city = {}
        for log_id, path in id2map.items():
            self.id2map[log_id] = ArgoverseStaticMap.from_json(Path(path))
            city = path.split('____')[-1].split('_')[0]
            self.id2city[log_id] = city
        if sd_map is not None:
            self.sd_maps = {} # city : data
            for name in CityNames:
                self.sd_maps[name] = mmcv.load(os.path.join(sd_map, f'sd_map_data_{name}.pkl'))
            
    def sample_fixed_pts_from_line(self, line, padding=False, fixed_num=100):
        '''padding=True 间隔1m均匀采样 根据fixed_num不足进行补0 超过直接舍弃
        padding=False 根据fixed_num变步长进行采样 以满足长度要求'''
        if padding:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            distances = np.linspace(0, line.length, fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        num_valid = len(sampled_points)

        if num_valid < fixed_num:
            padding = np.zeros((fixed_num - len(sampled_points), 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        elif num_valid > fixed_num:
            sampled_points = sampled_points[:fixed_num, :]
            num_valid = fixed_num
        if False:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
            num_valid = len(sampled_points)

        num_valid = len(sampled_points)
        return sampled_points, num_valid
    
    def get_osm_geom(self, log_id, e2g_translation, e2g_rotation):
        city = self.id2city[log_id]
        osm_map = self.sd_maps[city][city]
        patch_box = (e2g_translation[0], e2g_translation[1], self.roi_size[1], self.roi_size[0])
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        rotation = Quaternion(matrix=e2g_rotation)
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        patch = get_patch_coord(patch_box, patch_angle)
        line_list = []
        for geom_line in osm_map.geoms:
            if geom_line.is_empty:
                continue
            new_line = geom_line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)
        osm_vector_list = []
        for line in line_list:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        # if is_intersection(line):
                        osm_vector_list.append(self.sample_fixed_pts_from_line(single_line, padding=False, fixed_num=50))
                elif line.geom_type == 'LineString':
                    # if is_intersection(line):
                    osm_vector_list.append(self.sample_fixed_pts_from_line(line, padding=False, fixed_num=50))
                else:
                    raise NotImplementedError
        return osm_vector_list

    def get_map_geom(self,
                     log_id: str, 
                     e2g_translation: NDArray, 
                     e2g_rotation: NDArray, 
                     polygon_ped=True) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `log_id` and ego pose.
        
        Args:
            log_id (str): log id
            e2g_translation (array): ego2global translation, shape (3,)
            e2g_rotation (array): ego2global rotation matrix, shape (3, 3)
            polygon_ped: if True, organize each ped crossing as closed polylines. \
                Otherwise organize each ped crossing as two parallel polylines. \
                Default: True
        
        Returns:
            geometries (Dict): extracted geometries by category.
        '''

        avm = self.id2map[log_id]
        
        g2e_translation = e2g_rotation.T.dot(-e2g_translation)
        g2e_rotation = e2g_rotation.T

        roi_x, roi_y = self.roi_size[:2]
        local_patch = box(-roi_x / 2, -roi_y / 2, roi_x / 2, roi_y / 2)

        all_dividers = []
        # for every lane segment, extract its right/left boundaries as road dividers
        for _, ls in avm.vector_lane_segments.items():
            # right divider
            right_xyz = ls.right_lane_boundary.xyz
            right_mark_type = ls.right_mark_type
            right_ego_xyz = transform_from(right_xyz, g2e_translation, g2e_rotation)

            right_line = LineString(right_ego_xyz)
            right_line_local = right_line.intersection(local_patch)

            if not right_line_local.is_empty and not right_mark_type in ['NONE', 'UNKNOWN']:
                all_dividers += split_collections(right_line_local)
                
            # left divider
            left_xyz = ls.left_lane_boundary.xyz
            left_mark_type = ls.left_mark_type
            left_ego_xyz = transform_from(left_xyz, g2e_translation, g2e_rotation)

            left_line = LineString(left_ego_xyz)
            left_line_local = left_line.intersection(local_patch)

            if not left_line_local.is_empty and not left_mark_type in ['NONE', 'UNKNOWN']:
                all_dividers += split_collections(left_line_local)
        
        # remove repeated dividers since each divider in argoverse2 is mentioned twice
        # by both left lane and right lane
        all_dividers = remove_repeated_lines(all_dividers)
        
        ped_crossings = [] 
        for _, pc in avm.vector_pedestrian_crossings.items():
            edge1_xyz = pc.edge1.xyz
            edge2_xyz = pc.edge2.xyz
            ego1_xyz = transform_from(edge1_xyz, g2e_translation, g2e_rotation)
            ego2_xyz = transform_from(edge2_xyz, g2e_translation, g2e_rotation)

            # if True, organize each ped crossing as closed polylines. 
            if polygon_ped:
                vertices = np.concatenate([ego1_xyz, ego2_xyz[::-1, :]])
                p = Polygon(vertices)
                line = get_ped_crossing_contour(p, local_patch)
                if line is not None:
                    ped_crossings.append(line)

            # Otherwise organize each ped crossing as two parallel polylines.
            else:
                line1 = LineString(ego1_xyz)
                line2 = LineString(ego2_xyz)
                line1_local = line1.intersection(local_patch)
                line2_local = line2.intersection(local_patch)

                # take the whole ped cross if all two edges are in roi range
                if not line1_local.is_empty and not line2_local.is_empty:
                    ped_crossings.append(line1_local)
                    ped_crossings.append(line2_local)

        drivable_areas = []
        for _, da in avm.vector_drivable_areas.items():
            polygon_xyz = da.xyz
            ego_xyz = transform_from(polygon_xyz, g2e_translation, g2e_rotation)
            polygon = Polygon(ego_xyz)
            polygon_local = polygon.intersection(local_patch)

            drivable_areas.append(polygon_local)

        # union all drivable areas polygon
        drivable_areas = ops.unary_union(drivable_areas)
        drivable_areas = split_collections(drivable_areas)

        # boundaries are defined as the contour of drivable areas
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        # some dividers overlaps with boundaries in argoverse2 dataset
        # we need to remove these dividers
        all_dividers = remove_boundary_dividers(all_dividers, boundaries)

        # some dividers are split into multiple small parts
        # we connect these lines
        all_dividers = connect_lines(all_dividers)

        return dict(
            divider=all_dividers, # List[LineString]
            ped_crossing=ped_crossings, # List[LineString]
            boundary=boundaries, # List[LineString]
            drivable_area=drivable_areas, # List[Polygon],
        )
