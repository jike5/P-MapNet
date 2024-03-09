import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
import geopandas as gpd
import pandas as pd
from .const import CLASS2LABEL
import os 

class VectorizedLocalMap(object):
    def __init__(self,
                 dataroot,
                 patch_size,
                 canvas_size,
                 sd_map_path='./data_osm/osm',
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes # ['road_divider', 'lane_divider'] 车道边界线,车道分隔线
        self.ped_crossing_classes = ped_crossing_classes # ['ped_crossing'] 人行道
        self.polygon_classes = contour_classes # ['road_segment', 'lane']   道路段 车道线
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

        # 处理osm数据
        self.sd_maps = {}
        proj = 3857
        # 筛选道路主干道
        # ref: https://wiki.openstreetmap.org/wiki/Map_features#Highway
        options = ['trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', # road
                'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'# road link
                'living_street',  'road',  # Special road  'service'
                ]
        
        map_origin_df = pd.DataFrame(
            {'City': ['boston-seaport', 'singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown'],
            'Latitude': [42.336849169438615, 1.2882100868743724, 1.2993652317780957, 1.2782562240223188],
            'Longitude': [-71.05785369873047, 103.78475189208984, 103.78217697143555, 103.76741409301758]}) #
        map_origin_gdf = gpd.GeoDataFrame(
            map_origin_df, geometry=gpd.points_from_xy(map_origin_df.Longitude, map_origin_df.Latitude), crs=4326)
        map_origin_gdf = map_origin_gdf.to_crs(proj) 
        for loc in self.MAPS:
            sd_map = gpd.read_file(os.path.join(sd_map_path, '{}.shp'.format(loc)))
            sd_map = sd_map.to_crs(proj)
            sd_map = sd_map[sd_map['type'].isin(options)]
            sd_map = MultiLineString(list(sd_map.geometry))
            origin_geo = map_origin_gdf[map_origin_gdf['City']==loc].geometry
            origin = (float(origin_geo.x), float(origin_geo.y))
            matrix = [1.0, 0.0, 0.0, 1.0, -origin[0], -origin[1]]
            self.sd_maps[loc] = affinity.affine_transform(sd_map, matrix)
            if loc == 'boston-seaport':
                '''
                '''
                scale = 0.7143
                matrix = [scale, 0.0, 0.0, scale, 0.0, 0.0]
                self.sd_maps[loc] = affinity.affine_transform(self.sd_maps[loc], matrix)
                matrix = [1.0351, 0.0014, -0.0002, 1.0326, 0.0, 0.0]
                self.sd_maps[loc] = affinity.affine_transform(self.sd_maps[loc], matrix)
                
    def get_osm_geom(self, patch_box, patch_angle, location):
        osm_map = self.sd_maps[location]
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
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
        return line_list

    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)
        # 取该车附近的patch_box内的车道线(30*60大小的patch)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        # a. 获取线类型(road_divider, lane_divider)
        line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location) 
        line_vector_dict = self.line_geoms_to_vectors(line_geom)
        # b. 获取人行道
        ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
        # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)['ped_crossing']
        # c. 获取道路线(roda_segment, lane)
        polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
        poly_bound_list, union_roads = self.poly_geoms_to_vectors(polygon_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length, CLASS2LABEL.get(line_type, -1))) 
        # CLASS2LABEL = {'road_divider': 0,'lane_divider': 0,'ped_crossing': 1,'contours': 2,'others': -1}
        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, CLASS2LABEL.get('ped_crossing', -1)))

        for contour, length in poly_bound_list:
         
            vectors.append((contour.astype(float), length, CLASS2LABEL.get('contours', -1)))

        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    'pts': pts,
                    'pts_num': pts_num,
                    'type': type
                })

        osm_geom = self.get_osm_geom(patch_box, patch_angle, location)
        osm_vector_list = []

        def is_intersection(line):
            if line.intersects(union_roads):
                return True
            else:
                return False
            
        for line in osm_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        osm_vector_list.append(self.sample_fixed_pts_from_line(single_line, padding=False, fixed_num=50))
                elif line.geom_type == 'LineString':
                    osm_vector_list.append(self.sample_fixed_pts_from_line(line, padding=False, fixed_num=50))
                else:
                    raise NotImplementedError

        return filtered_vectors, polygon_geom, osm_vector_list 

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1] 
        lanes = polygon_geom[1][1] 
        union_roads = ops.unary_union(roads) 
        union_lanes = ops.unary_union(lanes) 
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior) 
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw: 
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch) 
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results), union_roads

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(record['polygon_token'])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid

    def sample_fixed_pts_from_line(self, line, padding=False, fixed_num=100):
        '''padding=True,间隔1m均匀采样,根据fixed_num不足进行补0,超过直接舍弃
        padding=False,根据fixed_num变步长进行采样,以满足长度要求'''
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
        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
            num_valid = len(sampled_points)

        num_valid = len(sampled_points)
        return sampled_points, num_valid
