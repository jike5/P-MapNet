import mmcv
import numpy as np
from functools import reduce
from pyquaternion import Quaternion
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type

@PIPELINES.register_module(force=True)
class LoadMultiViewImagesFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                 to_float32=False, 
                 color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filenames']
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        results['img'] = img
        results['img_shape'] = [i.shape for i in img]
        results['ori_shape'] = [i.shape for i in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [i.shape for i in img]
        # results['scale_factor'] = 1.0
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__} (to_float32={self.to_float32}, '\
            f"color_type='{self.color_type}')"

@PIPELINES.register_module()
class NuscLoadPointsFromFile(object):
    """Load Points From File.
    """
    def __init__(self,
                 nsweeps,
                 min_distance,
                 coord_type):
        self.nsweeps = nsweeps
        self.min_distance = min_distance
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import transform_matrix
        self.LidarPointCloud = LidarPointCloud
        self.transform_matrix = transform_matrix

    def _load_points(self, results):
        points = np.zeros((5, 0))
        # Get reference pose and timestamp.
        ref_time = 1e-6 * results['timestamp']
        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = self.transform_matrix(
            results['ego2global_translation'], 
            Quaternion(results['ego2global_quaternion']),
            inverse=True)
        current_sd_rec = results['adj_info'][0]
        for i in range(1, self.nsweeps+1):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = self.LidarPointCloud.from_file(current_sd_rec['lidar_path'])
            current_pc.remove_close(self.min_distance)
            # Get past pose.
            global_from_car = self.transform_matrix(
                current_sd_rec['e2g_translation'], 
                Quaternion(current_sd_rec['e2g_rotation']),
                inverse=True)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            car_from_current = self.transform_matrix(
                current_sd_rec['lidar2ego_translation'], 
                Quaternion(current_sd_rec['lidar2ego_rotation']),
                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)
            points = np.concatenate((points, new_points), 1)

            # Abort if there are no previous sweeps.
            if i < len(results['adj_info']):
                current_sd_rec = results['adj_info'][i]
            else:
                break

        return points
    
    def pad_or_trim_to_np(self, x, shape, pad_val=0):
        shape = np.asarray(shape)
        pad = shape - np.minimum(np.shape(x), shape)
        zeros = np.zeros_like(pad)
        x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
        return x[:shape[0], :shape[1]]

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_data = self._load_points(results)
        lidar_data = lidar_data.transpose(1, 0)
        num_points = lidar_data.shape[0]
        lidar_data = self.pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        points_class = get_points_type(self.coord_type)
        points = points_class(
            lidar_data, points_dim=lidar_data.shape[-1], attribute_dims=None)
        results['points'] = points
        results['lidar_mask'] = lidar_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'nsweeps={self.nsweeps}, '
        repr_str += f'min_distance={self.min_distance}, '
        return repr_str
