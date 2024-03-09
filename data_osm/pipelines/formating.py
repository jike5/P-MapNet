import numpy as np
from mmcv.parallel import DataContainer as DC
import torch
import mmcv
from collections.abc import Sequence
# from mmdet3d.core.points import BasePoints
# from mmdet.datasets.pipelines import to_tensor

# copy from mmdet:https://mmdetection.readthedocs.io/en/v2.0.0/_modules/mmdet/datasets/pipelines/formating.html
def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

class FormatBundleMap(object):
    """Format data for map tasks and then collect data for model input.

    These fields are formatted as follows.

    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - semantic_mask (if exists): (1) to tensor, (2) to DataContainer (stack=True)
    - vectors (if exists): (1) to DataContainer (cpu_only=True)
    - img_metas: (1) to DataContainer (cpu_only=True)
    """

    def __init__(self, process_img=True, 
                keys=['img', 'semantic_mask', 'vectors'], 
                meta_keys=['intrinsics', 'extrinsics']):
        
        self.process_img = process_img
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if 'img' in results and self.process_img:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)
        
        if 'semantic_mask' in results:
            results['semantic_mask'] = DC(to_tensor(results['semantic_mask']), stack=True)

        if 'vectors' in results:
            # vectors may have different sizes
            vectors = results['vectors']
            results['vectors'] = DC(vectors, stack=False, cpu_only=True)
        
        if 'polys' in results:
            results['polys'] = DC(results['polys'], stack=False, cpu_only=True)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(process_img={self.process_img}, '
        return repr_str
