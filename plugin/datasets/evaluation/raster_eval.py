import torch
from mmdet3d.datasets import build_dataset, build_dataloader
import mmcv
from functools import cached_property
import prettytable
from numpy.typing import NDArray
from typing import Dict, Optional
from logging import Logger
from mmcv import Config
from copy import deepcopy
import os

N_WORKERS = 16

class RasterEvaluate(object):
    """Evaluator for rasterized map.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
    """

    def __init__(self, 
                 dataset_cfg: Config, 
                 n_workers: int=N_WORKERS,
                 map_ann_file: str=None):
        assert map_ann_file is not None
        self.map_ann_file = map_ann_file
        self.dataset_cfg = dataset_cfg
        self.n_workers = n_workers
        self.cat2id = self.dataset_cfg.cat2id
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        self.n_workers = n_workers
        self._format_gt()

    def _format_gt(self):
        assert self.map_ann_file is not None
        print('collecting gts...')
        if (not os.path.exists(self.map_ann_file)):
            self.dataset = build_dataset(self.dataset_cfg)
            self.dataloader = build_dataloader(
                self.dataset, samples_per_gpu=1, 
                workers_per_gpu=self.n_workers, shuffle=False, dist=False)
            gts = {}
            for data in mmcv.track_iter_progress(self.dataloader):
                token = deepcopy(data['img_metas'].data[0][0]['token'])
                gt = deepcopy(data['semantic'].data[0][0])
                gts[token] = gt[1:] # TODO: check this
                del data # avoid dataloader memory crash
            self.gts = gts
            mmcv.dump(gts, self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')
            self.gts = mmcv.load(self.map_ann_file)

    # @cached_property
    # def gts(self) -> Dict[str, NDArray]:
    #     print('collecting gts...')
    #     gts = {}
    #     for data in mmcv.track_iter_progress(self.dataloader):
    #         token = deepcopy(data['img_metas'].data[0][0]['token'])
    #         gt = deepcopy(data['semantic'].data[0][0])
    #         gts[token] = gt
    #         del data # avoid dataloader memory crash
        
    #     return gts

    def evaluate(self, 
                 result_path: str, 
                 logger: Optional[Logger]=None) -> Dict[str, float]:
        ''' Do evaluation for a submission file and print evalution results to `logger` if specified.
        The submission will be aligned by tokens before evaluation. 
        
        Args:
            result_path (str): path to submission file
            logger (Logger): logger to print evaluation result, Default: None
        
        Returns:
            result_dict (Dict): evaluation results. IoU by categories.
        '''
        
        results = mmcv.load(result_path)
        meta = results['meta']
        results = results['results']

        result_dict = {}

        gts = []
        preds = []
        for token, gt in self.gts.items():
            gts.append(gt)
            if token in results:
                pred = results[token]['semantic_mask']
            else:
                pred = torch.zeros((len(self.cat2id), 
                    self.canvas_size[1], self.canvas_size[0])).bool()
            
            preds.append(pred)
        
        preds = torch.stack(preds).bool()
        gts = torch.stack(gts).bool()

        # for every label
        total = 0
        for i in range(gts.shape[1]):
            category = self.id2cat[i]
            pred = preds[:, i]
            gt = gts[:, i]
            intersect = (pred & gt).sum().float().item()
            union = (pred | gt).sum().float().item()
            result_dict[category] = intersect / (union + 1e-7)
            total += result_dict[category]
        
        mIoU = total / gts.shape[1]
        result_dict['mIoU'] = mIoU
        
        categories = list(self.cat2id.keys())
        table = prettytable.PrettyTable([' ', *categories, 'mean'])
        table.add_row(['IoU', 
            *[round(result_dict[cat], 4) for cat in categories], 
            round(mIoU, 4)])
        
        if logger:
            from mmcv.utils import print_log
            print_log('\n'+str(table), logger=logger)
            print_log(f'mIoU = {mIoU:.4f}\n', logger=logger)

        return result_dict
