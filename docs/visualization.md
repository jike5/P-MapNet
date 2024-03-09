# Visualization

We provide all the visualization scripts under `tools/vis_*.py`

## Visualize prediction

- Set `modelf = /path/to/experiment/ckpt` in config file.

```shell
python tools/vis_map.py /path/to/experiment/config
```
**Notes**: 

- All the visualization samples will be saved in `P_MAPNET/Work_dir/experiment/vis` automatically. If you want to customize the saving path, you can add `vis_path = /customized_path` in config file.

## Merge them into video

We also provide the script to merge the input, output and GT into video to benchmark the performance qualitatively.

```shell
# visualize nuscenes dataset
python tools/vis_video_nus.py /path/to/experiment/config path/to/experiment/vis
#visualize argoverse2 dataset
python tools/vis_video_av2.py /path/to/experiment/config path/to/experiment/vis
```
**Notes**: 
- The video will be saved in `P-MAPNET/Work_dir/experiment/demo.mp4`