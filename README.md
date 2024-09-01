<div align="center">
<h1>P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors </h1>
  
[[Paper](https://arxiv.org/pdf/2403.10521.pdf)]   [[Project Page](https://jike5.github.io/P-MapNet/)]

</div >

**Abstract:**
Autonomous vehicles are gradually entering city roads today, with the help of high-definition maps (HDMaps). However, the reliance on HDMaps prevents autonomous vehicles from stepping into regions without this expensive digital infrastructure. This fact drives many researchers to study online HDMap construction algorithms, but the performance of these algorithms at far regions is still unsatisfying. We present P-MapNet, in which the letter P highlights the fact that we focus on incorporating map priors to improve model performance. Specifically, we exploit priors in both SDMap and HDMap. On one hand, we extract weakly aligned SDMap from OpenStreetMap, and encode it as an additional conditioning branch. Despite the misalignment challenge, our attention-based architecture adaptively attends to relevant SDMap skeletons and significantly improves performance. On the other hand, we exploit a masked autoencoder to capture the prior distribution of HDMap, which can serve as a refinement module to mitigate occlusions and artifacts. We benchmark on the nuScenes and Argoverse2 datasets.
Through comprehensive experiments, we show that: (1) our SDMap prior can improve online map construction performance, using both rasterized (by up to +18.73 mIoU) and vectorized (by up to +8.50 mAP) output representations. (2) our HDMap prior can improve map perceptual metrics by up to 6.34%. (3)
P-MapNet can be switched into different inference modes that covers different regions of the accuracy-efficiency trade-off landscape. (4) P-MapNet is a far-seeing solution that brings larger improvements on longer ranges. 

## Model

### Results on nuScenes-val set
We provide results on nuScenes-val set.

|    Range    |  Method   |  M  |   Div.   |   Ped.   |  Bound.  |   mIoU    |   Model    |   Config    |
|:-----------:|:--------:|:---:|:---:|:---:|:-----:|:--------:|:--------:|:--------:|
|  60 × 30 | HDMapNet | L+C | 45.9 | 30.5 | 56.8 | 44.40 | [ckpt](https://drive.google.com/file/d/1yYCRk_as7Vhvi_rL5BxqVrmEf_u7mB3b/view?usp=drive_link) | [cfg](config/nusc/baseline/baseline_60m.py) | 
|  60 × 30 | P-MapNet(SD Prio.) | L+C | **55.4** | **40.2** | **63.9** | **53.17** | [ckpt](https://drive.google.com/file/d/1iwCxHVafQaEwgTWVTgDVcJvjeW8c9uvz/view?usp=sharing) | [cfg](https://github.com/jike5/P-MapNet/blob/60m/output/config.txt) | 

> The model weights under **other settings** can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1P6LuhsHy3yy4sGwlDCGT9tjVzYpcaqEb?usp=drive_link) or [百度云](https://pan.baidu.com/s/1OVI3aWgOGGg6_iGCs_gxDg?pwd=65aa).

## Getting Started
### Training

Run `python train_corssattn.py`, for example:

```
# SDMap Prior model
python train_corssattn.py

# with multi gpus
python train_corssattn.py --gpus 0 1 2 3 --bsz 16
```


### Citation
If you found this paper or codebase useful, please cite our paper:
```
@misc{jiang2024pmapnet,
      title={P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors}, 
      author={Zhou Jiang and Zhenxin Zhu and Pengfei Li and Huan-ang Gao and Tianyuan Yuan and Yongliang Shi and Hang Zhao and Hao Zhao},
      year={2024},
      eprint={2403.10521},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
