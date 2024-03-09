# P-MapNet

**P-MapNet: Far-seeing Map Constructor Enhanced by both SDMap and HDMap Priors**



**[[Paper](https://arxiv.org/abs/2107.06307)] [[Project Page](https://tsinghua-mars-lab.github.io/HDMapNet/)] [[5-min video](https://www.youtube.com/watch?v=AJ-rToTN8y8)]**

**Abstract:**
Autonomous vehicles are gradually entering city roads today, with the help of high-definition maps (HDMaps). However, the reliance on HDMaps prevents autonomous vehicles from stepping into regions without this expensive digital infrastructure. This fact drives many researchers to study online HDMap construction algorithms, but the performance of these algorithms at far regions is still unsatisfying. We present P-MapNet, in which the letter P highlights the fact that we focus on incorporating map priors to improve model performance. Specifically, we exploit priors in both SDMap and HDMap. On one hand, we extract weakly aligned SDMap from OpenStreetMap, and encode it as an additional conditioning branch. Despite the misalignment challenge, our attention-based architecture adaptively attends to relevant SDMap skeletons and significantly improves performance. On the other hand, we exploit a masked autoencoder to capture the prior distribution of HDMap, which can serve as a refinement module to mitigate occlusions and artifacts. We benchmark on the nuScenes and Argoverse2 datasets.
Through comprehensive experiments, we show that: (1) our SDMap prior can improve online map construction performance, using both rasterized (by up to +18.73 mIoU) and vectorized (by up to +8.50 mAP) output representations. (2) our HDMap prior can improve map perceptual metrics by up to 6.34%. (3)
P-MapNet can be switched into different inference modes that covers different regions of the accuracy-efficiency trade-off landscape. (4) P-MapNet is a far-seeing solution that brings larger improvements on longer ranges. 

![visualization](figs/teaser.jpg)

Please check [installation](docs/installation.md) for environment installation and nuScenes dataset preparation.


### Citation
If you found this paper or codebase useful, please cite our paper:
```
@misc{li2021hdmapnet,
      title={HDMapNet: An Online HD Map Construction and Evaluation Framework}, 
      author={Qi Li and Yue Wang and Yilun Wang and Hang Zhao},
      year={2021},
      eprint={2107.06307},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
