### Environment

1. Create conda environment
```
conda env create -f environment.yml
conda activate pmapnet
```
2. Install pytorch
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Install dependencies
```
pip install -r requirements.txt
```

### Datasets preparing
Download  [nuScenes dataset](https://www.nuscenes.org/) and put it to `dataset/` folder.