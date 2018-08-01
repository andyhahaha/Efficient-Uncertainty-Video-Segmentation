# Efficient-Uncertainty-Video-Segmentation


This is the official codes for the paper: [Efficient Uncertainty Estimation for Semantic Segmentation in Videos](https://arxiv.org/abs/1807.11037).

## Requirements
- Python 2.7
- Pytorch 0.2.0
- tqdm
- matplotlib
- Visdom 0.1.7
- pypng
- protobuf
- [Opencv](https://anaconda.org/conda-forge/opencv)

## CamVid dataset
Normall CamVid dataset can download from [here](https://github.com/alexgkendall/SegNet-Tutorial).

However, our Method leverage consecutive frames to speed up uncertainty estimation. Therefore we need a CamVid dataset contain all consecutive frames(fps 30) instead of labeled frames(fps 1).
We extract fps 30 frames from original videos and build new version [here]()



## Train script

### Evaluate Script



## Results

### Trained model


