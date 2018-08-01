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
We extract fps 30 frames from original videos and build new version [here](https://drive.google.com/file/d/13IJqu2nTaFbYPaT3IhoCjH7dte-gbSSz/view?usp=sharing)



## Train script
- Tiramisu
    ```
    python exp_train.py
    ```
## Evaluate Script
- Tiramisu MC dropout
    Important hyper-parameter
    ```
    mode = 'test'
    ckpt_epoch = 900
    video_unct = False
    sample_num = 10
    ```
    Command
    ```
    python exp_test_MC.py
    ```
- Tiramisu TA-MC
    Important hyper-parameter
    ```
    mode = 'test'
    ckpt_epoch = 900
    video_unct = False
    error_thres = 300
    alpha_normal = 0.2
    alpha_error = 0.7
    ```
    Command
    ```
    python exp_test_TA.py
    ```
- Tiramisu RTA-MC
    Important hyper-parameter
    ```
    mode = 'test'
    ckpt_epoch = 900
    video_unct = False
    error_thres = 40
    alpha_normal = 0.2
    alpha_error = 0.5
    ```
    Command
    ```
    python exp_test_RTA.py
    ```


## Results

## Trained model


