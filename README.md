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

Download and unzip the dataset
Edit config.json
```
{
  "camvid":
  {
    "data_path": "/YOUR/PATH/camvid/"
  }
}
```



## Train script
- Tiramisu
    ```
    python exp_train.py
    ```
## Evaluate Script
- Tiramisu MC dropout (sample 5 times)
    Important hyper-parameter
    ```
    mode = 'test'
    ckpt_epoch = 900
    video_unct = False
    sample_num = 5
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
    python exp_test_RTA.py
    ```


- Tiramisu RTA-MC
    Important hyper-parameter
    ```
    mode = 'test'
    ckpt_epoch = 900
    video_unct = False
    error_thres = 40
    alpha_normal = 0.2
    alpha_error = 0.7
    ```
    Command
    ```
    python exp_test_RTA.py
    ```


## Results

- Tiramisu MC dropout N=5(we use N=5 result because the same inference time as RTA-MC.)
    - Performance
        |  | Accuracy |
        |-------|:-----:|
        | Global Accuracy   |  89.3  |
        | Mean Accuracy     |   75.3 |
        | Mean IOU          |  62.6    |

    - PR-Curve
        ![Alt text](/images/MC_PR.PNG)
    - Ranking IOU
- Tiramisu TA-MC
    - Performance
        |  | Accuracy(%) |
        |-------|:-----:|
        | Global Accuracy   |  89.6  |
        | Mean Accuracy     | 73.5   |
        | Mean IOU          |  62.2    |
    - PR-Curve
        ![Alt text](/images/TA-MC_PR.PNG)
    - Ranking IOU

- Tiramisu RTA-MC
    - Performance
        |  | Accuracy(%) |
        |-------|:-----:|
        | Global Accuracy   |  89.6  |
        | Mean Accuracy     |  74.2 |
        | Mean IOU          |   62.6   |
    - PR-Curve
        ![Alt text](/images/RTA-MC_PR.png)
    - Ranking IOU

## Trained model
Our trained tiramisu model can be download [here](https://drive.google.com/file/d/1bUpaZoTugeVs4zK31MLVe3jrL5ILdQ4n/view?usp=sharing)
Download and unzip it at checkpoint dir. Then edit one variable in exp_test_MC.py and python exp_test_RTA.py.
```
description = 'lr0.001_b4_trained'
```
Then it can evaluate our release model.


