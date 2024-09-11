# TEMVis
Quantitative Dynamic (in situ) TEM Analysis Using Optical Flow and U-Net
<img src="https://github.com/user-attachments/assets/0e21cffb-1abf-441b-b5a4-98895f2dfc45" alt="Image Description" width="600" >



## About
This repository contains the code used for video preprocessing with Optical flow and the code used for TEM video analysis with U-net. The programs are written in python. The detailed information is described in the following paper;  
>Tanaka, M., Sasaki, K., Punyafu, J., Muramatsu, M., Murayama, M. Machine-learning-aided Analysis of Relationship Between Crystal Defects and Macroscopic Mechanical Properties of TWIP steel. (to be submittted)

If you use this code to write an academic paper, I would appreciate it if you cite this paper.
## Requirement
- CUDA
- pytorch
- wandb(0.15.11)
- opencv
## Usage
### 1. Data preparation
#### 1.1  trimvideo.py
Trimiming of the length scale and time display in the video.
#### 1.2 optical_flow_cut.py
Conversion of the trimmed video into a video with static coordinates.
### 2. Semantic segmentation
#### 2.1 train_*class.py
Training of U-net. For the training of the 0.86 μm grains, train_2class.py is used. For the training of the 2 μm and 8.4 μm grains, train_3class.py is used.
#### 2.2 predict_*.py
Semantic segmentation of the frames not used for training. Trained models that reproduce the results of the paper are in the folder of "trained_model". For the prediction of the 0.86 μm grains, predict_086um.py is used. For the prediction of the 2 μm grains, predict_2um.py is used. For the prediction of the 8.4 μm grains, predict_8um.py is used. 
## Note
Code of semantic segmentation is based on https://github.com/milesial/Pytorch-UNet/tree/master.
