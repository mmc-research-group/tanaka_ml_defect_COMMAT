# Data and code for analysis of TEM video using machine learning
This is the repository for the paper "Machine-learning-aided Analysis of Relationship Between Crystal Defects and Macroscopic Mechanical Properties of TWIP steel".
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
