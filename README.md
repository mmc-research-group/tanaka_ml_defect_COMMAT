# Data and code for analysis of TEM video using machine learning
This is the repository for the paper "Machine-learning-aided Analysis of Relationship Between Crystal Defects and Macroscopic Mechanical Properties of TWIP steel".
## Usage
### 1. Data preparation
#### 1.1  trimvideo.py
Trimiming of the length scale and time display in the video.
#### 1.2 optical_flow_cut.py
Conversion of the trimmed video into a video with static coordinates.
### 2. Semantic segmentation
#### 2.1 train_*class.py
Training of U-net.
#### 2.2 predict_*.py
Semantic segmentation of the frames not used for training.
