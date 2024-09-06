import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Name of the video to be converted into the one with static coordinate
video_name = 'after_trimvideo.mp4'


# List for storing the displacements of feature points
x_list = []
y_list = []

point_num=1#point_num=0~2
cap = cv2.VideoCapture(video_name)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc("m","p","4","v")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (20,20),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

p0 = old_p0[7:10]

p0_0=p0


# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
frame_num = 1

while(1):
    ret,frame = cap.read()
    img = frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    good_0=p0_0[st==1]
    ex_x = int(good_new[point_num,0]-good_0[point_num,0])
    ex_y = int(good_new[point_num,1]-good_0[point_num,1])
    
    x_list.append(ex_x)
    y_list.append(ex_y)
    
    frame_num = frame_num + 1
    if frame_num == totalframecount :
        break   
    
     # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)


# Trimming of the video
cap = cv2.VideoCapture(video_name)
frame_num = 1
trim_height = height-max(y_list) + min(y_list)
trim_width = width-max(x_list) + min(x_list)
frame_rate = 20.0
trim_video = cv2.VideoWriter("after_optical_flow_cut.mp4",fourcc,frame_rate,(trim_width, trim_height))

while(1):
    ret,frame = cap.read()
    img = frame
    trimming = img[-min(y_list)+y_list[frame_num-1]:height-max(y_list)+y_list[frame_num-1],-min(x_list)+x_list[frame_num-1]:width-max(x_list)+x_list[frame_num-1]]
    trim_video.write(trimming)
    frame_num = frame_num + 1
    if frame_num == totalframecount :
        break
    
trim_video.release()

cap.release()