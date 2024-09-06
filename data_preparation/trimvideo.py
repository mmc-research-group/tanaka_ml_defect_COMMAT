import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Used for trimming the dispray of time in the TEM videos, etc.

# Name of the video to be trimmed
original_video_name = 'original_video.mp4'
# Name of the video after trimmed
trimmed_video_name = 'after_trimvideo.mp4'
# Frame rate of the video after trimmed
frame_rate = 20.0
# Coordinates of the area to be left (top left is the origin, right is x positive direction, bottom is y positive direction)
left = 0
right = 450
top = 30
bottom = 450

    
try:
    trimmed_video_size = (right-left, bottom-top)
    cap = cv2.VideoCapture(original_video_name)
    fourcc = cv2.VideoWriter_fourcc("m","p","4","v")
    trim_video = cv2.VideoWriter(trimmed_video_name, fourcc, frame_rate, trimmed_video_size)
 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 1

    while True:
        ret,frame = cap.read()
        
        if ret == False :
            break
     
        dst = frame[top:bottom, left:right]
        trim_video.write(dst)
        
        if count ==1: 
            img_ndarray = dst
            a = img_ndarray*255        
            pil_img = Image.fromarray(dst.astype(np.uint8))
            plt.imshow(dst)
            plt.show() 
        
        count = count + 1
      
    
    trim_video.release()
    cap.release()

except:
    import sys
    print('Error:',sys.exc_info()[0])
    print(sys.exc_info()[1])
    import traceback
    print(traceback.format_tb(sys.exc_info()[2]))