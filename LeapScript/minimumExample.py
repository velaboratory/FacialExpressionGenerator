import numpy as np        # `pip install numpy`
import cv2                # `pip install opencv-python`
import leapuvc            #  Ensure leapuvc.py is in this folder


import time
import os
from collections import OrderedDict
from skimage import io, transform
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
import json





# load cnn model
with torch.no_grad():
    model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features #  
    model_ft.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model_ft.fc = nn.Linear(num_ftrs, 2)
    #model_ft = model_ft.to(device)
    model_ft.cuda()
    model_ft.load_state_dict(torch.load('resnet18.pth')) 
    model_ft.eval()
    
# load calibration file
with open("calibration.json", 'r') as j:
     contents = json.loads(j.read()) 

#mean =-0.0732
#stdev = 0.0437
mean = 40
stdev = 64

# set the leap 
leap = leapuvc.leapImageThread(source=1,resolution = (640,480)) # resolution in the manual 
leap.setExposure(0)
leap.setLeftLED(True)
leap.setCenterLED(True)
leap.setRightLED(True)
leap.start()

undistRect = True
# Start the Leap Capture Thread


# Capture images until 'q' is pressed
while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    newFrame, leftRightImage = leap.read()
    if(newFrame):
        # Rectify the image
        if(undistRect):
                maps = contents['left']["undistortMaps"]
                leftRightImage[0] = cv2.remap(leftRightImage[0], np.array(maps[0], dtype=np.int16), # 
                np.array(maps[1], dtype=np.int16), cv2.INTER_LINEAR)
                maps = contents['right']["undistortMaps"]
                leftRightImage[1] = cv2.remap(leftRightImage[1], np.array(maps[0], dtype=np.int16), # 
                np.array(maps[1], dtype=np.int16), cv2.INTER_LINEAR)
                
        
        
        # Resize
        #preleft= cv2.normalize(cv2.resize(leftRightImage[0],
        #            (320,240)),None,0.0,1.0,cv2.NORM_MINMAX).astype('float32')
        #preright=cv2.normalize(cv2.resize(leftRightImage[1],
        #            (320,240)),None,0.0,1.0,cv2.NORM_MINMAX).astype('float32')
        preleft = cv2.resize(leftRightImage[0], (320,240)) # integer [0,256]
        preright = cv2.resize(leftRightImage[1], (320,240))    #size (240, 320)
     
       
        
        
        # Pack the raw images
        l = TF.normalize(TF.to_tensor(preleft), [0.5], [0.5]).unsqueeze(0)
        r = TF.normalize(TF.to_tensor(preright), [0.5], [0.5]).unsqueeze(0)
        #l = TF.to_tensor(preleft).unsqueeze(0)
        #r = TF.to_tensor(preright).unsqueeze(0)
        bundle = torch.cat((l,r)) 
        
        # Run thro network
        with torch.no_grad():
            prediction = model_ft(bundle.cuda()) # invoke cuda() to use gpu run throu
            #prediction = prediction.view(-1,lowerFaceCounts,2)
            _, preds = torch.max(prediction, 1)
        
        
        # Display the raw frame
        if preds[0].cpu().numpy() == 1:
            leftviz = cv2.putText(cv2.rectangle(preleft,(20,20),(300,200),(0,1,0),2)
                    ,"close",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 0.5, 250), 2, cv2.LINE_AA).astype('float32')
        else:
            leftviz = cv2.putText(cv2.rectangle(preleft,(20,20),(300,200),(0,1,0),2)
                    ,"open",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 250, 250), 2, cv2.LINE_AA).astype('float32')
                   
        if preds[1].cpu().numpy() == 1:
            rightviz = cv2.putText(cv2.rectangle(preright,(20,20),(300,200),(0,1,0),2)
                    ,"close",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 1, 250), 2, cv2.LINE_AA).astype('float32')
        else:
            rightviz = cv2.putText(cv2.rectangle(preright,(20,20),(300,200),(0,1,0),2)
                    ,"openR",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 250, 250), 2, cv2.LINE_AA).astype('float32')
        cv2.imshow('Frame L', preleft)
        cv2.imshow('Frame R', preright)

cv2.destroyAllWindows()

"""
preleft = (preleft - mean)/stdev
preright = (preright - mean)/stdev

preleft -= np.amin(preleft) #shift to 0 
preleft = cv2.normalize(preleft,None,0,1,cv2.NORM_MINMAX).astype('float32') # [0,1]
#preleft /= np.amax(preleft)
preright -= np.amin(preright) #shift to 0 
preright = cv2.normalize(preright,None,0,1,cv2.NORM_MINMAX).astype('float32') # [0,1]
#preright /= np.amax(preright)

"""