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


class Network(nn.Module):
    def __init__(self,num_classes = 4 * 2): # 4 pairs of 2d coordinate as x,y
        super().__init__()
        self.model_name='resnet18'
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
            #groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) 
        #self.firstlayer=nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstlayer=nn.MaxPool2d(3,stride=2)
        #or maxpooling
        self.model=models.resnet18() # 224 x 224 <- 640 x480
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        # resize data, re-adjusted label coordinates.
        
    def forward(self, x):
        x=self.model(self.firstlayer(x))
        return x
        
with torch.no_grad():

    best_network = Network()
    best_network.cuda()
    best_network.load_state_dict(torch.load('../Models/lip_landmarks0.pth')) 
    best_network.eval()
    
LeapExposure = 22
with open("calibration.json", 'r') as j:
     contents = json.loads(j.read()) 


# set the leap 
leap = leapuvc.leapImageThread(source=0,resolution = (640,480)) # resolution in the manual 
leap.setExposure(LeapExposure) 
# exp 20 with resnet18Category.pth

leap.setLeftLED(True)
leap.setCenterLED(True)
leap.setRightLED(True)
leap.start()

undistRect = True

while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    newFrame, leftRightImage = leap.read() # in np.uint8
    if(newFrame):
        # Rectify the image
        #print("before the undistorted: ",leftRightImage[0].shape)
        if(undistRect):
                maps = contents['left']["undistortMaps"]
                leftRightImage[0] = cv2.remap(leftRightImage[0], np.array(maps[0], dtype=np.int16), # 
                np.array(maps[1], dtype=np.int16), cv2.INTER_LINEAR)
                maps = contents['right']["undistortMaps"]
                leftRightImage[1] = cv2.remap(leftRightImage[1], np.array(maps[0], dtype=np.int16), # 
                np.array(maps[1], dtype=np.int16), cv2.INTER_LINEAR)
                
                
                
                
                # pytorch tools need RGB numpy or better PIL data and PyTorch Tensor input 
                preleft = Image.fromarray(np.uint8(leftRightImage[0])) 
                preright = Image.fromarray(np.uint8(leftRightImage[1]))
                
                
                
                # Pack the raw images
                bundle = torch.cat((preleft,preright))
                
                
                # Run thro network
                with torch.no_grad():
                    prediction = (best_network(bundle.cuda()).cpu() + 0.5)  # invoke cuda() to use gpu run throu
                    prediction = prediction.view(-1,4,2) # 4 as 4 markers
                  
                #show time  
                for (LabelLeft, LabelRight) in  zip(prediction[0],prediction[1]):
                
                    cv2.drawMarker(preleft, (LabelLeft[0]*640, LabelLeft[1]*480), (200, 150, 150), markerType=cv2.MARKER_STAR, 
                                markerSize=2, thickness=2, line_type=cv2.LINE_AA)
                    cv2.drawMarker(preright, (LabelRight[0]*640, LabelRight[1]*480), (250, 50, 200), markerType=cv2.MARKER_STAR, 
                                markerSize=2, thickness=2, line_type=cv2.LINE_AA)

                cv2.imshow('Frame L', preleft)
                cv2.imshow('Frame R', preright)
                
cv2.destroyAllWindows()