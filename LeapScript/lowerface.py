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
from PIL import Image
import json


class IdentityBrick(nn.Module): #residual block concepts in the resnet
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        super(IdentityBrick,self).__init__()
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self,x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)
        return x

class lowerfacelocator(nn.Module):
    def __init__(self,ngpu):
        super(lowerfacelocator, self).__init__() # referring to the inherit class as nn.Module explicitly
        self.ngpu = ngpu
        self.main=nn.Sequential(
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
        #groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)    
        #input is  1x 480x 640?
        nn.Conv2d(1 , 32, 3, 2,1, bias=False),  #32,  240, 320 
        nn.BatchNorm2d(32),  # number of channels
        nn.LeakyReLU(0.2, inplace=True),  #32,  240, 320 
        nn.MaxPool2d(2,2), #32, 120, 160
        IdentityBrick(32,32), # number of channels
        IdentityBrick(32,32,2), 
        IdentityBrick(32,64),
        IdentityBrick(64,64,2),
        IdentityBrick(64,128),
        IdentityBrick(128,128,2),#128, 30, 40
        nn.AvgPool2d(2), #128, 7, 10
        nn.Flatten(), #38400]
        nn.Linear(8960,2)
        )
    def forward(self,x):
        return self.main(x)
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")		
with torch.no_grad():
	best_network= lowerfacelocator(ngpu).to(device)
	best_network.load_state_dict(torch.load('lowerface-2.pth')) 
	best_network.eval()
	
LeapExposure = 20 # 50 in lab
with open("calibration.json", 'r') as j:
     contents = json.loads(j.read()) 
# set the leap 
leap = leapuvc.leapImageThread(source=1,resolution = (640,480)) # resolution in the manual 
leap.setExposure(LeapExposure) 
# exp 20 

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
				
                preleft = Image.fromarray(np.uint8(leftRightImage[0]))
                preright = Image.fromarray(np.uint8(leftRightImage[1]))
				
                l = TF.normalize(TF.to_tensor(preleft),[0.5], [0.5]).unsqueeze(0)
                r = TF.normalize(TF.to_tensor(preright),[0.5], [0.5]).unsqueeze(0)                   
                
                
                
                # Pack the raw images
                bundle = torch.cat((l,r))
                
                
                # Run thro network
                with torch.no_grad():
                    prediction = best_network(bundle.cuda()).cpu() + 0.5  # invoke cuda() to use gpu run                   
                    prediction = prediction.view(-1,1,2).numpy()*(640,480) # lefttop
					
				#show time
                leftRightImage[0] = cv2.rectangle(leftRightImage[0],prediction[0][0].astype(int),prediction[0][0].astype(int)+(224,224),(255, 0, 0),2)
                leftRightImage[1] = cv2.rectangle(leftRightImage[1],prediction[1][0].astype(int),prediction[1][0].astype(int)+(224,224),(255, 0, 0),2)
				

                cv2.imshow('Frame L', leftRightImage[0])
                cv2.imshow('Frame R', leftRightImage[1])
                
cv2.destroyAllWindows()
					
					
					
					
					