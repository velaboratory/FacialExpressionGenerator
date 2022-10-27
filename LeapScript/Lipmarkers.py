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


class Network(nn.Module):
    def __init__(self,num_classes = 4 * 2): # 4 pairs of 2d coordinate as x,y
        super().__init__()
        self.model_name='resnet18'
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
            #groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) 
        #self.firstlayer=nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        #self.firstlayer=nn.MaxPool2d(3,stride=2)
        self.model=models.resnet18() # 224 x 224 <- 640 x480
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
       
        
    def forward(self, x):
        #x=self.model(self.firstlayer(x))
        x=self.model(x)
        return x
        
with torch.no_grad():

    best_network = Network()
    best_network.cuda()
    best_network.load_state_dict(torch.load('lip_landmarks4.pth')) 
    best_network.eval()
    
LeapExposure = 20 # 50 in lab
with open("calibration.json", 'r') as j:
     contents = json.loads(j.read()) 


# set the leap 
leap = leapuvc.leapImageThread(source=0,resolution = (640,480)) # resolution in the manual 
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
                
                
                
                
                # pytorch needs RGB numpy or better PIL data and PyTorch Tensor input 
                cut224l=cv2.resize(leftRightImage[0][50:386,150:486],(224,224)) #336 ->224
                cut224r=cv2.resize(leftRightImage[1][50:386,150:486],(224,224))
				
                preleft = Image.fromarray(np.uint8(cut224l)) 
                preright = Image.fromarray(np.uint8(cut224r))
                
                l = TF.normalize(TF.to_tensor(preleft),[0.5], [0.5]).unsqueeze(0)
                r = TF.normalize(TF.to_tensor(preright),[0.5], [0.5]).unsqueeze(0)                   
                
                
                
                # Pack the raw images
                bundle = torch.cat((l,r))
                
                
                # Run thro network
                with torch.no_grad():
                    prediction = best_network(bundle.cuda()).cpu() + 0.5  # invoke cuda() to use gpu run throu                   
                    prediction = prediction.view(-1,4,2).numpy() # 4 as 4 markers
                    prediction = prediction*224# /2*3 + (150,50)
                 
                 
                #show time
                for (LabelLeft, LabelRight) in  zip(prediction[0],prediction[1]):
                    #cv2.drawMarker draws a marker! One marker!!!! https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga482fa7b0f578fcdd8a174904592a6250
                    cv2.drawMarker(cut224l, ((LabelLeft[0]).astype(int), (LabelLeft[1]).astype(int)),
                    #cv2.drawMarker(leftRightImage[0], ((LabelLeft[0] ).astype(int), (LabelLeft[1] ).astype(int)),
                    (200, 150, 150), markerType=cv2.MARKER_STAR,markerSize=2, thickness=2, line_type=cv2.LINE_AA)
                    cv2.drawMarker(cut224r, ((LabelRight[0]).astype(int), (LabelRight[1]).astype(int)), 
                    #cv2.drawMarker(leftRightImage(1), ((LabelRight[0]).astype(int), (LabelRight[1]).astype(int)), 
                    (250, 50, 200), markerType=cv2.MARKER_STAR,markerSize=2, thickness=2, line_type=cv2.LINE_AA)

                cv2.imshow('Frame L', cut224l)
                cv2.imshow('Frame R', cut224r)
                
cv2.destroyAllWindows()