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
        self.model=models.resnet18() # 224 x 224 <- 640 x480
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
       
        
    def forward(self, x):
        x=self.model(x)
        return x
with torch.no_grad():

    best_network = Network()
    best_network.cuda()
    best_network.load_state_dict(torch.load('LipMark1102Left.pth')) 
    best_network.eval()

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

def cutpoint225(lownp):
    """
    lownp is numpy array
    """
    resarr = np.ones((lownp.shape[0],2))
    i = 0
    for eachone in lownp:
        leftmost = eachone[0][0]
        topmost =  eachone[0][1]
        rightbound = leftmost+224
        botbound = topmost+224
        if rightbound < 480 and botbound < 640:
            pass
        elif rightbound >= 480 and botbound < 640:
            leftmost = leftmost - (rightbound -480)
        elif rightbound < 480 and botbound > 640:
            topmost = topmost - (botbound - 640)
        else:
            leftmost = leftmost - (rightbound -480) 
            topmost = topmost - (botbound - 640)
        resarr[i]= np.array((int(leftmost),int(topmost)))
        i+=1
    return resarr.astype(int)
	
def croptensor(rawtensor,lefttopcut):
    tenshape = rawtensor.shape
    restensor = torch.ones((tenshape[0],tenshape[1],224,224))
    j = 0 
    for i in rawtensor:
        restensor[j] = TF.crop(i,lefttopcut[j][1],lefttopcut[j][0],224,224)
        j+=1
    return restensor	
def cropimg(rawimgleft, rawimgright, lefttopcut):
	l = rawimgleft[lefttopcut[0][0]:lefttopcut[0][0]+224,lefttopcut[0][1]:lefttopcut[0][1]+224]
	r = rawimgright[lefttopcut[1][0]:lefttopcut[1][0]+224,lefttopcut[1][1]:lefttopcut[1][1]+224]
	return l, r
	
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")		
with torch.no_grad():
	lowerface= lowerfacelocator(ngpu).to(device)
	lowerface.load_state_dict(torch.load('lowface1102.pth')) 
	lowerface.eval()
with open("calibration.json", 'r') as j:
     contents = json.loads(j.read()) 
leap = leapuvc.leapImageThread(source=1,resolution = (640,480)) # resolution in the manual 	
LeapExposure = 15 # 50 in lab
print('the exposure is {}'.format(LeapExposure))
leap.setExposure(LeapExposure) 
leap.setLeftLED(True)
leap.setCenterLED(True)
leap.setRightLED(True)
leap.start()
fileIndex = 180
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
                
                
                # Run thro lower face network to get the crop lefttop label point
                with torch.no_grad():#To perform inference without Gradient Calculation.
                    prediction = lowerface(bundle.cuda()).cpu() + 0.5  # invoke cuda() to use gpu run                   
                    prediction = prediction.view(-1,1,2).numpy()*(640,480) # lefttop
					
					
				#show time
                #leftRightImage[0] = cv2.rectangle(leftRightImage[0],prediction[0][0].astype(int),prediction[0][0].astype(int)+(224,224),(255, 0, 0),2)
                #leftRightImage[1] = cv2.rectangle(leftRightImage[1],prediction[1][0].astype(int),prediction[1][0].astype(int)+(224,224),(255, 0, 0),2)
				
				# crop the 
                lefttop = cutpoint225(prediction)
                cutten = croptensor(bundle,lefttop)
				
				#run throu the lip mark network 
                with torch.no_grad():#To perform inference without Gradient Calculation.
                    predictions = (best_network(cutten.cuda()).cpu() + 0.5) #* np.array([img_shape[2], img_shape[1]])
                    predictions = predictions.view(-1, 4 ,2).detach().numpy() # 4 as 4 markers
                    predictions = predictions *224 #/2*3 + (150,50)
				
				
				#show time for lip markers
                for (LabelLeft, LabelRight) in  zip(predictions[0],predictions[1]):
                    #cv2.drawMarker draws a marker! One marker!!!! https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga482fa7b0f578fcdd8a174904592a6250
                    cv2.drawMarker(leftRightImage[0], ((LabelLeft[0]).astype(int)+lefttop[0][0], (LabelLeft[1]).astype(int)+lefttop[0][1]),
                    #cv2.drawMarker(leftRightImage[0], ((LabelLeft[0] ).astype(int), (LabelLeft[1] ).astype(int)),
                    (200, 150, 150), markerType=cv2.MARKER_STAR,markerSize=2, thickness=2, line_type=cv2.LINE_AA)
                    #cv2.drawMarker(leftRightImage[1], ((LabelRight[0]).astype(int)+lefttop[1][0], (LabelRight[1]).astype(int)+lefttop[1][1]), 
                    #cv2.drawMarker(leftRightImage(1), ((LabelRight[0]).astype(int), (LabelRight[1]).astype(int)), 
                    #(250, 50, 200), markerType=cv2.MARKER_STAR,markerSize=2, thickness=2, line_type=cv2.LINE_AA)
	
                #if(cv2.waitKey(1) & 0xFF == ord('s')):
                #	imgl, imgr = cropimg(leftRightImage[0],leftRightImage[1], lefttop)
                #	cv2.imwrite('Captures/'+'LeapExposure'+str(LeapExposure)+'File'
				#			+str(fileIndex)+'left.png', imgl)
                #	cv2.imwrite('Captures/'+'LeapExposure'+str(LeapExposure)+'File'
				#			+str(fileIndex)+'right.png', imgr)
                #	fileIndex += 1

				

                cv2.imshow('Frame L', leftRightImage[0])
                cv2.imshow('Frame R', leftRightImage[1])
                
cv2.destroyAllWindows()	 