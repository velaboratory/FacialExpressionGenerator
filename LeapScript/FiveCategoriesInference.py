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





# load cnn model
with torch.no_grad():
    model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features #  
    model_ft.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model_ft.fc = nn.Linear(num_ftrs, 5)
    #model_ft = model_ft.to(device)
    model_ft.cuda()
    model_ft.load_state_dict(torch.load('../Models/resnet18Category.pth')) 
    model_ft.eval()
    
# load calibration file
with open("calibration.json", 'r') as j:
     contents = json.loads(j.read()) 


# set the leap 
leap = leapuvc.leapImageThread(source=0,resolution = (640,480)) # resolution in the manual 
leap.setExposure(20)
leap.setLeftLED(True)
leap.setCenterLED(True)
leap.setRightLED(True)
leap.start()

undistRect = True
# Start the Leap Capture Thread


# Capture images until 'q' is pressed
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
                
        #print("before the pass to torch",leftRightImage[0].shape)
        # cv2 raw data in GRAYSCALAE
        
        
        # pytorch tools need RGB numpy or better PIL data and PyTorch Tensor input    

     
        
            
        preleft = Image.fromarray(np.uint8(leftRightImage[0])) 
        l = TF.normalize(TF.to_tensor(TF.resize(preleft,(240,320))),[0.5], [0.5]).unsqueeze(0)                                                      
        preright = Image.fromarray(np.uint8(leftRightImage[1])) 
        r = TF.normalize(TF.to_tensor(TF.resize(preright,(240,320))),[0.5], [0.5]).unsqueeze(0)
        
        #print("before the pass to torch",preleft.size, " and ",preleft.getbands())
        #ABOUVE PRINTS (640, 480) , ('L',) : graysacle, no need for TF.rgb_to_grayscale(,1) or .convert('RGB')
   

     
       
        
        
        # Pack the raw images
        bundle = torch.cat((l,r)) 
        
        # Run thro network
        with torch.no_grad():
            prediction = model_ft(bundle.cuda()) # invoke cuda() to use gpu run throu
            #prediction = prediction.view(-1,lowerFaceCounts,2)
            _, preds = torch.max(prediction, 1)
        
        
        # Display the raw frame
        
        if preds[0].cpu().numpy() == 0 :     
                leftviz = cv2.putText(cv2.rectangle(leftRightImage[0],(20,20),(600,400),(0,1,0),2)
                    ,"Eighty",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 50, 250), 2, cv2.LINE_AA) ;
        elif preds[0].cpu().numpy() ==1:    
                leftviz = cv2.putText(cv2.rectangle(leftRightImage[0],(20,20),(600,400),(0,1,0),2)
                    ,"Fifth",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (200, 150, 150), 2, cv2.LINE_AA); 
        elif preds[0].cpu().numpy() ==2:
                leftviz = cv2.putText(cv2.rectangle(leftRightImage[0],(20,20),(600,400),(0,1,0),2)
                    ,"Full",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 50, 250), 2, cv2.LINE_AA); 
        elif preds[0].cpu().numpy() ==3:
                leftviz = cv2.putText(cv2.rectangle(leftRightImage[0],(20,20),(600,400),(0,1,0),2)
                    ,"Half",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 50, 250), 2, cv2.LINE_AA); 
        else:
                 leftviz = cv2.putText(cv2.rectangle(leftRightImage[0],(20,20),(600,400),(0,1,0),2)
                    ,"Zero",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 50, 250), 2, cv2.LINE_AA); 
                
        
        
                   
        if preds[1].cpu().numpy() ==0 :     
                rightviz = cv2.putText(cv2.rectangle(leftRightImage[1],(20,20),(600,400),(0,1,0),2)
                    ,"Eighty",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 50, 250), 2, cv2.LINE_AA) ;
        elif preds[1].cpu().numpy() ==1:    
                rightviz = cv2.putText(cv2.rectangle(leftRightImage[1],(20,20),(600,400),(0,1,0),2)
                    ,"Fifth",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (200, 150, 150), 2, cv2.LINE_AA); 
        elif preds[1].cpu().numpy() ==2:
                rightviz = cv2.putText(cv2.rectangle(leftRightImage[1],(20,20),(600,400),(0,1,0),2)
                    ,"Full",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 50, 250), 2, cv2.LINE_AA); 
        elif preds[1].cpu().numpy() ==3:
                rightviz = cv2.putText(cv2.rectangle(leftRightImage[1],(20,20),(600,400),(0,1,0),2)
                    ,"Half",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 50, 250), 2, cv2.LINE_AA); 
        else:
                 rightviz = cv2.putText(cv2.rectangle(leftRightImage[1],(20,20),(600,400),(0,1,0),2)
                    ,"Zero",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,
                    (250, 50, 250), 2, cv2.LINE_AA); 
        cv2.imshow('Frame L', leftviz)
        cv2.imshow('Frame R', rightviz)

cv2.destroyAllWindows()

