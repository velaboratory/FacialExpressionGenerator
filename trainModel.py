import time
import cv2
import os
import glob
import re
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from collections import OrderedDict
from skimage import io, transform
from math import *
#import xml.etree.ElementTree as ET
import imutils
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.optim import lr_scheduler
#import csv

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#=============== data loader class  ======================= 
class GaussNoise(object):
    def __init__(self,mean =0., std= 0.02):
        self.std = std
        self.mean = mean
        #print("going")
        # need a blurry right now sharper than real leap data.
    def __call__(self,tensor):
        #print("running from GaussNoise class") # wow!!!!
        # won't be invoked during the dataset instance was called
        # will be invoked during iteration on the dataset batch!!!!
        return tensor+torch.randn(tensor.size())*self.std+self.mean
    def __repr__(self):
        return self.__class__.__name__ + ' adding gaussian noise'
    
class CustomizedAugmentedDataset(Dataset):
    def __init__(self, root: str, transform): 
        #super().__init__(root,transform=transform)
        self.transform = transform
        self.imageList = sorted(glob.glob(root+'/*.png'),
                      key=lambda x:float(re.findall("(\d+)",x)[0]))[:30000] # sorted by name:
        self.labelList = sorted(glob.glob(root+'/*.csv'),
                      key=lambda x:float(re.findall("(\d+)",x)[0]))[:30000]  # sorted by name:
        if not len(self.imageList) == len(self.labelList):
            raise TypeError(f"images and labels amount do not match:\
                            got images:{len(self.imageList)} but labes: {len(self.labelList)} ")

        #self.labels = []
        #self.crops = []
        #self.transform = transform
        #self.root_dir = '.'
        #self.labels = np.array(self.labels).astype('float32')     
        #assert len(self.image_filenames) == len(self.labels)


    def __len__(self):
        return len(self.imageList)
    def __getitem__(self, index):
        image = Image.fromarray(cv2.resize(
            cv2.imread(self.imageList[index], 0)[50:386,150:486],(224,224)))
        if self.transform is not None:
            image = self.transform(image)
        #0: grayscale
        #pytorch tools need RGB numpy or better PIL data and PyTorch Tensor input      
        #df = pd.read_csv( self.labelList[index], sep=',', header=None) # no need to close -> GC
        df = pd.read_csv( self.imageList[index].replace('png','csv'), sep=',', header=None)
        nparray = (df.to_numpy() - (150,50))/3*2
        origin = nparray.copy()
        labels =  origin[np.sort(nparray[:,1].argsort() ) ].astype('float32')[3]
        img_shape = np.array(image).shape # may not be computed every time
        labels = labels / np.array([img_shape[2], img_shape[1]])
        # this is because img data is normalized into [-0.5, 0.5] so its better regression labels 
        # are falling into [-0.5, 0.5]  float 32 as well. and number are uniformal to compute
        labels = labels - 0.5 
        return image, labels.astype('float32')
    def __repr__(self):
        return " bespoken loader"

class CustomizedLowerDataset(Dataset):
    def __init__(self, root: str, lowerlabel: str ,whichSide, transform): 
        #super().__init__(root,transform=transform)
        self.transform = transform
        self.lowerpath = lowerlabel
        self.pathroot  = root+'/*' # with 0
        if whichSide == 1: #left side
            self.pathroot  = root+'/*Left*'
        elif whichSide == 2: # Right side
            self.pathroot  = root+'/*Right*'
        self.imageList = sorted(glob.glob(self.pathroot+'.png'),
                      key=lambda x:float(re.findall("(\d+)",x)[0])) # sorted by name:
        self.labelList = sorted(glob.glob(self.pathroot+'.csv'),
                      key=lambda x:float(re.findall("(\d+)",x)[0]))   # sorted by name:
        
        if not len(self.imageList) == len(self.labelList):
            raise TypeError(f"images and labels amount do not match:\
                            got images:{len(self.imageList)} but labes: {len(self.labelList)} ")
        with torch.no_grad():
            self.lowerface= lowerfacelocator(ngpu).to(device)
        #lowerface.cuda()
            self.lowerface.load_state_dict(torch.load('lowface3.pth'))
            self.lowerface.eval()
        print('loading NETWORK PARAMETERS successfully')


           
        #self.lowerlist = sorted(glob.glob(lowerlabel+'.csv'),
        #              key=lambda x:float(re.findall("(\d+)",x)[0]))   # sorted by name:
        #if not len(self.imageList) == len(self.lowerlist):
        #    raise TypeError(f"images and lower amount do not match:\
        #                    got images:{len(self.imageList)} but labes: {len(self.lowerlist)} ")
      

        #self.labels = []
        #self.crops = []
        #self.transform = transform
        #self.root_dir = '.'
        #self.labels = np.array(self.labels).astype('float32')     
        #assert len(self.image_filenames) == len(self.labels)


    def __len__(self):
        return len(self.imageList)
    def __getitem__(self, index):
        imgestr = self.imageList[index]
        
        image = Image.fromarray(cv2.imread(imgestr, 0))
        if self.transform is not None:
            image = self.transform(image)
        image = image.cuda()
        with torch.no_grad():
            predictions = (self.lowerface(image.unsqueeze(0)).detach().cpu() + 0.5) 
            predictions = predictions.view(-1, 1 ,2).numpy()*(640,480)
        lownp = predictions[0].astype(int) #  
        #lownp = predictions[0]
        csvstr = imgestr.replace('png','csv')
        
        #lowerpoint = pd.read_csv(self.lowerpath +'/'+ os.path.basename(csvstr), sep=',', header=None)
        #lownp = lowerpoint.to_numpy()
        
        #print(lownp.shape)
        leftmost = lownp[0][0]
        topmost =  lownp[0][1]
        rightbound = leftmost+224
        botbound = topmost+224
        if rightbound < 480 and botbound < 640:
            #image = image.crop(leftmost,topmost,leftmost+224,topmost+224)  #(left, upper, right, lower) 
            image = TF.crop(image,leftmost,topmost,224,224) #top: int, left: int, height: int, width:int
        elif rightbound >= 480 and botbound < 640:
            leftmost = leftmost - (rightbound -480)
            #image = Image.fromarray(cv2.imread(imgestr, 0)[leftmost:leftmost+224,topmost:topmost+224])
            #image = image.crop(leftmost,topmost,leftmost+224,topmost+224)
            image = TF.crop(image,leftmost,topmost,224,224)
        elif rightbound < 480 and botbound > 640:
            topmost = topmost - (botbound - 640)
            #image = Image.fromarray(cv2.imread(imgestr, 0)[leftmost:leftmost+224,topmost:topmost+224])  
            #image = image.crop(leftmost,topmost,leftmost+224,topmost+224)
            image = TF.crop(image,leftmost,topmost,224,224)
        else:
            leftmost = leftmost - (rightbound -480) 
            topmost = topmost - (botbound - 640)
            #image = Image.fromarray(cv2.imread(imgestr, 0)[leftmost:leftmost+224,topmost:topmost+224])
            #image = image.crop(leftmost,topmost,leftmost+224,topmost+224)
            image = TF.crop(image,leftmost,topmost,224,224)
                 

        #0: grayscale
        #pytorch tools need RGB numpy or better PIL data and PyTorch Tensor input      
        #df = pd.read_csv( self.labelList[index], sep=',', header=None) # no need to close -> GC
        df = pd.read_csv( csvstr, sep=',', header=None)
        labels = df.to_numpy() - (leftmost,topmost)
        labels = labels / np.array([224, 224])
        # this is because img data is normalized into [-0.5, 0.5] so its better regression labels 
        # are falling into [-0.5, 0.5]  float 32 as well. and number are uniformal to compute
        labels = labels - 0.5 
        return image, labels.astype('float32')
    def __repr__(self):
        return " bespoken loader to load lower face and train for lipmark"
    
class bespokenDataset(Dataset):
    def __init__(self, root: str, transform): 
        #super().__init__(root,transform=transform)
        self.transform = transform
        self.imageList = sorted(glob.glob(root+'/*.png'),
                      key=lambda x:float(re.findall("(\d+)",x)[0])) # sorted by name:
        self.labelList = sorted(glob.glob(root+'/*.csv'),
                      key=lambda x:float(re.findall("(\d+)",x)[0]))  # sorted by name:
        if not len(self.imageList) == len(self.labelList):
            raise TypeError(f"images and labels amount do not match:\
                            got images:{len(self.imageList)} but labes: {len(self.labelList)} ")
        #self.labels = []
        #self.crops = []
        #self.transform = transform
        #self.root_dir = '.'
        #self.labels = np.array(self.labels).astype('float32')     
        #assert len(self.image_filenames) == len(self.labels)

    def __len__(self):
        return len(self.imageList)
    def __getitem__(self, index):
        image = Image.fromarray(cv2.imread(self.imageList[index], 0))
        if self.transform is not None:
            image = self.transform(image)
        #0: grayscale
        #pytorch tools need RGB numpy or better PIL data and PyTorch Tensor input      
        #df = pd.read_csv( self.labelList[index], sep=',', header=None) # no need to close -> GC
        df = pd.read_csv( self.imageList[index].replace('png','csv'), sep=',', header=None)
        lipmarks = df.to_numpy() 
        lifspan = lipmarks[3][0] - lipmarks[1][0]
        leftoffset = lipmarks[1][0] - (224 -int(lifspan))/2 #default cut
         #uniform distribution random re cut  
        if lipmarks[3][0]>224 and (lipmarks[1][0]+224) <640 :
            #print('range is ',int(lipmarks[3][0])-224)
            #print('range is ',int(lipmarks[1][0])-1, ' ',int(lipmarks[3][0])-224)
            rightbound = int(lipmarks[3][0])-224
            leftbound = int(lipmarks[1][0])
            if leftbound < rightbound:
                rint = random.randint(leftbound+1,rightbound-1)
            #print('random r is ',rint)
                leftoffset = rint
            else:
                rint = random.randint(rightbound +1,leftbound-1)
                leftoffset = rint
        verticalVar = random.randint(-3,3)
        lefttop = np.array((leftoffset,int(lipmarks[0][1]-50+verticalVar)))
        labels =  lefttop 
        img_shape = np.array(image).shape # may not be computed every time
        labels = labels / np.array([img_shape[2], img_shape[1]]) # normalize
        # this is because img data is normalized into [-0.5, 0.5] so its better regression labels 
        # are falling into [-0.5, 0.5]  float 32 as well. and number are uniformal to compute
        labels = labels - 0.5 
        return image, labels.astype('float32')
    def __repr__(self):
        return " bespoken loader to load full face but with lower face label and train lower face"

    
class lowerfaceDataset(Dataset):
    def __init__(self, root: str, transform): 
        #super().__init__(root,transform=transform)
        self.transform = transform
        self.imageList = sorted(glob.glob(root+'/*.png'),
                      key=lambda x:float(re.findall("(\d+)",x)[0])) # sorted by name:
        #self.labelList = sorted(glob.glob(root+'/*.csv'),
        #              key=lambda x:float(re.findall("(\d+)",x)[0]))  # sorted by name:
        #if not len(self.imageList) == len(self.labelList):
        #    raise TypeError(f"images and labels amount do not match:\
        #                    got images:{len(self.imageList)} but labes: {len(self.labelList)} ")
        #self.labels = []
        #self.crops = []
        #self.transform = transform
        #self.root_dir = '.'
        #self.labels = np.array(self.labels).astype('float32')     
        #assert len(self.image_filenames) == len(self.labels)

    def __len__(self):
        return len(self.imageList)
    def __getitem__(self, index):
        image = Image.fromarray(cv2.imread(self.imageList[index], 0))
        if self.transform is not None:
            image = self.transform(image)
        #0: grayscale
        #pytorch tools need RGB numpy or better PIL data and PyTorch Tensor input      
        #df = pd.read_csv( self.labelList[index], sep=',', header=None) # no need to close -> GC
        labels = os.path.basename(self.imageList[index]).replace('png','csv')
        return image, labels
    def __repr__(self):
        return " bespoken loader to load images for prediction, and the label is file name handy for result save to file"
    
#=============== network class  ======================= 

class Network(nn.Module):
    def __init__(self,num_classes = 4 * 2): # 4 pairs of 2d coordinate as x,y
        super().__init__()
        self.model_name='resnet18'
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
            #groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) 
        #self.firstlayer=nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        #self.firstlayer=nn.MaxPool2d(3,stride=2)
        #or maxpooling RESIZE -> # 224 x 224 --> 
        self.model=models.resnet18() # 224 x 224 <- 640 x480 ->320X240
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        # resize data, re-adjusted label coordinates.
        
    def forward(self, x):
        #x=self.model(self.firstlayer(x))
        x=self.model(x)
        return x
    
    
def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))   
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
        
    sys.stdout.flush()
    
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
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
   
#=============== train run function  ======================= 

def trainResnet18():
    CustomDataset = CustomizedAugmentedDataset(root="Output/lipmark",
                                           transform=
                                           transforms.Compose([
                                            transforms.ToTensor(),                      
                                            GaussNoise(), 
                                            transforms.GaussianBlur((3,5),(3,5)),
                                            #torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
                                            #sigma (float or tuple of python:float (min, max)) – Standard deviation to 
                                            # be used for creating kernel to perform blurring.
                                            #If float, sigma is fixed. If it is tuple of float (min, max), 
                                            #sigma is chosen uniformly at random to lie in the given range.
                                           transforms.Normalize((0.5), (0.5))
                                            ]))
    print('the lens of data',len(CustomDataset))
    print('begin training')
    # split the dataset into validation and test sets
    len_valid_sets = int(0.1*len(CustomDataset))
    len_train_sets = len(CustomDataset) - len_valid_sets
    print("The length of Train set is {}".format(len_train_sets))
    print("The length of Valid set is {}".format(len_valid_sets))

    train_datasets , valid_datasets = torch.utils.data.random_split(CustomDataset,
                                                                   [len_train_sets, len_valid_sets ])
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=8, shuffle=True, num_workers=0)
    torch.autograd.set_detect_anomaly(True)
    network = Network()
    network.cuda()    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    loss_min = np.inf
    num_epochs = 15
    start_time = time.time()
    for epoch in range(1,num_epochs+1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for _, data in enumerate(valid_loader):

            images, labels = data

            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0),-1).cuda() 

            predictions = network(images)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train/step

            print_overwrite(step, len(train_loader), running_loss, 'train')

        network.eval() 
        with torch.no_grad():

            for step in range(1,len(valid_loader)+1):

                images, landmarks = next(iter(valid_loader))

                images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0),-1).cuda()

                predictions = network(images)

                # find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')

        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), 'lip_landmarks6.pth') 
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))
    
      
def trainMylowerface(modelParamName,dataroot="Output/lipmark"):
    mydataset=bespokenDataset(root=dataroot,
                                           transform=
                                           transforms.Compose([
                                            transforms.ToTensor(),                      
                                            GaussNoise(), 
                                            transforms.GaussianBlur((3,5),(1,3)),
                                            #torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
                                            #sigma (float or tuple of python:float (min, max)) – Standard deviation to 
                                            # be used for creating kernel to perform blurring.
                                            #If float, sigma is fixed. If it is tuple of float (min, max), 
                                            #sigma is chosen uniformly at random to lie in the given range.
                                           transforms.Normalize((0.5), (0.5))
                                            ]))
    f = open(modelParamName+'.txt','w')   
    f.write('the lens of data: '+str(len(mydataset))+'\n')
    print('the lens of data',len(mydataset))
    print('begin training')
    # split the dataset into validation and test sets
    len_valid_sets = int(0.1*len(mydataset))
    len_train_sets = len(mydataset) - len_valid_sets
    print("The length of Train set is {}".format(len_train_sets))
    print("The length of Valid set is {}".format(len_valid_sets))
    f.write('The length of Train set is: '+str(len_train_sets)+'\n')
    f.write('The length of Valid set is: '+str(len_valid_sets)+'\n')
    train_datasets , valid_datasets = torch.utils.data.random_split(mydataset,
                                                                   [len_train_sets, len_valid_sets ])
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=8, shuffle=True, num_workers=0)
    
    torch.autograd.set_detect_anomaly(True)
    criterion = nn.MSELoss()

    loss_min = np.inf
    num_epochs = 10


    # Learning rate for optimizers
    lr = 0.0001

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    lowerface= lowerfacelocator(ngpu).to(device)
    #network.to(cuda) == network.cuda()

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    lowerface.apply(weights_init)

    # Print the model
    #print(lowerface)
    # Setup Adam optimizers
    optimizer = optim.Adam(lowerface.parameters(), lr=lr, betas=(beta1, 0.999),weight_decay=0.0) #
    start_time = time.time()
    for epoch in range(1,num_epochs+1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0
        lowerface.train()
        for step, data in enumerate(train_loader,1):

            images, labels = data

            images = images.cuda()
            labels = labels.view(labels.size(0),-1).cuda() 

            predictions = lowerface(images)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            
            loss_train_step = criterion(predictions, labels)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train/step

            print_overwrite(step, len(train_loader), running_loss, 'train')

        lowerface.eval() 
        with torch.no_grad():
            for step, data in enumerate(valid_loader,1):

                images, labels = data

                images = images.cuda()
                labels = labels.view(labels.size(0),-1).cuda()

                predictions = lowerface(images)

                # find the loss for the current step
                loss_valid_step = criterion(predictions, labels)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')
        f.write('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(lowerface.state_dict(), modelParamName+'.pth') 
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))
    f.write("Total Complete Elapsed Time : {} s".format(time.time()-start_time))
    f.close()

def trainLipmarkLowerface(modelParamName,whichSide):
    mydataset=CustomizedLowerDataset(root="Output/lipmark", lowerlabel = 'Output/lowerfacelabel', 
                                     whichSide=whichSide,transform=transforms.Compose([
                                            transforms.ToTensor(),                      
                                            GaussNoise(), 
                                            transforms.GaussianBlur((3,5),(1,3)),
                                            #torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
                                            #sigma (float or tuple of python:float (min, max)) – Standard deviation to 
                                            # be used for creating kernel to perform blurring.
                                            #If float, sigma is fixed. If it is tuple of float (min, max), 
                                            #sigma is chosen uniformly at random to lie in the given range.
                                           transforms.Normalize((0.5), (0.5))
                                            ])
                                    )
    print('the lens of data',len(mydataset))
    f = open(modelParamName+'.txt','w')   
    f.write('the lens of data: '+str(len(mydataset))+'\n')
    #print('begin training')
    print(f"{bcolors.WARNING}begin training{bcolors.ENDC}")
    # split the dataset into validation and test sets
    len_valid_sets = int(0.1*len(mydataset))
    len_train_sets = len(mydataset) - len_valid_sets
    print("The length of Train set is {}".format(len_train_sets))
    print("The length of Valid set is {}".format(len_valid_sets))
    f.write("The length of Train set is {}".format(len_train_sets))
    f.write("The length of Valid set is {}".format(len_valid_sets))
    train_datasets , valid_datasets = torch.utils.data.random_split(mydataset,
                                                                   [len_train_sets, len_valid_sets ])
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=8, shuffle=True, num_workers=0)
    torch.autograd.set_detect_anomaly(True)
    network = Network()
    network.cuda()    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    loss_min = np.inf
    num_epochs = 10
    start_time = time.time()
    for epoch in range(1,num_epochs+1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step, data in enumerate(train_loader, 1):

            images, landmarks = data
            #print('images size: ', images.shape)
            #print('landmark size: ',landmarks.shape)

            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0),-1).cuda() 

            predictions = network(images)
            #print('predictions size: ',predictions.shape)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train/step

            print_overwrite(step, len(train_loader), running_loss, 'train')

        network.eval() 
        with torch.no_grad():
            for step, data in enumerate(valid_loader,1):

                images, landmarks = data
           
                images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0),-1).cuda()

                predictions = network(images)

                # find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')
        f.write('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))

        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(),modelParamName+'.pth') 
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')

    #print('Training Complete')
    print(f"{bcolors.OKCYAN}Training Complete{bcolors.ENDC}")
    print("Total Elapsed Time : {} s".format(time.time()-start_time))
    f.write("Total Complete Elapsed Time : {} s".format(time.time()-start_time))
    f.close()

def lowerfacepred(modelParam,outPath):
    """
    this function is to load the pre-trained model to yield lower face topleft coordinates
    """
    print('running lower face prediction')
    mydataset=lowerfaceDataset(root="Output/lipmark",
                                           transform=
                                           transforms.Compose([
                                            transforms.ToTensor(),                      
                                            GaussNoise(), 
                                            transforms.GaussianBlur((3,5),(1,3)),
                                            #torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
                                            #sigma (float or tuple of python:float (min, max)) – Standard deviation to 
                                            # be used for creating kernel to perform blurring.
                                            #If float, sigma is fixed. If it is tuple of float (min, max), 
                                            #sigma is chosen uniformly at random to lie in the given range.
                                           transforms.Normalize((0.5), (0.5))
                                            ]))
    print('loading full dataset successfully')
    print('the size of dataset is: ', len(mydataset)) #155288
    train_loader = torch.utils.data.DataLoader(mydataset, batch_size=64, shuffle=True, num_workers=0)
    print('the size of dataloader is: ', len(train_loader)) #2427 ;2427x64= 155328 >155288, the very last batch is not 64
    
    with torch.no_grad():
        lowerface= lowerfacelocator(ngpu).to(device)
        #lowerface.cuda()
        lowerface.load_state_dict(torch.load(modelParam))
        lowerface.eval()
    print('loading NETWORK PARAMETERS successfully')

    for _, data in enumerate(train_loader):
        images, labels = data
        for image, label in zip(images,labels):# iterate throu every data point to get the prediction.
            image = image.cuda()
            predictions = (lowerface(image.unsqueeze(0)).detach().cpu() + 0.5) 
            predictions = predictions.view(-1, 1 ,2).numpy()*(640,480)
            predictions = predictions[0].astype(int) #  
            #print(predictions.shape)
            #print(predictions.astype(int))
            np.savetxt('Output/'+outPath+'/'+label, predictions, fmt='%i', delimiter=",")
    """    
    for step in range(1,len(train_loader)+1):
        images, labels = next(iter(train_loader)) # the size of these would  be batch_size 64
        for image, label in zip(images,labels):# iterate throu every data point to get the prediction.
            image = image.cuda()
            predictions = (lowerface(image.unsqueeze(0)).detach().cpu() + 0.5) 
            predictions = predictions.view(-1, 1 ,2).numpy()*(640,480)
            predictions = predictions[0].astype(int) #  
            #print(predictions.shape)
            #print(predictions.astype(int))
            np.savetxt('Output/lowerfacelabel/'+label, predictions, fmt='%i', delimiter=",")

    """ 
    #alternative iterate 
    #for batch in  train_loader:
    #    for i in batch
    
    
    
    

def main():
    print("begin working")   
    #trainMylowerface('lowface3')
    #lowerfacepred('lowface3.pth','lowerlabels')
    trainLipmarkLowerface('LowerFaceLipMark1Left',1)
    print('finished the first one')
    trainLipmarkLowerface('LowerFaceLipMark1Full',0)
    print('finished the second one')
    trainLipmarkLowerface('LowerFaceLipMark1Right',2)
    print('finished the last one')
    

if __name__ == "__main__":
    main()