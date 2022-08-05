import numpy as np        # `pip install numpy`
import cv2                # `pip install opencv-python`
import leapuvc            #  Ensure leapuvc.py is in this folder

import json


#https://stackoverflow.com/questions/72706073/attributeerror-partially-initialized-module-cv2-has-no-attribute-gapi-wip-gs
 # for face

#pip install opencv-python==4.5.5.64
#pip install opencv-contrib-python==4.5.5.62 

LeapExposure = 22


cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")



#retval	= cv.face.createFacemarkAAM()	
#retval.loadModel("lbfmodel.yaml")


facemarkLBF = cv2.face.createFacemarkLBF()
facemarkLBF.loadModel("lbfmodel.yaml")
# load calibration file
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
# Start the Leap Capture Thread




# run landmark detector:



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
        #detect face        
        facesl = cascade.detectMultiScale(leftRightImage[0], 1.3, 5)
        facesr = cascade.detectMultiScale(leftRightImage[1], 1.3, 5)
        #determine landmark but model needs rgb face image???
        leftrgb = cv2.cvtColor(leftRightImage[0], cv2.COLOR_GRAY2BGR)
        rightrgb = cv2.cvtColor(leftRightImage[1], cv2.COLOR_GRAY2BGR)
        success, landmarksl = facemarkLBF.fit(leftrgb, facesl) 
        triumph, landmarksr = facemarkLBF.fit(rightrgb, facesr) 
        
        if(success and triumph): #The function cv::drawMarker draws a marker, ONE marker. maybe I am wrong. 
            for landmarkl,landmarkr in zip(landmarksl,landmarksr):
                cv2.drawMarker(leftRightImage[0], (landmarkl[0], landmarkl[1]),(0,0,255), markerType=cv2.MARKER_STAR, 
                markerSize=2, thickness=2, line_type=cv2.LINE_AA)
                cv2.drawMarker(leftRightImage[1], (landmarkr[0], landmarkr[1]),(0,0,255), markerType=cv2.MARKER_STAR, 
                markerSize=2, thickness=2, line_type=cv2.LINE_AA)
                
                
               
        cv2.imshow('Frame L', leftRightImage[0])
        cv2.imshow('Frame R', leftRightImage[1])

cv2.destroyAllWindows()