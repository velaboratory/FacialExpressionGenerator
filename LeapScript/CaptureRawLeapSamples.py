import numpy as np        # `pip install numpy`
import cv2                # `pip install opencv-python`
import leapuvc            #  Ensure leapuvc.py is in this folder
import time
import json


# load calibration file
with open("calibration.json", 'r') as j:
     contents = json.loads(j.read()) 

# Start the Leap Capture Thread
leap = leapuvc.leapImageThread(source=1,resolution = (640,480))
LeapExposure = 25
leap.setExposure(LeapExposure)
leap.setLeftLED(True)
leap.setCenterLED(True)
leap.setRightLED(True)
leap.start()
undistRect = True
TimeVar = time.time()
timeInterval = 3
fileIndex = 30
# Capture images until 'q' is pressed
while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    newFrame, leftRightImage = leap.read()
    if(newFrame):
        if(undistRect):
                maps = contents['left']["undistortMaps"]
                leftRightImage[0] = cv2.remap(leftRightImage[0], np.array(maps[0], dtype=np.int16), # 
                np.array(maps[1], dtype=np.int16), cv2.INTER_LINEAR)
                maps = contents['right']["undistortMaps"]
                leftRightImage[1] = cv2.remap(leftRightImage[1], np.array(maps[0], dtype=np.int16), # 
                np.array(maps[1], dtype=np.int16), cv2.INTER_LINEAR)
        # Display the raw frame
        cv2.imshow('Frame L', leftRightImage[0])
        cv2.imshow('Frame R', leftRightImage[1])
        if  (time.time() - TimeVar) > timeInterval:
                cv2.imwrite('Captures/'+'LeapExposure'+str(LeapExposure)+'File'
							+str(fileIndex)+'left.png',leftRightImage[0])
                cv2.imwrite('Captures/'+'LeapExposure'+str(LeapExposure)+'File'
							+str(fileIndex)+'right.png',leftRightImage[1])
                fileIndex+=1
                TimeVar = time.time() 

cv2.destroyAllWindows()