import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")


labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


while True:
    success, img = cap.read()   # Camera
    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False) # Detects 1 hand
    if hands:
        hand = hands[0] # Select hand
        x,y,w,h = hand['bbox'] # New box for just the hand

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255 #Matrix 300x300 with data type from 0-255
        imgCrop = img[y-offset:y + h+offset , x-offset:x + w+offset] # Crop box to have same size

        aspectRatio = h/w

        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w) #round ceiling
            imgResize = cv2.resize(imgCrop, (wCal, imgSize)) #Resize image
            wGap = math.ceil((imgSize-wCal)/2) # Gap to push forward to center image
            imgWhite[:, wGap:wCal+wGap] = imgResize #Overlay image on top of white Img
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            
            


        else:
            k = imgSize/w
            hCal = math.ceil(k*h) #round ceiling
            imgResize = cv2.resize(imgCrop, (imgSize,hCal )) #Resize image
            hGap = math.ceil((imgSize - hCal)/2) # Gap to push forward to center image
            imgWhite[hGap:hCal + hGap, :] = imgResize #Overlay image on top of white Img
            prediction, index = classifier.getPrediction(imgWhite, draw=False) # Get prediction with model and dont show draw
    
        cv2.putText(imgOutput, labels[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
        cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset), (255,0,255),4)




    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)