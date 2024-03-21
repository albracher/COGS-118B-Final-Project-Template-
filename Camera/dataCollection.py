import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data/Y"
counter = 0

while True:
    success, img = cap.read()   # Camera
    hands, img = detector.findHands(img) # Detects 1 hand
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
        else:
            k = imgSize/w
            hCal = math.ceil(k*h) #round ceiling
            imgResize = cv2.resize(imgCrop, (imgSize,hCal )) #Resize image
            hGap = math.ceil((imgSize - hCal)/2) # Gap to push forward to center image
            imgWhite[hGap:hCal + hGap, :] = imgResize #Overlay image on top of white Img

        cv2.imshow("ImageWhite",imgWhite)



    #cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg",imgWhite) #save image when press "S"
        print(counter)