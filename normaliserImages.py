# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:16:27 2021

@author: zakariaElghamch
"""
import os
from typing import Counter
from PIL import Image
import numpy as np
import cv2
import pickle
root = os.path.dirname(os.path.abspath(__file__))
rootImages = os.path.join(root,"dataset/with_mask")

face_detect = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

cur_id = 0
N=0
for racin, dirs, files in os.walk(rootImages):
    for file in files:
        path = os.path.join(racin,file)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces=face_detect.detectMultiScale(gray,1.3, 5)
        if faces is None :
            for (x,y,w,h) in faces:
                N=N+1
                cv2.imwrite(rootImages+"/"+str(N) + ".jpg", gray[y:y+h,x:x+w])
            
        
        os.remove(path)
        
            
          
            
           
