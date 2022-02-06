# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 17:54:44 2022

@author: zakariaElghamch
"""

import os
import numpy as np
from PIL import Image
import cv2
import pickle
from LBP import LocalBinaryPatterns as lbp



root = os.path.dirname(os.path.abspath(__file__))
imgs = os.path.join(root,"dataset")

face_detect = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

x= []
y=[]
lb = lbp(100,8)
label_id = {}
cur_id = 0
for racin, dirs, files in os.walk(imgs):
    for file in files:
        if file.endswith("jpg") :      
            path = os.path.join(racin,file)
            label = os.path.basename(racin)
            img = Image.open(path)
            img_mat = np.array(img)
            hist = lb.describe(img_mat)
            x.append(hist)
            y.append(label)
print(len(x))
print(len(x[0]))
print(len(y))            
with open("baseIndex.ss","wb") as f:
    pickle.dump(x,f)
    pickle.dump(y,f)
          