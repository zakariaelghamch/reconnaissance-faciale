# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 20:27:27 2021

@author: zakariaElghamch
"""
import pickle
from LBP import LocalBinaryPatterns as lbp
import cv2
camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
with open("modelKnn", "rb") as fi:
    model = pickle.load(fi)
lb = lbp(100,8) 
while(True): 
    ret, image = camera.read()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),2)
        face_gray=gray[y:y+h,x:x+w]
        hist = lb.describe(face_gray)
        conf = model.predict(hist.reshape(1,-1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = str(conf[0])
        color = (255,255,255)
        stroke = 2
        cv2.putText(image, name, (x-20,y-20), font, 1,color, stroke, cv2.LINE_AA)
    cv2.imshow('frame',image)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()