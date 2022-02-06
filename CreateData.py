# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:04:05 2021

@author: zakariaElghamch
"""

import cv2
camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

Nframes=0
while(True): 
    ret, img = camera.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
        Nframes=Nframes+1 
        cv2.imwrite("dataset/said/said_"+str(Nframes) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif Nframes==600:
        break
camera.release()
cv2.destroyAllWindows()