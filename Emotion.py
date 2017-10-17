import cv2
import numpy as np
faceCascade=cv2.CascadeClassifier("D:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")
lefteyeCascade=cv2.CascadeClassifier("D:\opencv\sources\data\haarcascades\haarcascade_mcs_lefteye.xml")
righteyeCascade=cv2.CascadeClassifier("D:\opencv\sources\data\haarcascades\haarcascade_mcs_righteye.xml")
img=cv2.imread("alice.png")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(img_gray,1.3,4,(60,60),(300,300,cv2.HAAR))
