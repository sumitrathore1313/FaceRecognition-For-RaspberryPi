# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('train/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
font = cv2.FONT_HERSHEY_SIMPLEX
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    im = np.array(img)
    #gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, 'uint8')
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        result = cv2.face.MinDistancePredictCollector()
        recognizer.predict(gray[y: y + h, x: x + w],result, 0)
        Id = result.getLabel()
        conf = result.getDist()
        if(conf<50):
            if(Id==1):
                Id="sumit"
            elif(Id==2):
                Id="sparsh"
        else:
            Id="Unknown"
        #cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
        cv2.putText(im, str(Id), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('im',im) 
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()# -*- coding: utf-8 -*-

