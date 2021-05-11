import cv2
import numpy as np
from PIL import Image
import os

#!/usr/bin/python3
import RPi.GPIO as GPIO # dieu khien GPIO
#import time # su dung duoc thoi gian delay

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

#trainning hinh anh nhan dien vs thu vien nhan dien khuon mat
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('recognizer/trainingDatas.yml')

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['none', 'ThDanh'] 

while True:
    ret, frame =  cap.read()
    #frame = cv2.imread("ThanhDanh.jpg")
    #frame = cv2.imread("anhtu.jpg")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        roi_gray = gray[y:y +h, x:x +w]
        
        id, confidence = recognizer.predict(roi_gray)
        
        if confidence < 50:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            
            GPIO.output(17,GPIO.HIGH)
            #time.sleep(0.5)
            #GPIO.output(17,GPIO.LOW)
            #time.sleep(0.5)
            
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,0,0), 2)
            
            GPIO.output(17,GPIO.LOW)
            
        
        #cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,0,0), 1)  
        print(x,y,w,h)
        
        
    # display a frame    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
	    break

#cap.release()
cv2.destroyAllWindows()   
        
