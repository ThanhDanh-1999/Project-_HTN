# import the necessary packages
import cv2
import numpy as np
from PIL import Image
import os

#!/usr/bin/python3
import RPi.GPIO as GPIO # Controller GPIO

# Use the BCM channel numbers
GPIO.setmode(GPIO.BCM)
# Set up a pin (17) as an output
GPIO.setup(17, GPIO.OUT)

# Use file .xml to face recognition
# Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Use LBPH(LOCAL BINARY PATTERNS HISTOGRAMS) in the OpenCV library.
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Read training data from file.yml created
recognizer.read('recognizer/trainingDatas.yml')

# Initialize the camera and grab a reference to the video capture
cap = cv2.VideoCapture(0)

# Font for label
font = cv2.FONT_HERSHEY_SIMPLEX

# Iniciate id counter
id = 0

# Names related to ids: example ==> ThDanh: id=1,  etc
names = ['none', 'ThDanh'] 

# loop over frames from the video file stream
while True:
    # Grab the frame from the threaded video stream
    ret, frame =  cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray)
    
    #Draw a rectangle around every found face
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        roi_gray = gray[y:y +h, x:x +w]
        
        id, confidence = recognizer.predict(roi_gray)
        
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            
            GPIO.output(17,GPIO.HIGH) #Led_ON
            
        else:
            id = "Unknown"     #if face is not recognized, then print Unknown
            confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,0,0), 2)
            
            GPIO.output(17,GPIO.LOW) #Led_OFF
            
        cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,0,0), 1)  
        print(x,y,w,h)
        
        
    # Display a frame    
    cv2.imshow("Camera", frame)
    
    # Wait for 'q' key was pressed and break from the loop
    if cv2.waitKey(1) & 0xff == ord("q"):
	    break

# Clear the stream in preparation for the next frame
cap.release()
cv2.destroyAllWindows()  
