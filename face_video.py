import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

id = input("\n Enter user id :") 
sampleNum = 0
while True:
    _, frame = cap.read()   #  ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
            
        sampleNum +=1
        
        cv2.imwrite("dataset/User."+str(id)+ '.' + str(sampleNum)+".jpg",gray[y:y +h, x:x +w])
   
    cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    if sampleNum > 500:
        break
cap.release()
cv2.destroyAllWindows()
