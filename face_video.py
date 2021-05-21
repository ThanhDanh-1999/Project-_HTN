# import the necessary packages
import cv2      
import numpy as np
import os

# Use file .xml to face recognition
# Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the camera and grab a reference to the video capture
cap = cv2.VideoCapture(0)

id = input("\n Enter user id :") 

sampleNum = 0

# loop over frames from the video file stream
while True:

    # Grab the frame from the threaded video stream
    _, frame = cap.read()   
    # ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw a rectangle around every found face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Save the result image
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
            
        sampleNum +=1
        
        cv2.imwrite("dataset/User."+str(id)+ '.' + str(sampleNum)+".jpg",gray[y:y +h, x:x +w])
   
    # Display a frame    
    cv2.imshow('camera', frame)

    # Wait for 'q' key was pressed and break from the loop
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    if sampleNum > 500:
        break

# Clear the stream in preparation for the next frame
cap.release()
cv2.destroyAllWindows()
