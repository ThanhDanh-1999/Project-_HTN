# import the necessary packages
import cv2
import numpy as np
from PIL import Image
import os

# Use LBPH(LOCAL BINARY PATTERNS HISTOGRAMS) in the OpenCV library.
recognizer = cv2.face.LBPHFaceRecognizer_create()


# Path for face image database
path = 'dataset'

# function to get the images and label data
def getImages(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]   
    
    #print(imagePaths)
    
    faces = []
    ids = []
    
    for imagePath in imagePaths:
    
        faceImg = Image.open(imagePath).convert('L')   # convert it to grayscale
        
        faceNp = np.array(faceImg, 'uint8')
        
        print(faceNp)
        
        id = int(imagePath.split('/')[1].split('.')[1])
        
        # update the list of id
        faces.append(faceNp)
        ids.append(id)
        
        # Display a face training
        cv2.imshow('Training', faceNp)
        cv2.waitKey(10)
        
    return faces, ids

    
faces, ids = getImages(path)

recognizer.train(faces, np.array(ids))

# Save the model into recognizer/trainingDatas.yml
if not os.path.exists('recognizer'):
    os.makedirs('recognizer')
recognizer.save('recognizer/trainingDatas.yml')

# Clear the stream in preparation for the next frame
cv2.destroyAllWindows()
