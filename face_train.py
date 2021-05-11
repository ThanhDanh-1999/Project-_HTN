import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
#detector = cv2.CascadeClassifier('/home/pi/Desktop/Face_recognition/haarcascade_frontalface_default.xml')

# Path for face image database
path = 'dataset'
# function to get the images and label data
def getImages(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]   
    
    #print(imagePaths)
    
    faces = []
    ids = []
    
    for imagePath in imagePaths:
    
        faceImg = Image.open(imagePath).convert('L')
        
        faceNp = np.array(faceImg, 'uint8')
        
        print(faceNp)
        
        id = int(imagePath.split('/')[1].split('.')[1])
        
        faces.append(faceNp)
        ids.append(id)
        
        cv2.imshow('Training', faceNp)
        cv2.waitKey(10)
        
    return faces, ids

    
faces, ids = getImages(path)

recognizer.train(faces, np.array(ids))

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')
    
recognizer.save('recognizer/trainingDatas.yml')

cv2.destroyAllWindows()