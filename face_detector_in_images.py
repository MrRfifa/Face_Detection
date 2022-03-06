import cv2
from random import randrange

#load some pre-trained data on face frontals from opencv
trained_face_data= cv2.CascadeClassifier('frontal_face.xml')

#choose image to detect faces in
img=cv2.imread('rdj.jpg')

#convert the image to greyscale to make it easier 
greyscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates= trained_face_data.detectMultiScale(greyscaled_img)

#draw rectangles around the faces 352 135 474 474

for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),10)
#print(face_coordinates)

#display the image with the faces
cv2.imshow('rfifa face detector',img)
#waitkey pauses the execution until a key pressed
cv2.waitKey()
print("\ncode comleted")