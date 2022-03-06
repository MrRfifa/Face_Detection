import cv2 
from random import randrange

trained_face_data= cv2.CascadeClassifier('frontal_face.xml')

#to capture video from the cam (0) or a video cv2.VideoCapture('full_path/video_name.ext')
webcam=cv2.VideoCapture(0)

while True:

    #read the current frame
    successful_frame_read , frame   =  webcam.read()
    #convert the image to greyscale to make it easier 
    greyscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #detect faces
    face_coordinates= trained_face_data.detectMultiScale(greyscaled_img)
    #draw rectangles around the faces

    for (x,y,w,h) in face_coordinates:
     cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(0,256),randrange(0,256),randrange(0,256)),10)

    #display the image with the faces
    cv2.imshow('Rfifa face detector',frame)
    #waitkey pauses the execution until a key pressed
    key=cv2.waitKey(1)
    #stop if Q key is pressed
    if key==81 or key==113:
        break
    
#Release the VideoCapture object
webcam.release()
print('\n face detection completed')
