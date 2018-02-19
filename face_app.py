import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import csv

cap = cv2.VideoCapture(0)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    i=0;
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color,(50,50))
        X = roi_color
        X = X.reshape(1, 3, 50, 50).astype('float32')
        score = loaded_model.predict(X)
        num=np.argmax(score)
        if num==0:
            #print("Aakash")
            cv2.putText(frame,'Aakash',(x,y), font, 2,(0,0,255),4,cv2.LINE_AA)
        if num==1:
            #print("Aishwarya")
            cv2.putText(frame,'Aishwary',(x,y), font, 2,(0,0,255),4,cv2.LINE_AA)
        if num==2:
            #print("Animesh")
            cv2.putText(frame,'Animesh',(x,y), font, 2,(0,0,255),4,cv2.LINE_AA)
        if num==3:
            #print("Onkar")
            cv2.putText(frame,'Onkar',(x,y), font, 2,(0,0,255),4,cv2.LINE_AA)
        if num==4:
            #print("Sanjuksha")
            cv2.putText(frame,'Sanjuksha',(x,y), font, 2,(0,0,255),4,cv2.LINE_AA)
        i=i+1;

    if i==0:
        cv2.putText(frame,'No Face',(100,100), font, 2,(0,0,255),4,cv2.LINE_AA)
    # Our operations on the frame come here

    # Display the resulting frame
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
