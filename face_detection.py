import cv2 as cv
import time
import pickle
import os
import numpy as np

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv.face.LBPHFaceRecognizer_create()
cap = cv.VideoCapture(0)

#img = cv.imread("sam.jpg")
recognizer.read("trainz.yml")
labels = {"person_name": 1}
with open("labels.pickles", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

while True:
    
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
    
    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        #print(roi)
        id_, conf = recognizer.predict(roi_gray)
        if conf >=45 and conf <= 85:                                                                                                                                                      
            print(id_)
            print(labels[id_]) 
            color = (100, 45, 30)
            cv.putText(frame, labels[id_], (x, y), cv.FONT_HERSHEY_PLAIN, 2, color, 2)
        else:
            color = (0, 0, 100)
            cv.putText(frame, "UnKnown", (x, y), cv.FONT_HERSHEY_PLAIN, 2, color,2)
    cv.imshow("feed", frame)

    k = 27
    if cv.waitKey(1) & 0xFF == k:
        c = time.time()
        r = round(c)
        print(r)
        path1 = r"C:\Users\Samuel Mensah\Firstpy\Face_data\ChrisB"
        path2 = r"C:\Users\Samuel Mensah\Firstpy\Face_data\Samuel"
        if labels[id_] == "ChrisB":
            filename = os.path.join(path1, str(r) + ".jpg")
            cv.imwrite(filename, roi)
        elif labels[id_] == "Samuel":
            filename = os.path.join(path2, str(r) + ".jpg")
            cv.imwrite(filename, roi)
        else:
            filename = str(r) + "UnKnown.jpg"
            cv.imwrite(filename, roi) 
        break


cap.release()
cv.destroyAllWindows()