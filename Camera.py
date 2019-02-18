import cv2
import numpy as np
import pytesseract
import imutils

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if img.any():
        config = ('-l eng --oem 1 --psm 3')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        text = pytesseract.image_to_string(img, config=config)
        if text:
            print(text.encode("utf-8"))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (10, 25), font, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y), (x+w, y+h), (255,0,0), 3)
            roi_gray = gray[y: y+h, x: x+h]
            roi_color = img[y: y+h, x: x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        cv2.imshow('img',img)
    else:
        print("video not captured")
        break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
