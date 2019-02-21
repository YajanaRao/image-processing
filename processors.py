import pytesseract
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from keras.models import load_model

face_cascade = cv2.CascadeClassifier(
    'models/detection_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'models/detection_models/haarcascade_eye.xml')
emotion_classifier = load_model(
    'models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)


def text_detector(img):
    text = pytesseract.image_to_string(img, lang='eng')
    return text


def emotion_detector(img):
    emotion_map = {}
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                      4: 'sad', 5: 'surprise', 6: 'neutral'}
    emotion_prediction = emotion_classifier.predict(img)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_arg = 0
    for emotion_score in emotion_prediction[0]:
        # print('emotion_score:', emotion_score)
        # emotion_label_arg = np.argmax(emotion_score)
        # print('emotion_label_arg:', emotion_label_arg)
        emotion_text = emotion_labels[emotion_arg]
        # print (emotion_text, ':', emotion_score)
        emotion_map[emotion_text] = emotion_score
        emotion_arg += 1
    print(emotion_map)
    return(emotion_text)

def img_processor(img):
    if img.any():

        config = ('-l eng --oem 1 --psm 3')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            roi_gray = gray[y: y+h, x: x+h]
            roi_color = img[y: y+h, x: x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 0), 2)
        return img





