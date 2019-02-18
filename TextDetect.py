import os,sys
import numpy as np
import cv2
import pytesseract



def parse_text(img):
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    # for black text , cv.THRESH_BINARY_INV
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    # dilate , more the iteration more the dilation
    dilated = cv2.dilate(new_img, kernel, iterations=9)

    # for cv2.x.x

    # findContours returns 3 variables for getting contours
    _, contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # for cv3.x.x comment above line and uncomment line below

    #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    return img




cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    if image.any():
        rect = parse_text(image)
        cv2.imshow('img', rect)
    else:
        print("video not captured")
        break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()




