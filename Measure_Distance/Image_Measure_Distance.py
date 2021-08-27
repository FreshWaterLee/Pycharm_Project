import cv2
import numpy as np
import Utilss as ut

webcam =False
path = '1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)

scale = 3
wP = 210 * scale
hP = 297 * scale




while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    img, conts = ut.getContours(img, minArea=50000, filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        imgWarp = ut.WarpImg(img, biggest, wP, hP)
        imgContours2, conts2 = ut.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=True)
        cv2.imshow('A4', imgContours2)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()