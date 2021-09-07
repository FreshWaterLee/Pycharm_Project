import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cvzone

fpsReader = cvzone.FPS()
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while cap.isOpened():
    success, img = cap.read()
    img = detector.findHands(img)
    hands = detector.findPosition(img)
    print(len(hands[0]))
    if len(hands[0]) > 0 & len(hands[1]) > 0:
        print('양손이 존재')
    else:
        print(" ")

    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()