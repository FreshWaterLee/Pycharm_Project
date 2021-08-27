import cv2
import numpy as np

webcam =False
path = '1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3,1920)
cap.set(4,1080)

while True:
    success, img = cap.read()

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()