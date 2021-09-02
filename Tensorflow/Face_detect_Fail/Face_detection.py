from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os

protoPath = "deploy.prototxt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#vs = VideoStream(src=0).start()
time.sleep(2.0)
#fps = FPS().start()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]
    '''imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()'''
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                      swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
