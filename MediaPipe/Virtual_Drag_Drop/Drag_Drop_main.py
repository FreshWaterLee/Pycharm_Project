import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8)

colorR = (255, 0, 255)

cx, cy, w, h = 50, 50, 100, 100



class DragRect():
    def __init__(self, posCenter, size=[100, 100]):
        self.posCenter = posCenter
        self.size = size


    def update(self, cursor):
        cx, cy = self.posCenter
        w, h= self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor
        else:
            colorR = 255, 0, 255


rectList = []
for x in range(5):
    rectList.append(DragRect([x*125+75, 75]))


while cap.isOpened():
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img)

    if lmList:
        l, _, _ = detector.findDistance(8, 12, img, draw=False)
        if l < 25:
            cursor = lmList[8]
            for rect in rectList:
                rect.update(cursor)
    '''
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
        cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)
    '''
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow('Image', out)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
