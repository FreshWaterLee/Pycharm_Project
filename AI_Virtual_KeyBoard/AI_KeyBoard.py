import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Controller ## 파이썬을 이용한 키보드 컨트롤러 라이브러리

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

detector = HandDetector(detectionCon=0.8)
buttonList = []
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

finalText = ""
keyboard = Controller()

'''
def drawAll(img,buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    return img'''

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 40, y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0 )[mask]
    return out

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

while cap.isOpened():
    success, img = cap.read()
    img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_AREA)
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img) ## 손가락 좌표데이터, 손 바운딩 박스 좌표
    
    for i in range(len(keys)): ## 버튼 리스트 생성
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key))
    img = drawAll(img, buttonList) ## 위에서 생성한 버튼리스트를 이용해 버튼 시각화
    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            if x < lmList[8][0] < x+w and y < lmList[8][1] < y+h : ## [8][0]은 엄지손가락 좌표
                cv2.rectangle(img, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 4)

                l, _, _ = detector.findDistance(8, 12, img, draw=False) ## 두 좌표간 거리를 구하는 함수
                print('Distance is {}'.format(l))
                if l < 40: ## 두 손가락의 거리가 일정 거리 이하일때 선택으로 인식
                    cv2.rectangle(img, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 4)
                    if len(finalText) == 0 : ## 처음 문자를 입력할때
                        finalText += button.text
                        keyboard.press(button.text)
                    elif finalText[-1] == button.text: # 한개의 버튼에서 중복 입력을 통한 메모리 덤프 방지
                        pass
                    else: ## 다른 문자를 선택했을때
                        finalText += button.text
                        keyboard.press(button.text)
    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
