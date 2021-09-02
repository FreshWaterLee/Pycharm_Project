from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import cv2
import os
### 문자 검출을 위한 함수 선언
ocr = PaddleOCR(lang='korean')
font_Path = 'korean.ttf'
#imgPath = os.path.join('images', 'IU2017.jpg')
#img = cv2.imread(imgPath, cv2.IMREAD_COLOR)

cap = cv2.VideoCapture(0) ## 실시간 검출
### 검출 확인
while cap.isOpened():
    ret, frame_ori = cap.read()
    frame = frame_ori.copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(img)
    if len(result) > 0:
        boxes = [res[0] for res in result]
        texts = [res[1][0] for res in result]
        scores = [res[1][1] for res in result]
        img = draw_ocr(img, boxes, texts, scores, font_path=font_Path)
    else:
        print("없습니다!!")
        img = frame_ori.copy()
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
##print(len(result))
#print(result[0])
'''
cv2.destroyAllWindows()'''
