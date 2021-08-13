import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import os
import cv2
import time

mp_holistic = mp.solutions.holistic ## Holistic model
mp_drawing = mp.solutions.drawing_utils ## MediaPipe 라이브러리를 통해 검출된 포즈, 손 등을 시각화 시켜주는 변수

def mediapipe_detection(image,model): ## 입력 모델에 따라 객체 출력 함수(ex: 손 검출 모델을 넣으면 손이 검출,포즈면 포즈, )
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) ## 모델링에 입력하기위해 이미지의 픽셀 타입 변경
    image.flags.writeable = False ##  사용 불가
    results = model.process(image) ## 입력 모델에 현 프레임을 입력하여 모델에서 검출되는 결과를 변수화
    image.flags.writeable = True ##
    image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR) ## 전처리 이전의 이미지로 복구
    return image, results ## 현 프레임과 검출 결과값 리턴 

cap = cv2.VideoCapture(0) ## 웹캠 연결

with mp_holistic.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        image,results = mediapipe_detection(frame,holistic)

        print(results.pose_landmarks)
        if results.pose_landmarks == None:
            print("검출된 것이 없습니다.")
        else:
            print(len(results.pose_landmarks.landmark))
        cv2.imshow('Opencv Feed', image)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

'''import tensorflow as tf'''