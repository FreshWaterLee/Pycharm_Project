import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import os
import cv2
import time

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # MediaPipe 라이브러리를 통해 검출된 포즈, 손 등을 시각화 시켜주는 변수

def mediapipe_detection(image,model): # 입력 모델에 따라 객체 출력 함수(ex: 손 검출 모델을 넣으면 손이 검출,포즈면 포즈, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 모델링에 입력하기위해 이미지의 픽셀 타입 변경
    image.flags.writeable = False
    results = model.process(image) # 입력 모델에 현 프레임을 입력하여 모델에서 검출되는 결과를 변수화
    image.flags.writeable = True
    image =cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # 전처리 이전의 이미지로 복구
    return image, results
    # 현 프레임과 검출 결과값 리턴

def draw_landmarks(image,results): # 검출된 랜드마크(모델에 따라 검출된 포인트(손, 얼굴, 신체등))를 시각화 해주는 함수
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # 얼굴 랜드마크 시각화
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # 포즈 랜드마크 시각화(신체)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # 왼손으로 검출된 포인트 시각화
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # 오른손으로 검출된 포인트 시각화

def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1)
                              )

def extract_keypoint(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


# Path for exported data ,numpy arrays
DATA_PATH = os.path.join('MP_Data') #

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# hello
## 0
## 1
## 2
## ...
## 29
# thanks
##

# i love you




cap = cv2.VideoCapture(0) ## 웹캠 연결
with mp_holistic.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5) as holistic: 
    ## Holistic 모델의 메모리를 임시로 할당 받음(종료시 반납 or 한번의 반복문이 종료되면 반납일수도??)
    ## 모델 설정 및 holistic이라는 변수로 지정
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        if results.pose_landmarks == None:
            print("검출된 것이 없습니다.")
        else:
            draw_styled_landmarks(image, results)
            points = extract_keypoint(results)
            for action in actions:
                for sequence in range(no_sequences):
                    try:
                        os.makedirs((os.path.join(DATA_PATH,action,str(sequence))))
                    except:
                        pass

        cv2.imshow('Opencv Feed', image)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

'''import tensorflow as tf'''