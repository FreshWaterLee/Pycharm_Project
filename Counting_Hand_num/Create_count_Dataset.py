import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
## 손의 최대 검출 갯수는 2개일까?
mp_drawing = mp.solutions.drawing_utils
with mp_hands.Hands(max_num_hands=4,min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        ## 손 검출을 위한 전처리 (RGB로 변경해야함)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ## 검출 여부 불리언 세팅
        image.flags.writeable = False

        ## 전처리한 이미지에서 손 검출
        results = hands.process(image)
        
        ## 검출 여부 값 변경(검출을 진행했기때문에)
        image.flags.writeable = True

        ## 이미지 전처리 이전으로 복구
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ## 손 검출 결과 결과값 출력
        #print(results)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                                          )
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(2) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()