# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('카메라 온')
    while True:
        ret,img = cap.read()
        cv2.imshow('sss',img)
        if cv2.waitKey(1) == ord('q'):
            break
else:
    print('카메라 오프!!')

'''
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
'''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
