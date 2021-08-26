import cv2
import os
import time
import uuid

IMAGES_PATH = 'workspace/images'

labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']

number_imgs = 15

for label in labels:
    cap = cv2.VideoCapture(0)
    print('Collecting image for {}'.format(label))
    time.sleep(2)
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        image_name = os.path.join(IMAGES_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(image_name, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()