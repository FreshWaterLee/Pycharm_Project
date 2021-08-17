import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import time

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence_length = 30

label_map ={label:num for num, label in enumerate(actions)}

print(label_map)