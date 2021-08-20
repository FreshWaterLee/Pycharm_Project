import mediapipe as mp
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence_length = 30
no_sequences = 30


## 라벨링 및 랜드마크(검출 포인트) 불러오기
label_map ={label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)

Y = to_categorical(labels).astype(int)
## 테스트, 트레인 데이터 분류
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size= 0.05)

### 트레이닝 시작
log_dir =os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
#model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])

res = model.predict(x_test)
actions[np.argmax(res[4])]

actions[np.argmax(y_test[4])]

model.save('Action_0817.h5')
del model
model = load_model('Action_0817.h5')
yhat =model.predict(x_test)
ytrue = np.argmax(y_test,axis=1).tolist()
yhat = np.argmax(yhat,axis=1).tolist()
multilabel_confusion_matrix(ytrue,yhat)
print(accuracy_score(ytrue,yhat))