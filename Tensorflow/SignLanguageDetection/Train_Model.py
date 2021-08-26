import os

WORKSPACE_PATH = './workspace'
SCRIPT_PATH = './script'
APIMODEL_PATH = 'C:\models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
'''
labels = [
    {'name': 'hello', 'id': 1},
    {'name': 'yes', 'id': 2},
    {'name': 'no', 'id': 3},
    {'name': 'thank', 'id': 4},
    {'name': 'iloveyou', 'id': 5},
]

with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
#os.system("python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}")
'''

##!python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}
## Train Record 작성 명령문
##!python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x{IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}
## Test Record 작성 명령문


CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
print(config)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

## PipeLine을 원하는 옵션으로 변경
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
pipeline_config.model.ssd.num_classes = 5 ## 본인이 작성한 Custom 모델의 라벨링 수
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0' ## Train 참고자료
pipeline_config.train_config.fine_tune_checkpoint_type = "detection" 
pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']
config_text = text_format.MessageToString(pipeline_config)

## 위에 설정한 Custom PipeLine 저장
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text)

#
# python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=5000
# 트레이닝 실행 코드(첫번째는 model_Main(tf2) 코드, 모델 저장위치, 참고할 pipeLine 위치, Step은 트레이닝 횟수,