import tensorflow as tf
import tensorflowjs as tfjs

# JSON 파일과 H5 파일을 이용해 Keras 모델 로드
with open('facial_expression_model_structure.json', 'r') as json_file:
    model_json = json_file.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights('facial_expression_model_weights.h5')

# TensorFlow.js 형식으로 변환 및 저장
tfjs.converters.save_keras_model(model, 'tfjs_model')
