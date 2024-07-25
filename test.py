import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# 모델 로드
model = load_model('final_model_kaggle_version1/model.h5')

# 이미지 로드 및 전처리
img = Image.open('path/to/your/test/image.jpg')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 예측
prediction = model.predict(img_array)
print(prediction)
