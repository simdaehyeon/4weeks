import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 경로
ck_plus_path = 'C:/Users/user/Desktop/ck+48'

# 감정 라벨
emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

# 데이터 로드 함수
def load_data(data_path, emotion_labels):
    images = []
    labels = []
    for label in emotion_labels:
        label_path = os.path.join(data_path, label)
        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (224, 224))  
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  
                labels.append(label)
    return np.array(images), np.array(labels)

# CK+ 데이터 로드
images, labels = load_data(ck_plus_path, emotion_labels)

# 라벨 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# 데이터 정규화
images = images.astype('float32') / 255.0

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# 모델 정의
input = Input(shape=(224, 224, 3))

cnn1 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(input)
cnn1 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(cnn1)
cnn1 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(cnn1)
cnn1 = MaxPool2D(pool_size=3, strides=2)(cnn1)
cnn1 = BatchNormalization()(cnn1)

cnn2 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(cnn1)
cnn2 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(cnn2)
cnn2 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(cnn2)
cnn2 = MaxPool2D(pool_size=3, strides=2)(cnn2)
cnn2 = BatchNormalization()(cnn2)

cnn3 = Conv2D(256, kernel_size=3, activation='relu', padding='same')(cnn2)
cnn3 = Conv2D(256, kernel_size=3, activation='relu', padding='same')(cnn3)
cnn3 = Conv2D(256, kernel_size=3, activation='relu', padding='same')(cnn3)
cnn3 = MaxPool2D(pool_size=3, strides=2)(cnn3)
cnn3 = BatchNormalization()(cnn3)

cnn4 = Conv2D(512, kernel_size=3, activation='relu', padding='same')(cnn3)
cnn4 = Conv2D(512, kernel_size=3, activation='relu', padding='same')(cnn4)
cnn4 = Conv2D(512, kernel_size=3, activation='relu', padding='same')(cnn4)
cnn4 = MaxPool2D(pool_size=3, strides=2)(cnn4)
cnn4 = BatchNormalization()(cnn4)

dense = Flatten()(cnn4)
dense = Dropout(0.5)(dense)
dense = Dense(1024, activation='relu')(dense)
dense = Dropout(0.5)(dense)
dense = Dense(1024, activation='relu')(dense)
output = Dense(len(emotion_labels), activation='softmax', name='emotion')(dense)

model = Model(input, output)
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 데이터 증강
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# 모델 학습
model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=50)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# 모델 저장
model.save('C:/Users/user/Documents/GitHub/4weeks/emotion_model_with_bn.h5')
