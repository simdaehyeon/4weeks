from tensorflow.keras.models import load_model

# 모델 로드
model = load_model('C:/Users/user/Documents/GitHub/4weeks/models/model_v6_23.hdf5')

# 모델 저장
model.save('C:/Users/user/Documents/GitHub/4weeks/models/v6_model')
