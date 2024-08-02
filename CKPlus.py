import tensorflow as tf
from tensorflow import keras

def fix_model(model_path, new_model_path):
    model = keras.models.load_model(model_path)
    
    # 모델의 구성(config)을 JSON 형식으로 가져옵니다
    model_config = model.to_json()
    
    # JSON 문자열에서 'batch_shape'를 제거하고 'input_shape'를 추가합니다
    import json
    model_config_dict = json.loads(model_config)
    for layer in model_config_dict['config']['layers']:
        if 'batch_input_shape' in layer['config']:
            batch_input_shape = layer['config'].pop('batch_input_shape')
            layer['config']['input_shape'] = batch_input_shape[1:]
    
    # 수정된 구성으로 새 모델을 생성합니다
    fixed_model = keras.models.model_from_json(json.dumps(model_config_dict))
    
    # 원래 모델의 가중치를 새 모델에 복사합니다
    fixed_model.set_weights(model.get_weights())
    
    # 새 모델을 저장합니다
    fixed_model.save(new_model_path)

# 나이 모델을 수정합니다
fix_model('C:/Users/user/Document/GitHub/4weeks/new_age_model_fixed.h5', 'C:/Users/user/Documents/GitHub/4weeks/fixed_age_model.h5')

# 성별 모델을 수정합니다
fix_model('C:/Users/user/Document/GitHub/4weeks/new_gender_model_fixed.h5', 'C:/Users/user/Documents/GitHub/4weeks/fixed_gender_model.h5')
