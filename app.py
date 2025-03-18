# app.py

import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow import keras
from image_preprocessing import preprocess_image  # 이미지 전처리 함수
from gpt_handler import gpt_message
import config

# ----------------------
# 저장된 모델 로드
# ----------------------
print("모델 로딩 중...")
model = tf.keras.models.load_model(config.MODEL_PATH)
classes = ["cat", "dog"]
print("모델 로드 완료!")

# ----------------------    
# FastAPI 서버 시작
# ----------------------
app = FastAPI()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    - PNG, JPG 이미지를 업로드 받음
    - `image_processor.py`의 `preprocess_image()`를 사용하여 변환
    - 모델 예측 수행 후 JSON 형태로 결과 반환
    """
    # 파일 읽기
    contents = await file.read()





    # `image_preprocessing.py`에서 전처리 수행
    img_array = preprocess_image(contents)

    # 모델 예측
    pred = model.predict(img_array)
    pred_label = np.argmax(pred[0])

    predicted_class = classes[pred_label]
    confidence = float(pred[0][pred_label])

    message = gpt_message(predicted_class, confidence)


    # 4) 결과 반환
    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "message" : message
    }

if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
