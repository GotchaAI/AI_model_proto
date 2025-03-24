# app.py
import uvicorn
from fastapi import FastAPI, File, UploadFile
from image_preprocessing import preprocess_image  # 이미지 전처리 함수
from gpt_handler import gpt_message
import config
import torch_models
import torch
import torch.nn.functional as F
import time

# ----------------------
# 저장된 모델 로드
# ----------------------
print("모델 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch_models.CNNModel(output_classes = len(config.CATEGORIES))
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device)) 
model.to(device)
model.eval()  # 평가 모드
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
    img = preprocess_image(contents)

    o1 = time.time()
    print("모델 예측중 ....")
    with torch.no_grad():
        outputs = model(img)  # 모델 추론
        probabilities = F.softmax(outputs, dim=1)  # 확률 변환
        top3_prob, top3_indices = torch.topk(probabilities, 3)  # 상위 3개 예측 가져오기
    o2 = time.time()
    print(f"모델 예측 걸린 시간 : {o2-o1:.2f}초.")

    # message = gpt_message(predicted_class, confidence)

    # 4) 결과 반환
    return {
        "filename": file.filename,
        "1st_predicted_class": config.CATEGORIES[top3_indices[0][0].item()],
        "1st_confidence": top3_prob[0][0].item() * 100,
        "2nd_predicted_class": config.CATEGORIES[top3_indices[0][1].item()],
        "2nd_confidence": top3_prob[0][1].item() * 100,
        "3rd_predicted_class": config.CATEGORIES[top3_indices[0][2].item()],
        "3rd_confidence": top3_prob[0][2].item() * 100,
    }

if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
