import io
import numpy as np
import matplotlib.pyplot as plt
import torch
from craft import CRAFT
import cv2

from fastapi import HTTPException
import config
from collections import OrderedDict
from text_recog import test_net, copyStateDict

model = CRAFT()

print("Loading model...")
model.load_state_dict(copyStateDict(torch.load('craft_mlt_25k.pth', map_location='cpu')))
model.eval()



def preprocess_image(image_bytes: bytes, image_size=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])):
    """
    - 업로드된 이미지 파일의 바이트 데이터를 받아 처리
    - 1. 흑백 변환
    - 2. 크기 조정 (기본: 28x28)
    - 3. NumPy 배열로 변환 후 정규화 (0~1)
    - 4. CNN 입력 형태로 차원 확장
    - 예외 발생 시 FastAPI의 HTTPException을 발생시켜 오류 반환
    """

    try:

        # byte data to NumPy ndarray
        image_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)  # 흑백 변환

        bboxes, _, _ = test_net(
            net=model,
            image=img, 
            text_threshold=config.TEXT_THRESHOLD, 
            link_threshold=config.LINK_THRESHOLD, 
            low_text=config.LOW_TEXT, 
            cuda=config.CUDA, 
            poly=False
        )

        for box in bboxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)  # 흰색으로 채움


        # 리사이즈 (모델 입력 크기와 동일해야 함)
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)


        # 3) NumPy 배열 변환 및 정규화
        img_array = img.astype(np.float32) / 255.0  # 스케일링 (0~1)

        # 4) CNN 입력 차원 확장 (28,28,1) -> (1,28,28,1)
        img_array = np.expand_dims(img_array, axis=-1)  # (28,28,1)
        img_array = np.expand_dims(img_array, axis=0)   # (1,28,28,1)
        return img_array
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 오류: {str(e)}")
    
