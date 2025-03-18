import io
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from fastapi import HTTPException
import config
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    empty_cuda_cache
)

refine_net = load_refinenet_model(cuda=config.CUDA)
craft_net = load_craftnet_model(cuda=config.CUDA)



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
        # 이미지 로드 및 흑백 변환
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        show_image(img)



        # read image for craft
        img_data = read_image(image_bytes)


        print("텍스트 검출/마스킹 중...")
        # perform text prediction
        prediction_res = get_prediction(
            image = img_data,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=config.TEXT_THRESHOLD,
            link_threshold=config.LINK_THRESHOLD,
            low_text=config.LOW_TEXT,
            cuda=config.CUDA,
            long_size=config.LONG_SIZE
        )

        show_image(img)
        boxes= prediction_res['boxes']
        draw = ImageDraw.Draw(img)
        for box in boxes:
            box = [(int(point[0]), int(point[1])) for point in box]

            draw.polygon(box, fill=255)
        show_image(img)


        print('텍스트 검출/마스킹 완료.')
        print(prediction_res['times'])


        # 리사이즈 (모델 입력 크기와 동일해야 함)
        img = img.resize(image_size)


        # 3) NumPy 배열 변환 및 정규화
        img_array = np.array(img) / 255.0  # 스케일링 (0~1)

        # 4) CNN 입력 차원 확장 (28,28,1) -> (1,28,28,1)
        img_array = np.expand_dims(img_array, axis=-1)  # (28,28,1)
        img_array = np.expand_dims(img_array, axis=0)   # (1,28,28,1)
        return img_array
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 오류: {str(e)}")
    

    
def show_image(image):
    """
    Args:
        image (PIL.Image): 출력할 이미지
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()