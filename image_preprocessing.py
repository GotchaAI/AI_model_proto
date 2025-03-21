import io
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image, ImageDraw
from fastapi import HTTPException
import config
import torch
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction
)

print("CRAFT 모델 로딩중..")
refine_net = load_refinenet_model(cuda=config.CUDA)
craft_net = load_craftnet_model(cuda=config.CUDA)
print("CRAFT 모델 로딩 완료!")

encode_image = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    #   T.Normalize(mean=[0.485, 0.456, 0.406],
    #               std=[0.229, 0.224, 0.225]),
    T.Normalize(0.5, 0.5)
])



def preprocess_image(image_bytes: bytes):
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
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # show_image(img)



        # read image for craft
        img_data = read_image(image_bytes)


        print("텍스트 검출 중...")
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
        boxes= prediction_res['boxes']
        print('텍스트 검출 완료.')
        print(f'텍스트 검출 걸린 시간: {sum(prediction_res["times"].values()):.2f}초.')




        # show_image(img)
        boxes= prediction_res['boxes']
        draw = ImageDraw.Draw(img)
        for i, box in enumerate(boxes):
            print(f"바운딩 박스 {i+1} 마스킹")
            box = [(int(point[0]), int(point[1])) for point in box]
            print(box)
            draw.polygon(box, fill=(255, 255, 255))
        # show_image(img)



        return encode_image(img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    
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