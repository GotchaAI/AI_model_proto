from collections import OrderedDict
import time, config
from torch.autograd import Variable
import imgproc
import torch
import cv2
import craft_utils
import numpy as np




def copyStateDict(state_dict):
    """
    Pytorch 모델의 가중치를 받아 "module" 접두사 제거
    멀티 GPU 환경에서 학습된 가중치를 싱글 GPU에서 활용할 수 있도록
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict




def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    """
    CRAFT 모델을 실행하여 텍스트 감지 수행
    Args:
        net: 모델
        image: np.ndarray
        text_threshold: 
    """
    print("starting text recognition...")
    t0 = time.time()

    # 입력 이미지를 적절한 크기로 resizing
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, config.CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=config.MAG_RATIO)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized) # 데이터 정규화
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    if cuda:
        x = x.cuda()

    # forward pass, CRAFT 모델 실행
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map 
    score_text = y[0,:,:,0].cpu().data.numpy() # 텍스트 확률 맵 : 텍스트가 있을 확률이 높은 영역
    score_link = y[0,:,:,1].cpu().data.numpy() # 단어 또는 문장이 서로 연결된 확률

    # refine link
    if refine_net is not None: # refiner network 사용
        with torch.no_grad(): # 연결 점수를 더욱 정밀하게
            y_refiner = refine_net(y, feature) # refiner 결과로 score_link를 더 자세하게 얻음
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly) # 텍스트 감지 결과(score_text, score_link를 바운딩 박스, 다각형으로 변환)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h) # 바운딩 박스, 다각형 좌표를 원본 이미지 크기에 맞게 보정정
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy() # 결과 시각화
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img) # 확률 맵을 컬러 이미지로 변환

    print("completed text recognition.")
    print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text