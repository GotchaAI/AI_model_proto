import quickdraw as qd
import torch
import config
import time

# 1. 전체 클래스 불러오는 Dataset (config에 등록된 CATEGORIES)

# 모든 클래스 넣으면 메모리 터짐
class QuickDrawAllDataSet(torch.utils.data.Dataset):
    # 객체 초기화 시 데이터 로드드
    def __init__(self, max_drawings=1000, transform=None):
        self.data = [] 
        self.labels = [] 
        self.transform = transform if transform else transform.ToTensor() 
        self.max_drawings = max_drawings
        

        o1 = time.time()
        print("Quickdraw 데이터 로드 중 ...")
        for label_id, label_name in enumerate(config.CATEGORIES): # 각 Label마다
            print(f" 클래스 [{label_id + 1}]: {label_name} 로드 중 ...")
            data_group = qd.QuickDrawDataGroup(label_name, max_drawings=max_drawings, recognized=True) # 최대 max_drawings 만큼의 데이터 가져옴
            for i in range(data_group.drawing_count): # get_drawing: PIL Image 변환환
                self.data.append(data_group.get_drawing(i)) # 흑백 변환 X, RGB 3채널로
                self.labels.append(label_id)  # 레이블: 클래스 ID
            print(f" 클래스 [{label_id + 1}]: {label_name} 로드 완료!")
        o2 = time.time()
        print("Quickdraw 데이터 로드 완료.")
        print(f"{len(config.CATEGORIES)} 개 클래스 {len(self.data)}개 샘플, 실행 시간 : {o2 - o1} 초")

    
    def __len__(self):
        return len(self.data)

    # 
    def __getitem__(self, index: int):
        """
        특정 index의 데이터 반환
        :return: (이미지 Tensor, 정수형 레이블)
        """
        image = self.data[index]
        label = self.labels[index]
        return self.transform(image), label




# 2. 하나의 클래스에 대해서만 가져오는 DataSet
class QuickDrawDataSet(torch.utils.Dataset):
    def __init__(self, name, max_drawings=1000, transform=None):
        self.index = config.CATEGORIES.index(name) # 해당 클래스의 idx 가져옴
        self.data = []

        print(f" 클래스 [{self.index + 1}]: {name} 로드 중 ...")
        data_group = qd.QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=True) # 최대 max_drawings 만큼의 데이터 가져옴
        for i in range(data_group.drawing_count): # get_drawing: PIL Image 변환
            self.data.append(data_group.get_drawing(i)) # 흑백 변환 X, RGB 3채널로
        print(f" 클래스 [{self.index + 1}]: {name} 로드 완료!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        특정 index의 데이터 반환
        :return: (이미지 Tensor, 정수형 레이블)
        """
        image = self.data[index]
        return self.transform(image), self.index
