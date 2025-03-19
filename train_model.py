# train_model.py

import numpy as np
from quickdraw import QuickDrawDataGroup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import config

# ----------------------
# QuickDraw 데이터 로드 
# ----------------------
print("QuickDraw 데이터 로딩 중 ...")

categories = config.CATEGORIES

X_list = [] # Datas
y_list = [] # Labels

for label_id, label_name in enumerate(categories):
    data_group = QuickDrawDataGroup(label_name, max_drawings=config.MAX_DRAWINGS)
    drawings = data_group.drawings
    
    # 이미지 변환
    images = np.array([
        np.array(d.get_image(stroke_width=2)
                 .resize((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
                 .convert("L"))
        for d in drawings
    ])
    
    labels = np.full(len(images), label_id, dtype=np.int32)
    
    X_list.append(images)
    y_list.append(labels)

X = np.vstack(X_list).astype("float32") / 255.0
y = np.hstack(y_list).astype("int32")


# 차원 확장
X = np.expand_dims(X, axis=-1)  # (4000, 28, 28, 1)

# ----------------------
# 학습용/검증용 데이터 분할
# ----------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------
# CNN 모델 정의
# ----------------------

"""
max_drawings
image_size
batch_size
epochs
"""


model = keras.Sequential([
    layers.Conv2D(
        filter=32, 
        kernel_size=(3,3), 
        activation='relu', 
        input_shape=(config.IMAGE_SIZE[0],config.IMAGE_SIZE[1],1)
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)), 

    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.dropout(0.5), # overfitting 방지지
    layers.Dense(len(categories), activation='softmax')  # n개의 클래스
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------
# 모델 학습
# ----------------------
print("모델 학습 시작...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

# ----------------------
# 모델 저장
# ----------------------
model.save(config.MODEL_PATH)
print("모델이 'picture_model.h5' 파일로 저장되었습니다!")