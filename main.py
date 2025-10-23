import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# 기본 경로 설정
base_dir = './MNIST300'
train_dir = os.path.join(base_dir, 'training')
test_dir = os.path.join(base_dir, 'testing')

# 사용할 클래스만 지정 (2, 4, 8)
target_classes = ['2', '4', '8']

# ImageDataGenerator 설정
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2  # 80% train, 20% val
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# 학습용 데이터 로딩
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    classes=target_classes,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    classes=target_classes,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset='validation'
)

# 테스트 데이터 로딩
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(300, 300),
    classes=target_classes,
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
# VGG16 불러오기 (ImageNet 학습된 가중치 사용, 최상위 분류 레이어는 제거)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

# 기존 VGG16의 가중치는 그대로 사용 (학습 안함)
for layer in base_model.layers:
    layer.trainable = False

# 새 분류기 붙이기
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)  # 클래스 수 = 3 (2, 4, 8)

model = Model(inputs=base_model.input, outputs=predictions)

# 컴파일
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 구조 확인
model.summary()

# 콜백 설정
checkpoint = ModelCheckpoint('VGGbest_model_2.keras', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             verbose=1)
earlystop = EarlyStopping(monitor='val_loss', 
                          patience=5, 
                          restore_best_weights=True)

# 모델 학습
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[checkpoint, earlystop]
)

# 정확도 그래프
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()