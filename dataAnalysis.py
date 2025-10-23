import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# 사용할 클래스
target_classes = ['2', '4', '8']

# 학습 데이터 경로
train_dir = './MNIST300/training'

# 클래스별 이미지 수 카운트
class_counts = {}
for class_name in target_classes:
    class_path = os.path.join(train_dir, class_name)
    image_count = len(os.listdir(class_path))
    class_counts[class_name] = image_count

# 시각화
plt.figure(figsize=(6, 4))
plt.bar(class_counts.keys(), class_counts.values(), color=['skyblue', 'salmon', 'limegreen'])
plt.title('Number of Images per Class in Training Set')
plt.xlabel('Class Label')
plt.ylabel('Number of Images')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    classes=target_classes,
    class_mode='categorical',
    batch_size=32,
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    classes=target_classes,
    class_mode='categorical',
    batch_size=32,
    subset='validation'
)

# 훈련/검증 데이터 수 시각화
counts = {
    'Train': train_generator.samples,
    'Validation': val_generator.samples
}

plt.figure(figsize=(5, 4))
plt.bar(counts.keys(), counts.values(), color=['steelblue', 'orange'])
plt.title('Training vs Validation Sample Counts')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.show()