import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 기본 경로 설정
base_dir = './MNIST300'
test_dir = os.path.join(base_dir, 'testing')

# 사용할 클래스
target_classes = ['2', '4', '8']

# ImageDataGenerator (테스트 전용)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# 테스트 데이터 로딩
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(300, 300),
    classes=target_classes,
    class_mode='categorical',
    batch_size=32,
    shuffle=False  # 중요: 결과 정렬을 위해 shuffle=False
)

# 학습된 모델 로드
model = load_model('VGGbest_model_2.keras')

# 모델 평가
loss, accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {accuracy * 100:.6f}%")
print(f"❌ Test Loss: {loss:.4f}")

# 예측값 구하기
pred_probs = model.predict(test_generator)
pred_classes = np.argmax(pred_probs, axis=1)        # 예측 클래스
true_classes = test_generator.classes                # 실제 클래스
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix 시각화
cm = confusion_matrix(true_classes, pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(true_classes, pred_classes, target_names=class_labels))