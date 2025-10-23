# MNIST 숫자 분류 모델 (VGG16 전이학습)

## 📖 프로젝트 개요

이 프로젝트는 MNIST 손글씨 숫자 이미지 데이터셋 중 숫자 **2, 4, 8**만을 분류하는 딥러닝 모델을 개발하는 것을 목표로 합니다. ImageNet으로 사전 학습된 **VGG16 모델**을 기반으로 **전이학습(Transfer Learning)** 기법을 적용하여 높은 분류 정확도를 달성했습니다.

## 📊 데이터셋

* **원본 데이터:** MNIST 손글씨 숫자 이미지
* **사용 데이터:** 숫자 2, 4, 8에 해당하는 이미지만 사용.
* **이미지 특징:** 각 이미지는 **300x300 픽셀** 크기의 **컬러(RGB)** 이미지로 변환되었습니다.
* **데이터 분할:**
    * **학습 데이터 (Training Set):** 14,122장 (전체 학습 데이터의 80%).
    * **검증 데이터 (Validation Set):** 3,529장 (전체 학습 데이터의 20%).
    * **테스트 데이터 (Testing Set):** 2,988장.
* **데이터 로딩 및 전처리:**
    * `ImageDataGenerator`를 사용하여 디렉토리에서 이미지를 불러오고 VGG16 모델에 맞는 형식으로 전처리(`preprocess_input`)를 수행했습니다.
    * 학습 데이터는 `validation_split=0.2` 옵션을 사용하여 80%는 학습에, 20%는 검증에 사용하도록 자동 분할했습니다.
    * 클래스 라벨은 `class_mode='categorical'`을 통해 원-핫 인코딩 형식으로 변환했습니다.

** 데이터 분포 시각화:**
<img width="548" height="368" alt="Image" src="https://github.com/user-attachments/assets/9dff2207-f56a-4ffe-b2ce-0cd2d146f7e6" />

## 🛠️ 모델 구조

* **베이스 모델:** ImageNet으로 사전 학습된 VGG16 모델(`include_top=False`)을 사용했습니다. VGG16의 합성곱 레이어들은 이미지의 특징을 추출하는 데 효과적입니다.
* **전이학습:** VGG16의 가중치는 동결(`layer.trainable = False`)하여 사전 학습된 특징 추출 능력을 유지했습니다.
* **커스텀 분류기:** VGG16 출력 위에 새로운 분류 레이어를 추가했습니다.
    * `GlobalAveragePooling2D`: 공간 정보를 평균내어 벡터화합니다.
    * `Dense(256, activation='relu')`: 256개의 뉴런을 가진 완전 연결 레이어.
    * `Dropout(0.5)`: 과적합 방지를 위해 50%의 뉴런을 랜덤하게 비활성화합니다.
    * `Dense(3, activation='softmax')`: 최종 출력 레이어로, 3개 클래스(2, 4, 8)에 대한 확률을 출력합니다.

## ⚙️ 학습 과정

* **Optimizer:** `Adam` (learning_rate=1e-4).
* **Loss Function:** `categorical_crossentropy` (다중 클래스 분류에 적합).
* **Metrics:** `accuracy`.
* **Epochs:** 15 (EarlyStopping 콜백에 의해 조기 종료될 수 있음).
* **Batch Size:** 32.
* **Callbacks:**
    * `ModelCheckpoint`: 검증 정확도(`val_accuracy`)가 가장 높은 모델(`VGGbest_model_2.keras`)을 저장합니다.
    * `EarlyStopping`: 검증 손실(`val_loss`)이 5 epoch 동안 개선되지 않으면 학습을 조기 종료합니다.

## 📈 결과

* **최종 테스트 정확도:** **99.53%**.
* **최종 테스트 손실:** 0.0172.

### 학습 곡선 (Accuracy & Loss)
(PDF 보고서의 학습 곡선 이미지를 여기에 추가하세요.)

* **정확도:** 학습 초반 급격히 상승하여 빠르게 95% 이상 도달 후 점진적으로 개선되어 약 99% 이상에 수렴합니다. 학습 정확도와 검증 정확도 간 차이가 거의 없어 과적합이 발생하지 않았습니다.
* **손실:** 학습 초반 급격히 감소 후 점차 안정적으로 수렴합니다. 학습 손실과 검증 손실 곡선이 유사하게 움직여 과적합이 없음을 보여줍니다.

### 성능 평가 (Confusion Matrix & Classification Report)
(PDF 보고서의 Confusion Matrix와 Classification Report 이미지를 여기에 추가하세요.)

* **Confusion Matrix:** 대각선 상의 값(정답)이 매우 높고, 비대각선 값(오답)이 매우 낮아 모델이 각 클래스를 매우 잘 구별함을 알 수 있습니다. 숫자 8을 2로 잘못 예측한 경우가 5건으로 가장 많았지만, 전체적으로 오분류는 매우 적습니다.
* **Classification Report:** 모든 클래스(2, 4, 8)에서 Precision, Recall, F1-score가 0.99 또는 1.00으로 매우 높게 나타났습니다. 이는 모델이 각 숫자를 매우 정확하게 분류하고 있음을 의미합니다.

## 🚀 실행 방법

1.  **저장소 복제:**
    ```bash
    git clone [https://github.com/chakihwan/mnist-classification-vgg16.git](https://github.com/chakihwan/mnist-classification-vgg16.git)
    cd mnist-classification-vgg16
    ```
2.  **데이터셋 준비:**
    * `MNIST300` 데이터셋 압축을 해제하여 프로젝트 루트 디렉토리에 배치합니다. 디렉토리 구조는 다음과 같아야 합니다:
        ```
        mnist-classification-vgg16/
        ├── MNIST300/
        │   ├── training/
        │   │   ├── 2/
        │   │   ├── 4/
        │   │   └── 8/
        │   └── testing/
        │       ├── 2/
        │       ├── 4/
        │       └── 8/
        ├── main.py
        ├── evaluate_model.py
        ├── dataAnalysis.py
        └── README.md
        ```
3.  **필요 라이브러리 설치:**
    (직접 설치)
    ```bash
    pip install tensorflow matplotlib scikit-learn numpy
    ```
4.  **(선택) 데이터 분석 스크립트 실행:**
    ```bash
    python dataAnalysis.py
    ```
5.  **모델 학습 스크립트 실행:**
    ```bash
    python main.py
    ```
    * 학습이 진행되며, 가장 성능이 좋은 모델이 `VGGbest_model_2.keras` 파일로 저장됩니다.
    * 학습 완료 후 정확도 및 손실 그래프가 표시됩니다.
6.  **모델 평가 스크립트 실행:**
    ```bash
    python evaluate_model.py
    ```
    * 저장된 `VGGbest_model_2.keras` 모델을 로드하여 테스트 데이터셋으로 평가합니다.
    * 테스트 정확도, 손실, Confusion Matrix, Classification Report가 출력됩니다.
