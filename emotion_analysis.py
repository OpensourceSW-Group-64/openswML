import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import pipeline
import streamlit as st

# --- 1. 환경 설정 ---
BASE_DIR = "C:/Miniconda3/envs/oss"
DATASET_DIR = os.path.join(BASE_DIR, "FER2013_dataset")  # 데이터셋 경로
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")  # 모델 저장 경로

# --- 2. 데이터 로드 및 전처리 ---
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# 데이터 증강 및 전처리
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# --- 3. 모델 정의 ---
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),  # RGB 입력
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 모델 학습 (필요 시 주석 해제)
if not os.path.exists(MODEL_PATH):
    model = create_model()
    model.fit(train_generator, validation_data=test_generator, epochs=10)
    model.save(MODEL_PATH)  # 모델 저장

# --- 4. 모델 로드 ---
model = load_model(MODEL_PATH)  # 저장된 모델 로드
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- 5. Hugging Face 텍스트 감정 분석 ---
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- 6. Streamlit 대시보드 ---
st.title("Emotion Analysis Dashboard")

# 텍스트 감정 분석
user_text = st.text_input("Enter some text:")
if user_text:
    sentiment = sentiment_analyzer(user_text)
    st.write(f"Text Sentiment: {sentiment[0]['label']}")

# 얼굴 감정 분석
st.write("Facial Emotion Recognition Activated")
cap = cv2.VideoCapture(0)

# Streamlit 컨테이너 생성
frame_container = st.empty()  # 프레임을 표시할 컨테이너
stop_button = st.button("Stop", key="stop_button_unique")  # 종료 버튼

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video")
        break

    # OpenCV로 얼굴 감지 및 감정 분석
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0

        # 데이터 타입을 uint8로 변환
        face = (face * 255).astype('uint8')  # OpenCV는 uint8 형식 필요

        # 그레이스케일 이미지를 RGB로 변환
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)
        emotion = emotions[np.argmax(prediction)]

        # 얼굴 영역 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Streamlit으로 프레임 표시
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Streamlit은 RGB 형식 필요
    frame_container.image(frame_rgb, channels="RGB")  # 업데이트

    # 종료 버튼 클릭 시 루프 중단
    if stop_button:
        break

cap.release()
