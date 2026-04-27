import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mobilenet_final.onnx')
HAIR_DIR = os.path.join(BASE_DIR, 'hair_images')

CLASSES = ['하트형', '긴얼굴형', '달걀형', '둥근형', '각진형']

HAIR_TIPS = {
    '남성': {
        '하트형': '이마가 넓고 턱이 좁은 얼굴형이에요. 앞머리로 이마를 가리거나 사이드를 짧게 치는 투블럭 스타일이 잘 어울려요.',
        '긴얼굴형': '얼굴이 길고 좁은 형태예요. 옆 볼륨을 살려주는 가르마 스타일이나 짧은 투블럭이 잘 어울려요.',
        '달걀형': '가장 균형잡힌 얼굴형이에요. 투블럭, 리젠트 등 대부분의 헤어스타일이 잘 어울려요.',
        '둥근형': '얼굴이 둥글고 넓은 형태예요. 위로 볼륨감을 주는 리젠트 스타일이나 긴 앞머리가 잘 어울려요.',
        '각진형': '턱선이 각진 얼굴형이에요. 부드러운 느낌의 내추럴 펌이나 레이어드 스타일이 잘 어울려요.',
    },
    '여성': {
        '하트형': '이마가 넓고 턱이 좁은 얼굴형이에요. 턱 쪽에 볼륨을 주는 레이어드 스타일이 잘 어울려요.',
        '긴얼굴형': '얼굴이 길고 좁은 형태예요. 옆으로 볼륨감을 주는 웨이브 스타일이 잘 어울려요.',
        '달걀형': '가장 균형잡힌 얼굴형이에요. 대부분의 헤어스타일이 잘 어울려요.',
        '둥근형': '얼굴이 둥글고 넓은 형태예요. 높이감을 주는 레이어드 스타일이 잘 어울려요.',
        '각진형': '턱선이 각진 얼굴형이에요. 부드러운 웨이브나 레이어드 스타일이 잘 어울려요.',
    }
}

def preprocess_image(image):
    img = image.resize((224, 224)).convert('RGB')
    img_array = np.array(img).astype(np.float32)
    # MobileNetV2 preprocess_input 수동 구현
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@st.cache_resource
def load_face_model():
    return ort.InferenceSession(MODEL_PATH)

def predict_face_shape(image, session):
    img_array = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: img_array})[0][0]
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx] * 100
    return CLASSES[predicted_idx], confidence, predictions

def get_hair_images(face_shape, gender):
    folder = os.path.join(HAIR_DIR, gender, face_shape)
    images = []
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                images.append(os.path.join(folder, f))
    return images

# ======================== UI ========================
st.set_page_config(page_title='얼굴형 헤어스타일 추천', page_icon='💇', layout='centered')

st.title('💇 얼굴형 기반 헤어스타일 추천')
st.markdown('전면 사진을 업로드하면 얼굴형을 분석하고 어울리는 헤어스타일을 추천해드려요!')
st.divider()

with st.spinner('모델 불러오는 중...'):
    session = load_face_model()

st.subheader('성별을 선택하세요')
gender = st.radio('', ['남성', '여성'], horizontal=True)
st.divider()

uploaded_file = st.file_uploader('전면 사진을 업로드하세요', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('업로드한 사진')
        st.image(image, use_container_width=True)

    with st.spinner('얼굴형 분석 중...'):
        face_shape, confidence, all_probs = predict_face_shape(image, session)

    with col2:
        st.subheader('분석 결과')
        st.markdown(f'### 얼굴형: **{face_shape}**')
        st.markdown(f'신뢰도: **{confidence:.1f}%**')
        st.info(HAIR_TIPS[gender][face_shape])

    st.divider()

    st.subheader('📊 얼굴형 분석 상세')
    for i, cls in enumerate(CLASSES):
        st.progress(float(all_probs[i]), text=f'{cls}: {all_probs[i]*100:.1f}%')

    st.divider()

    st.subheader(f'✂️ {gender} {face_shape} 추천 헤어스타일')
    hair_images = get_hair_images(face_shape, gender)

    if hair_images:
        cols = st.columns(min(len(hair_images), 3))
        for i, img_path in enumerate(hair_images):
            with cols[i % 3]:
                filename = os.path.splitext(os.path.basename(img_path))[0]
                st.image(img_path, caption=filename, use_container_width=True)
    else:
        st.warning(f'hair_images/{gender}/{face_shape}/ 폴더에 이미지를 넣어주세요!')
