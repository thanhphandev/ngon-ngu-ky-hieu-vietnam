import cv2
import mediapipe as mp
import sys
import time
import warnings
import streamlit as st

# Import các module xử lý
from utils.feature_extraction import extract_features
from utils.strings import ExpressionHandler
from utils.tts import TextToSpeech
from utils.model import ASLClassificationModel
from utils.visualizer import Visualizer
from config import MODEL_NAME, MODEL_CONFIDENCE, PREDICTION_CONFIDENCE_THRESHOLD

# Bỏ qua các cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="ASL Recognition Pro", layout="wide", page_icon="🖐️")

st.markdown("""
    <style>
        .big-font {
            color: #e76f51 !important;
            font-size: 50px !important;
            font-weight: bold;
            border: 2px solid #fcbf49;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            background-color: #ffffff;
        }
        .stProgress > div > div > div > div {
            background-color: #2a9d8f;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HÀM LOAD TÀI NGUYÊN
# ==========================================
@st.cache_resource
def load_ai_model():
    """Load model AI"""
    return ASLClassificationModel.load_model(f"models/{MODEL_NAME}")

@st.cache_resource
def load_visualizer():
    """Load công cụ vẽ"""
    return Visualizer()

# Khởi tạo
try:
    model = load_ai_model()
    visualizer = load_visualizer()
except Exception as e:
    st.error(f"⚠️ Lỗi khởi tạo: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR & CẤU HÌNH
# ==========================================
st.sidebar.title("🔧 Bảng Điều Khiển")

run_camera = st.sidebar.checkbox("📷 Bật Camera", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Cấu hình MediaPipe")
detection_confidence = st.sidebar.slider("Độ nhạy phát hiện (Detection)", 0.0, 1.0, MODEL_CONFIDENCE, 0.05)
tracking_confidence = st.sidebar.slider("Độ nhạy theo dõi (Tracking)", 0.0, 1.0, MODEL_CONFIDENCE, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Giọng nói (TTS)")
tts_enabled = st.sidebar.checkbox("Bật đọc kết quả", value=False)
tts_engine_choice = st.sidebar.selectbox("Công cụ đọc", ["pyttsx3 (Offline)", "gTTS (Vietnamese, Online)"], index=0)
min_interval = st.sidebar.slider("Khoảng cách đọc (giây). Khuyến nghị 2 giây", 1.0, 5.0, 2.0, 0.5)

# Xử lý TTS Session
if 'tts' not in st.session_state:
    st.session_state.tts = None
    st.session_state.tts_engine = None

desired_engine = 'pyttsx3' if 'pyttsx3' in tts_engine_choice else 'gtts'

if tts_enabled:
    if st.session_state.tts is None or st.session_state.tts_engine != desired_engine:
        try:
            with st.spinner("Đang khởi tạo giọng nói..."):
                st.session_state.tts = TextToSpeech(engine=desired_engine, lang='vi')
                st.session_state.tts_engine = desired_engine
        except Exception as e:
            st.sidebar.error(f"Lỗi TTS: {e}")
            tts_enabled = False
elif not tts_enabled and st.session_state.tts is not None:
    st.session_state.tts = None

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🎥 Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("### 🧠 Phân tích AI")
    prediction_placeholder = st.empty()
    
    st.markdown("#### Độ tin cậy (Confidence)")
    confidence_bar = st.progress(0)
    confidence_text = st.empty()
    
    st.markdown("---")
    fps_display = st.empty()

# ==========================================
# 5. LOGIC XỬ LÝ CAMERA (LOOP)
# ==========================================
if run_camera:
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    
    cap = cv2.VideoCapture(0)
    expression_handler = ExpressionHandler()
    prev_time = 0

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence) as face_mesh, \
         mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence) as hands:

        while cap.isOpened() and run_camera:
            success, image = cap.read()
            if not success:
                st.warning("Không tìm thấy camera.")
                break

            # Tính FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            fps_display.metric("FPS", f"{int(fps)}")

            # Xử lý hình ảnh
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. Detect
            face_results = face_mesh.process(image)
            hand_results = hands.process(image)

            # 2. Vẽ (Sử dụng Visualizer mới)
            image.flags.writeable = True
            image = visualizer.draw_landmarks(image, face_results, hand_results)

            # 3. Dự đoán
            # Chỉ dự đoán nếu phát hiện được tay hoặc mặt
            if face_results.multi_face_landmarks or hand_results.multi_hand_landmarks:
                try:
                    feature = extract_features(mp_hands, face_results, hand_results)
                    
                    # Dùng hàm mới predict_with_confidence
                    label, confidence = model.predict_with_confidence(feature)
                    
                    # --- LOGIC MỚI: DUAL CONFIDENCE THRESHOLD ---
                    # Nếu độ tin cậy thấp hơn ngưỡng cho phép -> Coi là "binh_thuong" (Idle)
                    if confidence < PREDICTION_CONFIDENCE_THRESHOLD:
                        label = "binh_thuong"
                    
                    expression_handler.receive(label)
                    ui_text = expression_handler.get_message()

                    # Cập nhật UI
                    prediction_placeholder.markdown(f'<div class="big-font">{ui_text}</div>', unsafe_allow_html=True)
                    
                    # Cập nhật thanh Confidence
                    confidence_bar.progress(float(confidence))
                    confidence_text.text(f"Độ chính xác: {confidence*100:.1f}%")

                    # Đọc giọng nói
                    if tts_enabled and st.session_state.tts:
                        speech_text = expression_handler.get_speech_message()

                        # Do not speak if label is "binh_thuong"
                        if label != "binh_thuong" and speech_text.strip():
                            st.session_state.tts.speak_if_allowed(speech_text, min_interval=min_interval)

                except Exception as e:
                    # print(f"Error: {e}") # Debug only
                    pass
            else:
                # Nếu không có landmarks (không người, không tay), reset UI
                prediction_placeholder.markdown(f'<div class="big-font">...</div>', unsafe_allow_html=True)
                confidence_bar.progress(0)
                confidence_text.text("Đang chờ tín hiệu...")

            # Hiển thị
            video_placeholder.image(image, channels="RGB", use_column_width=True)

    cap.release()
    # cv2.destroyAllWindows() # Không cần thiết trên Streamlit Cloud và gây lỗi với headless
else:
    st.info("👋 Hãy bật camera để bắt đầu trải nghiệm.")