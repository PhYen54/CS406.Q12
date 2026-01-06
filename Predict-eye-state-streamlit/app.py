# app.py
import streamlit as st
import cv2
import numpy as np
from blink_model import load_blink_model, predict_eye_state
from blink_utils import (
    crop_eye, draw_eye_box,
    LEFT_EYE_IDX, RIGHT_EYE_IDX,
    mp_face, update_blink
)

st.set_page_config(page_title="Blink Detection App", layout="wide")

# Load model
def load_v2():
    weight_path = r"D:\UIT\Xử lý ảnh\Predict-eye-state-streamlit\model_weight\densenet121-union-64.pt"
    return load_blink_model(weight_path)

model = load_v2()


# -------------------------------
# STREAMLIT UI – TABS
# -------------------------------
tab1, tab2 = st.tabs(["📸 Upload Ảnh", "🎥 Camera Realtime"])



# ============================================================
# 📌 TAB 1 — Upload ảnh và detect
# ============================================================
with tab1:
    st.header("📸 Upload Ảnh – Detect Mắt")

    uploaded = st.file_uploader("Upload ảnh (jpg/png)", type=["jpg", "png", "jpeg"])

    if uploaded is not None:
        img = np.array(cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        with mp_face.FaceMesh(max_num_faces=10, refine_landmarks=True) as fm:
            result = fm.process(rgb)

        if not result.multi_face_landmarks:
            st.error("❌ Không tìm thấy khuôn mặt trong ảnh!")
        else:
            for face_id, lm in enumerate(result.multi_face_landmarks):

                left_img, left_box = crop_eye(rgb, lm, LEFT_EYE_IDX, w, h)
                right_img, right_box = crop_eye(rgb, lm, RIGHT_EYE_IDX, w, h)

                if left_img is None or right_img is None:
                    continue

                left_state, p_left = predict_eye_state(model, left_img)
                right_state, p_right = predict_eye_state(model, right_img)

                # vẽ box
                draw_eye_box(img, left_box, left_state)
                draw_eye_box(img, right_box, right_state)

                # text theo từng mặt
                lx, ly, _, _ = left_box
                cv2.putText(img, f"Face {face_id} L: {left_state} ({p_left:.2f})",
                            (lx, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0) if left_state=="OPEN" else (0,0,255), 2)

                rx1, ry1, _, _ = right_box
                cv2.putText(img, f"Face {face_id} R: {right_state} ({p_right:.2f})",
                            (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0) if right_state=="OPEN" else (0,0,255), 2)


            st.image(img, channels="BGR", caption="Kết quả detect")

# ============================================================
# 📌 TAB 2 — Camera realtime (code bạn có)
# ============================================================
with tab2:
    st.header("🎥 Nhận Diện Nhắm/Mở Bằng Camera")

    # Session states
    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False
    if "left_blinks" not in st.session_state:
        st.session_state.left_blinks = []
    if "right_blinks" not in st.session_state:
        st.session_state.right_blinks = []
    if "prev_left" not in st.session_state:
        st.session_state.prev_left = "OPEN"
    if "prev_right" not in st.session_state:
        st.session_state.prev_right = "OPEN"
    if "t_left" not in st.session_state:
        st.session_state.t_left = None
    if "t_right" not in st.session_state:
        st.session_state.t_right = None

    BLINK_THRESHOLD = 0.5

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Camera"):
            st.session_state.run_camera = True

    with col2:
        if st.button("⏹ Stop Camera"):
            st.session_state.run_camera = False

    FRAME_WINDOW = st.empty()
    INFO = st.empty()

    def run_camera():
        cap = cv2.VideoCapture(0)
        with mp_face.FaceMesh(max_num_faces=10, refine_landmarks=True) as fm:
            while st.session_state.run_camera:
                ok, frame = cap.read()
                if not ok:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = fm.process(rgb)

                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0]

                    left_img, left_box = crop_eye(rgb, lm, LEFT_EYE_IDX, w, h)
                    right_img, right_box = crop_eye(rgb, lm, RIGHT_EYE_IDX, w, h)

                    if left_img and right_img:
                        left_state, p_left = predict_eye_state(model, left_img)
                        right_state, p_right = predict_eye_state(model, right_img)

                        draw_eye_box(frame, left_box, left_state)
                        draw_eye_box(frame, right_box, right_state)

                        # Update blink state
                        st.session_state.prev_left, st.session_state.t_left, st.session_state.left_blinks = \
                            update_blink(st.session_state.prev_left, left_state,
                                         st.session_state.t_left, st.session_state.left_blinks, BLINK_THRESHOLD)

                        st.session_state.prev_right, st.session_state.t_right, st.session_state.right_blinks = \
                            update_blink(st.session_state.prev_right, right_state,
                                         st.session_state.t_right, st.session_state.right_blinks, BLINK_THRESHOLD)

                        # Text info
                        cv2.putText(frame, f"L: {left_state} ({p_left:.2f}) Blink:{len(st.session_state.left_blinks)}",
                                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0,255,0) if left_state=="OPEN" else (0,0,255),2)
                        cv2.putText(frame, f"R: {right_state} ({p_right:.2f}) Blink:{len(st.session_state.right_blinks)}",
                                    (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0,255,0) if right_state=="OPEN" else (0,0,255),2)

                FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()

    if st.session_state.run_camera:
        run_camera()

    INFO.info(f"""
    👁️ **Left Eye Blinks:** {len(st.session_state.left_blinks)}
    👁️ **Right Eye Blinks:** {len(st.session_state.right_blinks)}
    """)
