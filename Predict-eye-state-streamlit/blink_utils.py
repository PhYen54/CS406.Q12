# blink_utils.py
import mediapipe as mp
import cv2
from PIL import Image
import time

LEFT_EYE_IDX = [
    33,7,163,144,145,153,154,155,
    133,173,157,158,159,160,161,246
]
RIGHT_EYE_IDX = [
    263,249,390,373,374,380,381,382,
    362,398,384,385,386,387,388,466
]

mp_face = mp.solutions.face_mesh

def crop_eye(img_rgb, landmarks, idxs, w, h, padding=5):
    xs, ys = [], []
    for i in idxs:
        lm = landmarks.landmark[i]
        xs.append(lm.x)
        ys.append(lm.y)

    x_min = int(max(0, min(xs) * w) - padding)
    x_max = int(min(w, max(xs) * w) + padding)
    y_min = int(max(0, min(ys) * h) - padding)
    y_max = int(min(h, max(ys) * h) + padding)

    crop = img_rgb[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None, None

    return Image.fromarray(crop).convert("RGB"), (x_min, y_min, x_max, y_max)


def draw_eye_box(frame, box, state):
    if box:
        x1, y1, x2, y2 = box
        color = (0,255,0) if state=="OPEN" else (0,0,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def update_blink(prev_state, curr_state, t_close, blink_list, threshold=0.5):
    """Return: new_prev, new_t_close, updated_blink_count"""
    current_time = time.time()

    if prev_state == "OPEN" and curr_state == "CLOSED":
        t_close = current_time

    elif prev_state == "CLOSED" and curr_state == "OPEN":
        if t_close is not None:
            duration = current_time - t_close
            if duration < threshold:
                blink_list.append(1)
        t_close = None

    return curr_state, t_close, blink_list
