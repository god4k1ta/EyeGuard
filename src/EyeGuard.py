import cv2
import numpy as np
import time
from collections import deque
import os
import pygame

pygame.mixer.init()

base_path = os.path.join(os.path.dirname(__file__), "..", "sounds")

sound_l1 = pygame.mixer.Sound(os.path.join(base_path, "level1.mp3"))
sound_l2 = pygame.mixer.Sound(os.path.join(base_path, "level2.mp3"))
sound_l3 = pygame.mixer.Sound(os.path.join(base_path, "level3.mp3"))

# --- Каскады ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# --- Настройки ---
FRAMES_THRESHOLD = 2
LONG_BLINK_FRAMES = int(fps * 0.29)
LONG_CLOSE_SEC = 1.0
HEAD_ANGLE_THRESHOLD = 15
SMOOTH_ALPHA_FACE = 0.3
SMOOTH_ALPHA_LEVEL = 0.4
SMOOTH_ALPHA_EYE = 0.6
HYSTERESIS_UP = 0.65
HYSTERESIS_DOWN = 0.35
COOLDOWN_SEC = 3

# --- История морганий и углов ---
blink_times = deque()
blink_durations = deque(maxlen=200)
angle_history = deque(maxlen=5)
eye_positions_history = deque(maxlen=5)
smooth_face = None
last_valid_eyes = None  # для линии наклона головы

level_1_state = level_2_state = level_3_state = False
level_1_smooth = level_2_smooth = 0.0
eyes_closed_frames = 0

prev_l1 = prev_l2 = prev_l3 = False
cooldown_l1 = cooldown_l2 = cooldown_l3 = 0

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def detect_eyes(roi_gray, w):
    eyes_detected = 0
    eye_centers = []
    eyes = eye_cascade.detectMultiScale(roi_gray[:roi_gray.shape[0] // 2, :], 1.07, 7, minSize=(15, 15))
    for (ex, ey, ew, eh) in eyes:
        if 0.07 * w < ew < 0.4 * w:
            eyes_detected += 1
            eye_centers.append((ex + ew // 2, ey + eh // 2))
    return eyes_detected, eye_centers, eyes


def hysteresis(value, prev):
    return value > HYSTERESIS_DOWN if prev else value > HYSTERESIS_UP


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    # --- Детекция лица ---
    faces = face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(60, 60))
    if len(faces) == 0:
        prof_r = profile_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60, 60))
        gray_flipped = cv2.flip(gray, 1)
        prof_l_flipped = profile_cascade.detectMultiScale(gray_flipped, 1.05, 3, minSize=(60, 60))
        prof_l = [(gray.shape[1] - x - w, y, w, h) for (x, y, w, h) in prof_l_flipped]
        faces = list(prof_r) + list(prof_l)

    if len(faces) == 0:
        cv2.putText(frame, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("EyeGuard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # --- Сглаживание лица ---
    if smooth_face is None:
        smooth_face = np.array([x, y, w, h], dtype=float)
    else:
        smooth_face = SMOOTH_ALPHA_FACE * smooth_face + (1 - SMOOTH_ALPHA_FACE) * np.array([x, y, w, h])
    x, y, w, h = smooth_face.astype(int)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]

    # --- Детекция глаз ---
    eyes_detected, eye_centers, eyes_rects = detect_eyes(roi_gray, w)
    if eye_centers:
        if eye_positions_history:
            last_pos = eye_positions_history[-1]
            eye_centers_smoothed = [(int(SMOOTH_ALPHA_EYE * xc + (1 - SMOOTH_ALPHA_EYE) * lx),
                                     int(SMOOTH_ALPHA_EYE * yc + (1 - SMOOTH_ALPHA_EYE) * ly))
                                    for (xc, yc), (lx, ly) in zip(eye_centers, last_pos)]
        else:
            eye_centers_smoothed = eye_centers
        eye_positions_history.append(eye_centers_smoothed)
        last_valid_eyes = eye_centers_smoothed[:2]  # сохраняем последние глаза для наклона
    else:
        eye_positions_history.append([])
        if last_valid_eyes is not None:
            eye_centers = last_valid_eyes

    for (ex, ey, ew, eh) in eyes_rects:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    now = time.time()

    # --- Логика морганий ---
    if eyes_detected >= 1:
        if eyes_closed_frames >= FRAMES_THRESHOLD:
            blink_times.append(now)
            blink_durations.append(eyes_closed_frames)
        eyes_closed_frames = 0
    else:
        eyes_closed_frames += 1

    while blink_times and now - blink_times[0] > 60:
        blink_times.popleft()
        blink_durations.popleft()

    long_blinks = [d for d in blink_durations if d >= LONG_BLINK_FRAMES]
    avg_blink_duration = (np.mean(blink_durations) / fps) if blink_durations else 0

    # --- Наклон головы (боковой + вперед/назад) ---
    HEAD_ALPHA = 0.5
    HEAD_CONSEC_FRAMES = 5
    MIN_EYE_DISTANCE = 10
    VERTICAL_THRESHOLD = 0.15  # процент от высоты лица для определения наклона вперед/назад

    head_tilt_detected = False
    head_forward_detected = False

    current_eyes = eye_centers if len(eye_centers) >= 2 else last_valid_eyes

    if current_eyes is not None and len(current_eyes) >= 2:
        (x1, y1), (x2, y2) = current_eyes[:2]
        last_valid_eyes = current_eyes

        # --- Боковой наклон ---
        if abs(x2 - x1) >= MIN_EYE_DISTANCE:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            smooth_angle = HEAD_ALPHA * angle + (1 - HEAD_ALPHA) * angle_history[-1] if angle_history else angle
            angle_history.append(smooth_angle)

            if abs(smooth_angle) > HEAD_ANGLE_THRESHOLD:
                head_frames_counter = head_frames_counter + 1 if 'head_frames_counter' in locals() else 1
                head_tilt_detected = head_frames_counter >= HEAD_CONSEC_FRAMES
            else:
                head_frames_counter = 0
                head_tilt_detected = False

        # --- Наклон вперед/назад --
        eye_center_y = (y1 + y2) / 2
        face_top, face_bottom = 0, roi_gray.shape[0]
        vertical_ratio = (eye_center_y - face_top) / (face_bottom - face_top)

        # если глаза слишком низко → голова вперед; слишком высоко → голова назад
        if vertical_ratio > 0.6 or vertical_ratio < 0.4:
            head_forward_frames = head_forward_frames + 1 if 'head_forward_frames' in locals() else 1
            head_forward_detected = head_forward_frames >= HEAD_CONSEC_FRAMES
        else:
            head_forward_frames = 0
            head_forward_detected = False

        # рисуем линию глаз
        cv2.line(roi_color, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # --- Уровни усталости ---
    level_3_state = (eyes_closed_frames / fps) >= LONG_CLOSE_SEC

    target_l1 = 1 if not level_3_state and len(long_blinks) >= 3 else 0
    level_1_smooth = SMOOTH_ALPHA_LEVEL * level_1_smooth + (1 - SMOOTH_ALPHA_LEVEL) * target_l1
    level_1_state = hysteresis(level_1_smooth, prev_l1)

    target_l2 = 1 if level_1_state and head_tilt_detected and not level_3_state else 0
    level_2_smooth = SMOOTH_ALPHA_LEVEL * level_2_smooth + (1 - SMOOTH_ALPHA_LEVEL) * target_l2
    level_2_state = hysteresis(level_2_smooth, prev_l2)

    # --- Звуки ---
    if level_3_state and not prev_l3 and now - cooldown_l3 > COOLDOWN_SEC:
        sound_l3.play()
        cooldown_l3 = now
    if level_2_state and not prev_l2 and now - cooldown_l2 > COOLDOWN_SEC:
        sound_l2.play()
        cooldown_l2 = now
    if level_1_state and not prev_l1 and now - cooldown_l1 > COOLDOWN_SEC:
        sound_l1.play()
        cooldown_l1 = now

    prev_l1, prev_l2, prev_l3 = level_1_state, level_2_state, level_3_state

    # --- Визуализация ---
    if level_3_state:
        status = "Level 3 - Micro sleep"
        color = (0, 0, 255)
    elif level_2_state:
        status = "Level 2 - Head tilt + fatigue"
        color = (0, 165, 255)
    elif level_1_state:
        status = "Level 1 - Early fatigue"
        color = (0, 255, 255)
    else:
        status = "OK"
        color = (0, 255, 0)

    cv2.putText(frame, f"Blinks: {len(blink_durations)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Long blinks: {len(long_blinks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Avg blink: {avg_blink_duration:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),
                2)
    cv2.putText(frame, f"Current long blink frames: {eyes_closed_frames}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2)
    if head_tilt_detected:
        cv2.putText(frame, "Head tilt detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Status: {status}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("EyeGuard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
