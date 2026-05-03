import cv2
import mediapipe as mp
import numpy as np
import threading
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ── New Colour Theme (Dark Blue UI) ───────────────────────────
PALETTE = {
    "bg":         (12, 18, 28),
    "panel":      (28, 40, 60),
    "border":     (70, 110, 160),
    "text_main":  (230, 240, 255),
    "text_dim":   (150, 170, 200),
    "accent":     (0, 170, 255),
    "sitting":    (255, 190, 90),
    "standing":   (120, 255, 180),
    "unknown":    (140, 140, 160),
    "skel_joint": (0, 170, 255),
    "skel_bone":  (200, 220, 255),
}

LANDMARK_SPEC = mp_drawing.DrawingSpec(
    color=PALETTE["skel_joint"], thickness=2, circle_radius=4)
CONNECTION_SPEC = mp_drawing.DrawingSpec(
    color=PALETTE["skel_bone"], thickness=2)

quit_flag = False
screenshot_flag = False

def listen_for_quit():
    global quit_flag, screenshot_flag
    while True:
        user_input = input().strip().lower()
        if user_input in ('q', 'quit', ''):
            quit_flag = True
            break
        if user_input == 's':
            screenshot_flag = True

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle
\
def classify_posture(landmarks, frame_height):
    try:
        def pt(id): return landmarks[id.value]

        LS = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
        RS = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        LH = pt(mp_pose.PoseLandmark.LEFT_HIP)
        RH = pt(mp_pose.PoseLandmark.RIGHT_HIP)
        LK = pt(mp_pose.PoseLandmark.LEFT_KNEE)
        RK = pt(mp_pose.PoseLandmark.RIGHT_KNEE)
        LA = pt(mp_pose.PoseLandmark.LEFT_ANKLE)
        RA = pt(mp_pose.PoseLandmark.RIGHT_ANKLE)

        hip_y = (LH.y + RH.y) / 2
        knee_y = (LK.y + RK.y) / 2
        ankle_y = (LA.y + RA.y) / 2
        shoulder_y = (LS.y + RS.y) / 2

        lka = calculate_angle([LH.x, LH.y], [LK.x, LK.y], [LA.x, LA.y])
        rka = calculate_angle([RH.x, RH.y], [RK.x, RK.y], [RA.x, RA.y])
        avg_knee_angle = (lka + rka) / 2

        lha = calculate_angle([LS.x, LS.y], [LH.x, LH.y], [LK.x, LK.y])
        rha = calculate_angle([RS.x, RS.y], [RH.x, RH.y], [RK.x, RK.y])
        avg_hip_angle = (lha + rha) / 2

        body_height_ratio = ankle_y - shoulder_y
        hip_knee_diff = knee_y - hip_y

        sit, std = 0, 0

        if avg_knee_angle < 120: sit += 3
        elif avg_knee_angle > 160: std += 3
        else: sit += 1

        if hip_knee_diff > -0.1: sit += 2
        else: std += 2

        if avg_hip_angle < 120: sit += 2
        else: std += 2

        if body_height_ratio > 0.55: std += 2
        else: sit += 2

        total = sit + std
        if sit > std:
            return "Sitting", round(sit / total, 2), avg_knee_angle
        else:
            return "Standing", round(std / total, 2), avg_knee_angle
    except:
        return "Unknown", 0.0, 0.0

def draw_rect(img, x, y, w, h, color, alpha=1.0):
    if alpha < 1.0:
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)

def put_text(img, text, x, y, scale, color, thickness=1):
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)

def draw_ui(frame, posture, confidence, knee_angle, fps):
    h, w = frame.shape[:2]

    p_color = {
        "Sitting": PALETTE["sitting"],
        "Standing": PALETTE["standing"],
    }.get(posture, PALETTE["unknown"])

    # Header
    draw_rect(frame, 0, 0, w, 60, PALETTE["panel"], alpha=0.9)
    put_text(frame, "POSTURE DETECTION SYSTEM", 20, 35, 0.7, PALETTE["text_main"])

    # Left panel
    draw_rect(frame, 20, 80, 260, 180, PALETTE["panel"], alpha=0.85)

    put_text(frame, "POSTURE", 40, 110, 0.5, PALETTE["text_dim"])
    put_text(frame, posture, 40, 150, 1.0, p_color, 2)

    put_text(frame, "KNEE ANGLE", 40, 185, 0.5, PALETTE["text_dim"])
    put_text(frame, f"{int(knee_angle)} deg", 40, 220, 0.8, PALETTE["text_main"])

    put_text(frame, f"CONFIDENCE: {int(confidence*100)}%", 40, 250, 0.5, PALETTE["accent"])

    # FPS
    put_text(frame, f"{int(fps)} FPS", w-120, 40, 0.5, PALETTE["text_dim"])

    return frame

# Camera
cap = cv2.VideoCapture(0)

listener_thread = threading.Thread(target=listen_for_quit, daemon=True)
listener_thread.start()

prev_time = time.time()
fps = 0

while True:
    if quit_flag:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time))
    prev_time = now

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    posture, confidence, knee_angle = "Unknown", 0.0, 0.0

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_SPEC,
            connection_drawing_spec=CONNECTION_SPEC,
        )

        posture, confidence, knee_angle = classify_posture(
            results.pose_landmarks.landmark, frame.shape[0])

    frame = draw_ui(frame, posture, confidence, knee_angle, fps)

    cv2.imshow("Posture Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()