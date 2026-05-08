import cv2
import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import mediapipe as mp
from datetime import datetime
from mediapipe.tasks.python import vision

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode
mp_image = mp.Image
ImageFormat = mp.ImageFormat

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SESSION_DIR = os.path.join(BASE_DIR, "sessions")

# --- Landmark Config ---
SELECTED_LANDMARKS = {
    469: "left_iris_1",
    470: "left_iris_2",
    471: "left_iris_3",
    472: "left_iris_4",
    474: "right_iris_1",
    475: "right_iris_2",
    476: "right_iris_3",
    477: "right_iris_4",
    362: "left_canthus",
    133: "right_canthus",
    168: "between_eyes",
    2:   "nose_tip",
}

DISTRACTION_ALERT_THRESHOLD = 65


def extract_landmarks_from_frame(frame, results):
    if not results.face_landmarks:
        return None
    h, w = frame.shape[:2]
    landmarks = results.face_landmarks[0]
    row = {}
    for idx, name in SELECTED_LANDMARKS.items():
        lm = landmarks[idx]
        row[f"{name}_x"] = lm.x * w
        row[f"{name}_y"] = lm.y * h
        row[f"{name}_z"] = lm.z * w
    return row


def format_for_model(row):
    df = pd.DataFrame([row])
    df = df[feature_names]
    return df


def predict_attention(df_row):
    return model.predict(df_row)[0]


def draw_landmark_dots(frame, results):
    # Draws eye and iris landmark dots on the frame
    if not results.face_landmarks:
        return
    h, w = frame.shape[:2]
    for landmarks in results.face_landmarks:
        for idx in [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
        for idx in [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)


def draw_overlay(frame, prediction, distraction_timer, alert_active):
    h, w = frame.shape[:2]

    if prediction == 1:
        label, colour = "Attentive", (0, 200, 0)
    elif prediction == 0:
        label, colour = "Distracted", (0, 0, 255)
    else:
        label, colour = "No Face", (180, 180, 180)

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 2)

    if prediction == 0 and distraction_timer > 0:
        cv2.putText(frame, f"Distracted for: {distraction_timer:.1f}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if alert_active:
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
        cv2.putText(frame, "REFOCUS! You've been distracted.", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def save_session(session_data):
    os.makedirs(SESSION_DIR, exist_ok=True)
    filename = f"session_{session_data['start_time'].replace(':', '-').replace(' ', '_')}.json"
    filepath = os.path.join(SESSION_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)
    print(f"Session saved -> {filepath}")


def main():

    model_path = os.path.join(BASE_DIR, "face_landmarker.task")
    if not os.path.exists(model_path):
        print(f"ERROR: face_landmarker.task missing!")
        return

    # Load model here instead of at import time
    global model, feature_names
    model = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_names.txt"), "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
        
    # model_path = os.path.join(BASE_DIR, "face_landmarker.task")
    # if not os.path.exists(model_path):
    #     print(f"ERROR: face_landmarker.task missing!")
    #     return

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3
    )

    # Session tracking
    session_start = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_attentive = 0.0
    total_distracted = 0.0
    distraction_count = 0
    distraction_timer = 0.0
    alert_active = False
    last_frame_time = time.time()
    last_prediction = None

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            delta = now - last_frame_time
            last_frame_time = now

            # Flip and convert
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp_image(image_format=ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect(mp_img)

            # Extract landmarks and predict
            prediction = None
            row = extract_landmarks_from_frame(frame, results)
            if row is not None:
                df_row = format_for_model(row)
                prediction = predict_attention(df_row)

            # Update session timers
            if prediction == 0:
                distraction_timer += delta
                total_distracted += delta
                if last_prediction != 0:
                    distraction_count += 1
                if distraction_timer >= DISTRACTION_ALERT_THRESHOLD:
                    alert_active = True
            elif prediction == 1:
                total_attentive += delta
                if distraction_timer > 0:
                    distraction_timer = 0.0
                    alert_active = False

            last_prediction = prediction

            # Draw everything onto the same frame
            draw_landmark_dots(frame, results)
            draw_overlay(frame, prediction, distraction_timer, alert_active)

            cv2.imshow("Attentionator", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    # Save session summary
    session_duration = time.time() - session_start
    session_data = {
        "start_time": start_time_str,
        "session_duration_seconds": round(session_duration, 2),
        "total_attentive_seconds": round(total_attentive, 2),
        "total_distracted_seconds": round(total_distracted, 2),
        "distraction_count": distraction_count,
        "time_on_task_percent": round((total_attentive / session_duration) * 100, 1) if session_duration > 0 else 0
    }

    print("\n--- Session Summary ---")
    for key, value in session_data.items():
        print(f"{key}: {value}")

    save_session(session_data)


if __name__ == "__main__":
    main()