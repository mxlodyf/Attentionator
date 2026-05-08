import cv2
import os
import time
import joblib
import pandas as pd
import mediapipe as mp
from datetime import datetime
from tkinter import Tk, Label, Button, Canvas
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode
mp_image = mp.Image
ImageFormat = mp.ImageFormat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBCAM_DIR = os.path.join(BASE_DIR, "data", "webcam")
MODEL_DIR = os.path.join(BASE_DIR, "model")

CAPTURE_INTERVAL = 0.5       # seconds between image saves
COLLECTION_DURATION = 60     # seconds per label
BACKGROUND_COLOUR = "#000000"

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


# --- Step 1: Show instruction window and wait for user to click ready ---
def show_instruction_window(title, instructions):
    ready = {"value": False}

    window = Tk()
    window.title(title)
    window.geometry("500x300")
    window.configure(bg=BACKGROUND_COLOUR)
    window.resizable(False, False)

    Label(window, text=title, font=("Arial", 16, "bold"),
          bg=BACKGROUND_COLOUR).pack(pady=(30, 10))

    Label(window, text=instructions, font=("Arial", 12),
          bg=BACKGROUND_COLOUR, wraplength=440, justify="left").pack(pady=10)

    def on_ready():
        ready["value"] = True
        window.destroy()

    Button(window, text="I'm Ready", font=("Arial", 12),
           command=on_ready, width=20).pack(pady=20)

    window.mainloop()
    return ready["value"]


# --- Step 2: Collect webcam images for one label ---
def collect_images(label, landmarker):
    output_dir = os.path.join(WEBCAM_DIR, label)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCollecting '{label}' data for {COLLECTION_DURATION} seconds...")

    count = 0
    last_capture = time.time()
    session_start = time.time()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - session_start
        remaining = COLLECTION_DURATION - elapsed

        # Stop after collection duration
        if remaining <= 0:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect(mp_img)

        face_detected = bool(results.face_landmarks)

        # Countdown and status overlay
        cv2.putText(frame, f"Time remaining: {remaining:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Collecting: {label}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0) if face_detected else (0, 0, 255), 2)
        cv2.putText(frame, f"Saved: {count} images", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not face_detected:
            cv2.putText(frame, "No face detected - adjust position", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Save frame at interval when face is detected
        now = time.time()
        if face_detected and (now - last_capture) >= CAPTURE_INTERVAL:
            filename = f"{label}_{datetime.now().strftime('%H%M%S%f')}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            count += 1
            last_capture = now

        cv2.imshow(f"Calibration - {label}", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} '{label}' images.")
    return count


# --- Step 3: Extract landmarks from collected images into a CSV ---
def extract_landmarks_to_csv(label, mp_face_mesh):
    image_dir = os.path.join(WEBCAM_DIR, label)
    records = []

    images = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    print(f"Extracting landmarks from {len(images)} '{label}' images...")

    for filename in images:
        path = os.path.join(image_dir, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        results = mp_face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        row = {}
        for idx, name in SELECTED_LANDMARKS.items():
            lm = landmarks[idx]
            row[f"{name}_x"] = lm.x * w
            row[f"{name}_y"] = lm.y * h
            row[f"{name}_z"] = lm.z * w

        row["label"] = 1 if label == "attentive" else 0
        records.append(row)

    df = pd.DataFrame(records)
    os.makedirs(MODEL_DIR, exist_ok=True)
    csv_path = os.path.join(WEBCAM_DIR, f"{label}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows -> {csv_path}")
    return df


# --- Step 4: Train Random Forest on extracted CSVs ---
def train_model(attentive_df, distracted_df):
    print("\nTraining model...")

    df = pd.concat([attentive_df, distracted_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.1f}%")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "random_forest.joblib"))

    # Save feature names
    with open(os.path.join(MODEL_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(X.columns.tolist()))

    print("Model saved.")


# --- Show a simple progress window during processing ---
def show_processing_window(message):
    window = Tk()
    window.title("Processing")
    window.geometry("500x150")
    window.configure(bg=BACKGROUND_COLOUR)
    window.resizable(False, False)
    Label(window, text=message, font=("Arial", 13),
          bg=BACKGROUND_COLOUR, wraplength=440).pack(pady=50)
    window.update()
    return window


# --- Main calibration flow ---
def run():
    model_path = os.path.join(BASE_DIR, "face_landmarker.task")

    landmarker_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3
    )

    # --- Attentive collection ---
    show_instruction_window(
        title="Calibration — Step 1 of 2: Attentive",
        instructions=(
            "We need to learn what YOU look like when you're focused.\n\n"
            "When you click 'I'm Ready', look at the screen naturally "
            "for 60 seconds — as if you're reading or working.\n\n"
            "Try slight natural head movements, but keep your gaze on the screen."
        )
    )

    with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
        collect_images("attentive", landmarker)

    # --- Distracted collection ---
    show_instruction_window(
        title="Calibration — Step 2 of 2: Distracted",
        instructions=(
            "Now we need to learn what YOU look like when you're distracted.\n\n"
            "When you click 'I'm Ready', look away from the screen for 60 seconds.\n\n"
            "Mix it up — look left, right, down at your phone, turn your head, "
            "close your eyes, look at the ceiling."
        )
    )

    with FaceLandmarker.create_from_options(landmarker_options) as landmarker:
        collect_images("distracted", landmarker)

    # --- Extract landmarks ---
    processing_window = show_processing_window(
        "Processing your calibration data...\nThis will only take a moment."
    )

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        min_detection_confidence=0.3
    )

    attentive_df = extract_landmarks_to_csv("attentive", mp_face_mesh)
    distracted_df = extract_landmarks_to_csv("distracted", mp_face_mesh)
    mp_face_mesh.close()

    processing_window.destroy()

    # --- Train model ---
    training_window = show_processing_window(
        "Training your personal attention model...\nThis will only take a moment."
    )

    train_model(attentive_df, distracted_df)
    training_window.destroy()

    # --- Done ---
    done_window = Tk()
    done_window.title("Calibration Complete")
    done_window.geometry("500x200")
    done_window.configure(bg=BACKGROUND_COLOUR)
    done_window.resizable(False, False)
    Label(done_window, text="Calibration Complete!",
          font=("Arial", 16, "bold"), bg=BACKGROUND_COLOUR).pack(pady=(30, 10))
    Label(done_window, text="Your personal model has been trained.\nLaunching Attentionator...",
          font=("Arial", 12), bg=BACKGROUND_COLOUR).pack(pady=10)
    done_window.after(2500, done_window.destroy)
    done_window.mainloop()