import os
from pathlib import Path
import mediapipe as mp
import pandas as pd
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    min_detection_confidence=0.3   # default is 0.5, lowering it is more lenient
)

LANDMARK_NAMES = {
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

SELECTED_LANDMARKS = (
    list(range(469, 473)) +   # left eye
    list(range(474, 478)) +   # right eye
    [362, 133, 168, 2]        # medial canthi, midpoint, nose tip
)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"No face detected: {image_path}")
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    row = {}
    for idx in SELECTED_LANDMARKS:
        name = LANDMARK_NAMES[idx]
        lm = landmarks[idx]
        row[f"{name}_x"] = lm.x * w
        row[f"{name}_y"] = lm.y * h
        row[f"{name}_z"] = lm.z * w

    return row

def process_dataset(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for label_name, label_value in [("attentive", 1), ("distracted", 0)]:
        folder = os.path.join(input_root, label_name)
        if not os.path.exists(folder):
            print(f"Folder not found, skipping: {folder}")
            continue

        records = []
        images = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        print(f"Processing {len(images)} images in '{label_name}'...")

        for filename in images:
            path = os.path.join(folder, filename)
            row = extract_landmarks(path)
            if row is None:
                continue
            row["label"] = label_value
            records.append(row)

        df = pd.DataFrame(records)
        out_path = os.path.join(output_root, f"{label_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows -> {out_path}")

process_dataset(
    input_root=os.path.join(BASE_DIR, "..", "data", "abstracted"),
    output_root=os.path.join(BASE_DIR, "..", "data", "processed")
)