import cv2
import os
import time
import mediapipe as mp
from mediapipe.tasks.python import vision
from datetime import datetime

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode
mp_image = mp.Image
ImageFormat = mp.ImageFormat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


LABEL = "attentive"
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "webcam", LABEL)

CAPTURE_INTERVAL = 0.5


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(BASE_DIR, "..", "face_landmarker.task")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3
    )

    print(f"Collecting '{LABEL}' data. Press ESC to stop.")
    print(f"Saving to: {OUTPUT_DIR}")

    count = 0
    last_capture = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp_image(image_format=ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect(mp_img)

            # Show whether a face is detected
            face_detected = bool(results.face_landmarks)
            status = f"Face: {'YES' if face_detected else 'NO'} | Saved: {count} | Label: {LABEL}"
            colour = (0, 200, 0) if face_detected else (0, 0, 255)
            cv2.putText(frame, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

            # Save frame at the capture interval, only when face is detected
            now = time.time()
            if face_detected and (now - last_capture) >= CAPTURE_INTERVAL:
                filename = f"{LABEL}_{datetime.now().strftime('%H%M%S%f')}.jpg"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, frame)
                count += 1
                last_capture = now
                print(f"Saved {count}: {filename}")

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone. Collected {count} '{LABEL}' images.")


if __name__ == "__main__":
    main()