import cv2
import os
import mediapipe as mp
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode
mp_image = mp.Image
ImageFormat = mp.ImageFormat


def main():
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} missing! Download: curl -L -o {model_path} 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'")
        return
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3
    )
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame first (so you see a mirror image)
            frame_flipped = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)

            image = mp_image(image_format=ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect(image)

            if results.face_landmarks:
                h, w = rgb_frame.shape[:2]
                for landmarks in results.face_landmarks:
                    # Left eye (green)
                    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    for idx in left_eye_indices:
                        lm = landmarks[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame_flipped, (x, y), 2, (0, 255, 0), -1)
                    
                    # Right eye (green)
                    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    for idx in right_eye_indices:
                        lm = landmarks[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame_flipped, (x, y), 2, (0, 255, 0), -1)
                    
                    # Left iris (red)
                    left_iris_indices = [468, 469, 470, 471, 472]
                    for idx in left_iris_indices:
                        lm = landmarks[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame_flipped, (x, y), 3, (0, 0, 255), -1)
                    
                    # Right iris (red)
                    right_iris_indices = [473, 474, 475, 476, 477]
                    for idx in right_iris_indices:
                        lm = landmarks[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame_flipped, (x, y), 3, (0, 0, 255), -1)

            cv2.imshow('Face Landmarker', frame_flipped)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            # Feedback Area
            cv2.putText(
                frame,
                "Live Feedback",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
