from pathlib import Path
import mediapipe as mp
import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
image_path = BASE_DIR / "data/abstracted/attentive/confused_0020.jpg"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True    # enables iris landmarks 468-477
)

SELECTED_LANDMARKS = (
    list(range(469, 473)) +   # left eye
    list(range(474, 478)) +   # right eye
    [362, 133, 168, 2]        # medial canthi, midpoint, nose tip
)

def test_one_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("No face detected!")
        return

    landmarks = results.multi_face_landmarks[0].landmark

    # Draw each selected landmark as a dot and label it
    for idx in SELECTED_LANDMARKS:
        lm = landmarks[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (px, py), 3, (0, 255, 0), -1)          # green dot
        cv2.putText(image, str(idx), (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Also print pixel values to console
        print(f"Landmark {idx}: x={px}, y={py}, z={lm.z * w:.4f}")

    cv2.imshow("Landmark Test", image)
    cv2.waitKey(0)   # press any key to close
    cv2.destroyAllWindows()

test_one_image(str(image_path))