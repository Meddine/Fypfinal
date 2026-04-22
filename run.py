import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------
# Settings
# ----------------------------
WIDTH, HEIGHT = 1920, 1200

# Change these if the projector is on a second screen
PROJECTOR_X = 1920   # example: projector is to the right of a 1920px main display
PROJECTOR_Y = 0

SHOW_DEBUG_CAMERA_VIEW = True
SHOW_PROJECTOR_OUTPUT = True

# ----------------------------
# Load calibration files
# ----------------------------
M = np.load("M.npy")
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

# ----------------------------
# MediaPipe Tasks setup
# ----------------------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ----------------------------
# Camera
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

# ----------------------------
# Windows
# ----------------------------
if SHOW_PROJECTOR_OUTPUT:
    cv2.namedWindow("Projector Output", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Projector Output", PROJECTOR_X, PROJECTOR_Y)
    cv2.setWindowProperty("Projector Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

if SHOW_DEBUG_CAMERA_VIEW:
    cv2.namedWindow("Debug Warped View", cv2.WINDOW_NORMAL)

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue

    # Undistort camera frame
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Warp camera into tabletop/projector coordinate space
    warped = cv2.warpPerspective(undistorted, M, (WIDTH, HEIGHT))

    # Use a copy for overlay debugging
    debug_view = warped.copy()

    # Use a separate black canvas for projector skeleton only
    projector_view = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    x1 = int(hand_landmarks[start_idx].x * WIDTH)
                    y1 = int(hand_landmarks[start_idx].y * HEIGHT)
                    x2 = int(hand_landmarks[end_idx].x * WIDTH)
                    y2 = int(hand_landmarks[end_idx].y * HEIGHT)

                    # draw on debug overlay
                    cv2.line(debug_view, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # draw on projector-only canvas
                    cv2.line(projector_view, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw landmark points
            for landmark in hand_landmarks:
                x = int(landmark.x * WIDTH)
                y = int(landmark.y * HEIGHT)

                cv2.circle(debug_view, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(projector_view, (x, y), 5, (0, 255, 0), -1)

    if SHOW_DEBUG_CAMERA_VIEW:
        cv2.imshow("Debug Warped View", debug_view)

    if SHOW_PROJECTOR_OUTPUT:
        cv2.imshow("Projector Output", projector_view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()