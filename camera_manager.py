import cv2
import numpy as np
import mediapipe as mp

class CameraManager:
    def __init__(self, transformation_matrix_path, width, height):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(0)
        
        # Load your existing calibration data
        self.M = np.load(transformation_matrix_path)
        self.camera_matrix = np.load("camera_matrix.npy")
        self.dist_coeffs = np.load("dist_coeffs.npy")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.results = None
        self.frame_shape = None

    def update(self):
        ret, frame = self.cap.read()
        if not ret: return False

        # Undistort the raw camera feed
        frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        self.frame_shape = frame.shape

        # Process landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        return True

    def get_transformed_landmarks(self):
        if self.results and self.results.multi_hand_landmarks:
            transformed_hands = []
            for hand_landmarks in self.results.multi_hand_landmarks:
                landmark_coords = []
                for lm in hand_landmarks.landmark:
                    x = lm.x * self.frame_shape[1]
                    y = lm.y * self.frame_shape[0]
                    landmark_coords.append([x, y])
                
                # Transform camera points to projector points
                pts = np.array([landmark_coords], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pts, self.M)[0]
                transformed = np.clip(transformed, [0, 0], [self.width - 1, self.height - 1])
                transformed_hands.append(transformed)
            return transformed_hands
        return None

    def release(self):
        self.cap.release()