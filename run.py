import cv2
import numpy as np 
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.1,
                       min_tracking_confidence=0.1)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)



M = np.load("M.npy")

camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

width, height = 1920, 1200

while True:
    ret, frame = cap.read()


    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    warped_image = cv2.warpPerspective(frame, M, (width, height))


    rgb_frame = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    warped_image = np.zeros((height, width, 3), np.uint8)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(warped_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    
    

    cv2.namedWindow("Final Image" , cv2.WND_PROP_FULLSCREEN)
    
    cv2.imshow("Final Image", warped_image) 
    cv2.waitKey(1)