import cv2
import numpy as np
import time


camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")


#cap = cv2.VideoCapture(0)


width = 1920
height = 1200

#calibration_image = np.zeros((height, width,3), dtype=np.uint8)
#calibration_image = cv2.rectangle(calibration_image, (20, 20), (width-20, height-20), (0, 255, 0), 20)


#cv2.imshow("Calibration Image", calibration_image)
#cv2.waitKey(0)

#cv2.namedWindow("Calibration Frame", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("Calibration Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#cv2.imshow("Calibration Frame", calibration_image)
#cv2.waitKey(0)
#time.sleep(1)
#success, image = cap.read()   


#cv2.imshow("Captured Image", image)  
#cv2.waitKey(0)



#cv2.imwrite("calibration_frame.jpg", image)


image = cv2.imread('./calibration_frame.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold_value = 98


_, thresholded_image  = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

thresholded_images = thresholded_image.astype('uint8')


#cv2.imshow("Thresholded Image", thresholded_image)
#cv2.waitKey(0)


contours, _  = cv2.findContours(thresholded_images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


largest_contour = max(contours, key=cv2.contourArea)


cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)


#cv2.imshow("Largest Contour", image)
#cv2.waitKey(0)


epsilon = 0.02 * cv2.arcLength(largest_contour, True)
projection_approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

if len(projection_approx_corners) == 4:
    for i, corner in enumerate(projection_approx_corners):
        x, y = corner.ravel()
        print(f"Corner {i + 1}: ({x}, {y})")
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)



else:
    print("Could not find 4 corners in the contour.")



cv2.imshow("Corners", image)
cv2.waitKey(0)



if len(projection_approx_corners) == 4:
    

    points = projection_approx_corners.reshape((4,2)).astype(np.float32)


    points_sorted = points[np.argsort(points[:, 0])]



    top_points = points_sorted[:2, :]
    bottom_points = points_sorted[2:, :]


    top_points = top_points[np.argsort(top_points[:, 1]), :]

    bottom_points = bottom_points[np.argsort(bottom_points[:, 1]), :]


    ordered_points = np.vstack((top_points, bottom_points[::-1]))


    dst_points = np.array([
        [0, 0],
        [0, height - 1],
        [width - 1, height - 1],
        [width - 1, 0]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered_points, dst_points)

    warped_image = cv2.warpPerspective(image, M, (width, height))

    cv2.imshow("Warped Image", warped_image)

    cv2.waitKey(0)

else:
    print("Could not find 4 corners to compute perspective transform.")

np.save("M.npy", M)