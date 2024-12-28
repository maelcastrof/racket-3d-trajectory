import cv2
import numpy as np

# Load camera calibration parameters fro 45 degrees
camera_matrix = np.array([
    [1.12944047e+03, 0.00000000e+00, 1.04936865e+03],
    [0.00000000e+00, 1.06185468e+03, 5.55655991e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist_coeffs = np.array([[-0.43673283, -2.01713805, -0.04056567, 0.08416871, 9.82369894]])

# Video input path
video_path = 'data/input/Black_red_rigth2_45.MOV'  
cap = cv2.VideoCapture(video_path) #It reads frame by frame

if not cap.isOpened():
    raise ValueError(f"Could not open video file: {video_path}")

# Tracker initialization
tracker = cv2.legacy.TrackerCSRT_create()  # Use CSRT for more robust tracking
tracking = False  # Indicates if the racket is being tracked

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Undistort the frame
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    if not tracking:
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the color range for the racket, in this case is red
        lower_color = np.array([170, 120, 80])  # Lower bound based on [173, 149, 106]
        upper_color = np.array([176, 200, 150])  # Upper bound based on [173, 149, 106]

        # Create a mask for the defined color
        mask = cv2.inRange(hsv, lower_color, upper_color) #A mask is created to isolate pixels within the defined color range

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #The contours represent potential racket regions

        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 500:
                continue

            # Draw the contour and bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Initialize the tracker with the bounding box
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)
            tracking = True
            break  # Track the first detected racket
    else:
        # Update the tracker
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Lost Track", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            tracking = False  # Reset if tracking fails

    # Show the result
    cv2.imshow("Racket Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
