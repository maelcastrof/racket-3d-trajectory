import cv2
import numpy as np

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 25

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points
imgpoints = []  # 2D points

# Video input path
video_path = '../data/calibration/Calibration_45.MOV' 

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Could not open video file: {video_path}")

frame_count = 0
gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1

    # Process every nth frame to save time (e.g., every 10th frame)
    if frame_count % 10 != 0:
        continue

    print(f"Processing frame {frame_count}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        print(f"Chessboard detected in frame {frame_count}")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Chessboard Detection', frame)
        cv2.waitKey(1)  # Adjust waitKey for smoother playback if needed
    else:
        print(f"No chessboard detected in frame {frame_count}. Skipping.")

cap.release()
cv2.destroyAllWindows()

# Verify calibration data
if len(objpoints) == 0 or len(imgpoints) == 0:
    raise ValueError("No valid data for calibration. Check your video and chessboard visibility.")

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

if ret:
    print("Calibration successful!")
    print("Camera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
else:
    print("Calibration failed.")
