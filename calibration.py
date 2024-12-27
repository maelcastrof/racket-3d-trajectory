import cv2
import numpy as np
import glob

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 25

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load chessboard images
images = glob.glob('img/chessboard/*.jpg')
print(f"Found {len(images)} images for calibration.")

gray = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not load image: {fname}")
        continue

    print(f"Processing image: {fname}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        print(f"Chessboard detected in {fname}")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
    else:
        print(f"No chessboard detected in {fname}. Skipping.")

cv2.destroyAllWindows()

# Verify calibration data
if len(objpoints) == 0 or len(imgpoints) == 0:
    raise ValueError("No valid data for calibration. Check your images and chessboard visibility.")

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
