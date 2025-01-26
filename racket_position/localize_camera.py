import cv2
import numpy as np

def localize_camera(camera_matrix, dist_coeffs):
    """
    Estimates the camera position and rotation matrix.
    """
    table_length = 2.74  # Table length (meters)
    table_width = 1.51   # Table width (meters)

    # Real-world table corners
    world_points = np.array([
        [0, 0, 0],
        [-table_width, 0, 0],
        [0, (table_length / 2) - 0.3, 0],
        [-table_width, (table_length / 2) - 0.3, 0]
    ], dtype=np.float32)

    # Image points corresponding to the table corners
    image_points = np.array([
        [457, 822],
        [1553, 822],
        [37, 1044],
        [1904, 1044]
    ], dtype=np.float32)

    # Estimate pose using SolvePnP
    success, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        raise ValueError("Camera localization failed.")

    # Convert rotation vector to matrix
    rotM = cv2.Rodrigues(rvec)[0]

    # Calculate the camera position
    camera_position = -np.matrix(rotM).T @ np.matrix(tvec)
    camera_position = np.array(camera_position).flatten()
    camera_x, camera_y, camera_z = camera_position

    return rotM, tvec, camera_x, camera_y, camera_z
