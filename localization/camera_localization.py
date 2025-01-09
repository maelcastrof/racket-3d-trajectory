import numpy as np
import cv2

def localize_camera(camera_matrix, dist_coeffs):

    # Step 1: Real-world table dimensions (in meters)
    table_length = 2.74  # Table length
    table_width = 1.51   # Table width
    table_height = 0.76  # Height of the tabletop (optional)
    # Real-world coordinates for 1/4 of the table (3D)
    world_points = np.array([
        [0, 0, 0],                           # Bottom-left corner (P0)
        [-table_width, 0, 0],                # Bottom-right corner (P1)
        [0, (table_length / 2), 0],            # 1/2 top-left corner (P2)
        [-table_width, (table_length / 2)-0.22, 0]  # 1/4 top-right corner (P4)
    ], dtype=np.float32)
    # Image coordinates (in pixels)
    image_points = np.array([
        [733, 735],   # Bottom-left corner (P0)
        [1333, 630],  # Bottom-right corner (P1)
        [1876, 938],  # 4/5 top-left corner (P2)
        [1891, 666]  # 4/5 top-right corner (P4)
    ], dtype=np.float32)

    dist_coeffs = np.array([3.57467207, -14.86052367, -0.2714199, 0.03699078, -9.08131138])
    # SolvePnP to estimate rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        print("Failed to estimate camera pose!")
        exit()
    # Convert rvec to rotation matrix
    rotM = cv2.Rodrigues(rvec)[0]
    # Calculate camera position in world coordinates
    camera_position = -np.matrix(rotM).T @ np.matrix(tvec)
    # Extract X, Y, Z camera coordinates
    camera_position = np.array(camera_position).flatten()  # Convert to a 1D array
    camera_x, camera_y, camera_z = camera_position

    return rotM, tvec
