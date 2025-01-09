import cv2
import numpy as np

def localize_racket(image_points, camera_matrix, dist_coeffs, rotM, tvec):
    """
    Convert image points of the racket to world coordinates.
    """
    # Inverse of the camera matrix
    inv_K = np.linalg.inv(camera_matrix)

    # Convert racket image points to 3D world coordinates
    racket_points_3D = []
    for point in image_points:
        # Convert each point to homogeneous coordinates (3x1)
        pixel_point = np.array([[point[0]], [point[1]], [1]], dtype=np.float32)

        # Inverse projection to camera space
        camera_point = inv_K @ pixel_point  # Convert to camera space
        camera_point = camera_point.flatten()

        # Transform to world coordinates using the camera rotation and translation
        world_point_3D = rotM.T @ (camera_point - tvec.flatten())
        racket_points_3D.append(world_point_3D)

    racket_points_3D = np.array(racket_points_3D)

    # Calculate the center of the racket (mean of the 4 points)
    racket_center_3D = np.mean(racket_points_3D, axis=0)

    return racket_points_3D, racket_center_3D
