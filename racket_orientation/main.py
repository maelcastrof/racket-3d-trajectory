import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from racket_position.localize_camera import localize_camera
from racket_position.racket_detection import detect_racket_contour
from racket_orientation.racket_position import calculate_scale_and_world_coordinates, find_racket_world_coordinates
from racket_orientation.racket_orientation import fit_ellipse_to_contour, plot_3d_with_rotated_ellipse, plot_ideal_racket_with_table

# Table dimensions in meters
table_length = 2.74  
table_width = 1.51   
table_height = 0  

# Calibration results for the front camera (2000 frames)
camera_matrix = np.array([
    [1.10929294e+03, 0.00000000e+00, 9.68244406e+02],
    [0.00000000e+00, 1.10828076e+03, 5.32739654e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Distortion coefficients (example)

rotM, tvec, camera_x, camera_y, camera_z = localize_camera(camera_matrix, dist_coeffs)
extrinsics = np.hstack((rotM, tvec.reshape(3, 1)))

# Compute projection matrix
projection_matrix = np.dot(camera_matrix, extrinsics)

# Ellipse dimensions for the racket
ellipse_a = (0.1851 / 2)  # Semi-major axis (radius)
ellipse_b = (0.1748 / 2)  # Semi-minor axis (radius)
y_cutoff = -0.08  # Horizontal cutoff line

def main_countour(image_path):
    """
    Process a single frame to detect the racket and estimate its 3D position relative to the camera.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load the image: {image_path}")
        return

    # Detect the racket contours
    contour_points_2d, line_points_2d = detect_racket_contour(frame)

    # Camera position in 3D space
    camera_position = np.array([-np.linalg.inv(rotM) @ tvec.flatten()]).flatten()
    print("Camera Position (in meters):\n", camera_position)

    if contour_points_2d:
        contour_points_3d = []
        for image_point in contour_points_2d:
            if len(image_point) == 2:
                world_point_homogeneous = np.array([0, 0, 0, 1])

                # Calculate scale factor and world coordinates
                scale_factor, _ = calculate_scale_and_world_coordinates(projection_matrix, image_point, world_point_homogeneous)
                racket_position_camera = find_racket_world_coordinates(projection_matrix, image_point, scale_factor)
                contour_points_3d.append(tuple(racket_position_camera))

        contour_points_3d = np.array(contour_points_3d)

        # Convert 2D line points to 3D
        line_points_3d = []
        for image_point in line_points_2d:
            if len(image_point) == 2:
                world_point_homogeneous = np.array([0, 0, 0, 1])

                scale_factor, _ = calculate_scale_and_world_coordinates(projection_matrix, image_point, world_point_homogeneous)
                racket_position_camera = find_racket_world_coordinates(projection_matrix, image_point, scale_factor)
                line_points_3d.append(tuple(racket_position_camera))
        
        line_points_3d = np.array(line_points_3d)

        # Combine contour and line points
        all_points_3d = np.concatenate((contour_points_3d, line_points_3d), axis=0)

        # Calculate the center of contour points
        center_all = np.mean(all_points_3d, axis=0)
        center_contour = np.mean(contour_points_3d, axis=0)
        print(f"Center of Contour Points (3D): {center_contour}")

        # Fit ellipse and compute angles
        best_angles = fit_ellipse_to_contour(all_points_3d, ellipse_a, ellipse_b, y_cutoff, center_all)
        print(f"Optimal Angles (Degrees): Roll={best_angles[0]:.2f}, Pitch={best_angles[1]:.2f}, Yaw={best_angles[2]:.2f}")
        
        # Plot the 3D data and compute orientation
        plot_3d_with_rotated_ellipse(all_points_3d, ellipse_a, ellipse_b, y_cutoff, best_angles, center_contour)
        plot_ideal_racket_with_table(best_angles[0], best_angles[1], best_angles[2], center_contour, ellipse_a, ellipse_b, y_cutoff, table_length, table_width, camera_position)
    else:
        print("No contours detected.")

    plt.show()

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), '../data/frame_0002.png')
    main_countour(image_path)
