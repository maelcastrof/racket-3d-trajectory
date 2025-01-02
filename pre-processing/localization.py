import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Real-world table dimensions (in meters)
table_length = 2.74  # Table length
table_width = 1.51   # Table width
table_height = 0.76  # Height of the tabletop (optional)

# Real-world coordinates of the table corners (3D)
world_points = np.array([
    [0, 0, 0.66],  # Bottom-left corner
    [table_width, 0, 0.66],  # Bottom-right corner
    [table_width, table_length, 0.66],  # Top-right corner
    [0, table_length, 0.66],  # Top-left corner
    [0, 0, table_height],  # Bottom-left tabletop corner
    [table_width, 0, table_height],  # Bottom-right tabletop corner
    [table_width, table_length, table_height],  # Top-right tabletop corner
    [0, table_length, table_height]  # Top-left tabletop corner
], dtype=np.float32)

# Step 2: Image coordinates (in pixels)
# Visible points and placeholders for missing points
image_points = np.array([
    [691, 552],  # Bottom-left corner (visible)
    [747, 694],  # Bottom-right corner (visible)
    [np.nan, np.nan],  # Top-right corner (missing)
    [np.nan, np.nan],  # Top-left corner (missing)
    [690, 521],  # Bottom-left tabletop corner (visible)
    [747, 643],  # Bottom-right tabletop corner (visible)
    [np.nan, np.nan],  # Top-right tabletop corner (missing)
    [np.nan, np.nan]   # Top-left tabletop corner (missing)
], dtype=np.float32)


# Filter valid points
valid_points = ~np.isnan(image_points).any(axis=1)
valid_image_points = image_points[valid_points]
valid_world_points = world_points[valid_points]

# Step 3: Camera calibration results
camera_matrix = np.array([
    [1.19733085e+03, 0.00000000e+00, 9.59615426e+02],
    [0.00000000e+00, 1.23197655e+03, 4.08899254e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
dist_coeffs = np.array([3.57467207, -14.86052367, -0.2714199, 0.03699078, -9.08131138])

# Step 4: Estimate camera pose
success, rvec, tvec = cv2.solvePnP(valid_world_points, valid_image_points, camera_matrix, dist_coeffs)

if not success:
    print("Camera pose estimation failed!")
    exit()

# Convert rotation vector to rotation matrix
rotation_matrix, _ = cv2.Rodrigues(rvec)

# Step 5: Compute camera position in the world coordinate system
camera_position = -np.dot(rotation_matrix.T, tvec).flatten()

# Print results
print("Rotation Matrix:\n", rotation_matrix)
print("Translation Vector (in meters):\n", tvec)
print("Camera Position (in meters):\n", camera_position)

# Step 6: Visualization in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the table corners
ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], c='b', label='Table Corners')
for i, point in enumerate(world_points):
    ax.text(point[0], point[1], point[2], f'P{i}', color='blue')

# Plot the camera position
ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', label='Camera')
ax.text(camera_position[0], camera_position[1], camera_position[2], 'Camera', color='red')

# Set axes labels
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# Set plot limits
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_zlim(-1, 3)

plt.legend()
plt.title('Camera Localization Using Calibration Matrix')
plt.show()


# # Simulated corresponding 2D image points (in pixels)
# # Assuming a rough projection to simulate detected points
# image_points = np.array([
#     [693, 300],  # Corresponds to bottom-left corner
#     [800, 320],  # Bottom-right corner
#     [820, 600],  # Top-right corner
#     [460, 590],  # Top-left corner
#     [470, 280],  # Bottom-left tabletop corner
#     [810, 300],  # Bottom-right tabletop corner
#     [830, 580],  # Top-right tabletop corner
#     [480, 570]   # Top-left tabletop corner
# ], dtype=np.float32)

# # Solve for extrinsic parameters (rotation and translation)
# success, rvec, tvec = cv2.solvePnP(world_points, image_points, lateral_camera_matrix, lateral_dist_coeffs)

# if success:
#     # Convert rotation vector to rotation matrix
#     rotation_matrix, _ = cv2.Rodrigues(rvec)

#     # Camera position in world coordinates
#     camera_position = -np.dot(rotation_matrix.T, tvec)
#     print("Camera Position (in world coordinates):")
#     print(camera_position.flatten())

#     # Visualization
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the table corners in 3D
#     ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], c='b', label='Table Corners')
#     for i, point in enumerate(world_points):
#         ax.text(point[0], point[1], point[2], f'P{i}', color='blue')

#     # Plot the camera position
#     ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', label='Camera')
#     ax.text(camera_position[0], camera_position[1], camera_position[2], 'Camera', color='red')

#     # Set axes labels
#     ax.set_xlabel('X (m)')
#     ax.set_ylabel('Y (m)')
#     ax.set_zlabel('Z (m)')

#     # Set plot limits
#     ax.set_xlim(-1, 3)
#     ax.set_ylim(-1, 3)
#     ax.set_zlim(-1, 2)

#     plt.legend()
#     plt.title('Camera Localization Using Calibration Matrix')
#     plt.show()
# else:
#     print("Failed to solve PnP problem.")
