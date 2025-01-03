import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Step 1: Real-world table dimensions (in meters)
table_length = 2.74  # Table length
table_width = 1.51   # Table width
table_height = 0.76  # Height of the tabletop (optional)

# Real-world coordinates for 1/4 of the table (3D)
world_points = np.array([
    [0, 0, 0],                           # Bottom-left corner (P0)
    [-table_width, 0, 0],                # Bottom-right corner (P1)
    [0, table_length / 2, 0],            # 1/2 top-left corner (P2)
    [-table_width, table_length / 4, 0]  # 1/4 top-right corner (P4)
], dtype=np.float32)

# Image coordinates (in pixels)
image_points = np.array([
    [733, 735],   # Bottom-left corner (P0)
    [1333, 630],  # Bottom-right corner (P1)
    [1881, 937],  # 1/2 top-left corner (P2)
    [1637, 649],  # 1/4 top-right corner (P4)
], dtype=np.float32)

# Camera matrix and distortion coefficients (example values)
camera_matrix = np.array([
    [1.12944047e+03, 0.00000000e+00, 1.04936865e+03],
    [0.00000000e+00, 1.06185468e+03, 5.55655991e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
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

# Display results
print("Camera Position (world coordinates):")
print(f"X: {camera_x:.2f} meters")
print(f"Y: {camera_y:.2f} meters")
print(f"Z: {camera_z:.2f} meters")

# Visualization in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot only P0 as a point
ax.scatter(world_points[0, 0], world_points[0, 1], world_points[0, 2], c='b', label='P0 (Table Corner)')
ax.text(world_points[0, 0], world_points[0, 1], world_points[0, 2], 'P0', color='blue')

# Full table coordinates for shading (corners)
full_table_points = np.array([
    [0, 0, 0],                         # Bottom-left corner
    [-table_width, 0, 0],              # Bottom-right corner
    [0, table_length, 0],              # Top-left corner
    [-table_width, table_length, 0]    # Top-right corner
])

# Shade the full table
vertices = [
    [full_table_points[0], full_table_points[1], full_table_points[3], full_table_points[2]]
]
table_surface = Poly3DCollection(vertices, alpha=0.3, facecolor='blue')
ax.add_collection3d(table_surface)

# Plot the table edges (lines around the table)
edge_lines = [
    [full_table_points[0], full_table_points[1]],  # Bottom edge
    [full_table_points[1], full_table_points[3]],  # Right edge
    [full_table_points[3], full_table_points[2]],  # Top edge
    [full_table_points[2], full_table_points[0]]   # Left edge
]
for edge in edge_lines:
    edge = np.array(edge)
    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color='blue', linewidth=2)

# Plot a horizontal line in the middle of the table (blue)
middle_line = np.array([
    [0, table_length / 2, 0],         # Midpoint on the left edge
    [-table_width, table_length / 2, 0]  # Midpoint on the right edge
])
ax.plot(middle_line[:, 0], middle_line[:, 1], middle_line[:, 2], color='blue', linewidth=2, label='Middle Line')

# Plot the camera position
ax.scatter(camera_x, camera_y, camera_z, c='r', label='Camera')
ax.text(camera_x, camera_y, camera_z, 'Camera', color='red')

# Axis labels
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# Adjust axis limits to include the camera
x_min = min(-table_width - 0.5, camera_x - 0.5)
x_max = max(0.5, camera_x + 0.5)
y_min = min(-0.5, camera_y - 0.5)
y_max = max(table_length + 0.5, camera_y + 0.5)
z_max = max(2, camera_z + 0.5)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(0, z_max)

# Add a legend and title
plt.legend()
plt.title('Camera Localization and Ping-Pong Table (Edges, Middle Line)')
plt.show()
