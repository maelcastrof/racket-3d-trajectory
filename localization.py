import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Step 1: Real-world table dimensions (in meters)
table_length = 2.74  # Table length
table_width = 1.51   # Table width
table_height = 0  # Height of the tabletop (optional)

# Real-world coordinates for 1/4 of the table (3D)
world_points = np.array([
    [0, 0, 0],                           # corner left top (REF_P) #esquina superior izquierda
    [-table_width, 0, 0],                # corner right top (punto_3) #esquina superior derecha
    [0, (table_length / 2)-0.3, 0],            # 1/2 top-left corner (punto_0) #esquina inferior izquierda
    [-table_width, (table_length / 2)-0.3, 0]  # 1/4 top-right corner (punto_1) #esquina inferior derecha
], dtype=np.float32)

# Image coordinates (in pixels)
image_points = np.array([
    REF_P,
    punto_3,
    punto_0,
    punto_1,
], dtype=np.float32)


# SolvePnP to estimate rotation and translation vectors
success, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, np.zeros(5))

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
ax.scatter(world_points[0, 0], world_points[0, 1], world_points[0, 2], c='b', label='P0')
ax.text(world_points[0, 0], world_points[0, 1], world_points[0, 2], 'P0', color='blue')

ax.scatter(world_points[1, 0], world_points[1, 1], world_points[1, 2], c='b', label='P1')
ax.text(world_points[1, 0], world_points[1, 1], world_points[1, 2], 'P1', color='blue')

ax.scatter(world_points[2, 0], world_points[2, 1], world_points[2, 2], c='b', label='P2')
ax.text(world_points[2, 0], world_points[2, 1], world_points[2, 2], 'P2', color='blue')

ax.scatter(world_points[3, 0], world_points[3, 1], world_points[3, 2], c='b', label='P3')
ax.text(world_points[3, 0], world_points[3, 1], world_points[3, 2], 'P3', color='blue')

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

# Ajustar los límites de los ejes para que todos tengan la misma escala
x_limits = [-table_width, 0]
y_limits = [0, table_length]
z_limits = [0, max(camera_z, 0)]  # Considerar la posición de la cámara o el plano de la mesa

# Determinar el rango máximo entre los ejes
max_range = max(abs(x_limits[1] - x_limits[0]), abs(y_limits[1] - y_limits[0]), abs(z_limits[1] - z_limits[0]))

# Ajustar los límites para todos los ejes con el mismo rango
mid_x = (x_limits[0] + x_limits[1]) / 2
mid_y = (y_limits[0] + y_limits[1]) / 2
mid_z = (z_limits[0] + z_limits[1]) / 2

ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

# Añadir proporción uniforme a los ejes
ax.set_box_aspect([1, 1, 1])  # Escalas iguales en X, Y, Z

# Add a legend and title
plt.legend()
plt.title('Camera Localization and Ping-Pong Table (Edges, Middle Line)')
plt.show()
