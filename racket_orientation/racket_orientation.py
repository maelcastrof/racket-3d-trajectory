from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def generate_rotated_truncated_ellipse(center_3d, ellipse_a, ellipse_b, y_cutoff, roll, pitch, yaw):
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_ellipse = ellipse_b * np.cos(theta)
    z_ellipse = ellipse_a * np.sin(theta)

    # Aplicar el corte (truncar la elipse en 2D)
    mask_above_cutoff = z_ellipse >= y_cutoff
    x_truncated = x_ellipse[mask_above_cutoff]
    z_truncated = z_ellipse[mask_above_cutoff]
    y_truncated = np.zeros_like(x_truncated)

    # Agrupar en un solo array
    ellipse_points = np.vstack((x_truncated, y_truncated, z_truncated)).T

    # Crear la rotación a partir de los ángulos roll, pitch, yaw
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rotated_points = rotation.apply(ellipse_points)

    # Alinear el centro de la elipse con el centro de los puntos del contorno
    rotated_points += center_3d

    return np.array(rotated_points)  # Garantizar salida como array NumPy

def objective_function(angles, contour_points_3d, center_3d, ellipse_a, ellipse_b, y_cutoff):
    roll, pitch, yaw = angles
    rotated_ellipse = generate_rotated_truncated_ellipse(center_3d, ellipse_a, ellipse_b, y_cutoff, roll, pitch, yaw)

    # Asegurar que ambas variables sean arrays NumPy
    contour_points_3d = np.asarray(contour_points_3d)
    rotated_ellipse = np.asarray(rotated_ellipse)

    # Calcular la distancia promedio de la elipse a los puntos del contorno
    distances = np.linalg.norm(
        contour_points_3d[:, None, :] - rotated_ellipse[None, :, :], axis=2
    )
    mean_distance = np.mean(np.min(distances, axis=1))
    return mean_distance

def fit_ellipse_to_contour(contour_points_3d, ellipse_a, ellipse_b, y_cutoff, center_3d):
    # Calcular el centro de los puntos 3D
    #center_3d = np.mean(np.asarray(contour_points_3d), axis=0)
   
    # Optimización para encontrar los ángulos óptimos
    result = minimize(
        objective_function,
        x0=[0, 0, 0],  # Ángulos iniciales (roll, pitch, yaw)
        args=(contour_points_3d, center_3d, ellipse_a, ellipse_b, y_cutoff),
        bounds=[(-180, 180), (-180, 180), (-180, 180)],
        method='L-BFGS-B'
    )

    return result.x  # Ángulos óptimos

def plot_3d_with_rotated_ellipse(contour_points_3d, ellipse_a, ellipse_b, y_cutoff, best_angles, center_3d):
    contour_points_3d = np.array(contour_points_3d)
    #center_3d = np.mean(contour_points_3d, axis=0)

    # Generar la elipse con la rotación óptima
    roll, pitch, yaw = best_angles
    rotated_ellipse = generate_rotated_truncated_ellipse(center_3d, ellipse_a, ellipse_b, y_cutoff, roll, pitch, yaw)

    # Graficar en 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar los puntos del contorno
    ax.scatter(contour_points_3d[:, 0], contour_points_3d[:, 1], contour_points_3d[:, 2],
               c='blue', s=10, label='Contour Points')

    # Graficar la elipse rotada y alineada
    ax.plot(rotated_ellipse[:, 0], rotated_ellipse[:, 1], rotated_ellipse[:, 2],
            c='red', lw=2, label='Best Fit Rotated Ellipse')

    # Configurar límites de los ejes
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    ax.legend()
    plt.show()


def euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """
    Converts Euler angles (yaw, pitch, roll) in degrees to a 3x3 rotation matrix.
    
    Args:
    - yaw_deg: Rotation angle around the Z-axis (in degrees)
    - pitch_deg: Rotation angle around the Y-axis (in degrees)
    - roll_deg: Rotation angle around the X-axis (in degrees)
    
    Returns:
    - A 3x3 rotation matrix.
    """
    # Convert degrees to radians
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    
    # Rotation matrix for yaw (rotation around the Z-axis)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Rotation matrix for pitch (rotation around the Y-axis)
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation matrix for roll (rotation around the X-axis)
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    return R

def plot_ideal_racket_with_table(roll, pitch, yaw, center, ellipse_a, ellipse_b, y_cutoff, table_length, table_width, camera_position):
    """
    Plots the ideal racket shape in 3D space with a given rotation (using Euler angles), table, and camera position.
    
    Args:
    - roll, pitch, yaw: Euler angles (in radians) for the racket's orientation.
    - center: 3D center of the racket (x, y, z).
    - ellipse_a: Semi-major axis of the ellipse (meters).
    - ellipse_b: Semi-minor axis of the ellipse (meters).
    - y_cutoff: Truncation level (meters).
    - table_length: Length of the table (meters).
    - table_width: Width of the table (meters).
    - camera_position: 3D position of the camera (x, y, z).
    """
    # Convert Euler angles to rotation matrix
    R = euler_to_rotation_matrix(yaw, pitch, roll)
    
    # Create points for the ideal ellipse in the local coordinate frame
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_ellipse = ellipse_b * np.cos(theta)
    z_ellipse = ellipse_a * np.sin(theta)

    # Aplicar el corte (truncar la elipse en 2D)
    mask_above_cutoff = z_ellipse >= y_cutoff
    x_truncated = x_ellipse[mask_above_cutoff]
    z_truncated = z_ellipse[mask_above_cutoff]
    y_truncated = np.zeros_like(x_truncated)

    # Stack into 3D points and apply rotation and translation
    ellipse_points = np.vstack((x_truncated, y_truncated, z_truncated))
    rotated_ellipse_points = np.dot(R, ellipse_points).T + center

    # Visualization in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the ping-pong table
    full_table_points = np.array([ 
        [0, 0, 0],                         # Bottom-left corner
        [-table_width, 0, 0],              # Bottom-right corner
        [0, table_length, 0],              # Top-left corner
        [-table_width, table_length, 0]    # Top-right corner
    ])

    # Table surface
    vertices = [[
        full_table_points[0], full_table_points[1],
        full_table_points[3], full_table_points[2]
    ]]
    table_surface = Poly3DCollection(vertices, alpha=0.3, facecolor="blue")
    ax.add_collection3d(table_surface)

    # Draw table edges
    edge_lines = [
        [full_table_points[0], full_table_points[1]],
        [full_table_points[1], full_table_points[3]],
        [full_table_points[3], full_table_points[2]],
        [full_table_points[2], full_table_points[0]]
    ]
    for edge in edge_lines:
        edge = np.array(edge)
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="blue", linewidth=2)

    # Draw the middle line of the table
    middle_line = np.array([
        [0, table_length / 2, 0],
        [-table_width, table_length / 2, 0]
    ])
    ax.plot(middle_line[:, 0], middle_line[:, 1], middle_line[:, 2], color="blue", linewidth=2, label="Middle Line")

    # Draw the rotated ideal ellipse (racket shape)
    ax.plot(rotated_ellipse_points[:, 0], rotated_ellipse_points[:, 1],
            rotated_ellipse_points[:, 2], color="red", lw=2, label="Ideal Racket Shape")

    # Draw the camera position
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c="red", label="Camera")
    ax.text(camera_position[0], camera_position[1], camera_position[2], "Camera", color="red")

    # Configure the plot
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (Depth, m)")
    ax.set_zlabel("Z (Height, m)")
    ax.set_title("Racket Shape with Rotation")
    ax.legend()

    # Adjust plot limits
    x_limits = [-table_width, 0]
    y_limits = [0, table_length]
    z_limits = [0, max(camera_position[2], 0)]
    max_range = max(abs(x_limits[1] - x_limits[0]), abs(y_limits[1] - y_limits[0]), abs(z_limits[1] - z_limits[0]))
    mid_x = (x_limits[0] + x_limits[1]) / 2
    mid_y = (y_limits[0] + y_limits[1]) / 2
    mid_z = (z_limits[0] + z_limits[1]) / 2

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    plt.show()
