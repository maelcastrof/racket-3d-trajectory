import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from racket_detection import detect_racket
from racket_localization import localize_racket
from camera_localization import localize_camera
# Table dimensions
table_length = 2.74  # Length in meters
table_width = 1.51   # Width in meters
table_height = 0  # Height of the table in meters

# Camera calibration parameters
camera_matrix = np.array([
    [1.12944047e+03, 0.00000000e+00, 1.04936865e+03],
    [0.00000000e+00, 1.06185468e+03, 5.55655991e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
dist_coeffs = np.array([3.57467207, -14.86052367, -0.2714199, 0.03699078, -9.08131138])

# Rotation and translation vectors (example values)
rotM, tvec = localize_camera(camera_matrix, dist_coeffs )


def draw_table(ax):
    """
    Draws a 3D ping-pong table on the given matplotlib axis.
    """
    # Full table coordinates for shading (corners)
    full_table_points = np.array([
        [0, 0, table_height],                         # Bottom-left corner
        [-table_width, 0, table_height],              # Bottom-right corner
        [0, table_length, table_height],              # Top-left corner
        [-table_width, table_length, table_height]    # Top-right corner
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
        [0, table_length / 2, table_height],         # Midpoint on the left edge
        [-table_width, table_length / 2, table_height]  # Midpoint on the right edge
    ])
    ax.plot(middle_line[:, 0], middle_line[:, 1], middle_line[:, 2], color='blue', linewidth=2, label='Middle Line')

def main(video_path):
    """
    Process the video to detect and localize the racket while plotting in 3D in real-time.
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Create a 3D plot for visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

   # Set Z-axis limits
    ax.set_zlim(-0.76, 2)  # Updated Z-axis limits
    
    # Draw the table on the plot
    draw_table(ax)

    # Plot the camera position
    camera_position = -np.linalg.inv(rotM) @ tvec.flatten()
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', label='Camera', s=100)
    ax.text(camera_position[0], camera_position[1], camera_position[2], f'CameraPosition({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f})', color='red')

    # Draw reference point P0 (0, 0, 0) in the 3D plot (corner of the table)
    ax.scatter(0, 0, table_height, c='b', label='P0 (0, 0, 0)', s=100)
    ax.text(0, 0, table_height, 'P0 (0, 0, 0)', color='blue')

    # Initialize racket position plot
    racket_plot = ax.scatter([], [], [], c='g', s=100, label='Racket')
    racket_text = None  # Variable to store text for racket position

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the racket in the frame
        racket_points = detect_racket(frame)
        if racket_points is not None:
            # Localize the racket in 3D space
            _, racket_center = localize_racket(racket_points, camera_matrix, dist_coeffs, rotM, tvec)

            # Update the racket's position in 3D plot
            racket_plot._offsets3d = (
                [-racket_center[0]],  # Posición X (invertida para el sistema de coordenadas)
                [-racket_center[1]],  # Posición Y (invertida para el sistema de coordenadas)
                [racket_center[2] + table_height]  # Posición Z (con la altura de la mesa)
            )

            # If there is an existing text label, remove it
            if racket_text is not None:
                racket_text.remove()

            # Add the new racket position label
            racket_text = ax.text(
                -racket_center[0], -racket_center[1], racket_center[2] + table_height,
                'Racket', color='green'
            )

            # Create text for racket position in the format RacketPosition(X, Y, Z)
            racket_position_text = f"RacketPosition({racket_center[0]:.2f}, {racket_center[1]:.2f}, {racket_center[2]:.2f})"
            
            # Add racket position text to the frame (video)
            cv2.putText(frame, racket_position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw detected racket corners on the video frame
        if racket_points is not None:
            for point in racket_points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

        # Display the video frame
        cv2.imshow('Racket Detection', frame)

        # Update the 3D plot (with the latest racket position)
        plt.pause(0.01)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Add labels and finalize the plot
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Real-Time Racket Localization in 3D Space')
    plt.legend()
    plt.show()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "../data/input/Black_red_rigth2_45_play.mp4"  # Replace with your video file
    main(video_path)
