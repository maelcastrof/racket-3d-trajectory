import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from racket_detection import detect_racket
from visualization import draw_table, draw_camera
from localize_camera import localize_camera
from racket_position import find_racket_world_coordinates, calculate_scale_and_world_coordinates

# Table dimensions in world coordinates
table_length = 2.74  # Table length in meters
table_width = 1.51   # Table width in meters
table_height = 0  # Table height in meters (assumed flat on the ground)

# Real-world coordinates for 1/4 of the table (3D)
world_points = np.array([
    [0, 0, 0],                           # Corner: top-left (REF_P)
    [-table_width, 0, 0],                # Corner: top-right (punto_3)
    [0, (table_length / 2) - 0.3, 0],    # Halfway: mid-left (punto_0)
    [-table_width, (table_length / 2) - 0.3, 0]  # Halfway: mid-right (punto_1)
], dtype=np.float32)

# Camera calibration parameters (intrinsic matrix)
camera_matrix = np.array([
    [1.10929294e+03, 0.00000000e+00, 9.68244406e+02],
    [0.00000000e+00, 1.10828076e+03, 5.32739654e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# Distortion coefficients (set to zero for simplicity)
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Get camera position and rotation
rotM, tvec, camera_x, camera_y, camera_z = localize_camera(camera_matrix, dist_coeffs)
camera_position = [camera_x, camera_y, camera_z]

# Print calibration results
print("Rotation Matrix:\n", rotM)
print("Translation Vector (in meters):\n", tvec)
print("Camera Position (in meters):\n", camera_position)

# Combine rotation and translation into [R | t] matrix
extrinsics = np.hstack((rotM, tvec.reshape(3, 1)))

# Compute the projection matrix
projection_matrix = np.dot(camera_matrix, extrinsics)

# Path to the input video
video_path = os.path.join(os.path.dirname(__file__), '../data/Black_red_left_front.MOV')
cap = cv2.VideoCapture(video_path)

# Define output directory and filename
output_directory = os.path.join(os.path.dirname(__file__), 'results')  # Output folder named "results"
output_filename = 'output_video.mp4'
output_video_path = os.path.join(output_directory, output_filename)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Create figure for displaying video and 3D plot
fig = plt.figure(figsize=(10, 10))

# Subplot for video
ax1 = fig.add_subplot(2, 1, 1)  # Video on the top
ax1.axis("off")  # No axes for the video
video_img = None  # Placeholder for the video frame

# Subplot for 3D plot
ax2 = fig.add_subplot(2, 1, 2, projection='3d')  # 3D plot on the bottom
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")
ax2.set_title("Racket Position with Table and Camera")

# Set plot limits and proportions
ax2.set_xlim(-2, 1)
ax2.set_ylim(-0.5, 2.5)
ax2.set_zlim(-1.5, 1.5)
ax2.set_box_aspect([1, 1, 1])  # Equal scaling for X, Y, Z

# Draw the table and camera in the 3D plot
draw_table(ax2)  # Assuming `draw_table` draws the table edges
draw_camera(camera_position, ax2)  # Assuming `draw_camera` draws the camera position

frame_idx = 0
racket_point = None  # For plotting the racket's current position
previous_position = None  # For storing the racket's previous position

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Detect racket in the frame
    ellipse = detect_racket(frame)
    if ellipse:
        # Get racket center in image coordinates
        racket_center_image = (ellipse[0][0], ellipse[0][1])

        # Calculate the 3D position of the racket
        world_point_homogeneous = np.array([0, 0, 0, 1])
        image_point = [457, 822]
        scale_factor = calculate_scale_and_world_coordinates(
            projection_matrix, image_point, world_point_homogeneous
        )
        racket_position_camera = find_racket_world_coordinates(projection_matrix, racket_center_image, scale_factor, ellipse, camera_matrix)
        
        # Get racket 3D coordinates
        x, y, z = (
            float(racket_position_camera[0]),
            float(racket_position_camera[1] + camera_position[1]),
            float(racket_position_camera[2])
        )

        # Validate if y exceeds the threshold
        if previous_position and y > table_length / 4:
            # Use the previous position if it exceeds the threshold
            x, y, z = previous_position
        else:
            # Update the current position
            previous_position = (x, y, z)
        
        # Draw the detected racket's ellipse on the frame
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

        # Update racket position in the 3D plot
        if racket_point:
            racket_point.remove()  # Remove the previous point
        racket_point = ax2.scatter(x, y, z, c='g', s=50, label="Racket Position")

        # Display the coordinates on the video frame
        coord_text = f"Frame: {frame_idx}, Center: ({x:.2f}, {y:.2f}, {z:.2f})"
        text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x, text_y = 10, 30  # Text position
        box_coords = ((text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5))
        cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)  # White background
        cv2.putText(frame, coord_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text

    # Display video frame in the subplot
    if video_img is None:
        video_img = ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        video_img.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Write the frame to the output file
    out.write(frame)

    # Update Matplotlib plot
    plt.pause(0.01)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Show 3D plot legend
ax2.legend()
plt.show()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
