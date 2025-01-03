import cv2
import numpy as np

# List to store the points clicked by the user
clicked_points = []

# Callback function for mouse click events
def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click
        # Append the clicked point to the list
        clicked_points.append((x, y))
        
        # Draw a small circle at the clicked point
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image)

        # If 4 points are clicked, close the window
        if len(clicked_points) == 4:
            print("Four points selected.")
            print("Points:", clicked_points)
            cv2.destroyAllWindows()

# Path to your image
image_path = '../data/frames/before_impact.png'

# Load the image
image = cv2.imread(image_path)

# Create a window with resize capability
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# Show the image and wait for user to click on points
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)

# Wait for the user to select points and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the clicked points to a NumPy array
image_points = np.array(clicked_points, dtype=np.float32)

# Print the points
print("Selected Image Points (in pixels):")
print(image_points)
