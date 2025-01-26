import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_racket(frame):
    """
    Detects the racket in the frame based on its red color and returns ellipse parameters.
    """

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color in both ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assumed to be the racket)
        largest_contour = max(contours, key=cv2.contourArea)

        # Check if the contour has enough points to fit an ellipse
        if len(largest_contour) >= 100:
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(largest_contour)

            return ellipse

    # Return None if no racket is detected
    return None

def detect_racket_contour(frame, distance_threshold=10):
    """
    Detects the racket in the frame based on its red color, returning contour points and the bottom line with equidistant points.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for detecting red color
    lower_red1 = np.array([0, 70, 50])   # First range for red
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([150, 50, 50])  # Second range for red
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the frame to draw contours
    frame_with_contours = frame.copy()

    # Initialize a list to store contour points
    contour_points = []

    # Draw significant contours and collect their points
    for contour in contours:
        if len(contour) >= 100:  # Filter out small contours
            cv2.drawContours(frame_with_contours, contour, -1, (255, 0, 0), 4)  # Draw in blue
            # Add contour points to the list
            for point in contour:
                x, y = point[0]
                contour_points.append((x, y))  # Collect 2D points in image coordinates

    # Create an empty mask image to visualize contours
    mask_img = np.zeros_like(frame)
    filtered_contours = [contour for contour in contours if len(contour) >= 100]
    cv2.drawContours(mask_img, filtered_contours, -1, (0, 255, 0), 3)  # Draw filtered contours in green

    # Apply Canny Edge Detection on the mask
    edges = cv2.Canny(mask_img, 100, 200)

    # Detect lines using the Hough Transform
   #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=2, minLineLength=50, maxLineGap=15) #Frame 0001
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=2, minLineLength=50, maxLineGap=10) #Frame 0002
   # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=3, minLineLength=70, maxLineGap=20) #Frame 0003

    # Initialize a list to store line points
    line_points = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_with_contours, (x1, y1), (x2, y2), (0, 255, 0), 10)  # Draw lines in green

            # Generate equidistant points along the line
            num_points = 500  # Number of equidistant points to generate
            for t in np.linspace(0, 1, num_points):
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                line_points.append((x, y))  # Add the line points

    # Filter contour points that are close to the detected line
    filtered_contour_points = []
    for contour_point in contour_points:
        cx, cy = contour_point
        is_near_line = False

        # Check if the point is close to any line point
        for line_point in line_points:
            lx, ly = line_point
            # Compute Euclidean distance between contour point and line point
            distance = np.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
            if distance < distance_threshold:  # If close to a line
                is_near_line = True
                break  # No need to check further points

        # If the point is not near any line, add it to the filtered list
        if not is_near_line:
            filtered_contour_points.append((cx, cy))

    # Display the frame with contours and lines
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(frame_with_contours, cv2.COLOR_BGR2RGB))
    plt.title("Contours and Detected Lines")
    plt.axis("off")
    plt.show()

    # Return the filtered contour points and line points
    return filtered_contour_points, line_points
