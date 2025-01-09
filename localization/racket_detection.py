import cv2
import numpy as np

def detect_racket(frame):
    """
    Detect the racket in the frame based on its red color and return the corner points.
    """
    # Convert to HSV color space for color-based segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for the red face of the racket
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Combine both ranges to detect red color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.add(mask_red1, mask_red2)  # Combine red masks

    # Find contours of the detected regions
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assumed to be the racket)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle for the racket
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Define the four corner points: top-left, top-right, bottom-left, bottom-right
        top_left = (x, y)
        top_right = (x + w, y)
        bottom_left = (x, y + h)
        bottom_right = (x + w, y + h)

        return [top_left, top_right, bottom_left, bottom_right]
    else:
        return None
