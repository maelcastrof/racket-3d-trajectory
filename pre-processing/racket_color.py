import cv2
import numpy as np
import os
# Relative path to the data folder
data_folder = os.path.join(os.path.dirname(__file__), '../data')
# Upload the image
image_path = os.path.join(os.path.dirname(__file__), '../data/frames/frame_0244.png')
image = cv2.imread(image_path)


if image is None:
    raise ValueError(f"Could not load image: {image_path}")

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the HSV color at the clicked point
        pixel = hsv_image[y, x]
        print(f"HSV Color at ({x}, {y}): {pixel}")

# Show the image and set the mouse callback
cv2.namedWindow("Pick Color (Click on Racket)", cv2.WINDOW_NORMAL)
cv2.imshow("Pick Color (Click on Racket)", image)

# Set the mouse callback
cv2.setMouseCallback("Pick Color (Click on Racket)", pick_color)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
