import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to ignore the top part of the image
def ignore_top_part(image, percentage_ignore):
    h, w = image.shape[:2]
    recorte = int(h * percentage_ignore)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[recorte:, :] = 255  # Solo conserva la parte inferior
    return cv2.bitwise_and(image, image, mask=mask)



# Relative path to the data folder
data_folder = os.path.join(os.path.dirname(__file__), '../data')
# Upload the image
image_path = os.path.join(os.path.dirname(__file__), '../data/frames/frame_0244.png')
image = cv2.imread(image_path)

percentage_ignore = 0.7  # 50%
image_inferior = ignore_top_part(image, percentage_ignore)

# Pre-process the image
gray = cv2.cvtColor(image_inferior, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, threshold1=50, threshold2=150)

# Detect lines using the Hough Transform
lines = cv2.HoughLinesP(
    edges, 
    rho=1, 
    theta=np.pi / 180, 
    threshold=80, 
    minLineLength=100, 
    maxLineGap=10
)

# Create an empty image to draw lines
output_image = image.copy()

# Find table edges
if lines is not None:
    # Collect all detected lines
    table_edges = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        table_edges.append(((x1, y1), (x2, y2)))
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Function to find intersections
def find_intersection(line1, line2):
    x1, y1, x2, y2 = *line1[0], *line1[1]
    x3, y3, x4, y4 = *line2[0], *line2[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)


# Display the final output
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Ping-Pong Table Edges")
plt.axis("off")
plt.show()


##################
# Function to extend detected lines to cover the entire image
def extend_lines_simple(lines, width, height):
    extended_lines = []
    for line in lines:
        x1, y1, x2, y2 = line.flatten()

        # Calculate the slope and extend the line
        if y1 != y2:  # If not a horizontal line
            m = (x2 - x1) / (y2 - y1)
            x_top = int(x1 + m * (0 - y1))  # Extend to the top edge (y=0)
            x_bottom = int(x1 + m * (height - 1 - y1))  # Extend to the bottom edge (y=height-1)
            extended_lines.append([(x_top, 0), (x_bottom, height - 1)])
        else:  # Horizontal line
            extended_lines.append([(0, y1), (width - 1, y1)])
    return extended_lines

# Extend the detected lines
if lines is not None:
    height, width = image.shape[:2]
    extended_lines = extend_lines_simple(lines, width, height)

    # Draw all extended lines on a copy of the image
    output_image = image.copy()
    for line in extended_lines:
        cv2.line(output_image, line[0], line[1], (255, 0, 0), 1)  # Blue for extended lines

    # Show the image with the extended lines
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Extended Lines")
    plt.axis("off")
    plt.show()
else:
    print("No lines were detected.")

def group_extended_lines(extended_lines, threshold_dist=10, threshold_angle=np.deg2rad(10)):
    used = set()
    grouped_lines = []

    for i, line1 in enumerate(extended_lines):
        if i in used:
            continue

        current_group = [line1]
        used.add(i)
        x1a, y1a, x2a, y2a = map(float, (*line1[0], *line1[1]))

        for j, line2 in enumerate(extended_lines):
            if j in used:
                continue

            x1b, y1b, x2b, y2b = map(float, (*line2[0], *line2[1]))

            # Calculate the average distance between the two lines
            dist1 = np.sqrt((x1a - x1b)**2 + (y1a - y1b)**2)
            dist2 = np.sqrt((x2a - x2b)**2 + (y2a - y2b)**2)
            avg_dist = (dist1 + dist2) / 2

            # Calculate the angle between the two lines
            angle1 = np.arctan2(y2a - y1a, x2a - x1a)
            angle2 = np.arctan2(y2b - y1b, x2b - x1b)
            angle_diff = np.abs(angle1 - angle2)

            if avg_dist < threshold_dist and angle_diff < threshold_angle:
                current_group.append(line2)
                used.add(j)

        grouped_lines.append(current_group)

    return grouped_lines

# Extend lines across the entire image (already defined previously)

# Group extended lines
if lines is not None:
    height, width = image.shape[:2]

    # Extend the detected lines
    extended_lines = extend_lines_simple(lines, width, height)

    # Group extended lines
    grouped_lines = group_extended_lines(extended_lines, threshold_dist=400, threshold_angle=np.deg2rad(200))
    index_to_remove = 1
    filtered_grouped_lines = [line for i, line in enumerate(grouped_lines) if i != index_to_remove]

    # Create a new image to show only the grouped lines
    output_image_grouped = image.copy()

    for group in filtered_grouped_lines:
        for line in group:
            pt1 = tuple(map(int, line[0]))
            pt2 = tuple(map(int, line[1]))
            cv2.line(output_image_grouped, pt1, pt2, (0, 0, 255), 2)  # Red for grouped lines

    # Show the resulting image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image_grouped, cv2.COLOR_BGR2RGB))
    plt.title("Grouped Lines (Red)")
    plt.axis("off")
    plt.show()
else:
    print("No lines were detected.")

# # Image size
# image_height, image_width = image.shape[:2]

# # Function to calculate the parametric form (a, b, c) for each line
# def calculate_line_parameters(x1, y1, x2, y2):
#     a = y2 - y1
#     b = x1 - x2
#     c = x2 * y1 - x1 * y2
#     return a, b, c

# # Find the intersection of two lines given their parameters
# def calculate_intersection(l1, l2):
#     a1, b1, c1 = l1
#     a2, b2, c2 = l2
#     determinant = a1 * b2 - a2 * b1
#     if determinant == 0:  # Lines are parallel
#         return None
#     x = (b1 * c2 - b2 * c1) / determinant
#     y = (a2 * c1 - a1 * c2) / determinant
#     return int(x), int(y)

# # Calculate the parameters of all lines
# line_parameters = [
#     calculate_line_parameters(x1, y1, x2, y2)
#     for line in filtered_grouped_lines
#     for (x1, y1), (x2, y2) in [line]  # Unpacking two tuples
# ]



# # Calculate intersection points
# intersected_points = []
# for i in range(len(line_parameters)):
#     for j in range(i + 1, len(line_parameters)):
#         point = calculate_intersection(line_parameters[i], line_parameters[j])
#         if point is not None:
#             x, y = point
#             # Check if the intersection is within the image
#             if 0 <= x < image_width and 0 <= y < image_height:
#                 intersected_points.append((x, y))

# intersected_points = intersected_points[0], intersected_points[1], intersected_points[3], intersected_points[4]

# # Draw the intersection points on the image
# output_image = image.copy()
# for (x, y) in intersected_points:
#     cv2.circle(output_image, (x, y), 15, (0, 255, 0), -1)  # Green for the points

# # Draw grouped lines (in red)
# for line in filtered_grouped_lines:
#     cv2.line(output_image, line[0], line[1], (0, 0, 255), 2)  # Red for grouped lines

# # Show the image with the intersection points
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
# plt.title("Intersection Points Within the Image")
# plt.axis("off")
# plt.show()

# # Display the intersection points
# print("Intersection Points:")
# print(intersected_points)

# # Assign meaningful variable names to intersection points
# corner_bottom_left = intersected_points[0]
# corner_bottom_right = intersected_points[1]
# ref_point = intersected_points[2]  # Top-left corner
# corner_top_right = intersected_points[3]

# # Draw lines connecting the selected points on the image
# output_image_lines = image.copy()
# cv2.line(output_image_lines, corner_bottom_left, corner_bottom_right, (0, 0, 255), 2)  # Red line
# cv2.line(output_image_lines, corner_bottom_left, ref_point, (0, 0, 255), 2)  # Red line
# cv2.line(output_image_lines, ref_point, corner_top_right, (0, 0, 255), 2)  # Red line
# cv2.line(output_image_lines, corner_bottom_right, corner_top_right, (0, 0, 255), 2)  # Red line

# # Draw intersection points (in green)
# for (x, y) in intersected_points:
#     cv2.circle(output_image_lines, (x, y), 15, (0, 255, 0), -1)  # Green for the points

# # Show the image with connected lines
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(output_image_lines, cv2.COLOR_BGR2RGB))
# plt.title("Lines Connecting Selected Points")
# plt.axis("off")
# plt.show()


