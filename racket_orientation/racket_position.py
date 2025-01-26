import numpy as np

table_length = 2.74  
table_width = 1.51   
table_height = 0  

def calculate_scale_and_world_coordinates(projection_matrix, image_point, world_point_homogeneous):
    """
    Calculate the scale factor (s) and reconstruct the 3D world coordinates from the projection matrix.
    
    :param projection_matrix: 3x4 projection matrix (P)
    :param image_point: 2D image point (u, v) in pixels
    :param world_point_homogeneous: 3D world point in homogeneous coordinates (X^w, Y^w, Z^w, 1)
    :return: Scale factor (s) and reconstructed 3D world coordinates
    """
    # Extract u, v from the image point
    u, v = image_point

    # Compute the scale factor (s)
    P = projection_matrix
    s_numerator = P[2, 0] * world_point_homogeneous[0] + \
                  P[2, 1] * world_point_homogeneous[1] + \
                  P[2, 2] * world_point_homogeneous[2] + \
                  P[2, 3]
    s_denominator = 1.0  # Normalization factor for homogeneous coordinates
    s = s_numerator / s_denominator

    # Reconstruct the world coordinates in pixels
    reconstructed_u = (P[0, 0] * world_point_homogeneous[0] +
                       P[0, 1] * world_point_homogeneous[1] +
                       P[0, 2] * world_point_homogeneous[2] +
                       P[0, 3]) / s
    reconstructed_v = (P[1, 0] * world_point_homogeneous[0] +
                       P[1, 1] * world_point_homogeneous[1] +
                       P[1, 2] * world_point_homogeneous[2] +
                       P[1, 3]) / s

    # Return the scale factor and the reconstructed world coordinates
    return s, (reconstructed_u, reconstructed_v)

def find_racket_world_coordinates(projection_matrix, image_point, scale_factor):
    """
    Calculate the 3D world coordinates of the racket given its 2D image coordinates and scale factor.
    
    :param projection_matrix: 3x4 projection matrix (P)
    :param image_point: 2D image point (u, v) in pixels
    :param scale_factor: Scale factor (s)
    :return: 3D world coordinates (X^w, Y^w, Z^w)
    """
    # Convert 2D image point to homogeneous coordinates
    u, v = image_point
    uv1 = np.array([u, v, 1]) * scale_factor  # Scale the homogeneous coordinates

    # Extend projection matrix to 4x4 by adding a row [0, 0, 0, 1]
    P_extended = np.vstack([projection_matrix, [0, 0, 0, 1]])

    # Compute the inverse of the projection matrix
    P_inv = np.linalg.pinv(P_extended)

    # Calculate the 3D world coordinates in homogeneous form
    world_coordinates_homogeneous = P_inv @ np.append(uv1, 1)

    # Convert from homogeneous to Euclidean coordinates
    Xw, Yw, Zw, W = world_coordinates_homogeneous
    Xw /= W
    Yw /= W
    Zw /= W

    return Xw, Yw, Zw