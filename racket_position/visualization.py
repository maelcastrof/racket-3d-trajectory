import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_table(ax):
    """
    Draws the 3D table on the given axis.
    """
    table_length = 2.74
    table_width = 1.51

    table_corners = np.array([
        [0, 0, 0],
        [-table_width, 0, 0],
        [0, table_length, 0],
        [-table_width, table_length, 0]
    ])

    vertices = [[table_corners[0], table_corners[1], table_corners[3], table_corners[2]]]
    table_surface = Poly3DCollection(vertices, alpha=0.3, facecolor='blue')
    ax.add_collection3d(table_surface)

    middle_line = np.array([
        [0, table_length / 2, 0],
        [-table_width, table_length / 2, 0]
    ])
    ax.plot(middle_line[:, 0], middle_line[:, 1], middle_line[:, 2], color='blue', linewidth=2, label='Middle Line')

def draw_camera(camera_position, ax):
    """
    Draws the camera position in the 3D plot.
    """
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', label='Camera', s=100)
    ax.text(camera_position[0], camera_position[1], camera_position[2], f'Camera\n({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f})', color='red')
