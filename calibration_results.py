# Calibration Results

import numpy as np

# Calibration results for front camera (2000 frames)
front_camera_matrix = np.array([
    [1.10929294e+03, 0.00000000e+00, 9.68244406e+02],
    [0.00000000e+00, 1.10828076e+03, 5.32739654e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

front_dist_coeffs = np.array([[-0.07595659,  0.90237918,  0.00257907,  0.02142784, -1.27778586]])

# Calibration results for lateral camera (1100 frames)
lateral_camera_matrix = np.array([
    [1.19733085e+03, 0.00000000e+00, 9.59615426e+02],
    [0.00000000e+00, 1.23197655e+03, 4.08899254e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

lateral_dist_coeffs = np.array([[3.57467207, -14.86052367, -0.2714199, 0.03699078, -9.08131138]])

# Calibration results for 45-degree camera (2160 frames)
degree45_camera_matrix = np.array([
    [1.12944047e+03, 0.00000000e+00, 1.04936865e+03],
    [0.00000000e+00, 1.06185468e+03, 5.55655991e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

degree45_dist_coeffs = np.array([[-0.43673283, -2.01713805, -0.04056567, 0.08416871, 9.82369894]])
