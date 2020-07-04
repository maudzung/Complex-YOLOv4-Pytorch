import numpy as np

class_list = ["Car", "Pedestrian", "Cyclist"]

CLASS_NAME_TO_ID = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Van': 0,
    'Person_sitting': 1,
}

# Front side (of vehicle) Point Cloud boundary for BEV
boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

# Back back (of vehicle) Point Cloud boundary for BEV
boundary_back = {
    "minX": -50,
    "maxX": 0,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

BEV_WIDTH = 608  # across y axis -25m ~ 25m
BEV_HEIGHT = 608  # across x axis 0m ~ 50m

DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]

# Following parameters are calculated as an average from KITTI dataset for simplicity
#####################################################################################
Tr_velo_to_cam = np.array([
    [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0, 0, 0, 1]
])

# cal mean from train set
R0 = np.array([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
])

P2 = np.array([[719.787081, 0., 608.463003, 44.9538775],
               [0., 719.787081, 174.545111, 0.1066855],
               [0., 0., 1., 3.0106472e-03],
               [0., 0., 0., 0]
               ])

R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)
#####################################################################################
