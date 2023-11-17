import numpy as np
from pyquaternion import Quaternion

lidar_center = np.array([0.985793, 0.0, 1.84019])
lidar_orientation = np.array([0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719])


lidar_quaternion = Quaternion(lidar_orientation).inverse
# lidar_center -= np.array(lidar_center)
lidar_center = np.dot(lidar_quaternion.rotation_matrix, lidar_center)
lidar_orientation = lidar_quaternion * lidar_orientation

cam_center = np.array([1.72200568478, 0.00475453292289, 1.49491291905])
cam_orientation = np.array([0.5077241387638071, -0.4973392230703816, 0.49837167536166627, -0.4964832014373754])

import pdb
pdb.set_trace()
cam_quaternion = Quaternion(cam_orientation).inverse
lidar_center -= np.array(cam_center)
lidar_center = np.dot(cam_quaternion.rotation_matrix, lidar_center)
lidar_orientation = cam_quaternion * lidar_orientation

print(lidar_orientation)