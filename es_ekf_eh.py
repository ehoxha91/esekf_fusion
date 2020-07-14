import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import *
# from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


# Load data 
with open('data/pt3_data.pkl', 'rb') as file:
    data = pickle.load(file)


gt = data['gt']         # 6DOF Pose ground truth.
imu_f = data['imu_f']   # IMU accelerometer measurements
imf_w = data['imu_w']   # IMU gyroscope measurements
gnss = data['gnss']     # GPS measured position.
lidar = data['lidar']   # Lidar measured position. 

# Print a row of data just to understand what we have as input.
# print(gt.p[0,:])        # Ground truth p   - [gt_px, gt_py, gt_pz]
# print(gt.r[0,:])        # Euler angle      - [roll, pitch, yaw]
# print(imu_f.data[0,:])  # IMU Acceleration - [a_x, a_y, a_z]
# print(imu_w.data[0,:])  # IMU Gyro rate.   - [g_x, g_y, g_z]
# print(gnss.data[0,:])   # GPS position     - [g_px, g_py, g_pz]
# print(lidar.data[0,:])  # Lidar position   - [l_px, l_py, l_pz]

# Plot ground truth:
# gt_fig = plt.figure()
# ax = gt_fig.add_subplot(111, projection='3d')
# ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.set_title('Ground Truth trajectory')
# ax.set_zlim(-1, 5)
# plt.show()

# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
# Calibration of lidar ... matching the lidar frame with the global frame.
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])
t_i_li = np.array([0.5, 0.1, 0.5])

lidar.data = C_li.dot(lidar.data.T).T + t_i_li # Transform lidar measurements to the world frame.

# White Gaussian Noise parameters for all sensors:
var_imu_f = 0.01
var_imu_w = 0.01
var_gnss = 10.0
var_lidar = 1.0

g = np.array([0, 0, -9.81]) # Gravity vector
L = np.zeros([9, 6])
L[3:, :] = np.eye(6)  # motion model noise jacobian
H = np.zeros([3, 9])
H[:, :3] = np.eye(3)  # measurement model jacobian

# print("Motion Model Jacobian:")
# print(L)
# print("Measurement Model Jacobian:")
# print(H)

# Define vectors and matrices
p_est = np.zeros([imu_f.data.shape[0], 3])      # keep all position history
v_est = np.zeros([imu_f.data.shape[0], 3])      # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])      # orientation estimates as quaternions
P_cov = np.zeros([imu_f.data.shape[0], 9, 9])   # covariance matrices at each timestep


# Initial values -- taken from ground truth.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]

# As input we have ground truth in euler angles (roll, pitch, yaw)
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
# print(q_est[0])

P_cov[0] = np.zeros(9)  # covariance of estimate
# print(P_cov[0])

gnss_i = 0
lidar_i = 0


### Main loop:

for k in range(1, imu_f.data.shape[0]):
    delta_t = imu_f.t[k] - imu_f.t[k-1] # time is given in imu_f

# 1. Prediction 
    # 1.1 Get Rotation matrix from quaternion
    rotation_matrix = Quaternion(*q_est[k-1]).to_mat() 
    # print(rotation_matrix)

    # 1.2 Estimate the motion
    accel = rotation_matrix.dot(imu_f.data[k-1]) + g
    p_est[k] = p_est[k-1]+delta_t*v_est[k-1] + ((delta_t**2)/2)*accel
    v_est[k] = v_est[k-1] + delta_t*accel
    q_est[k] = Quaternion(axis_angle=imf_w.data[k-1]*delta_t).
    print(p_est[k])
    print(v_est[k])
    