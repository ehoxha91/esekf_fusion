import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import *

# Load data from: camera, imu and ground truth.

cam_pose = pd.read_csv('data/vodataset.txt')
cam_x = cam_pose['x'][:]
cam_y = cam_pose['y'][:]
cam_z = cam_pose['z'][:]
cam_qx = cam_pose['qx'][:]
cam_qy = cam_pose['qy'][:]
cam_qz = cam_pose['qz'][:]
cam_qw = cam_pose['qw'][:]

imu = pd.read_csv('data/imu.txt')
imu_t = imu['w_x'][:]
imu_w_x = imu['w_x'][:]
imu_w_y = imu['w_y'][:]
imu_w_z = imu['w_z'][:]
imu_a_x = imu['a_x'][:]
imu_a_y = imu['a_y'][:]
imu_a_z = imu['a_z'][:]

gt = pd.read_csv('data/gt.txt')
gt_x = gt['x'][:]
gt_y = gt['y'][:]
gt_z = gt['z'][:]

# Transform data to world frame:
# Camera
R_cam = np.array([
   [ 1, 0, 0],
   [ 0, 1, 0],
   [ 0, 0, 1 ]
])
t_cam = np.array([0, 0, 0])

# IMU
R_imu = np.array([
   [ 1, 0, 0],
   [ 0, 1, 0],
   [ 0, 0, 1 ]
])
t_imu = np.array([0, 0, 0])

# Encoder
R_enc = np.array([
   [ 1, 0, 0],
   [ 0, 1, 0],
   [ 0, 0, 1 ]
])
t_enc = np.array([0, 0, 0])

# IMU noise:
var_a = 0.003
var_w = 0.003

# gravity vector
g = np.array([0.0, 0.0, 9.81])

# IMU Motion Model Noise Jacobian: df/dn
L = np.zeros([9, 6])
L[3:9, :] = np.eye(6)  # motion model noise jacobian

# Measurement Jacobian
H_cam = np.zeros([3, 9])     # [x,y,z]
H_cam[0:3, 0:3] = np.eye(3)   # measurement model jacobian
 
# Declare state variables and initialize the arrays:
p_est = np.zeros([105, 3])       # keep all position history
v_est = np.zeros([105, 3])       # velocity estimates
q_est = np.zeros([105, 4])       # orientation estimates as quaternions
a_b_est = np.zeros([105, 3])     # acceleration bias state
w_b_est = np.zeros([105, 3])     # gyro bias state

P_cov = np.zeros([105, 9, 9])    # covariance matrices at each timestep
p_gt = np.zeros([105, 3])        # ground truth data
p_cam = np.zeros([105, 3])       # camera data

p_est[0] = np.array([gt_x[0], gt_y[0], gt_z[0]])          
v_est[0] = np.array([0, 0, 0])
q_est[0] = Quaternion(1.0, 0.0, 0.0, 0.0).to_numpy()
P_cov[0] = np.identity(9)*0.01

cam_i = 0                              # count camera data
print_info = False
# Main loop:
for k in range(1, 105):
   delta_t = .1   # This is the period for Eric's dataset.

   # fill ground truth vector and camera vector for plotting
   if k < len(gt_x):
      p_gt[k] = np.array([gt_x[k], gt_y[k], gt_z[k]])
   if k < len(gt_x):
      p_cam[k] = np.array([cam_x[k], cam_y[k], cam_z[k]])

   # Load gyroscope and accelerometer values
   imu_w = np.array([imu_w_x[k-1], imu_w_y[k-1], imu_w_z[k-1]])
   imu_a = np.array([imu_a_x[k-1], imu_a_y[k-1], imu_a_z[k-1]])

   # Get rotation matrix from quaternion
   C_ns = Quaternion(*q_est[k-1]).to_mat()

   # Update the nominal state by using IMU
   accel = C_ns.dot(imu_a) + g
   
   p_est[k] = p_est[k-1] + delta_t*v_est[k-1] + 0.5*(delta_t**2)*accel
   v_est[k] = v_est[k-1] + delta_t*accel
   q_est[k] = Quaternion(axis_angle=(imu_w*delta_t)).quat_mul(q_est[k-1])
   


   F = np.identity(9)
   Q = np.identity(6)

   F[:3, 3:6] = delta_t * np.identity(3)
   F[3:6, 6:] = -(C_ns.dot(skew_symmetric(imu_a.reshape((3,1)))))*delta_t
   Q = delta_t**2 * np.diag([var_w, var_w, var_w, var_a, var_a, var_a])
   P_cov[k] = (F.dot(P_cov[k-1])).dot(F.T) + (L.dot(Q)).dot(L.T)

   
   # Update with camera's measurement:
   # y_k = np.array([cam_x[k], cam_y[k], cam_z[k]])

   # if print_info == True: 
   #    print("w[rad/s] = {0}".format(imu_w))
   #    print("a[m/s**2] = {0}".format(imu_a))
   #    print("C_ns = \n {0}".format(C_ns)) 
   #    print("Accel [m/s**2] = {0}".format(accel))
   #    print("p_est_imu[m] = {0}".format(p_est[k]))
   #    print("v_est_imu[m/s] = {0}".format(v_est[k]))
   #    print("q_est_imu = {0}".format(q_est[k]))
   #    print("###### Measurement ######")
   #    print("y_k[m] = {0}".format(y_k))
   
   # # Measurement's covariance matrix
   # R = np.identity(3)*0.001

   # # Compute Kalman Gain:
   # K = P_cov[k].dot(H_cam.T).dot(np.linalg.inv(H_cam.dot(P_cov[k]).dot(H_cam.T) + R))

   # # Compute the error state:
   # dx = K.dot(y_k - p_est[k])

   # # Correct the predicted state
   # dp = dx[:3]
   # dv = dx[3:6]
   # dphi = dx[6:9]

   # p_est[k] = p_est[k] + dp
   # v_est[k] = v_est[k] + dv
   # q_est[k] = Quaternion(axis_angle=dphi).quat_mul(q_est[k])

   # # Correct the covariance:
   # P_cov[k] = (np.identity(9) - K.dot(H_cam)).dot(P_cov[k])

   # if print_info == True:
   #    print("###### Corrected ######")
   #    print("dx = {0}".format(dx))
   #    print("p[m] = {0}".format(p_est[k]))
   #    print("v[m/s] = {0}".format(v_est[k]))
   #    print("q = {0}".format(q_est[k]))

est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111)
ax.plot(p_est[:,1], -p_est[:,0], label='Estimated')
ax.plot(-p_gt[:,1], p_gt[:,0], label='Ground Truth')
ax.plot(p_cam[:,1], p_cam[:,0], label='Camera')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Final Estimated Trajectory')
ax.legend()
plt.show()