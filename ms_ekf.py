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

# Noise for acceleration and gyroscope
na_var   = 2.0e-1   # acc noise (m/s^2)
ng_var   = 5.0e-4   # gyro noise (rad/s)
nba_var  = 1.0e-2   # acc bias noise (m/s^2)
nbg_var  = 1.0e-5   # gyro bias noise (rad/s)
a_bias = 1.9393e-05
w_bias = 3.0000e-3

camera_var_pgi_x  = 1.0e-2
camera_var_pgi_y  = 15**(-2)
camera_var_pgi_z  = 12**(-2)

# Define vectors and matrices
p_est = np.zeros([105, 3])       # keep all position history
v_est = np.zeros([105, 3])       # velocity estimates
q_est = np.zeros([105, 4])       # orientation estimates as quaternions
a_b_est = np.zeros([105, 3])     # acceleration bias estimate
w_b_est = np.zeros([105, 3])     # angular velocity bias estimate
g_est = np.zeros([105, 3])       # gravity estimate
P_cov = np.zeros([105, 18, 18])  # covariance matrices at each timestep
p_gt = np.zeros([105, 3])        # ground truth data
p_cam = np.zeros([105, 3])       # save camera pose

# Initialize 
p_est[0] = np.array([0, 0, 0])
v_est[0] = np.array([13.1720, 0.1429, 0.1139])   
q_est[0] = Quaternion(1.0, 0.0, 0.0, 0.0).to_numpy()
a_b_est[0] = np.array([0, 0, 0])
w_b_est[0] = np.array([0, 0, 0])
g = np.array([-0.0885830961309836, 0.350638111726515, 9.80333136998259])
# P_cov[0] = np.identity(18)*0.01  # covariance of estimate

def preditct(i, dt, w, a):
    global g
    OM = 0.5*np.array(
        [[0, -w[0], -w[1], -w[2]],
         [w[0], 0, w[2], -w[1]],
         [w[1], -w[2], 0, w[0]],
         [w[2], w[1], -w[0], 0]], dtype=np.float64)

    p_est[i] = p_est[i-1] + dt*v_est[i-1]    # position update
    v_est[i] = v_est[i-1] + dt*((Quaternion(*q_est[i-1]).q2r()).dot(a-a_b_est[i-1])-g)
    q_est[i] = q_est[i-1] + dt*OM.dot(q_est[i-1])
    a_b_est[i] = 0
    w_b_est[i] = 0
    return p_est[i], v_est[i], q_est[i], a_b_est[i], w_b_est[i]


for k in range(1, 105):
    
    if k < len(gt_x):
        p_gt[k] = np.array([gt_x[k-1], gt_y[k-1], gt_z[k-1]])
    if k < len(gt_x):
        p_cam[k] = np.array([cam_x[k-1], cam_y[k-1], cam_z[k-1]])
    
    # Load gyroscope and accelerometer values
    w = np.array([imu_w_x[k-1], imu_w_y[k-1], imu_w_z[k-1]])
    a = np.array([imu_a_x[k-1], imu_a_y[k-1], imu_a_z[k-1]])
    p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k] = preditct(k, 0.1, w, a)

est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111)
ax.plot(-p_est[:,1], p_est[:,0], label='Estimated')
ax.plot(-p_cam[:,2], p_cam[:,0], label='Camera')
ax.plot(-p_gt[:,1], p_gt[:,0], label='Ground Truth')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
ax.set_title('Final Estimated Trajectory')
ax.legend()
# ax.set_zlim(-1, 5)
plt.show()