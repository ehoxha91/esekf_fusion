import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import *

# Use ground truth as initialization...
'''
gt = pd.read_csv('data/imu0/ground_truth.csv')
gt_x = gt['x'][:]
gt_y = gt['y'][:]
gt_z = gt['z'][:]
gt_qw = gt['qw'][0]
gt_qx = gt['qx'][0]
gt_qy = gt['qy'][0]
gt_qz = gt['qz'][0]
gt_vx = gt['vx'][0]
gt_vy = gt['vy'][0]
gt_vz = gt['vz'][0]
gt_wbx = gt['w_b_x'][0]
gt_wby = gt['w_b_y'][0]
gt_wbz = gt['w_b_z'][0]
gt_abx = gt['a_b_x'][0]
gt_aby = gt['a_b_y'][0]
gt_abz = gt['a_b_z'][0]

imu = pd.read_csv('data/imu0/data.csv')
imu_t = imu['t'][:]
imu_w_x = imu['w_x'][:]
imu_w_y = imu['w_y'][:]
imu_w_z = imu['w_z'][:]
imu_a_x = imu['a_x'][:]
imu_a_y = imu['a_y'][:]
imu_a_z = imu['a_z'][:]
'''
# cam_t = pd.read_csv('data/f_mono.txt',names='t')
# print(cam_t)
cam_pose = pd.read_csv('data/vodataset.txt')
# cam_t = cam_pose['t'][:]
cam_x = cam_pose['x'][:]
cam_y = cam_pose['y'][:]
cam_z = cam_pose['z'][:]
# cam_qx = cam_pose['qx'][:]
# cam_qy = cam_pose['qy'][:]
# cam_qz = cam_pose['qz'][:]
# cam_qw = cam_pose['qw'][:]

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

# Calibration of lidar ... matching the lidar frame with the global frame.
# C_cam = np.array([
#    [ 0.0148655429818, -0.999880929698, 0.00414029679422],
#    [ 0.999557249008, 0.0149672133247, 0.025715529948],
#    [-0.0257744366974, 0.00375618835797, 0.999660727178 ]
# ])

# t_cam = np.array([-0.0216401454975, -0.064676986768, 0.00981073058949])

# init state variance

na_var   = 2.0e-1  # acc noise (m/s^2)
ng_var   = 5.0e-4  # gyro noise (rad/s)
nba_var  = 1.0e-2    # acc bias noise (m/s^2)
nbg_var  = 1.0e-5    # gyro bias noise (rad/s)

camera_var_pgi_x  = 1.0e-2
camera_var_pgi_y  = 15**(-2)
camera_var_pgi_z  = 12**(-2)

# White Gaussian Noise parameters for all sensors:
var_a = 2.0000e-3
var_w = 1.6968e-04

a_bias = 1.9393e-05
w_bias = 3.0000e-3

g = np.array([0, 0, 9.81]) # Gravity vector
L = np.zeros([18, 12])
L[3:15, :] = np.eye(12)  # motion model noise jacobian

# H_cam = np.zeros([6, 18])
# H_cam[0:3, 0:3] = np.eye(3)  # measurement model jacobian
# H_cam[3:6, 6:9] = np.eye(3)

H_cam = np.zeros([2, 18])    # Simpler model, just x,y,z
H_cam[0:2, 0:2] = np.eye(2)  # measurement model jacobian

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

# Initial values -- taken from ground truth.imu_f.shape[0]
p_est[0] = np.array([0, 0, 0])
v_est[0] = np.array([13.1720, 0.1429, 0.1139])   
q_est[0] = Quaternion(1.0, 0.0, 0.0, 0.0).to_numpy()
a_b_est[0] = np.array([0, 0, 0])
w_b_est[0] = np.array([0, 0, 0])
g_est[0] = np.array([-0.0885830961309836, 0.350638111726515, 9.80333136998259])
P_cov[0] = np.identity(18)*0.01  # covariance of estimate

cam_i = 0   # Count camera data

print_info = True
### Main loop:
for k in range(1, 105):
    # delta_t = imu_t[k] - imu_t[k-1] # time is given in imu_f
    # delta_t = delta_t/1000000000.0  # convert time to seconds from [ns]
    delta_t = .1
    if k < len(gt_x):
        p_gt[k] = np.array([gt_x[k], gt_y[k], gt_z[k]])
    
    if k < len(gt_x):
        p_cam[k] = np.array([cam_x[k], cam_y[k], cam_z[k]])
    
    # Load gyroscope and accelerometer values
    imu_w = np.array([imu_w_x[k-1], imu_w_y[k-1], imu_w_z[k-1]])
    imu_a = np.array([imu_a_x[k-1], imu_a_y[k-1], imu_a_z[k-1]])
    # print(imu_a)
    # print(imu_w)

    # 1. Prediction 
    # 1.1 Get Rotation matrix from quaternion
    C_ns = Quaternion(*q_est[k-1]).to_mat() 
    
    # 1.2 Estimate the motion
    accel = C_ns.dot(imu_a-a_b_est[k-1]) + g
    # print(delta_t*v_est[k-1])
    p_est[k] = p_est[k-1] + delta_t*v_est[k-1] + 0.5*(delta_t**2)*accel
    # print(p_est[k])
    v_est[k] = v_est[k-1] + delta_t*accel
    q_est[k] = Quaternion(euler=(imu_w-w_b_est[k-1])*delta_t).quat_mul(q_est[k-1])
    a_b_est[k] = a_b_est[k-1]
    w_b_est[k] = w_b_est[k-1]
    g_est[k] = g_est[k-1]

    # 2. Propagate uncertainty
    F = np.identity(18)
    Q = np.identity(12)

    # F matrix
    F[0:3, 3:6] = delta_t * np.identity(3)
    F[3:6,6:9] = skew_symmetric(np.dot(C_ns,imu_a-a_b_est[k-1]).reshape(3,1))
    Rdt = np.multiply(C_ns,-delta_t)
    F[3:6,9:12] = Rdt
    F[3:6,15:18] = delta_t * np.identity(3)
    F[6:9,12:15] = Rdt

    # Q noise matrix
    Q = delta_t**2 * np.diag([ng_var, ng_var, ng_var, na_var, na_var, na_var, nba_var, nba_var, nba_var, nbg_var, nbg_var, nbg_var])

    # Update with camera's measurement:
    y_k = np.array([-cam_z[k], cam_x[k]])
    # print(y_k)
    if print_info == True: 
       print("w[rad/s] = {0}".format(imu_w))
       print("a[m/s**2] = {0}".format(imu_a))
       print("C_ns = \n {0}".format(C_ns)) 
       print("Accel [m/s**2] = {0}".format(accel))
       print("p_est_imu[m] = {0}".format(p_est[k]))
       print("v_est_imu[m/s] = {0}".format(v_est[k]))
       print("q_est_imu = {0}".format(q_est[k]))
       print("###### Measurement ######")
       print("y_k[m] = {0}".format(y_k))
    
    # Measurement's covariance matrix
    R = np.identity(2)
    R[0,0] = camera_var_pgi_x
    R[1,1] = camera_var_pgi_y

    # Compute Kalman Gain:
    K = P_cov[k].dot(H_cam.T).dot(np.linalg.inv(H_cam.dot(P_cov[k]).dot(H_cam.T) + R))
    
    # Compute the error state:
    dx = K.dot(y_k - np.array([p_est[k][0], p_est[k][1]]))

    # Correct the predicted state
    dp = dx[:3]
    dv = dx[3:6]
    dphi = dx[6:9]
    dab = dx[9:12]
    dwb = dx[12:15]
    dg = dx[15:]

    p_est[k] = p_est[k] + dp
    v_est[k] = v_est[k] + dv
    q_est[k] = Quaternion(euler=dphi).quat_mul(q_est[k])
    a_b_est[k] = a_b_est[k] + dab
    w_b_est[k] = w_b_est[k] + dwb
    g_est[k] = g_est[k] + dg

    # Correct the covariance:
    P_cov[k] = (np.identity(18) - K.dot(H_cam)).dot(P_cov[k])

    # if print_info == True:
    #    print("###### Corrected ######")
    #    print("dx = {0}".format(dx))
    #    print("p[m] = {0}".format(p_est[k]))
    #    print("v[m/s] = {0}".format(v_est[k]))
    #    print("q = {0}".format(q_est[k]))


#     P_cov[k] = (F.dot(P_cov[k-1])).dot(F.T) + (L.dot(Q)).dot(L.T)
#     if cam_i < len(cam_t) and cam_t[cam_i] == imu_t[k]:
#         # Transform camera data to world frame.
#         # cam_rpy = Quaternion(cam_qw[cam_i], cam_qx[cam_i], cam_qy[cam_i], cam_qz[cam_i]).quat_to_euler()
#         # cam_rpy = C_cam.dot(cam_rpy) + t_cam
#         cam_xyz = np.array([cam_x[cam_i], cam_y[cam_i], cam_z[cam_i]])
#         cam_xyz = C_cam.dot(cam_xyz) + t_cam
        
#         # cam_data = np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], cam_rpy[0], cam_rpy[1], cam_rpy[2]])
#         # p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k], P_cov[k] = measurement_update(0.001, H_cam, P_cov[k], cam_data, p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k])
#         p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k], P_cov[k] = measurement_update(0.001, H_cam, P_cov[k], cam_xyz, p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k])
#         cam_i += 1
# print("We had {0} camera corrections".format(cam_i))

est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111)
ax.plot(p_est[:,1], p_est[:,0], label='Estimated')
ax.plot(-p_cam[:,2], p_cam[:,0], label='Camera')
ax.plot(-p_gt[:,1], p_gt[:,0], label='Ground Truth')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
ax.set_title('Final Estimated Trajectory')
ax.legend()
# ax.set_zlim(-1, 5)
plt.show()