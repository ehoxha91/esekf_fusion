import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import *

# Load data 
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

gt = data['gt']         # 6DOF Pose ground truth.
imu_f = data['imu_f']   # IMU accelerometer measurements
imf_w = data['imu_w']   # IMU gyroscope measurements

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
var_imu_f = 0.1
var_imu_w = 0.1
var_gnss = 10.0
var_lidar = 1.0
# Square them...
var_f = var_imu_f**2
var_w = var_imu_w**2

var_cam = 0.1
var_enc = 0.1

g = np.array([0, 0, -9.81]) # Gravity vector
L = np.zeros([18, 12])
L[3:15, :] = np.eye(12)  # motion model noise jacobian

H_cam = np.zeros([6, 18])
H_cam[0:3, 0:3] = np.eye(3)  # measurement model jacobian
H_cam[3:6, 6:9] = np.eye(3)

H_enc = np.zeros([7,18])
H_enc[0:3, 0:3] = np.eye(3)
H_enc[3:6, 3:6] = np.eye(3)
H_enc[6, 8] = 1

print("Motion Model Jacobian:")
print(L)
print("Measurement H_cam:")
print(H_cam)
print("Measurement H_enc:")
print(H_enc)

# Define vectors and matrices
p_est = np.zeros([imu_f.data.shape[0], 3])      # keep all position history
v_est = np.zeros([imu_f.data.shape[0], 3])      # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])      # orientation estimates as quaternions
a_b_est = np.zeros([imu_f.data.shape[0], 3])    # acceleration bias estimate
w_b_est = np.zeros([imu_f.data.shape[0], 3])    # angular velocity bias estimate
g_est = np.zeros([imu_f.data.shape[0], 3])      # gravity estimate
P_cov = np.zeros([imu_f.data.shape[0], 18, 18]) # covariance matrices at each timestep

# Initial values -- taken from ground truth.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy() # As input we have ground truth in euler angles (roll, pitch, yaw)
a_b_est[0] = gt.p[0]
w_b_est[0] = gt.p[0]
g_est[0] = g

print("Initial Values")
print("##############")
print(p_est[0])
print(v_est[0])
print(q_est[0])
print(a_b_est[0])
print(w_b_est[0])
print(g_est[0])

P_cov[0] = np.zeros(18)  # covariance of estimate
print(P_cov[0])
print("##############")

# Measurement Update:
def measurement_update(sensor_var, H, P_cov_est, y_k, p_est, v_est, q_est, ab_estimate, wb_estimate, g_estimate, camera=True):
    
    # Covariance matrix of the sensor
    if camera == True:
        R = np.identity(6) * sensor_var # Covariance of Camera
    else:
        R = np.identity(7) * sensor_var # Covariance of Encoder

    # Compute Kalman Gain:
    K = P_cov_est.dot(H.T).dot(np.linalg.inv(H.dot(P_cov_est).dot(H.T) + R))

    # Compute the error state:
    dx = K.dot(y_k - p_est)
    
    # Correct the predicted state
    dp = dx[:3]
    dv = dx[3:6]
    dphi = dx[6:9]
    dab = dx[9:12]
    dwb = dx[12:15]
    dg = dx[15:]

    p_upd = p_est + dp
    v_upd = v_est + dv
    q_upd = Quaternion(euler=dphi).quat_mul(q_est)
    a_b_upd = ab_estimate + dab
    w_b_upd = wb_estimate + dwb
    g_upd = g_estimate + dg

    # Correct the covariance:
    P_cov_upd = (np.identity(18) - K.dot(H)).dot(P_cov_est)

    return p_upd, v_upd, q_upd, a_b_upd, w_b_upd, g_upd, P_cov_upd

### Main loop:

for k in range(1, imu_f.data.shape[0]):
    delta_t = imu_f.t[k] - imu_f.t[k-1] # time is given in imu_f

    # 1. Prediction 
    # 1.1 Get Rotation matrix from quaternion
    C_ns = Quaternion(*q_est[k-1]).to_mat() 
    # print(rotation_matrix)

    # 1.2 Estimate the motion
    accel = C_ns.dot(imu_f.data[k-1]) + g
    p_est[k] = p_est[k-1]+delta_t*v_est[k-1] + 0.5*(delta_t**2)*accel
    v_est[k] = v_est[k-1] + delta_t*accel
    q_est[k] = Quaternion(axis_angle=imf_w.data[k-1]*delta_t).quat_mul(q_est[k-1])
    a_b_est[k] = a_b_est[k-1]
    w_b_est[k] = w_b_est[k-1]
    g_est[k] = g_est[k-1]

    # 2. Propagate uncertainty

    F = np.identity(18)
    Q = np.identity(12)

    # F matrix
    F[0:3, 3:6] = delta_t * np.identity(3)
    F[3:6,6:9] = skew_symmetric(np.dot(C_ns,(imu_f.data[k-1]-a_b_est[0])).reshape(3,1))
    Rdt = np.multiply(C_ns,-delta_t)
    F[3:6,9:12] = Rdt
    F[3:6,15:18] = delta_t * np.identity(3)
    F[6:9,12:15] = Rdt

    #print(F.astype(float))
    # Q noise matrix
    Q = delta_t**2 * np.diag([var_f, var_f, var_f, var_w, var_w, var_w, var_f, var_f, var_f, var_w, var_w, var_w])

    P_cov[k] = (F.dot(P_cov[k-1])).dot(F.T) + (L.dot(Q)).dot(L.T)

    camera = True   # This is true whenever we have camera measuerment
    encoder = True  # This is true whenever we have encoder measurement 
    camera_data = np.array([0, 0, 0, 0, 0, 0])      # [px, py, pz, roll, pitch, yaw]
    encoder_data = np.array([0, 0, 0, 0, 0, 0, 0])  # [px, py, pz, vx, vy, vz, theta(yaw)]

    if camera == True:
        p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k], P_cov[k] = measurement_update(var_cam, H_cam, P_cov[k], camera_data, p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k])
    if encoder == True:
        p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k], P_cov[k] = measurement_update(var_enc, H_enc, P_cov[k], encoder_data, p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k])

est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Final Estimated Trajectory')
ax.legend()
ax.set_zlim(-1, 5)
plt.show()