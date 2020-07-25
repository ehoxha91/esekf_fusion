import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import *

#TODO: Write a function to load data for vectors:
data_length_gt = 100                            
data_length_imu = 100
data_length_cam = 100
data_length_enc = 100

# Allocate vectors
gt = np.zeros([data_length_gt, 3])              # 6DOF Pose ground truth.
imu_f = np.zeros([data_length_imu, 3])          # IMU accelerometer measurements
imf_w = np.zeros([data_length_imu, 3])          # IMU gyroscope measurements
camera_data = np.zeros([data_length_cam, 6])    # [px, py, pz, roll, pitch, yaw]
encoder_data = np.zeros([data_length_enc, 6])   # [px, py, pz, vx, vy, vz]

# TODO: Update calibration matrices with real ones.

# Camera
R_cam = np.array([
   [ 1, 0, 0],
   [ 0, 1, 0],
   [ 0, 0, 1 ]
])
t_cam = np.array([0, 0, 0])
camera_data[:,0:3] = R_cam.dot(camera_data[:,0:3].T).T + t_cam
camera_data[:,3:6] = R_cam.dot(camera_data[:,3:6].T).T + t_cam

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

encoder_data[:,0:3] = R_cam.dot(encoder_data[:,0:3].T).T + t_cam # Transform [px, py, pz]
encoder_data[:,3:6] = R_cam.dot(encoder_data[:,3:6].T).T + t_cam # Transform [vx, vy, vz]


# White Gaussian Noise parameters for all sensors:
var_f = 0.1     # Acceleration noise            - IMU
var_w = 0.1     # Angular velocity noise        - IMU
var_a_b = 0.1   # Acceleration bias noise       - IMU
var_w_b = 0.1   # Angular velocity bias noise   - IMU
var_cam = 0.1   # Camera                        - Visual Odometry
var_enc = 0.1   # Encoder                       - Wheel Odometry

g = np.array([0, 0, -9.81]) # Gravity vector
L = np.zeros([18, 12])
L[3:15, :] = np.eye(12)  # motion model noise jacobian

H_cam = np.zeros([6, 18])
H_cam[0:3, 0:3] = np.eye(3)  # measurement model jacobian
H_cam[3:6, 6:9] = np.eye(3)

H_enc = np.zeros([6,18])
H_enc[0:6, 0:6] = np.eye(6)


print("Motion Model Jacobian:")
print(L)
print("Measurement H_cam:")
print(H_cam)
print("Measurement H_enc:")
print(H_enc)

# Define vectors and matrices
p_est = np.zeros([imu_f.shape[0], 3])      # keep all position history
v_est = np.zeros([imu_f.shape[0], 3])      # velocity estimates
q_est = np.zeros([imu_f.shape[0], 4])      # orientation estimates as quaternions
a_b_est = np.zeros([imu_f.shape[0], 3])    # acceleration bias estimate
w_b_est = np.zeros([imu_f.shape[0], 3])    # angular velocity bias estimate
g_est = np.zeros([imu_f.shape[0], 3])      # gravity estimate
P_cov = np.zeros([imu_f.shape[0], 18, 18]) # covariance matrices at each timestep

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
    R = np.identity(6) * sensor_var

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
for k in range(1, len(imu_f)):
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

    # Q noise matrix
    Q = delta_t**2 * np.diag([var_f, var_f, var_f, var_w, var_w, var_w, var_a_b, var_a_b, var_a_b, var_w_b, var_w_b, var_w_b])

    P_cov[k] = (F.dot(P_cov[k-1])).dot(F.T) + (L.dot(Q)).dot(L.T)

    camera = True   # This is true whenever we have camera measuerment
    encoder = True  # This is true whenever we have encoder measurement 

    if camera == True:
        p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k], P_cov[k] = measurement_update(var_cam, H_cam, P_cov[k], camera_data[k], p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k])
    if encoder == True:
        p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k], P_cov[k] = measurement_update(var_enc, H_enc, P_cov[k], encoder_data[k], p_est[k], v_est[k], q_est[k], a_b_est[k], w_b_est[k], g_est[k])

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