# Helper python class

import numpy as np

def skew_symmetric(v):
    """Skew symmetric form of a 3x1 vector."""
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]], dtype=np.float64)

class Quaternion():

    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0, euler=None, axis_angle=None):
        
        if euler is not None:
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)

            self.w = cr * cp * cy + sr * sp * sy
            self.x = sr * cp * cy - cr * sp * sy
            self.y = cr * sp * cy + sr * cp * sy
            self.z = cr * cp * sy - sr * sp * cy
        elif axis_angle is not None:   # as input is axis angle representation...
            axis_angle = np.array(axis_angle) # convert it to np.array
            norm = np.linalg.norm(axis_angle) # calculate the norm
            self.w = np.cos(norm/2)
            if norm < 1e-50:    # if change is too small.
                self.x = 0
                self.y = 0
                self.z = 0
            else:
                # q = [ q_w, q_v].T = [cos(norm/2), (axis_angle/norm)*sin(norm/2)]
                q_v = axis_angle / norm * np.sin(norm/2)
                self.x = q_v[0].item()
                self.y = q_v[1].item()
                self.z = q_v[2].item()      
        else:
            self.w = w
            self.x = x
            self.y = y
            self.z = z

    def to_numpy(self):
        return np.array([self.w, self.x, self.y, self.z])

    def to_mat(self):
        R = np.zeros([3, 3])
        qx2 = self.x**2
        qy2 = self.y**2
        qz2 = self.z**2

        qxy = self.x*self.y
        qxz = self.x*self.z
        qyz = self.y*self.z

        qxw = self.x*self.w
        qyw = self.y*self.w
        qzw = self.z*self.w

        R[0,0] = 1 - 2*(qy2+qz2)
        R[0,1] = 2*(qxy - qzw)
        R[0,2] = 2*(qxz + qyw)
        R[1,0] = 2*(qxy + qzw)
        R[1,1] = 1 - 2*(qx2 + qz2)
        R[1,2] = 2*(qyz - qxw)
        R[2,0] = 2*(qxz - qyw)
        R[2,1] = 2*(qyz + qxw)
        R[2,2] = 1 - 2*(qx2 + qy2)

        return R

    def quat_to_euler(self):
        ysqr = self.y * self.y

        t0 = +2.0 * (self.w * self.x + self.y * self.z)
        t1 = +1.0 - 2.0 * (self.x * self.x + ysqr)
        roll = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (self.w * self.y - self.z * self.x)
        t2 = np.where(t2>+1.0,+1.0,t2)

        t2 = np.where(t2<-1.0, -1.0, t2)
        pitch = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (self.w * self.z + self.x * self.y)
        t4 = +1.0 - 2.0 * (ysqr + self.z * self.z)
        yaw = np.degrees(np.arctan2(t3, t4))

        return np.array([roll, pitch, yaw])

    def quat_mul(self, q, out='np', right=True):
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]

        if right is True:
            sum_term[1:, 1:] = -skew_symmetric(v)
        else:
            sum_term[1:, 1:] = skew_symmetric(v)

        sigma = self.w * np.eye(4) + sum_term
        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj
    
    def q2r(self):
        ''' Convert from quaternion to direct cosine matrix
        '''
        #normalize quaternion
        q_norm = self.w**2 + self.x**2 + self.y**2 + self.z**2
        q_norm = np.sqrt(q_norm)

        self.w = self.w/q_norm
        self.x = self.x/q_norm
        self.y = self.y/q_norm
        self.z = self.z/q_norm

        a = self.w
        b = self.x
        c = self.y
        d = self.z
        R = np.zeros([3,3])
        R[0,0] = a*a+b*b-c*c-d*d
        R[0,1] = 2*(b*c-a*d)
        R[0,2] = 2*(b*d+a*c)
        R[1,0] = 2*(b*c+a*d)
        R[1,1] = a*a-b*b+c*c-d*d
        R[1,2] = 2*(c*d-a*b)
        R[2,0] =  2*(b*d-a*c)
        R[2,1] = 2*(c*d+a*b)
        R[2,2] = a*a-b*b-c*c+d*d
        return R
