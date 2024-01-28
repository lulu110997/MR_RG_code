import math
import sys
import time
import cv2
from spatialmath import SE3
from math import pi
from spatialmath.base import trplot, tranimate, ishom
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

a_line = lambda x, y: np.vstack([x,y]).T

def skew_beta(x, M):
    assert x.ndim == 1
    assert isinstance(M, np.ndarray)
    return np.linalg.inv(M)@skew(x)@M

def skew(x):
    assert x.ndim == 1
    if x.shape[0] == 3:
        s = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])
    elif x.shape[0] == 6:
        s = np.array([[0, -x[2], x[1], x[3]],
                      [x[2], 0, -x[0], x[4]],
                      [-x[1], x[0], 0, x[5]],
                      [0, 0, 0, 0]])
    else:
        raise "x must be of shape (3,) or (6,)"
    return s

def inter():
    plt.draw()
    plt.legend()
    input()

def to_SE3(omega, theta, v):
    I = np.eye(3)
    skew_omega = skew(omega.flatten())
    bot = np.array([[0, 0, 0, 1]])

    rot = expm(skew_omega*theta)  # Matrix exponential operator not same as **
    assert (np.abs(rot.transpose() - np.linalg.inv(rot)) < 0.0001).all()  # Ensure we have a matrix in the SO(3)
    G = I*theta + (1-math.cos(theta))*skew_omega + (theta - math.sin(theta))*(np.linalg.matrix_power(skew_omega,2))
    np.power
    transl = (G@v.transpose()).reshape(3,1)
    return np.vstack((np.hstack((rot, transl)), bot))

def get_screw_axis_revolute(s_hat, q):
    theta_dot = 1  # angular velocity

    # instantenous linear velocity. Also the translation due to rotation about s_hat.
    # Occurs in the plane orthogonal to s_hat
    v = -np.cross(s_hat, q)

    # Calculate omega
    omega = s_hat*theta_dot
    omega_norm = np.linalg.norm(omega)

    # Calculate h (ratio of the linear velocity along the screw axis to the angular velocity θ̇ about
    # the screw axis)
    if int(omega_norm) == 1:
        # Case 1: rigid body rotation occurs
        h = v/theta_dot
        transl_2 = h*s_hat*theta_dot  # translation along s_hat
        V_s = np.hstack((omega, v + transl_2)).flatten()
        return V_s/omega_norm
    elif abs(omega_norm) < 0.00001:
        # Case 2: no rigid body rotation occurs
        h = np.inf
        transl_2 = 0
        V_s = np.hstack((omega, v + transl_2)).flatten()
        v_norm = np.linalg.norm(v)
        return V_s/v_norm
    else:
        assert int(omega_norm) == 1 or abs(omega_norm) < 0.00001


def cla():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.set_xlim(-4.5, 4.5)
    # ax.set_ylim(-0.5, 4.5)
    # ax.set_zlim(-1, 1)

cla()

M = SE3([1, 0, -1]) @ SE3().Ry(pi/2)

T1_frame = SE3([0,0,0])
T2_frame = SE3([1,0,0]) @ SE3().Rx(pi/2) @ SE3().Rz(-pi/2)
T3_frame = SE3([1,0,-1]) @ SE3().Ry(pi/2)

S1 = get_screw_axis_revolute(np.array([[0,0,1]]), np.array([[0,0,0]]))
S2 = get_screw_axis_revolute(np.array([[0,-1,0]]), np.array([[1,0,0]]))
S3 = get_screw_axis_revolute(np.array([[1,0,0]]), np.array([[0,0,-1]]))

T1_rot_s = to_SE3(S1[:3], pi/2, S1[3:])
T2_rot_s = to_SE3(S2[:3], pi/2, S2[3:])
T3_rot_s = to_SE3(S3[:3], pi/2, S3[3:])

T1_rot_b = expm(skew_beta(S1, M.A)*pi/2)
T2_rot_b = expm(skew_beta(S2, M.A)*pi/2)
T3_rot_b = expm(skew_beta(S3, M.A)*pi/2)

############################## PLOTS ##############################
plt.ion(); plt.show()

# 8) Introduce different joint frames
# 9) Quick mention of how to obtain s_hat and theta (ie the unit axis s_hat and point q are defined in fixed frame)
# S1 is just s_hat = (0,0,1) as joint axis coincides with fixed frame and q=(0,0,0)
# S2 s_hat = (0, -1, 0) as the joint's z_axis points to the negative y_fixed frame axis
# S2 q = [1,0,0]. We could also have chosen (1, 0/2) since we just need any point along z_j2
# Remember that h=0 since we have zero pitch for revolute joints.
# ############ another sidenote: Joint 2 has a rot of SE3().Rx(pi/2) @ SE3().Rz(-pi/2)
# Put in eqn from earlier to obtain screw axis for joint 2. We can then perform the same process to obtain S3
# Once we have the screw axes and a displacement theta, we can calculate the forward kinematics
trplot(M.A, frame='ee', color='r', length=0.6);# inter()
trplot(T3_frame.A, frame='j3', color='g', length=0.5);# inter()
trplot(T2_frame.A, frame='j2', color='b', length=0.5);# inter()

# 10) Observe how the EE changes as we apply the transforms with pi/2 displacement on each joint. The other joints don't move so we'll have to imagine how it would look
# 11) Show how the ee pose changes as each operation is applied (pi/2 around the z_axis of each joint). the other joints were not plotted in this case
# 12) After joint 3 displ, notice how the screw axes of joint 1/2 is still the same relative to the fixed frame
# 13) for this formulation, notice how M is transform by the joints closest to the EE
trplot(T1_frame.A, frame='j1,fixedframe', color='black', length=0.5); inter()
trplot(T3_rot_s@M.A, frame='rot3', color='pink', length=0.7); inter()
trplot(T2_rot_s@T3_rot_s@M.A, frame='rot3rot2', color='purple', length=0.7); inter()
trplot(T1_rot_s@T2_rot_s@T3_rot_s@M.A, frame='rot3rot2rot1', color='cyan', length=0.7); inter()
ax = plt.gca()
ax.view_init(elev=100, azim=0, roll=0); inter()

cla()
# 14) 2 ways: 1 screw axes defined in body frame (ee in this case) or just convert the screw axes represented in the
# fixed frame to be represented in the body frame. More details in book
# 15) Show how the ee pose changes as each operation is applied (pi/2 around the z_axis of each joint)
# 16) Didn't plot but joint 2 also moves
# 17) After joint 1 displ, notice how the screw axes of joint 2/3 is still the same relative to the body (ee) frame
# 18) for this formulation, notice how M is transform by the joints closest to the EE
trplot(M.A, frame='ee', color='r', length=0.6);# inter()
trplot(T3_frame.A, frame='j3', color='g', length=0.5);# inter()
trplot(T2_frame.A, frame='j2', color='b', length=0.5);# inter()
trplot(T1_frame.A, frame='j1,fixedframe', color='black', length=0.5); inter()
trplot(M.A@T1_rot_b, frame='rot3', color='pink', length=0.7); inter()
trplot(M.A@T1_rot_b@T2_rot_b, frame='rot3rot2', color='purple', length=0.7); inter()
trplot(M.A@T1_rot_b@T2_rot_b@T3_rot_b, frame='rot3rot2rot1', color='cyan', length=0.7); inter()
ax = plt.gca()
ax.view_init(elev=100, azim=0, roll=0)

print(np.round(T1_rot_s@T2_rot_s@T3_rot_s@M.A, 3))
print(np.round(M.A@T1_rot_b@T2_rot_b@T3_rot_b, 3))

ax = plt.gca()
ax.view_init(elev=100, azim=0, roll=0); inter()
# Visualise joint reference frames
# trplot(T1_frame.A, frame='inital j1', color='r', length=0.5);# inter()
# trplot(T2_frame.A, frame='inital j2', color='g', length=0.5);# inter()
# trplot(T3_frame.A, frame='inital j3', color='b', length=0.5);# inter()
# trplot(M.A, frame='inital j3', color='black', length=0.6);# inter()
# plt.axis('tight')
# plt.show()

# Consider each revolute joint to be a zero-pitch screw axis
# The forward kinematics can be expressed as a product of matrix exponen-
# tials, each corresponding to a screw motion

# trplot(T_end, width=0.5, frame='s,d'); plt.show()
# sys.exit()
#  inter()
# trplot(T_c.A, width=0.5, color='g', frame='c'); inter()

# Define s_hat
# ax.plot(*a_line(np.array([[0, 2, -1]]), np.array([[0, 2, 1]])), color='g', linewidth=1.5, label='s_hat'); inter()

# Define q
# ax.scatter(q[0, 0], q[0, 1], q[0, 2], color='r', label='q'); inter()

# Translation due to rotation about s_hat. Occurs in the plane orthogonal to s_hat
# ax.plot(*a_line(q, transl_2), color='black', linewidth=1.5, label='transl1'); inter()
# tranimate(T_end, frame='A', arrow=False, dims=[0, 5], nframes=200, movie='out.mp4')

# X = SE3.Rx(0.3)
# X.animate(frame='A', color='green')
# X.animate(start=SE3.Ry(0.2))
# T_s.animate()

