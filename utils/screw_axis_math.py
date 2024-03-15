"""
Most of the code probs obtained from official MR github repo
https://github.com/NxRLab/ModernRobotics/tree/master
"""
from scipy.linalg import expm
import math
import numpy as np
np.set_printoptions(suppress=True, precision=4)

def adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(skew(p), R), R]]


def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]


def skew_beta(x, M):
    assert x.ndim == 1
    assert isinstance(M, np.ndarray)
    return np.linalg.inv(M)@skew(x)@M

def skew(x):
    """
    Converts 3 vector representation to so(3) representation using skew symmetric matrix
    Args:
        x: 3-vector

    Returns: skew symmetric representation of x

    """
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

def to_SE3(omega, theta, v):
    """
    Obtains the SE3 (homogenous transformation) based on the rotation axis, theta (joint config) and position
    Args:
        omega: axis of rotation
        theta: rotation value
        v: position

    Returns:

    """
    I = np.eye(3)
    skew_omega = skew(omega.flatten())
    bot = np.array([[0, 0, 0, 1]])

    rot = expm(skew_omega*theta)  # Matrix exponential operator not same as **
    assert (np.abs(rot.transpose() - np.linalg.inv(rot)) < 0.0001).all()  # Ensure we have a matrix in the SO(3)
    G = I*theta + (1-math.cos(theta))*skew_omega + (theta - math.sin(theta))*(np.linalg.matrix_power(skew_omega,2))

    transl = (G@v.transpose()).reshape(3,1)
    return np.vstack((np.hstack((rot, transl)), bot))

def get_screw_axis_revolute(s_hat, q):
    """
    Obtains the screw axis of a revolute joint
    Args:
        s_hat: axis of rotation
        q: point on the axis of rotation

    Returns: screw axis of revolute joint

    """
    # instantenous linear velocity. Also the translation due to rotation about s_hat.
    # Occurs in the plane orthogonal to s_hat
    v = -np.cross(s_hat, q)

    theta_dot = 1  # angular velocity

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

def create_T_matrix(rmatr, tvec, se3=False):
    """
    Create a transformation matrix from the rotation matrix and translation vector
    Args:
        rmatr: 3x3 rotation matrix
        tvec: 3x1 translation vector
        se3: if se3 is False, returns a homogenous matrix, else returns a lie group of SE(3)

    Returns: 4x4 transformation matrix
    """
    # Check the shapes of the rotation matrix and translation vector is correct
    assert tvec.shape == (3, 1), f"The translation vector is the wrong shape! It is {tvec.shape}"
    assert rmatr.shape == (3, 3), f"The rotation matrix is not 3x3! It is {rmatr.shape}"

    R = np.vstack((rmatr, np.zeros((1, 3))))
    if se3:
        t = np.vstack((tvec, np.zeros((1, 1))))
    else:
        t = np.vstack((tvec, np.ones((1, 1))))
    return np.hstack((R, t))

def v2se3(spatial_velocity, theta=None, use_textbook_calc=False):
    """
    converts spatial velocity to se3 matrix
    Args:
        spatial_velocity: np ndarray | 6-vector representation of spatial velocity

    Returns: 4x4 se3 matrix
    """
    if use_textbook_calc:
        return to_SE3(spatial_velocity[:3], theta, spatial_velocity[3:])
    else:
        assert spatial_velocity.ndim == 1
        assert spatial_velocity.size == 6
        omega_skew = skew(spatial_velocity[:3])
        v = spatial_velocity[3:].reshape(3, 1)
        return create_T_matrix(omega_skew, v, True)

def jacob_spatial(screw_axis, theta):
    """
    Calculates spatial jacobian
    Args:
        screw_axis: list of screw axis for each joint in the form (omega, v) when robot at home position
        theta: list of joint positions

    Returns: Spatial jacobian

    Example Input:
        Slist = np.array([[0, 0, 1,   0, 0.2, 0.2],
                          [1, 0, 0,   2,   0,   3],
                          [0, 1, 0,   0,   2,   1],
                          [1, 0, 0, 0.2, 0.3, 0.4]])
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])

    Output:
        np.array([[  0, 0.98006658, -0.09011564,  0.95749426]
                  [  0, 0.19866933,   0.4445544,  0.28487557]
                  [  1,          0,  0.89120736, -0.04528405]
                  [  0, 1.95218638, -2.21635216, -0.51161537]
                  [0.2, 0.43654132, -2.43712573,  2.77535713]
                  [0.2, 2.96026613,  3.23573065,  2.22512443]])
    """
    # TODO: from spatialmath.base import trexp
    Js = np.zeros((6, 6))
    T = np.eye(4)
    Js[:, 0] = screw_axis[0, :]
    for i in range(1, len(theta)):
        # exp_matr = expm(v2se3(screw_axis[i - 1, :]) * theta[i-1])
        exp_matr = v2se3(screw_axis[i - 1, :], theta[i-1], True)
        T = T @ exp_matr
        Js[:, i] = np.transpose(adjoint(T) @ np.transpose(screw_axis[i, :]))
    return Js