import numpy as np

from utils.screw_axis_math import *
from utils.create_robot import get_2d_robot
from roboticstoolbox.backends.PyPlot import EllipsePlot
from matplotlib import pyplot as plt
Q_INIT = (0.1, 0.4, 0.2)  # Initial robot position
Q_Z = (0.0, 0.0, 0.0)  # Initial robot position

robot = get_2d_robot(link_length=0.25)


robot.q = (Q_INIT)
jacob_w = robot.jacob0(robot.q)
# axis of rotation is around the z --> [0, 0, 1]
# slist = [rotation_axis, v_s]
slist = np.array([[0, 0, 1, 0, 0, 0],  # q1 is on origin
                  # q2 is link_length units in x dir --> vs2 = -omega_2 \cross q2 = - <0, 0, 1> <0.25, 0, 0> = norm(a)*norm(b)*sin(theta)*n_hat; theta=?, n_hat=?
                  [0, 0, 1, 0, -0.25, 0],
                  # q3 is link_length units in x dir --> vs3 = -omega_2 \cross q2 = - <0, 0, 1> <0.25, 0, 0> = norm(a)*norm(b)*sin(theta)*n_hat; theta=?, n_hat=?
                  [0, 0, 1, 0, -0.5, 0]])
jacob_s = jacob_spatial(slist, robot.q)
print(jacob_w)
print(jacob_s[:, :3])
robot.plot(q=Q_INIT, block=1)
