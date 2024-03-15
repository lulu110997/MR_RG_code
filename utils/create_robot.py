import roboticstoolbox as rtb
from spatialmath import SE3, SE2

def get_2d_robot(link_length=0.25, base_pos=(0,0,0)):
    robot = rtb.DHRobot(
        [
            rtb.RevoluteDH(a=link_length),
            rtb.RevoluteDH(a=link_length),
            rtb.RevoluteDH(a=link_length)
        ], name="2D_robot", base=SE3(base_pos))

    return robot