"""SO-101 forward kinematics from TheRobotStudio so101_new_calib.urdf, numpy only.

Joint angles are LeRobot degrees: (ticks - range mid) * 360 / 4095, the same numbers
save_pose.py prints. The tool point is the very front of the fixed jaw, taken from the
wrist_roll_follower_so101_v1.stl mesh; the URDF's gripper_frame_link sits 6.3 mm short of it.
All positions are in the arm base frame (base_link): x forward, z up, metres.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parent / "urdf/so101_new_calib.urdf"
# Fixed-jaw tip in gripper_link, from the STL vertex furthest along -z.
FIXED_JAW_TIP = np.array([-0.0108, -0.0002, -0.1044])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, angle):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _load_chain(tip_link="gripper_link"):
    joints = {j.find("child").get("link"): j for j in ET.parse(URDF).getroot().findall("joint")}
    chain, link = [], tip_link
    while link in joints:
        chain.append(joints[link])
        link = joints[link].find("parent").get("link")
    steps = []
    for j in reversed(chain):
        origin = j.find("origin")
        A = np.eye(4)
        A[:3, :3] = _rpy(*map(float, origin.get("rpy").split()))
        A[:3, 3] = list(map(float, origin.get("xyz").split()))
        axis = None
        if j.get("type") == "revolute":
            axis = np.array(list(map(float, j.find("axis").get("xyz").split())))
            axis /= np.linalg.norm(axis)
        steps.append((A, j.get("name"), axis))
    return steps


_CHAIN = _load_chain()


def gripper_pose(q_deg):
    """4x4 pose of gripper_link in the arm base frame. q_deg: dict joint -> LeRobot degrees."""
    T = np.eye(4)
    for A, name, axis in _CHAIN:
        T = T @ A
        if axis is not None:
            R = np.eye(4)
            R[:3, :3] = _axis_angle(axis, np.radians(q_deg.get(name, 0.0)))
            T = T @ R
    return T


def tip_position(q_deg):
    """Fixed-jaw tip in the arm base frame (m)."""
    T = gripper_pose(q_deg)
    return T[:3, :3] @ FIXED_JAW_TIP + T[:3, 3]


def tool_axis(q_deg):
    """Unit vector from the wrist towards the jaw tips, in the arm base frame."""
    return -gripper_pose(q_deg)[:3, 2]
