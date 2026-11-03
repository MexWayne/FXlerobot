"""Top-down grasp IK for the SO-101 (numpy only, on llm_vla/so101_kin.py).

The tool points straight down, or tilts up to max_tilt_deg when straight down is out of reach, and
the jaws close horizontally, perpendicular to a horizontal object axis. In
gripper_link the tool axis is -z and the moving jaw sits on the +x side of the fixed jaw (its pivot
is at x = +20 mm, the fixed-jaw tip at x = -11 mm), so the jaw-closing line is gripper_link x. The grasp point (TCP) sits between the jaws, GRASP_INSET above
the fixed-jaw tip.
All positions are in the arm base frame (base_link of the URDF), metres.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import so101_kin  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRASP_INSET = 0.012  # TCP this far up the jaws from the fixed-jaw tip
# Weights: 1 mm of position counts as much as 1 deg of tool tilt or jaw misalignment.
W_POS = 1000.0
W_ANG = np.degrees(1.0)
DOWN = np.array([0.0, 0.0, -1.0])


def tcp_offset(half_width):
    """Grasp point in gripper_link: half_width from the fixed jaw towards the moving jaw."""
    tip = so101_kin.FIXED_JAW_TIP
    return tip + np.array([half_width, 0.0, GRASP_INSET])


def pose(q, offset):
    """TCP position, tool axis and jaw-closing direction in the arm base frame."""
    T = so101_kin.gripper_pose(q)
    return T[:3, :3] @ offset + T[:3, 3], -T[:3, 2], T[:3, 0]


EXACT = 0.5  # residual (mm / deg) below which an upright solution is taken without trying more seeds
TILT_PREFER = 0.02  # inside the allowed tilt, each degree still costs this much (prefers upright)


def _residual(x, p_goal, axis, offset, max_tilt_deg):
    q = dict(zip(JOINTS, x))
    p, tool, close = pose(q, offset)
    if max_tilt_deg <= 0:
        tilt = (tool - DOWN) * W_ANG
    else:
        angle = np.degrees(np.arccos(np.clip(-tool[2], -1.0, 1.0)))
        tilt = [max(0.0, angle - max_tilt_deg), TILT_PREFER * angle, close[2] * W_ANG]
    return np.concatenate([(p - p_goal) * W_POS, tilt, [close @ axis * W_ANG]])


def solve(p_goal, axis, offset, limits, seeds, iters=200, max_tilt_deg=0.0):
    """Joint angles (deg) putting the TCP at p_goal, tool down, jaws closing across `axis`.

    axis: horizontal unit vector along the object (arm base frame). limits: joint -> (lo, hi) deg.
    max_tilt_deg > 0 lets the tool lean that far off vertical (jaws still closing horizontally).
    Returns (q dict, residual norm in mm/deg units; with a tilt allowed it includes
    TILT_PREFER * tilt)."""
    lo = np.array([limits[j][0] for j in JOINTS])
    hi = np.array([limits[j][1] for j in JOINTS])
    best = None
    for seed in seeds:
        x = np.clip(np.array([seed[j] for j in JOINTS], float), lo, hi)
        lam = 1.0
        r = _residual(x, p_goal, axis, offset, max_tilt_deg)
        for _ in range(iters):
            J = np.empty((len(r), len(x)))
            for k in range(len(x)):
                dx = np.zeros(len(x))
                dx[k] = 0.01
                J[:, k] = (_residual(x + dx, p_goal, axis, offset, max_tilt_deg) - r) / 0.01
            step = np.linalg.solve(J.T @ J + lam * np.eye(len(x)), -J.T @ r)
            x_new = np.clip(x + step, lo, hi)
            r_new = _residual(x_new, p_goal, axis, offset, max_tilt_deg)
            if r_new @ r_new < r @ r:
                x, r, lam = x_new, r_new, lam * 0.5
                if np.linalg.norm(step) < 1e-4:
                    break
            else:
                lam *= 4.0
        err = float(np.linalg.norm(r))
        if best is None or err < best[1]:
            best = (dict(zip(JOINTS, x.tolist())), err)
        if max_tilt_deg <= 0 and err < EXACT:
            break  # upright and exact: the other seeds cannot do better
    return best
