"""Hold a bottle by its cap over a trash can and let go, with the left arm.

The bottle hangs from the cap the jaws hold, so only where the grasp point is matters, and that the
jaws still close horizontally (the bottle swings about the closing line). The tool may lean; a small
cost keeps it near pointing down, which keeps the wrist above the bottle. The arm goes from PARK up
and out over the can in short steps, the height ahead of the way out (straight above PARK the arm
does not reach as high as further out), each IK started from the step before so the wrist never
turns over on the way. Over the can the bottle stays above the rim all the way.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "llm_vla" / "pick_bottle"), str(ROOT / "llm_vla")]
import ik  # noqa: E402
import mount  # noqa: E402
import pick  # noqa: E402
from arm import joint_limits  # noqa: E402

BOTTLE_HANG_M = 0.21   # held by the cap, a bottle of unknown length reaches this far below the grasp point
DROP_CLEAR_M = 0.04    # bottle bottom this far above the rim when it is let go
STEP_M = 0.03          # waypoints this far apart: one big step turns the wrist over
RISE_LEAD = 2.0        # on the way out the height is reached this many times sooner than the place
BOTTLE_R_M = 0.04      # the hanging bottle's half width, against the rim
# The jaws must not point far upwards: the bottle hanging from the cap would lie against the
# fingers. PARK carries a bottle well with the tool at 104 deg from straight down.
MAX_TILT_DEG = 115.0
MAX_POS_ERR_M = 0.003
MAX_JAW_DEG = 5.0      # closing line this close to horizontal
MAX_STEP_DEG = 40.0    # no joint turns more than this between two neighbouring steps
TILT_COST = 0.03       # per degree of tool lean, in mm of position error
LET_GO_S = 1.0
# Let go, the jaw must come this close to OPEN_DEG: an overheated gripper unloads and stays shut on
# the cap, and the bottle went back out still held (2026-11-04).
OPENED_TOL_DEG = 15.0


def _tilt(tool):
    return float(np.degrees(np.arccos(np.clip(-tool[2], -1.0, 1.0))))


def _residual(x, goal, offset):
    p, tool, close = ik.pose(dict(zip(ik.JOINTS, x)), offset)
    tilt = _tilt(tool)
    return np.concatenate([(p - goal) * ik.W_POS, [close[2] * ik.W_ANG],
                           [tilt * TILT_COST, max(0.0, tilt - MAX_TILT_DEG)]])


def solve_point(goal, seed, limits, offset, iters=150):
    """Joint angles putting the grasp point at `goal` (arm frame), from `seed`. Returns (q, ok)."""
    lo = np.array([limits[j][0] for j in ik.JOINTS])
    hi = np.array([limits[j][1] for j in ik.JOINTS])
    x = np.clip(np.array([seed[j] for j in ik.JOINTS], float), lo, hi)
    lam = 1.0
    r = _residual(x, goal, offset)
    for _ in range(iters):
        J = np.stack([(_residual(x + e, goal, offset) - r) / 0.01 for e in np.eye(len(x)) * 0.01], axis=1)
        step = np.linalg.solve(J.T @ J + lam * np.eye(len(x)), -J.T @ r)
        x_new = np.clip(x + step, lo, hi)
        r_new = _residual(x_new, goal, offset)
        if r_new @ r_new < r @ r:
            x, r, lam = x_new, r_new, lam * 0.5
            if np.linalg.norm(step) < 1e-4:
                break
        else:
            lam *= 4.0
    q = dict(zip(ik.JOINTS, x.tolist()))
    p, tool, close = ik.pose(q, offset)
    ok = (np.linalg.norm(p - goal) <= MAX_POS_ERR_M and _tilt(tool) <= MAX_TILT_DEG + 1.0
          and np.degrees(np.arcsin(min(1.0, abs(close[2])))) <= MAX_JAW_DEG)
    return q, ok


def plan_drop(arm_now, bin_xy, bin_r, rim_z, hang=BOTTLE_HANG_M):
    """Joint waypoints from the arm as it is (PARK, holding a bottle) out over the can (base frame
    x, y of its centre, its radius and rim height); the bottle reaches `hang` below the grasp point.
    Raises SystemExit when the can is out of reach or too tall."""
    T = mount.load()
    R, t = T[:3, :3], T[:3, 3]
    offset = ik.tcp_offset(pick.CAP_RADIUS_M)
    limits = joint_limits()
    z = rim_z + hang + DROP_CLEAR_M
    start = R @ ik.pose(arm_now, offset)[0] + t
    over = np.array([bin_xy[0], bin_xy[1], z])
    clear = rim_z + hang + 0.01  # lowest grasp point with the bottle over the can
    n = max(1, int(np.ceil(np.linalg.norm(over[:2] - start[:2]) / STEP_M)))
    goals, cur = [], start.copy()
    for k in range(1, n + 1):
        xy = start[:2] + (over[:2] - start[:2]) * k / n
        need = clear if np.hypot(*(xy - over[:2])) < bin_r + BOTTLE_R_M + STEP_M else -np.inf
        while cur[2] < need - 1e-6:  # up on the spot before the bottle comes over the rim
            cur[2] = min(need, cur[2] + STEP_M)
            goals.append(cur.copy())
        cur = np.r_[xy, max(cur[2], need, start[2] + (z - start[2]) * min(1.0, RISE_LEAD * k / n))]
        goals.append(cur.copy())
    q, path = dict(arm_now), []
    for g in goals:
        nxt, ok = solve_point(R.T @ (g - t), q, limits, offset)
        if not ok:
            raise SystemExit(f"够不着垃圾桶上方 ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}) m：桶口离地 {rim_z:.2f} m，"
                             f"瓶子挂在下面要 {hang + DROP_CLEAR_M:.2f} m")
        if max(abs(nxt[j] - q[j]) for j in ik.JOINTS) > MAX_STEP_DEG:
            raise SystemExit("往垃圾桶上方去的路上手腕要翻转，不动")
        q = nxt
        path.append(q)
    return path


def drop(arm, bin_xy, bin_r, rim_z, hang=BOTTLE_HANG_M):
    """Out over the can, open, back the same way, PARK (holding nothing)."""
    path = plan_drop(arm.deg(), bin_xy, bin_r, rim_z, hang)
    for k, q in enumerate(path):
        pick.move(arm, q, pick.CARRY_DEG_S, f"举到垃圾桶上方 {k + 1}/{len(path)}")
    got = pick.move(arm, {"gripper": pick.OPEN_DEG}, pick.FREE_DEG_S, "松开")
    if got["gripper"] < pick.OPEN_DEG - OPENED_TOL_DEG:
        arm.hold()  # not pressing on the cap any more
        temp, hot = arm.gripper_heat()
        raise SystemExit(f"夹爪没张开（停在 {got['gripper']:.0f}°，{temp}°C{'，过热卸力了' if hot else ''}），"
                         "瓶子还在手里，左臂停在垃圾桶上方")
    time.sleep(LET_GO_S)
    for q in reversed(path[:-1]):
        pick.move(arm, q, pick.FREE_DEG_S, "收回")
    pick.park(arm)
