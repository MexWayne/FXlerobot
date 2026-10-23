"""Correct the dead-reckoned pose by lining up fence points the head sees with the saved fence.

There is no IMU, lidar or wheel odometry, so the pose drifts with every move. The fence polygon
in map/area.json does not move between runs, which makes it a landmark: the seen fence-floor
contact points (robot frame) are rotated and shifted until they lie on the saved fence.
"""

import math

import cv2
import numpy as np

TRIM_M = 0.15  # a point further than this from the fence counts as this much (chairs, people)


def fence_field(cmap):
    """Distance in metres from every map cell to the nearest fence edge."""
    img = np.full((cmap.h, cmap.w), 255, dtype=np.uint8)
    pts = np.array([[(x - cmap.origin[0]) / cmap.res - 0.5, (y - cmap.origin[1]) / cmap.res - 0.5]
                    for x, y in cmap.boundary], dtype=np.float32)
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, 0, 1)
    return cv2.distanceTransform(img, cv2.DIST_L2, 5).astype(np.float32) * cmap.res


def _cost(field, cmap, xy):
    """Mean trimmed fence distance of world points xy (..., N, 2)."""
    c = np.floor((xy[..., 0] - cmap.origin[0]) / cmap.res).astype(np.int64)
    r = np.floor((xy[..., 1] - cmap.origin[1]) / cmap.res).astype(np.int64)
    inside = (c >= 0) & (r >= 0) & (c < cmap.w) & (r < cmap.h)
    d = np.full(c.shape, TRIM_M, dtype=np.float32)
    d[inside] = np.minimum(field[r[inside], c[inside]], TRIM_M)
    return d.mean(axis=-1)


def fit_pose(field, cmap, pose, local, xy_win, deg_win, xy_step=0.025, deg_step=1.0,
             per_m=0.0, per_deg=0.0, anchor=None):
    """Best (x, y, yaw) near `pose` for robot-frame fence points `local`.

    The saved fence is only an estimate (unseen bearings are filled in), so a part view can fit a
    wrong shift slightly better; per_m / per_deg add metres of cost per metre / degree moved away
    from `anchor` (default `pose`), and only a clearly better fit moves the pose.
    Returns (pose, cost, cost_at_anchor); costs are mean trimmed distances in metres, prior included.
    """
    pts = np.asarray(local, dtype=np.float32)
    x0, y0, yaw0 = pose

    def place(x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], axis=1)

    ax, ay, ayaw = anchor or pose

    def prior(x, y, yaw):
        return per_m * np.hypot(x - ax, y - ay) + per_deg * abs(math.degrees(yaw - ayaw))

    start = float(_cost(field, cmap, place(ax, ay, ayaw)))
    span = np.arange(-xy_win, xy_win + 1e-9, xy_step, dtype=np.float32)
    tx, ty = np.meshgrid(span, span)
    shifts = np.stack([tx.ravel(), ty.ravel()], axis=1)
    best = (start, (ax, ay, ayaw))
    for d in np.arange(-deg_win, deg_win + 1e-9, deg_step):
        yaw = yaw0 + math.radians(float(d))
        cost = _cost(field, cmap, place(x0, y0, yaw)[None, :, :] + shifts[:, None, :])
        cost = cost + prior(x0 + shifts[:, 0], y0 + shifts[:, 1], yaw)
        k = int(np.argmin(cost))
        if cost[k] < best[0]:
            best = (float(cost[k]), (x0 + float(shifts[k, 0]), y0 + float(shifts[k, 1]), yaw))
    return best[1], best[0], start


def locate(field, cmap, pose, local, xy_win, deg_win, per_m=0.0, per_deg=0.0):
    """Coarse grid over the window, then a fine one around the coarse best.

    Returns (pose, fit, start, gain): fit is the plain mean fence distance at the returned pose,
    start the one at `pose`, gain how much the prior-weighted cost dropped (metres).
    """
    kw = {"per_m": per_m, "per_deg": per_deg, "anchor": pose}
    coarse, _, start = fit_pose(field, cmap, pose, local, xy_win, deg_win,
                                xy_step=max(0.025, xy_win / 12.0), deg_step=max(1.0, deg_win / 10.0), **kw)
    fine, total, _ = fit_pose(field, cmap, coarse, local, 0.06, 3.0, xy_step=0.02, deg_step=0.5, **kw)
    _, fit, _ = fit_pose(field, cmap, fine, local, 0.0, 0.0)
    return fine, fit, start, start - total
