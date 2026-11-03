#!/usr/bin/env python
"""Locate two paper cups standing upside down on the board with both the lidar and the head stereo.

    python cup_compare.py snapshots/<time> --left 344,158 452,178 --right 446,166 561,204

Input is an arm_snapshot.py folder. Each cup has a pencil "+" on its bottom (now its top); the
pixels given per eye are rough clicks on those crosses, refined here to where the two pencil lines
meet. The crosses are triangulated with the stereo calibration and moved into the base frame with
head_extrinsic.yaml turned by the neck model (vln_nav.cam_at). The lidar sees each cup's wall as an
arc; a circle fitted to it gives the cup axis. Writes cups.json into the snapshot folder.
Base frame: x forward, y left, origin on the floor under the base centre.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "project_tool"), str(ROOT / "vln")]
import lidar_scan  # noqa: E402
import validate_head_extrinsic as val  # noqa: E402
from board_view import HEAD_CAMERAS  # noqa: E402
from vln_nav import cam_at  # noqa: E402

EXTRINSIC_DIR = ROOT / "calib" / "head"
UPSAMPLE = 4
WIN_PX = 5  # half window around a click: the cross centre is searched this close to it
ARM_PX = 10  # length of each pencil arm scored from the centre
LINK_M = 0.015  # lidar points closer than this to a neighbour belong to the same object
MIN_POINTS = 10  # fewer than this over all turns is a stray echo
GATE_M = 0.08  # the arc of a cup is the object whose points come this close to its stereo position


def refine_cross(gray, seed):
    """Sub-pixel point where two dark straight lines cross near seed (x, y)."""
    x0, y0 = int(round(seed[0])) - WIN_PX - ARM_PX, int(round(seed[1])) - WIN_PX - ARM_PX
    n = 2 * (WIN_PX + ARM_PX) + 1
    patch = gray[y0:y0 + n, x0:x0 + n].astype(np.float32)
    patch = cv2.resize(patch, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    dark = np.clip(cv2.medianBlur(patch.astype(np.uint8), 31).astype(np.float32) - patch, 0, None)
    r = ARM_PX * UPSAMPLE
    scores = []
    for a in np.deg2rad(np.arange(0, 180, 2)):
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        p, q = (r + r * np.cos(a), r + r * np.sin(a)), (r - r * np.cos(a), r - r * np.sin(a))
        cv2.line(k, tuple(int(round(v)) for v in p), tuple(int(round(v)) for v in q), 1.0, 1)
        scores.append(cv2.filter2D(dark, -1, k / k.sum()))
    scores = np.stack(scores)  # angle, y, x
    # The two lines must be at least 40 deg apart: best line plus best line outside +-20 deg of it.
    best = scores.argmax(0)
    first = scores.max(0)
    idx = np.arange(len(scores))[:, None, None]
    sep = np.abs((idx - best[None] + 45) % 90 - 45) >= 10
    total = first + np.where(sep, scores, -np.inf).max(0)
    m = r + 1
    inner = total[m:-m, m:-m]
    iy, ix = np.unravel_index(inner.argmax(), inner.shape)
    return x0 + (ix + m) / UPSAMPLE, y0 + (iy + m) / UPSAMPLE


def triangulate(calib, uv_left, uv_right):
    R = np.array(calib["stereo"]["R_right_left"])
    t = np.array(calib["stereo"]["T_right_left_m"]).reshape(3, 1)
    P1, P2 = np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, t])
    norm = {}
    for eye, uv in (("left", uv_left), ("right", uv_right)):
        K, dist = np.array(calib[eye]["K"]), np.array(calib[eye]["dist"])
        norm[eye] = cv2.undistortPoints(np.array(uv, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2).T
    X = cv2.triangulatePoints(P1, P2, norm["left"], norm["right"])
    X = (X[:3] / X[3]).T
    # Reprojection error in pixels, per point and eye.
    err = []
    for eye, P in (("left", P1), ("right", P2)):
        K, dist = np.array(calib[eye]["K"]), np.array(calib[eye]["dist"])
        rv, _ = cv2.Rodrigues(P[:, :3])
        px, _ = cv2.projectPoints(X, rv, P[:, 3], K, dist)
        err.append(np.linalg.norm(px.reshape(-1, 2) - np.array(uv_left if eye == "left" else uv_right), axis=1))
    return X, np.array(err).T


def objects(pts):
    """Split lidar points into connected groups (single linkage within LINK_M)."""
    label = -np.ones(len(pts), int)
    for seed in range(len(pts)):
        if label[seed] >= 0:
            continue
        label[seed], todo = seed, [seed]
        while todo:
            i = todo.pop()
            for j in np.flatnonzero((label < 0) & (np.linalg.norm(pts - pts[i], axis=1) < LINK_M)):
                label[j] = seed
                todo.append(j)
    groups = [pts[label == k] for k in np.unique(label)]
    return [g for g in groups if len(g) >= MIN_POINTS]


def fit_circle(xy):
    """Least-squares circle (centre, radius, rms) to points on an arc."""
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    centre = np.array([cx, cy])
    for _ in range(20):  # Gauss-Newton on geometric distance
        d = xy - centre
        rho = np.linalg.norm(d, axis=1)
        r = rho.mean()
        J = np.c_[-d / rho[:, None]]
        J = J - J.mean(0)
        step = np.linalg.lstsq(J, -(rho - r), rcond=None)[0]
        centre = centre + step
        if np.linalg.norm(step) < 1e-7:
            break
    rho = np.linalg.norm(xy - centre, axis=1)
    return centre, rho.mean(), float(np.sqrt(np.mean((rho - rho.mean()) ** 2)))


def pairs(text):
    return [tuple(float(v) for v in p.split(",")) for p in text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--left", nargs=2, required=True, help="x,y of each cup's cross in the left eye")
    ap.add_argument("--right", nargs=2, required=True, help="same cups in the same order, right eye")
    args = ap.parse_args()
    snap_dir = Path(args.snapshot)
    snap = json.loads((snap_dir / "snapshot.json").read_text())

    calib = yaml.safe_load(HEAD_CAMERAS.read_text())
    uv = {}
    for eye, seeds in (("left", pairs(args.left)), ("right", pairs(args.right))):
        gray = cv2.cvtColor(cv2.imread(str(snap_dir / f"{eye}.jpg")), cv2.COLOR_BGR2GRAY)
        uv[eye] = [refine_cross(gray, s) for s in seeds]
    X_cam, reproj = triangulate(calib, uv["left"], uv["right"])

    _, extr, cams = val.load(str(EXTRINSIC_DIR))
    neck = snap["neck_deg"]
    cam = cam_at(cams["left"], neck["neck_yaw"] - extr["neck_yaw_pos"],
                 neck["neck_pitch"] - extr["neck_pitch_pos"])
    T = cam["T_base_cam"]
    X_base = (T[:3, :3] @ X_cam.T).T + T[:3, 3]

    z = np.load(snap_dir / "lidar.npz")
    pts = np.vstack([z[k] for k in z.files])
    lidar_origin = np.array([lidar_scan.LIDAR_X_M, lidar_scan.LIDAR_Y_M])
    cups = []
    groups = objects(pts[np.linalg.norm(pts - lidar_origin, axis=1) < 1.0])
    for i, xb in enumerate(X_base):
        gap = [np.linalg.norm(g - xb[:2], axis=1).min() for g in groups]
        if min(gap) > GATE_M:
            raise SystemExit(f"杯{i + 1} 附近 {GATE_M} m 内没有雷达点")
        near = groups[int(np.argmin(gap))]
        centre, radius, rms = fit_circle(near)
        cups.append({
            "uv_left": list(uv["left"][i]), "uv_right": list(uv["right"][i]),
            "reproj_px": reproj[i].tolist(),
            "stereo_cam_m": X_cam[i].tolist(), "stereo_base_m": xb.tolist(),
            "lidar_points": near.tolist(), "lidar_centre_m": centre.tolist(),
            "lidar_radius_m": radius, "lidar_fit_rms_m": rms,
            "range_from_lidar_m": {"lidar": float(np.linalg.norm(centre - lidar_origin)),
                                   "stereo": float(np.linalg.norm(xb[:2] - lidar_origin))},
        })
        print(f"杯{i + 1}: 双目 base ({xb[0]:.3f}, {xb[1]:.3f}, {xb[2]:.3f}) m, 重投影 {np.round(reproj[i], 2)} px | "
              f"雷达圆心 ({centre[0]:.3f}, {centre[1]:.3f}) m 半径 {radius * 1000:.1f} mm "
              f"拟合残差 {rms * 1000:.1f} mm, {len(near)} 点")
    out = {
        "neck_deg": neck, "lidar_origin_base_m": lidar_origin.tolist(),
        "lidar_env_points": pts[np.linalg.norm(pts - lidar_origin, axis=1) < 1.0].tolist(),
        "cups": cups,
        "separation_m": {
            "lidar": float(np.linalg.norm(np.subtract(cups[0]["lidar_centre_m"], cups[1]["lidar_centre_m"]))),
            "stereo_3d": float(np.linalg.norm(X_cam[0] - X_cam[1])),
            "stereo_xy": float(np.linalg.norm(X_base[0, :2] - X_base[1, :2])),
        },
    }
    print("杯距 (m)", {k: round(v, 4) for k, v in out["separation_m"].items()})
    for i, c in enumerate(cups):
        print(f"杯{i + 1} 离雷达 (m)", {k: round(v, 4) for k, v in c["range_from_lidar_m"].items()})
    (snap_dir / "cups.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(snap_dir / "cups.json")


if __name__ == "__main__":
    main()
