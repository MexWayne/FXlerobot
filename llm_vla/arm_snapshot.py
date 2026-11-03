#!/usr/bin/env python
"""Record what the robot knows about the left arm right now, for figures and checks. Moves nothing.

    python arm_snapshot.py            # -> snapshots/<time>/

Saves the left arm joint angles and fixed-jaw tip from SO-101 forward kinematics (arm base frame),
the neck angles, both head images with the board pose and its projected grid points, and several
lidar turns with the near range kept (lidar_scan.py drops everything inside 0.25 m, which is where
the arm is). Lidar points are in the base frame as lidar_scan.py hands them out (LIDAR_X_M and
LIDAR_Y_M applied at the time of the snapshot).
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "project_tool"), str(ROOT / "vln")]
import calibrate_head_cameras as cal  # noqa: E402
import lidar_scan  # noqa: E402
import so101_kin  # noqa: E402
from board_view import HEAD_CAMERAS, board_pose  # noqa: E402
from save_pose import (  # noqa: E402
    ARMS, MOTORS, Servos, degrees, load_calibration, load_neck_calibration, open_neck, read_neck,
)

LIDAR_TURNS = 8
LIDAR_RMIN = 0.05
SQUARE_MM, MARKER_MM = 40.0, 30.0


def read_left_arm():
    bus = Servos(ARMS["left"])
    try:
        ticks = {n: bus.read(i, "Present_Position") for i, n in enumerate(MOTORS, start=1)}
    finally:
        bus.close()
    if None in ticks.values():
        raise RuntimeError(f"左臂没有读全: {ticks}")
    cal_left = load_calibration("left")
    return {n: degrees(t, cal_left[n]) for n, t in ticks.items()}


def read_neck_deg():
    bus = open_neck()
    try:
        ticks = read_neck(bus)
    finally:
        bus.close()
    c = load_neck_calibration()
    return {n: degrees(t, c[n]) for n, t in ticks.items()}


def lidar_turns():
    lidar_scan.RMIN = LIDAR_RMIN
    lidar_scan.VIEW_DEG = 180.0
    lidar = lidar_scan.Lidar()
    lidar.start()
    try:
        turns = lidar.scans(n=LIDAR_TURNS, timeout=LIDAR_TURNS + 3.0)
    finally:
        lidar.close()
    if not turns:
        raise RuntimeError("雷达没有给出完整的扫描")
    return turns


def main():
    out_dir = HERE / "snapshots" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)

    q = read_left_arm()
    neck = read_neck_deg()
    tip = so101_kin.tip_position(q)
    snap = {"left_arm_deg": q, "neck_deg": neck,
            "tip_in_arm_base_m": tip.tolist(), "tool_axis": so101_kin.tool_axis(q).tolist()}

    calib = yaml.safe_load(HEAD_CAMERAS.read_text())
    board, detector = cal.make_board(SQUARE_MM, MARKER_MM)
    caps = {n: cal.open_camera(d, calib.get("focus_absolute", 0)) for n, d in cal.CAMS.items()}
    try:
        for _ in range(10):
            frames = cal.grab_pair(caps)
    finally:
        for cap in caps.values():
            cap.release()
    cols, rows = board.getChessboardSize()
    grid = np.array([[c * SQUARE_MM / 1000, r * SQUARE_MM / 1000, 0.0]
                     for r in range(rows + 1) for c in range(cols + 1)])
    for eye, img in frames.items():
        cv2.imwrite(str(out_dir / f"{eye}.jpg"), img)
        K, dist = np.array(calib[eye]["K"]), np.array(calib[eye]["dist"])
        res, _ = board_pose(board, detector, img, K, dist)
        if res is None:
            continue
        T = res["T_cam_board"]
        px, _ = cv2.projectPoints(grid, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
        snap[eye] = {"T_cam_board": T.tolist(), "reproj_px": res["reproj_px"],
                     "grid_cols_rows": [cols + 1, rows + 1], "grid_px": px.reshape(-1, 2).tolist(),
                     "detected_px": {str(k): v.tolist() for k, v in res["corners"].items()}}

    turns = lidar_turns()
    np.savez(out_dir / "lidar.npz", *turns)
    snap["lidar_turns"] = len(turns)
    (out_dir / "snapshot.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n")
    print(f"tip in arm base (m) {np.round(tip, 4).tolist()}, lidar {len(turns)} turns")
    print(out_dir)


if __name__ == "__main__":
    main()
