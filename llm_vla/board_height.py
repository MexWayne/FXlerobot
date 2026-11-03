#!/usr/bin/env python
"""Check the head camera height model with the board lying level at known heights. Moves nothing.

    python board_height.py 14      # board surface 14 mm above the floor (board on the floor)
    python board_height.py 32      # on an 18 mm support
    python board_height.py 85      # on a 71 mm support
    python board_height.py report

Each run stores both eyes' board pose and the neck angles in calib/board_height/h<mm>.json. The
report puts the board into the base frame with head_extrinsic.yaml and the neck model
(vln_nav.cam_at) and compares its height with the stated one: an error that is the same at every
height is a camera height offset, one that grows with height is a depth scale error.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "project_tool"), str(HERE.parent / "vln")]
import validate_head_extrinsic as val  # noqa: E402
from arm_snapshot import SQUARE_MM, read_neck_deg  # noqa: E402
from touch_calib import board_poses, grab  # noqa: E402
from vln_nav import cam_at  # noqa: E402

OUT_DIR = HERE / "calib" / "board_height"
EXTRINSIC_DIR = HERE.parent / "calib" / "head"
CENTRE = np.array([3.5 * SQUARE_MM / 1000, 2.5 * SQUARE_MM / 1000, 0.0, 1.0])


def capture(height_mm):
    calib, frames = grab()
    poses = board_poses(calib, frames)
    if not poses:
        raise SystemExit("两眼都没看到标定板")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"height_mm": height_mm, "stamp": time.strftime("%Y-%m-%d %H:%M:%S"), "neck_deg": read_neck_deg()}
    for eye, res in poses.items():
        rec[eye] = {"T_cam_board": res["T_cam_board"].tolist(), "reproj_px": res["reproj_px"], "corners": res["n"]}
        cv2.imwrite(str(OUT_DIR / f"h{height_mm:g}_{eye}.jpg"), frames[eye])
        print(f"[{eye}] 角点 {res['n']} 个，重投影 {res['reproj_px']:.2f} px")
    (OUT_DIR / f"h{height_mm:g}.json").write_text(json.dumps(rec, indent=1) + "\n")
    report()


def report():
    _, extr, cams = val.load(str(EXTRINSIC_DIR))
    rows = []
    for path in sorted(OUT_DIR.glob("h*.json"), key=lambda p: json.loads(p.read_text())["height_mm"]):
        rec = json.loads(path.read_text())
        n = rec["neck_deg"]
        line = f"板面 {rec['height_mm']:5.1f} mm  脖子 yaw {n['neck_yaw']:6.2f} pitch {n['neck_pitch']:5.2f} |"
        for eye in ("left", "right"):
            if eye not in rec:
                line += f" {eye} 没看到 |"
                continue
            cam = cam_at(cams[eye], n["neck_yaw"] - extr["neck_yaw_pos"], n["neck_pitch"] - extr["neck_pitch_pos"])
            Tcb = np.array(rec[eye]["T_cam_board"])
            T = cam["T_base_cam"] @ Tcb
            z = (T @ CENTRE)[2] * 1000
            tilt = np.degrees(np.arccos(abs(T[2, 2])))
            dist = abs(Tcb[:3, 2] @ Tcb[:3, 3]) * 1000  # camera to board plane, from PnP alone
            line += f" {eye} 算出 {z:6.1f} mm（差 {z - rec['height_mm']:+5.1f}），倾斜 {tilt:.1f}°，相机到板面 {dist:5.1f} mm |"
            if eye == "left":
                rows.append((rec["height_mm"], z, dist))
        print(line)
    if len(rows) >= 2:
        h, z, d = (np.array(c) for c in zip(*rows))
        slope, offset = np.polyfit(h, z, 1)
        print(f"左眼：算出高度 = {slope:.3f} × 真实高度 {offset:+.1f} mm"
              f"（理想 1.000 × 真实 + 0）；相机到板面距离对高度的斜率 {np.polyfit(h, d, 1)[0]:.3f}（理想 -1.000）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("height", help="board surface height above the floor in mm, or 'report'")
    args = ap.parse_args()
    if args.height == "report":
        report()
    else:
        capture(float(args.height))


if __name__ == "__main__":
    main()
