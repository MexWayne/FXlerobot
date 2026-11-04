#!/usr/bin/env python
"""Where the left arm base sits in the base frame, from gripper-tip touches on board corners.

    python mount.py        # -> arm_mount.yaml

Each touch gives the tip in the arm base frame (SO-101 kinematics, from the joint angles recorded at
the touch, so a change to so101_kin counts) and, through the board pose and
the head camera chain (head_extrinsic.yaml + vln_nav.cam_at), the same point in the base frame.
The arm base is taken as level, so the fit is a yaw and a translation. Touches come from
arm_snapshot.py folders that saw the board and from touch_calib.py.
The kinematics were off by 1-2.5 cm between touches (2026-11-02), so the residuals are that big;
pick.py corrects the remaining error with the cameras before it grasps.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
LLM_VLA = HERE.parent
sys.path[:0] = [str(LLM_VLA), str(LLM_VLA.parent / "project_tool"), str(LLM_VLA.parent / "vln")]
import so101_kin  # noqa: E402
import validate_head_extrinsic as val  # noqa: E402
from vln_nav import cam_at  # noqa: E402

OUT = HERE / "arm_mount.yaml"
SQUARE_M = 0.04
SNAPSHOT_TOUCHES = {"20261102_161359": (6, 1)}  # snapshot -> board corner the tip was on


def base_point(cams, extr, neck, T_cam_board, col, row):
    T = cam_at(cams["left"], neck["neck_yaw"] - extr["neck_yaw_pos"],
               neck["neck_pitch"] - extr["neck_pitch_pos"])["T_base_cam"] @ np.array(T_cam_board)
    return (T @ np.array([col * SQUARE_M, row * SQUARE_M, 0.0, 1.0]))[:3]


def touches(cams, extr):
    out = []
    for snap, (c, r) in SNAPSHOT_TOUCHES.items():
        s = json.loads((LLM_VLA / "snapshots" / snap / "snapshot.json").read_text())
        out.append((f"{snap} ({c},{r})", base_point(cams, extr, s["neck_deg"], s["left"]["T_cam_board"], c, r),
                    so101_kin.tip_position(s["left_arm_deg"])))
    session = LLM_VLA / "calib" / "touch" / "session.json"
    if session.is_file():
        s = json.loads(session.read_text())
        for t in s["touches"]:
            out.append((f"touch ({t['col']},{t['row']})",
                        base_point(cams, extr, s["neck_deg"], s["left"]["T_cam_board"], t["col"], t["row"]),
                        so101_kin.tip_position(t["left_arm_deg"])))
    return out


def fit(pairs):
    """yaw, t with p_base = Rz(yaw) @ p_arm + t."""
    A = np.array([a for _, _, a in pairs])
    B = np.array([b for _, b, _ in pairs])
    ma, mb = A[:, :2].mean(0), B[:, :2].mean(0)
    H = (A[:, :2] - ma).T @ (B[:, :2] - mb)
    yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t = (B - A @ R.T).mean(0)
    return R, t


def load():
    """4x4 T_base_arm from arm_mount.yaml."""
    return np.array(yaml.safe_load(OUT.read_text())["T_base_arm"])


def main():
    _, extr, cams = val.load(str(LLM_VLA.parent / "calib" / "head"))
    pairs = touches(cams, extr)
    if len(pairs) < 2:
        raise SystemExit("碰点不够，至少要 2 个")
    R, t = fit(pairs)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    res = {}
    for name, b, a in pairs:
        e = (R @ a + t - b) * 1000
        res[name] = np.round(e, 1).tolist()
        print(f"{name}: 残差 (mm) {res[name]}")
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    print(f"左臂底座在底盘坐标 (mm) {np.round(t * 1000, 1).tolist()}，朝向 {yaw:.1f}°")
    OUT.write_text(yaml.safe_dump({"T_base_arm": T.tolist(), "yaw_deg": yaw, "residual_mm": res},
                                  sort_keys=False, allow_unicode=True))
    print(OUT)


if __name__ == "__main__":
    main()
