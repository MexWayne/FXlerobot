#!/usr/bin/env python
"""Head camera -> left arm transform at one neck pose, from the gripper tip touching board corners.

    python touch_calib.py board        # arm off the board: board pose in both eyes, labelled grid image
    python touch_calib.py touch 6 1    # fixed-jaw tip held on grid corner (column 6, row 1)
    python touch_calib.py drop         # forget the last touch (tip was not on the corner)
    python touch_calib.py solve        # -> calib/cam_to_left_arm.yaml

The board lies flat and still and the neck stays where it is for the whole session. Grid corners
are counted from the board origin in whole squares, (0..7, 0..5); only the inner corners (1..6,
1..4) are sharp enough to touch. The tip is not seen by the cameras (the arm hides it); its camera
position comes from the board pose taken by `board`, its arm position from SO-101 kinematics.
Moves nothing; the left arm is read only, its torque is left as it is.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "project_tool")]
import calibrate_head_cameras as cal  # noqa: E402
import so101_kin  # noqa: E402
from arm_snapshot import MARKER_MM, SQUARE_MM, read_left_arm, read_neck_deg  # noqa: E402
from board_view import HEAD_CAMERAS, board_pose  # noqa: E402

SESSION_DIR = HERE / "calib" / "touch"
SESSION = SESSION_DIR / "session.json"
RESULT = HERE / "calib" / "cam_to_left_arm.yaml"
COLS, ROWS = 7, 5
NECK_TOL_DEG = 0.3
BOARD_TOL_MM = 3.0


def grab():
    calib = yaml.safe_load(HEAD_CAMERAS.read_text())
    caps = {n: cal.open_camera(d, calib.get("focus_absolute", 0)) for n, d in cal.CAMS.items()}
    try:
        for _ in range(10):
            frames = cal.grab_pair(caps)
    finally:
        for cap in caps.values():
            cap.release()
    return calib, frames


def board_poses(calib, frames):
    board, detector = cal.make_board(SQUARE_MM, MARKER_MM)
    out = {}
    for eye, img in frames.items():
        res, _ = board_pose(board, detector, img, np.array(calib[eye]["K"]), np.array(calib[eye]["dist"]))
        if res is not None:
            out[eye] = res
    return out


def corner(c, r):
    return np.array([c * SQUARE_MM / 1000, r * SQUARE_MM / 1000, 0.0])


def in_cam(T, p):
    return T[:3, :3] @ p + T[:3, 3]


def cmd_board(args):
    calib, frames = grab()
    poses = board_poses(calib, frames)
    if "left" not in poses:
        raise SystemExit("左眼没看到标定板：把手臂、杯子、卡尺都从板子上拿开再试")
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    session = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"), "neck_deg": read_neck_deg(),
               "square_mm": SQUARE_MM, "touches": []}
    for eye, res in poses.items():
        session[eye] = {"T_cam_board": res["T_cam_board"].tolist(), "reproj_px": res["reproj_px"],
                        "corners": res["n"]}
        cv2.imwrite(str(SESSION_DIR / f"board_{eye}.jpg"), frames[eye])
        print(f"[{eye}] 角点 {res['n']} 个，重投影 {res['reproj_px']:.2f} px")
    # Left eye with every grid corner labelled (column, row).
    K, dist = np.array(calib["left"]["K"]), np.array(calib["left"]["dist"])
    T = poses["left"]["T_cam_board"]
    vis = frames["left"].copy()
    for r in range(1, ROWS):
        for c in range(1, COLS):
            px, _ = cv2.projectPoints(corner(c, r).reshape(1, 3), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
            u, v = px.reshape(2).round().astype(int)
            cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"{c},{r}", (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
    cv2.imwrite(str(SESSION_DIR / "grid_left.png"), vis)
    SESSION.write_text(json.dumps(session, indent=1) + "\n")
    print(SESSION_DIR / "grid_left.png")


def cmd_touch(args):
    if not 1 <= args.col <= COLS - 1 or not 1 <= args.row <= ROWS - 1:
        raise SystemExit(f"只用内角点：列 1..{COLS - 1}，行 1..{ROWS - 1}")
    session = json.loads(SESSION.read_text())
    neck = read_neck_deg()
    moved = max(abs(neck[k] - session["neck_deg"][k]) for k in neck)
    if moved > NECK_TOL_DEG:
        raise SystemExit(f"脖子比 board 时动了 {moved:.2f}°，要重新跑 board")
    q = read_left_arm()
    tip = so101_kin.tip_position(q)
    # The arm covers part of the board; whatever is still seen must not have moved.
    calib, frames = grab()
    shift = {}
    for eye, res in board_poses(calib, frames).items():
        T0 = np.array(session[eye]["T_cam_board"])
        T1 = res["T_cam_board"]
        shift[eye] = max(np.linalg.norm(in_cam(T1, corner(c, r)) - in_cam(T0, corner(c, r))) * 1000
                         for c in (0, COLS) for r in (0, ROWS))
        print(f"[{eye}] 板子位姿与 board 时相差 {shift[eye]:.1f} mm（{res['n']} 个角点）")
    if not shift:
        print("两眼都没认出板子（被手臂挡住），这一点没法检查板子有没有动")
    elif min(shift.values()) > BOARD_TOL_MM:
        raise SystemExit("板子好像被碰动了：没有记录。把板子放回去后重新跑 board，之前的点作废")
    n = len(session["touches"])
    for eye in ("left", "right"):
        cv2.imwrite(str(SESSION_DIR / f"touch{n + 1}_{eye}.jpg"), frames[eye])
    session["touches"].append({"col": args.col, "row": args.row, "left_arm_deg": q,
                               "tip_in_arm_base_m": tip.tolist(), "board_shift_mm": shift})
    SESSION.write_text(json.dumps(session, indent=1) + "\n")
    print(f"第 {n + 1} 点 ({args.col},{args.row})：爪尖在左臂底座 (mm) {np.round(tip * 1000, 1).tolist()}")


def cmd_drop(args):
    session = json.loads(SESSION.read_text())
    if not session["touches"]:
        raise SystemExit("没有可删的点")
    n = len(session["touches"])
    t = session["touches"].pop()
    for eye in ("left", "right"):
        (SESSION_DIR / f"touch{n}_{eye}.jpg").unlink(missing_ok=True)
    SESSION.write_text(json.dumps(session, indent=1) + "\n")
    print(f"删掉第 {n} 点 ({t['col']},{t['row']})，还剩 {n - 1} 个")


def kabsch(src, dst):
    """R, t with dst ~ R @ src + t."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, md - R @ ms


def cmd_solve(args):
    session = json.loads(SESSION.read_text())
    touches = session["touches"]
    if len(touches) < 3:
        raise SystemExit(f"只有 {len(touches)} 个点，至少要 3 个（建议 6 个，铺开）")
    P_arm = np.array([t["tip_in_arm_base_m"] for t in touches])
    result = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"), "neck_deg": session["neck_deg"],
              "points": [[t["col"], t["row"]] for t in touches]}
    for eye in ("left", "right"):
        if eye not in session:
            continue
        T_cam_board = np.array(session[eye]["T_cam_board"])
        P_cam = np.array([in_cam(T_cam_board, corner(t["col"], t["row"])) for t in touches])
        R, t = kabsch(P_cam, P_arm)
        res = (P_cam @ R.T + t - P_arm) * 1000
        loo = []
        for i in range(len(touches)):
            keep = np.arange(len(touches)) != i
            Ri, ti = kabsch(P_cam[keep], P_arm[keep])
            loo.append(np.linalg.norm(Ri @ P_cam[i] + ti - P_arm[i]) * 1000)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        rms = float(np.sqrt((res ** 2).sum(1).mean()))
        result[eye] = {"T_arm_cam": T.tolist(), "rms_mm": rms, "residual_mm": res.round(1).tolist(),
                       "leave_one_out_mm": np.round(loo, 1).tolist()}
        print(f"[{eye}] 残差 RMS {rms:.1f} mm，留一法 {np.round(loo, 1).tolist()} mm")
        for (c, r), e in zip(result["points"], res):
            print(f"    ({c},{r}) 残差 (mm) {e.round(1).tolist()}")
    RESULT.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    print(RESULT)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("board")
    p = sub.add_parser("touch")
    p.add_argument("col", type=int)
    p.add_argument("row", type=int)
    sub.add_parser("drop")
    sub.add_parser("solve")
    args = ap.parse_args()
    {"board": cmd_board, "touch": cmd_touch, "drop": cmd_drop, "solve": cmd_solve}[args.cmd](args)


if __name__ == "__main__":
    main()
