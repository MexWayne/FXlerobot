#!/usr/bin/env python
"""Look at the ChArUco board lying flat on the floor from one neck pose and save its pose per eye.

    python board_view.py search 0 0       # level head, the pose head_extrinsic.yaml was calibrated at
    python board_view.py grasp -15 35     # head turned to the left arm's workspace

Writes calib/board_<tag>.json (T_cam_board for each eye, measured neck angles) and annotated
images calib/board_<tag>_<eye>.png. The square size is also measured by stereo triangulation,
which checks --square-mm against the real board.
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
sys.path.insert(0, str(HERE.parent / "project_tool"))
import calibrate_head_cameras as cal  # noqa: E402
from neck_look import move_neck  # noqa: E402

CALIB_DIR = HERE / "calib"
HEAD_CAMERAS = CALIB_DIR / "head_cameras.yaml"


def board_pose(board, detector, img, K, dist):
    corners, ids, mc, mi = cal.detect(detector, img)
    if ids is None or len(ids) < cal.MIN_CORNERS:
        return None, cal.draw(img, (corners, ids, mc, mi), "board")
    obj, imgp = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, cal.draw(img, (corners, ids, mc, mi), "board")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    vis = cal.draw(img, (corners, ids, mc, mi), "board")
    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, 0.05)
    return {"T_cam_board": T, "reproj_px": err, "n": len(ids),
            "corners": {int(i): c.reshape(2) for i, c in zip(ids.flatten(), corners)}}, vis


def stereo_square_mm(calib, left, right, board):
    """Triangulate corners seen by both eyes and return the mean neighbouring-corner distance."""
    common = sorted(set(left["corners"]) & set(right["corners"]))
    if len(common) < 6:
        return None
    R = np.array(calib["stereo"]["R_right_left"])
    t = np.array(calib["stereo"]["T_right_left_m"]).reshape(3, 1)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t])
    pts = {}
    for eye, P in (("left", P1), ("right", P2)):
        K, dist = np.array(calib[eye]["K"]), np.array(calib[eye]["dist"])
        raw = np.array([(left if eye == "left" else right)["corners"][i] for i in common], np.float32)
        pts[eye] = cv2.undistortPoints(raw.reshape(-1, 1, 2), K, dist).reshape(-1, 2).T
    X = cv2.triangulatePoints(P1, P2, pts["left"], pts["right"])
    X = (X[:3] / X[3]).T
    xyz = dict(zip(common, X))
    cols = board.getChessboardSize()[0] - 1
    d = [np.linalg.norm(xyz[i] - xyz[i + 1]) for i in common if i + 1 in xyz and (i + 1) % cols]
    d += [np.linalg.norm(xyz[i] - xyz[i + cols]) for i in common if i + cols in xyz]
    return float(np.mean(d) * 1000) if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("yaw", type=float)
    ap.add_argument("pitch", type=float)
    ap.add_argument("--cameras", default=None,
                    help="head_cameras.yaml; default llm_vla/calib if rescaled there, else calib/head")
    ap.add_argument("--square-mm", type=float, default=40.0)
    ap.add_argument("--marker-mm", type=float, default=30.0)
    args = ap.parse_args()

    neck = move_neck(args.yaw, args.pitch)
    print(f"neck yaw={neck['neck_yaw']:.2f}° pitch={neck['neck_pitch']:.2f}°")
    time.sleep(0.5)

    cameras = Path(args.cameras) if args.cameras else HEAD_CAMERAS
    if not cameras.is_file():
        cameras = Path(cal.DEFAULT_DATA) / "head_cameras.yaml"
    calib = yaml.safe_load(cameras.read_text())
    print(f"相机参数 {cameras}")
    board, detector = cal.make_board(args.square_mm, args.marker_mm)
    caps = {n: cal.open_camera(d, calib.get("focus_absolute", 0)) for n, d in cal.CAMS.items()}
    try:
        for _ in range(10):
            frames = cal.grab_pair(caps)
    finally:
        for cap in caps.values():
            cap.release()

    CALIB_DIR.mkdir(exist_ok=True)
    out = {"tag": args.tag, "neck_cmd": [args.yaw, args.pitch],
           "neck": [neck["neck_yaw"], neck["neck_pitch"]],
           "square_mm": args.square_mm, "marker_mm": args.marker_mm, "cameras": str(cameras),
           "stamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    seen = {}
    for eye in cal.CAMS:
        K, dist = np.array(calib[eye]["K"]), np.array(calib[eye]["dist"])
        res, vis = board_pose(board, detector, frames[eye], K, dist)
        cv2.imwrite(str(CALIB_DIR / f"board_{args.tag}_{eye}.png"), vis)
        if res is None:
            print(f"[{eye}] 没看到标定板（角点不足 {cal.MIN_CORNERS} 个）")
            continue
        seen[eye] = res
        p = res["T_cam_board"][:3, 3] * 1000
        print(f"[{eye}] corners={res['n']} reproj={res['reproj_px']:.2f}px "
              f"board origin in cam (mm) {np.round(p, 0).tolist()}")
        out[eye] = {"T_cam_board": res["T_cam_board"].tolist(), "reproj_px": res["reproj_px"],
                    "corners": res["n"]}
    if len(seen) == 2:
        sq = stereo_square_mm(calib, seen["left"], seen["right"], board)
        if sq is not None:
            out["stereo_square_mm"] = sq
            print(f"双目三角化量出的格子边长 {sq:.1f} mm（脚本用的是 {args.square_mm} mm）")
    path = CALIB_DIR / f"board_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
