#!/usr/bin/env python3
"""ChArUco calibration for the two head (front) C920 cameras.

  capture     live view of both eyes, auto-saves a pair when the board is held still, ESC quits
  cali_intri  intrinsics for each eye + left->right stereo extrinsics
  cali_extri  board standing upright on the floor facing the robot -> pose of each eye in the base frame

Board: 7x5 squares, DICT_5X5_50, square 40 mm, marker 30 mm.
Base frame matches LeKiwiBase: x forward, y left, z up, origin at base center on the floor.
"""

import argparse
import glob
import os
import select
import subprocess
import sys
import time

if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"
_xauth = "/run/user/%d/gdm/Xauthority" % os.getuid()
if not os.environ.get("XAUTHORITY") and os.path.exists(_xauth):
    os.environ["XAUTHORITY"] = _xauth

import cv2
import numpy as np
import yaml

CAMS = {"left": "/dev/cam_front_left", "right": "/dev/cam_front_right"}
WIDTH, HEIGHT = 640, 480
MIN_CORNERS = 8
DEFAULT_DATA = os.path.expanduser("~/workspaces/FXLerobot/FXlerobot/calib/head")


def make_board(square_mm, marker_mm):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    board = cv2.aruco.CharucoBoard((7, 5), square_mm / 1000.0, marker_mm / 1000.0, dictionary)
    return board, cv2.aruco.CharucoDetector(board)


def lock_focus(dev, focus):
    # C920 resets to autofocus after replug or reboot; intrinsics are only valid at one focus.
    for ctrl in ("focus_automatic_continuous=0", f"focus_absolute={focus}"):
        subprocess.run(["v4l2-ctl", "-d", dev, "-c", ctrl], capture_output=True)


def open_camera(dev, focus):
    lock_focus(dev, focus)
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {dev}")
    return cap


def grab_pair(caps):
    # grab() both first so the two frames are as close in time as USB allows.
    for cap in caps.values():
        cap.grab()
    frames = {}
    for name, cap in caps.items():
        ok, frame = cap.retrieve()
        if not ok:
            raise SystemExit(f"read failed: {name}")
        frames[name] = frame
    return frames


def detect(detector, img):
    corners, ids, marker_corners, marker_ids = detector.detectBoard(img)
    if ids is None:
        return None, None, marker_corners, marker_ids
    return corners, ids, marker_corners, marker_ids


def draw(img, det, label):
    vis = img.copy()
    corners, ids, marker_corners, marker_ids = det
    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (0, 255, 0))
    n = 0 if ids is None else len(ids)
    color = (0, 255, 0) if n >= MIN_CORNERS else (0, 0, 255)
    cv2.putText(vis, f"{label} corners={n}/24", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def corner_map(det):
    corners, ids = det[0], det[1]
    if ids is None:
        return {}
    return {int(i): c.reshape(2) for i, c in zip(ids.flatten(), corners)}


def mean_shift(a, b):
    common = set(a) & set(b)
    if len(common) < 4:
        return float("inf")
    return float(np.mean([np.linalg.norm(a[i] - b[i]) for i in common]))


def stdin_enter_pressed():
    if not sys.stdin.isatty():
        return False
    pressed = False
    while select.select([sys.stdin], [], [], 0)[0]:
        if not sys.stdin.readline():
            break
        pressed = True
    return pressed


def cmd_capture(args):
    board, detector = make_board(args.square_mm, args.marker_mm)
    os.makedirs(args.data, exist_ok=True)
    caps = {name: open_camera(dev, args.focus) for name, dev in CAMS.items()}
    idx = len(glob.glob(os.path.join(args.data, "left_*.png")))
    print(f"saving to {args.data}, starting at {idx}.")
    if args.auto:
        print(f"auto save: hold the board still; a new pose must move > {args.min_move_px:.0f}px from the last saved one.")
    print("manual save: SPACE/s in the image window, or ENTER in this terminal. ESC/q in the window or Ctrl+C quits.")
    prev = {n: {} for n in CAMS}
    last_saved = {n: {} for n in CAMS}
    still_since = None
    last_save_t = 0.0
    try:
        while True:
            frames = grab_pair(caps)
            dets = {n: detect(detector, f) for n, f in frames.items()}
            cur = {n: corner_map(dets[n]) for n in CAMS}
            usable = [n for n in CAMS if len(cur[n]) >= MIN_CORNERS]
            now = time.time()

            still = bool(usable) and all(mean_shift(cur[n], prev[n]) < args.still_px for n in usable)
            still_since = (still_since or now) if still else None
            moved = any(mean_shift(cur[n], last_saved[n]) > args.min_move_px for n in usable)
            auto_ready = (args.auto and still_since is not None and now - still_since >= args.hold_s
                          and moved and now - last_save_t >= 1.0)

            view = np.hstack([draw(frames[n], dets[n], n) for n in CAMS])
            status = "HOLD STILL" if usable and not still else ("MOVE TO NEW POSE" if usable and not moved else "")
            cv2.putText(view, f"saved={idx} {status}", (8, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)
            cv2.imshow("head cameras", view)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            manual = key in (32, ord("s"), 13) or stdin_enter_pressed()

            if manual or auto_ready:
                for n in CAMS:
                    cv2.imwrite(os.path.join(args.data, f"{n}_{idx:03d}.png"), frames[n])
                    if len(cur[n]) >= MIN_CORNERS:
                        last_saved[n] = cur[n]
                counts = {n: len(cur[n]) for n in CAMS}
                both = "both eyes" if len(usable) == 2 else (usable[0] if usable else "NO BOARD")
                print(f"saved {idx:03d} ({'manual' if manual else 'auto'}) corners {counts} -> {both}", flush=True)
                idx += 1
                last_save_t = now
                still_since = None
            prev = cur
    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        pairs = sum(
            1 for i in range(idx)
            if os.path.exists(os.path.join(args.data, f"left_{i:03d}.png"))
        )
        print(f"{pairs} pairs in {args.data}")


def collect(board, detector, data, name):
    obj_pts, img_pts, ids_by_view, files = [], [], [], []
    for path in sorted(glob.glob(os.path.join(data, f"{name}_*.png"))):
        img = cv2.imread(path)
        corners, ids, _, _ = detect(detector, img)
        if ids is None or len(ids) < MIN_CORNERS:
            continue
        obj, imgp = board.matchImagePoints(corners, ids)
        obj_pts.append(obj.astype(np.float32))
        img_pts.append(imgp.astype(np.float32))
        ids_by_view.append({int(i): c.reshape(2) for i, c in zip(ids.flatten(), corners)})
        files.append(os.path.basename(path))
    return obj_pts, img_pts, ids_by_view, files


def calibrate_one(board, detector, data, name, fit_k3):
    obj_pts, img_pts, ids_by_view, files = collect(board, detector, data, name)
    if len(obj_pts) < 10:
        raise SystemExit(f"{name}: only {len(obj_pts)} usable images, need >= 10 (aim for 20+)")
    # k3 only gets constrained by corners near the image edges; without them it overfits wildly.
    flags = 0 if fit_k3 else cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (WIDTH, HEIGHT), None, None, flags=flags)
    per_view = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    print(f"[{name}] images={len(obj_pts)} rms={rms:.3f}px fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} "
          f"cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    worst = sorted(zip(per_view, files), reverse=True)[:3]
    print(f"[{name}] worst views: " + ", ".join(f"{f}={e:.2f}px" for e, f in worst))
    return {"rms": float(rms), "K": K, "dist": dist, "images": len(obj_pts)}, ids_by_view, files


def cmd_calibrate(args):
    board, detector = make_board(args.square_mm, args.marker_mm)
    chess = board.getChessboardCorners().astype(np.float32)
    result = {}
    views = {}
    for name in CAMS:
        result[name], ids_by_view, files = calibrate_one(board, detector, args.data, name, args.fit_k3)
        views[name] = {f.split("_", 1)[1]: v for f, v in zip(files, ids_by_view)}

    obj_pts, left_pts, right_pts = [], [], []
    for key in sorted(set(views["left"]) & set(views["right"])):
        lv, rv = views["left"][key], views["right"][key]
        common = sorted(set(lv) & set(rv))
        if len(common) < MIN_CORNERS:
            continue
        obj_pts.append(chess[common].reshape(-1, 1, 3))
        left_pts.append(np.array([lv[i] for i in common], np.float32).reshape(-1, 1, 2))
        right_pts.append(np.array([rv[i] for i in common], np.float32).reshape(-1, 1, 2))

    stereo = None
    if len(obj_pts) >= 5:
        rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            obj_pts, left_pts, right_pts,
            result["left"]["K"], result["left"]["dist"],
            result["right"]["K"], result["right"]["dist"],
            (WIDTH, HEIGHT), flags=cv2.CALIB_FIX_INTRINSIC,
        )
        baseline_mm = float(np.linalg.norm(T) * 1000)
        axis_angle_deg = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
        print(f"[stereo] pairs={len(obj_pts)} rms={rms:.3f}px baseline={baseline_mm:.1f}mm "
              f"optical axes {axis_angle_deg:.1f} deg apart T(mm)={np.round(T.ravel() * 1000, 1).tolist()}")
        stereo = {"rms": float(rms), "pairs": len(obj_pts), "R_right_left": R.tolist(),
                  "T_right_left_m": T.ravel().tolist(), "baseline_mm": baseline_mm,
                  "optical_axis_angle_deg": axis_angle_deg}
    else:
        print(f"[stereo] only {len(obj_pts)} pairs with the board in both eyes, need >= 5; skipped")

    out = {
        "image_size": [WIDTH, HEIGHT],
        "focus_absolute": args.focus,
        "board": {"squares": [7, 5], "dict": "DICT_5X5_50",
                  "square_mm": args.square_mm, "marker_mm": args.marker_mm},
        "stereo": stereo,
    }
    for name in CAMS:
        r = result[name]
        out[name] = {"device": CAMS[name], "rms": r["rms"], "images": r["images"],
                     "K": r["K"].tolist(), "dist": r["dist"].ravel().tolist()}
    path = os.path.join(args.data, "head_cameras.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"wrote {path}")


def upright_axis_dir(axis_cam):
    # Head camera roll is small, so image right/down tell the board's in-plane orientation.
    if abs(axis_cam[0]) >= abs(axis_cam[1]):
        return "right" if axis_cam[0] > 0 else "left"
    return "down" if axis_cam[1] > 0 else "up"


def base_from_upright_board(x_m, y_m, z_m, yaw_deg, x_dir, y_dir):
    # Board stands on the floor facing the robot: OpenCV board z points away from the robot.
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    left = np.array([-s, c, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    dirs = {"left": left, "right": -left, "up": up, "down": -up}
    T = np.eye(4)
    T[:3, 0] = dirs[x_dir]
    T[:3, 1] = dirs[y_dir]
    T[:3, 2] = np.cross(T[:3, 0], T[:3, 1])
    T[:3, 3] = [x_m, y_m, z_m]
    if T[:3, 2] @ np.array([c, s, 0.0]) < 0.5:
        raise SystemExit(f"board axes x={x_dir} y={y_dir} do not face away from the robot; is the board mirrored?")
    return T


def upright_origin_corner(x_dir, y_dir):
    horiz = x_dir if x_dir in ("left", "right") else y_dir
    vert = y_dir if horiz == x_dir else x_dir
    side = {"right": "robot-left", "left": "robot-right"}[horiz]
    level = {"up": "bottom", "down": "top"}[vert]
    return f"{level} {side}"


def cmd_extrinsic(args):
    calib_path = os.path.join(args.data, "head_cameras.yaml")
    with open(calib_path) as f:
        calib = yaml.safe_load(f)
    board, detector = make_board(args.square_mm, args.marker_mm)
    caps = {name: open_camera(dev, calib.get("focus_absolute", args.focus)) for name, dev in CAMS.items()}
    try:
        for _ in range(10):
            frames = grab_pair(caps)
    finally:
        for cap in caps.values():
            cap.release()

    board_in_base = {"mode": "upright", "x_mm": args.board_x_mm, "y_mm": args.board_y_mm,
                     "z_mm": args.board_z_mm, "yaw_deg": args.board_yaw_deg}
    out = {"neck_yaw_pos": args.neck_yaw, "neck_pitch_pos": args.neck_pitch,
           "board_in_base": board_in_base, "stamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    missing = []
    orient = {}
    for name in CAMS:
        K = np.array(calib[name]["K"])
        dist = np.array(calib[name]["dist"])
        corners, ids, mc, mi = detect(detector, frames[name])
        if ids is None or len(ids) < MIN_CORNERS:
            n = 0 if ids is None else len(ids)
            print(f"[{name}] only {n} corners (need {MIN_CORNERS}); the whole board must sit inside this eye's "
                  f"view, away from the image edges")
            cv2.imwrite(os.path.join(args.data, f"extrinsic_{name}_{stamp}.png"), draw(frames[name], (corners, ids, mc, mi), name))
            missing.append(name)
            continue
        obj, imgp = board.matchImagePoints(corners, ids)
        ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            print(f"[{name}] solvePnP failed")
            missing.append(name)
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cam_board[:3, 3] = tvec.ravel()
        x_dir = upright_axis_dir(T_cam_board[:3, 0])
        y_dir = upright_axis_dir(T_cam_board[:3, 1])
        if ({x_dir, y_dir} & {"left", "right"}) == set() or ({x_dir, y_dir} & {"up", "down"}) == set():
            raise SystemExit(f"[{name}] board axes x={x_dir} y={y_dir} are not one horizontal + one vertical; "
                             f"is the board really upright?")
        orient[name] = (x_dir, y_dir)
        T_base_board = base_from_upright_board(args.board_x_mm / 1000, args.board_y_mm / 1000,
                                               args.board_z_mm / 1000, args.board_yaw_deg, x_dir, y_dir)
        T_base_cam = T_base_board @ np.linalg.inv(T_cam_board)
        pos_mm = T_base_cam[:3, 3] * 1000
        optical_axis = T_base_cam[:3, 2]
        pitch_down = float(np.degrees(np.arcsin(-optical_axis[2])))
        yaw_left = float(np.degrees(np.arctan2(optical_axis[1], optical_axis[0])))
        print(f"[{name}] corners={len(ids)} reproj={err:.2f}px cam in base (mm) "
              f"x={pos_mm[0]:.0f} y={pos_mm[1]:.0f} z={pos_mm[2]:.0f} "
              f"looks down {pitch_down:.1f} deg, turned left {yaw_left:+.1f} deg")
        vis = draw(frames[name], (corners, ids, mc, mi), name)
        cv2.drawFrameAxes(vis, K, dist, rvec, tvec, 2 * args.square_mm / 1000)
        img_path = os.path.join(args.data, f"extrinsic_{name}_{stamp}.png")
        cv2.imwrite(img_path, vis)
        out[name] = {"reproj_px": err, "T_base_cam": T_base_cam.tolist(), "image": img_path,
                     "pitch_down_deg": pitch_down, "yaw_left_deg": yaw_left}
    path = os.path.join(args.data, "head_extrinsic.yaml")
    if orient:
        if len(set(orient.values())) > 1:
            raise SystemExit(f"{path} NOT written: eyes disagree on the board orientation {orient}")
        x_dir, y_dir = next(iter(orient.values()))
        board_in_base.update({"x_dir": x_dir, "y_dir": y_dir})
        print(f"upright board: red x axis points {x_dir}, green y axis points {y_dir}. "
              f"Measure from the {upright_origin_corner(x_dir, y_dir)} corner of the checker pattern "
              f"(seen from behind the robot).")
    if missing:
        raise SystemExit(f"{path} NOT written: {', '.join(missing)} did not see the board. "
                         f"Check extrinsic_*_{stamp}.png, move the board and rerun.")
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"wrote {path}  (board axes drawn: red=x, green=y; origin is the board corner they start from)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--square-mm", type=float, default=40.0)
    p.add_argument("--marker-mm", type=float, default=30.0)
    p.add_argument("--focus", type=int, default=0, help="C920 focus_absolute, 0 = infinity")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--no-auto", dest="auto", action="store_false", help="only save on key press")
    c.add_argument("--hold-s", type=float, default=0.8, help="board must be still this long before auto save")
    c.add_argument("--still-px", type=float, default=1.5, help="max mean corner motion per frame to count as still")
    c.add_argument("--min-move-px", type=float, default=40.0, help="min corner shift from the last saved pose")
    k = sub.add_parser("cali_intri")
    k.add_argument("--fit-k3", action="store_true", help="also fit k3; only with many corners near image edges")
    e = sub.add_parser("cali_extri")
    e.add_argument("--board-x-mm", type=float, default=0.0, help="board origin, forward of base center")
    e.add_argument("--board-y-mm", type=float, default=0.0, help="board origin, left of base center")
    e.add_argument("--board-z-mm", type=float, default=0.0, help="board origin above the floor")
    e.add_argument("--board-yaw-deg", type=float, default=0.0,
                   help="board normal (pointing away from the robot) vs base forward, CCW")
    e.add_argument("--neck-yaw", type=float, default=None)
    e.add_argument("--neck-pitch", type=float, default=None)
    args = p.parse_args()
    {"capture": cmd_capture, "cali_intri": cmd_calibrate, "cali_extri": cmd_extrinsic}[args.cmd](args)


if __name__ == "__main__":
    main()
