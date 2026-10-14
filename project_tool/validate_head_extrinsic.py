#!/usr/bin/env python3
"""Check the head camera -> base extrinsics written by
`calibrate_head_cameras.py cali_extri` (head_extrinsic.yaml).

  1. stereo consistency  left/right extrinsics must reproduce the stereo calibration (no tape needed)
  2. upright board       stand the board vertically somewhere else than during cali_extri: each eye must
                         see it vertical (lean ~ 0) and both eyes must agree on where it is.
                         With --board-x-mm/--board-y-mm/--board-z-mm/--board-yaw-deg it is also compared
                         to the tape.
  3. floor grid + click  a 10 cm floor grid in base coordinates is drawn on both eyes; click a floor point
                         to read its base x/y, the same point is marked in the other eye.

Live window: SPACE/s in the window or ENTER in the terminal logs a row to extrinsic_check.csv,
ESC/q quits. --headless prints one report and exits.
Base frame: x forward, y left, z up, origin on the floor under the base center.
"""

import argparse
import csv
import os
import sys
import time

import calibrate_head_cameras as cal
import cv2
import numpy as np
import yaml

GRID_STEP_M = 0.1
GRID_X = (0.2, 2.0)
GRID_Y = (-1.0, 1.0)

# Pass thresholds for the printed OK / CHECK verdicts.
STEREO_ROT_DEG = 1.0
STEREO_TRANS_MM = 10.0
LEAN_DEG = 1.5
EYE_AGREE_MM = 8.0
TRUTH_MM = 20.0
TRUTH_YAW_DEG = 1.5


def verdict(ok):
    return "OK" if ok else "CHECK"


def load(data):
    with open(os.path.join(data, "head_cameras.yaml")) as f:
        calib = yaml.safe_load(f)
    extr_path = os.path.join(data, "head_extrinsic.yaml")
    if not os.path.exists(extr_path):
        raise SystemExit(f"{extr_path} not found, run `calibrate_head_cameras.py cali_extri` first")
    with open(extr_path) as f:
        extr = yaml.safe_load(f)
    cams = {}
    for name in cal.CAMS:
        if name not in extr:
            raise SystemExit(f"{extr_path} has no '{name}' entry; that eye did not see the board during cali_extri")
        cams[name] = {
            "K": np.array(calib[name]["K"], float),
            "dist": np.array(calib[name]["dist"], float),
            "T_base_cam": np.array(extr[name]["T_base_cam"], float),
        }
    return calib, extr, cams


def stereo_consistency(calib, cams):
    if not calib.get("stereo"):
        print("[stereo] no stereo calibration in head_cameras.yaml, skipped")
        return None
    # Maps left-camera coordinates to right-camera coordinates, same convention as stereoCalibrate.
    T_rl = np.linalg.inv(cams["right"]["T_base_cam"]) @ cams["left"]["T_base_cam"]
    R_st = np.array(calib["stereo"]["R_right_left"], float)
    t_st = np.array(calib["stereo"]["T_right_left_m"], float)
    dR = R_st.T @ T_rl[:3, :3]
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))))
    trans_mm = float(np.linalg.norm(T_rl[:3, 3] - t_st) * 1000)
    ok = rot_deg < STEREO_ROT_DEG and trans_mm < STEREO_TRANS_MM
    print(f"[stereo] extrinsics vs stereo calibration: rotation off {rot_deg:.2f} deg, "
          f"translation off {trans_mm:.1f} mm -> {verdict(ok)}")
    return {"rot_deg": rot_deg, "trans_mm": trans_mm, "ok": ok}


def board_in_base(board, detector, img, cam):
    det = cal.detect(detector, img)
    corners, ids = det[0], det[1]
    if ids is None or len(ids) < cal.MIN_CORNERS:
        return None, det
    obj, imgp = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, cam["K"], cam["dist"], flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, det
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cam_board[:3, 3] = tvec.ravel()
    T_base_board = cam["T_base_cam"] @ T_cam_board
    sx, sy = board.getChessboardSize()
    sq = board.getSquareLength()
    center = (T_base_board @ np.array([sx * sq / 2, sy * sq / 2, 0, 1]))[:3]
    normal = T_base_board[:3, 2]
    return {
        "corners": len(ids),
        "origin_mm": T_base_board[:3, 3] * 1000,
        "center_mm": center * 1000,
        "normal_yaw_deg": float(np.degrees(np.arctan2(normal[1], normal[0]))),
        "lean_deg": float(np.degrees(np.arcsin(np.clip(abs(normal[2]), 0.0, 1.0)))),
        "x_dir": cal.upright_axis_dir(T_cam_board[:3, 0]),
        "y_dir": cal.upright_axis_dir(T_cam_board[:3, 1]),
        "rvec": rvec,
        "tvec": tvec,
    }, det


def angle_diff(a, b):
    return float((a - b + 180) % 360 - 180)


def board_report(meas, truth):
    rows = {}
    for name, m in meas.items():
        if m is None:
            print(f"[{name}] board not found (need >= {cal.MIN_CORNERS} corners)")
            continue
        o = m["origin_mm"]
        vertical = m["lean_deg"] < LEAN_DEG
        line = (f"[{name}] corners={m['corners']} board origin x={o[0]:.0f} y={o[1]:.0f} z={o[2]:.0f} mm, "
                f"faces yaw={m['normal_yaw_deg']:+.1f} deg | lean {m['lean_deg']:.2f} deg "
                f"-> vertical {verdict(vertical)}")
        row = {"eye": name, "corners": m["corners"], "origin_x_mm": o[0], "origin_y_mm": o[1],
               "origin_z_mm": o[2], "yaw_deg": m["normal_yaw_deg"], "lean_deg": m["lean_deg"],
               "upright_ok": vertical}
        if truth is not None:
            err = float(np.linalg.norm(o - np.array([truth["x_mm"], truth["y_mm"], truth["z_mm"]])))
            yaw_err = angle_diff(m["normal_yaw_deg"], truth["yaw_deg"])
            t_ok = err < TRUTH_MM and abs(yaw_err) < TRUTH_YAW_DEG
            line += f" | vs tape {err:.1f} mm, yaw {yaw_err:+.1f} deg -> {verdict(t_ok)}"
            row.update({"truth_err_mm": err, "truth_yaw_err_deg": yaw_err, "truth_ok": t_ok})
        print(line)
        rows[name] = row
    found = [meas[n] for n in cal.CAMS if meas.get(n) is not None]
    if found:
        m = found[0]
        try:
            corner = cal.upright_origin_corner(m["x_dir"], m["y_dir"])
            print(f"[board] origin is the {corner} corner of the checker pattern (seen from behind the robot)")
        except KeyError:
            print(f"[board] axes x={m['x_dir']} y={m['y_dir']}: the board does not look upright")
    if len(found) == 2:
        d = meas["left"]["center_mm"] - meas["right"]["center_mm"]
        agree = float(np.linalg.norm(d))
        print(f"[eyes] left vs right board center differs {agree:.1f} mm "
              f"(dx={d[0]:+.1f} dy={d[1]:+.1f} dz={d[2]:+.1f}) -> {verdict(agree < EYE_AGREE_MM)}")
        for row in rows.values():
            row["eyes_agree_mm"] = agree
    return rows


def project_base_points(pts_base, cam):
    T_cam_base = np.linalg.inv(cam["T_base_cam"])
    p_cam = (T_cam_base[:3, :3] @ pts_base.T + T_cam_base[:3, 3:4]).T
    valid = p_cam[:, 2] > 0.05
    uv = np.full((len(pts_base), 2), np.nan)
    if valid.any():
        proj, _ = cv2.projectPoints(p_cam[valid], np.zeros(3), np.zeros(3), cam["K"], cam["dist"])
        uv[valid] = proj.reshape(-1, 2)
    return uv


def draw_polyline(img, uv, color, thickness):
    h, w = img.shape[:2]
    for a, b in zip(uv[:-1], uv[1:]):
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        if max(abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1])) > 5 * max(w, h):
            continue
        cv2.line(img, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), color, thickness)


def draw_floor_grid(img, cam):
    ys = np.linspace(*GRID_Y, 80)
    xs = np.linspace(*GRID_X, 80)
    for x in np.arange(GRID_X[0], GRID_X[1] + 1e-6, GRID_STEP_M):
        major = abs(x / 0.5 - round(x / 0.5)) < 1e-6
        pts = np.column_stack([np.full_like(ys, x), ys, np.zeros_like(ys)])
        draw_polyline(img, project_base_points(pts, cam), (0, 220, 255) if major else (0, 140, 160), 2 if major else 1)
        if major:
            uv = project_base_points(np.array([[x, 0.0, 0.0]]), cam)[0]
            if not np.isnan(uv).any() and 0 <= uv[0] < img.shape[1] and 0 <= uv[1] < img.shape[0]:
                cv2.putText(img, f"x={x:.1f}m", (int(uv[0]) + 4, int(uv[1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    for y in np.arange(GRID_Y[0], GRID_Y[1] + 1e-6, GRID_STEP_M):
        if abs(y) < 1e-6:
            continue
        pts = np.column_stack([xs, np.full_like(xs, y), np.zeros_like(xs)])
        draw_polyline(img, project_base_points(pts, cam), (0, 140, 160), 1)
    pts = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    draw_polyline(img, project_base_points(pts, cam), (0, 0, 255), 2)


def pixel_to_floor(u, v, cam):
    n = cv2.undistortPoints(np.array([[[u, v]]], float), cam["K"], cam["dist"]).reshape(2)
    R, t = cam["T_base_cam"][:3, :3], cam["T_base_cam"][:3, 3]
    d = R @ np.array([n[0], n[1], 1.0])
    if d[2] >= -1e-6:
        return None
    return t + (-t[2] / d[2]) * d


def append_csv(path, rows):
    if not rows:
        print("nothing to log: board not found in either eye")
        return
    fields = ["stamp", "eye", "corners", "origin_x_mm", "origin_y_mm", "origin_z_mm", "yaw_deg",
              "lean_deg", "upright_ok", "eyes_agree_mm", "truth_err_mm", "truth_yaw_err_deg", "truth_ok"]
    new = not os.path.exists(path)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for row in rows.values():
            w.writerow({"stamp": stamp, **{k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in row.items()}})
    print(f"logged {len(rows)} row(s) to {path}")


def get_frames(args, caps):
    if args.left_image:
        return {"left": cv2.imread(args.left_image), "right": cv2.imread(args.right_image)}
    return cal.grab_pair(caps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=cal.DEFAULT_DATA)
    p.add_argument("--square-mm", type=float, default=28.2)
    p.add_argument("--marker-mm", type=float, default=20.0)
    p.add_argument("--board-x-mm", type=float, default=None, help="tape: board origin forward of base center")
    p.add_argument("--board-y-mm", type=float, default=None, help="tape: board origin left of base center")
    p.add_argument("--board-z-mm", type=float, default=0.0, help="tape: board origin above the floor")
    p.add_argument("--board-yaw-deg", type=float, default=0.0,
                   help="tape: board normal (pointing away from the robot) vs base forward, CCW")
    p.add_argument("--headless", action="store_true", help="print one report and exit, no window")
    p.add_argument("--left-image", help="use saved images instead of the cameras")
    p.add_argument("--right-image")
    args = p.parse_args()
    if bool(args.left_image) != bool(args.right_image):
        raise SystemExit("give both --left-image and --right-image")

    calib, extr, cams = load(args.data)
    board, detector = cal.make_board(args.square_mm, args.marker_mm)
    truth = None
    if args.board_x_mm is not None and args.board_y_mm is not None:
        truth = {"x_mm": args.board_x_mm, "y_mm": args.board_y_mm, "z_mm": args.board_z_mm,
                 "yaw_deg": args.board_yaw_deg}

    print(f"extrinsic from {extr.get('stamp', '?')}, neck yaw={extr.get('neck_yaw_pos')} "
          f"pitch={extr.get('neck_pitch_pos')}. The neck must be at this pose for the check to mean anything.")
    for name, cam in cams.items():
        pos = cam["T_base_cam"][:3, 3] * 1000
        print(f"[{name}] camera in base: x={pos[0]:.0f} y={pos[1]:.0f} z={pos[2]:.0f} mm "
              f"(compare with a tape measurement of the lens)")
    stereo_consistency(calib, cams)

    caps = {}
    if not args.left_image:
        focus = calib.get("focus_absolute", 0)
        caps = {name: cal.open_camera(dev, focus) for name, dev in cal.CAMS.items()}
    csv_path = os.path.join(args.data, "extrinsic_check.csv")

    try:
        if args.headless:
            # Live cameras need a few frames before exposure settles.
            for _ in range(1 if args.left_image else 10):
                frames = get_frames(args, caps)
            meas = {n: board_in_base(board, detector, frames[n], cams[n])[0] for n in cal.CAMS}
            board_report(meas, truth)
            return

        clicks = []

        def on_mouse(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            name = "left" if x < cal.WIDTH else "right"
            u = x - (0 if name == "left" else cal.WIDTH)
            pt = pixel_to_floor(u, y, cams[name])
            if pt is None:
                print(f"[{name}] clicked above the horizon, no floor there")
                return
            print(f"[{name}] floor point x={pt[0] * 1000:.0f} mm y={pt[1] * 1000:.0f} mm")
            clicks[:] = [(name, pt)]

        window = "extrinsic check"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)
        print("click the floor to read base x/y; SPACE/s or ENTER logs the board check; ESC/q quits")
        while True:
            frames = get_frames(args, caps)
            views, meas = {}, {}
            for name in cal.CAMS:
                m, det = board_in_base(board, detector, frames[name], cams[name])
                meas[name] = m
                vis = cal.draw(frames[name], det, name)
                draw_floor_grid(vis, cams[name])
                if m is not None:
                    cv2.drawFrameAxes(vis, cams[name]["K"], cams[name]["dist"], m["rvec"], m["tvec"],
                                      2 * args.square_mm / 1000)
                    o = m["origin_mm"]
                    lines = (f"x={o[0]:.0f} y={o[1]:.0f} z={o[2]:.0f} yaw={m['normal_yaw_deg']:+.1f}",
                             f"lean={m['lean_deg']:.2f}")
                    for i, text in enumerate(lines):
                        cv2.putText(vis, text, (8, 50 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                views[name] = vis
            for name, pt in clicks:
                for eye in cal.CAMS:
                    uv = project_base_points(pt.reshape(1, 3), cams[eye])[0]
                    if np.isnan(uv).any():
                        continue
                    c = (int(uv[0]), int(uv[1]))
                    cv2.drawMarker(views[eye], c, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
                    cv2.putText(views[eye], f"({pt[0]:.2f},{pt[1]:.2f})m", (c[0] + 8, c[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if all(meas[n] is not None for n in cal.CAMS):
                d = meas["left"]["center_mm"] - meas["right"]["center_mm"]
                agree = np.linalg.norm(d)
                cv2.putText(views["left"], f"eyes differ {agree:.1f}mm", (8, cal.HEIGHT - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow(window, np.hstack([views[n] for n in cal.CAMS]))
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (32, ord("s"), 13) or cal.stdin_enter_pressed():
                append_csv(csv_path, board_report(meas, truth))
    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
