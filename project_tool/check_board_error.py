"""Live error of the current head calibration on the ChArUco board in view."""
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_head_cameras as cal

DATA = os.path.expanduser("~/workspaces/FXLerobot/FXlerobot/calib/head")
OUT = "/tmp/board_error"


def skew(t):
    x, y, z = np.asarray(t, float).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)


def pose_of(board, det, K, dist):
    corners, ids = det[0], det[1]
    if ids is None or len(ids) < cal.MIN_CORNERS:
        return None
    obj, imgp = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))
    R, _ = cv2.Rodrigues(rvec)
    return {"n": int(len(ids)), "reproj": err, "R": R, "t": tvec.reshape(3), "map": cal.corner_map(det)}


def epi_px(left_map, right_map, Kl, dl, Kr, dr, R, T):
    common = sorted(set(left_map) & set(right_map))
    if len(common) < 4:
        return common, []
    pl = np.array([left_map[i] for i in common], np.float32).reshape(-1, 1, 2)
    pr = np.array([right_map[i] for i in common], np.float32).reshape(-1, 1, 2)
    nl = cv2.undistortPoints(pl, Kl, dl).reshape(-1, 2)
    nr = cv2.undistortPoints(pr, Kr, dr).reshape(-1, 2)
    E = skew(T) @ R
    focal = float((Kr[0, 0] + Kr[1, 1]) * 0.5)
    errs = []
    for a, b in zip(nl, nr):
        line = E @ np.array([a[0], a[1], 1.0])
        dist = abs(float(np.dot(np.array([b[0], b[1], 1.0]), line)))
        errs.append(dist / max(float(np.hypot(line[0], line[1])), 1e-9) * focal)
    return common, errs


def rot_deg(Ra, Rb):
    R = Ra @ Rb.T
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(DATA, "head_cameras.yaml")) as f:
        calib = yaml.safe_load(f)
    # The board in front of the robot is 7x5 squares. make_board() still says 5x7, which finds no corners.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    board = cv2.aruco.CharucoBoard((7, 5), 0.040, 0.030, dictionary)
    params = cv2.aruco.CharucoParameters()
    params.minMarkers = 1
    detector = cv2.aruco.CharucoDetector(board, params)
    K = {n: np.array(calib[n]["K"]) for n in cal.CAMS}
    dist = {n: np.array(calib[n]["dist"]) for n in cal.CAMS}
    R_cal = np.array(calib["stereo"]["R_right_left"])
    T_cal = np.array(calib["stereo"]["T_right_left_m"], float).reshape(3)
    print(f"stored left rms {calib['left']['rms']:.3f}px  right {calib['right']['rms']:.3f}px  "
          f"stereo {calib['stereo']['rms']:.3f}px  baseline {calib['stereo']['baseline_mm']:.1f}mm")

    caps = {n: cal.open_camera(dev, calib.get("focus_absolute", 0)) for n, dev in cal.CAMS.items()}
    try:
        for _ in range(8):
            cal.grab_pair(caps)
        samples = []
        for i in range(6):
            frames = cal.grab_pair(caps)
            dets = {n: cal.detect(detector, frames[n]) for n in cal.CAMS}
            poses = {n: pose_of(board, dets[n], K[n], dist[n]) for n in cal.CAMS}
            samples.append((frames, dets, poses))
    finally:
        for cap in caps.values():
            cap.release()

    best = max(samples, key=lambda s: 0 if s[2]["left"] is None or s[2]["right"] is None
               else s[2]["left"]["n"] + s[2]["right"]["n"])
    frames, dets, poses = best
    for name in cal.CAMS:
        cv2.imwrite(os.path.join(OUT, f"{name}.jpg"), cal.draw(frames[name], dets[name], name))
        pose = poses[name]
        if pose is None:
            n = 0 if dets[name][1] is None else len(dets[name][1])
            print(f"[{name}] corners={n}  board not usable")
            continue
        print(f"[{name}] corners={pose['n']}  reproj={pose['reproj']:.2f}px  "
              f"board {np.linalg.norm(pose['t']) * 1000:.0f} mm from camera")

    if poses["left"] is None or poses["right"] is None:
        print("stereo not measured: one eye does not see enough corners")
        return

    R_meas = poses["right"]["R"] @ poses["left"]["R"].T
    T_meas = poses["right"]["t"] - R_meas @ poses["left"]["t"]
    common, errs = epi_px(poses["left"]["map"], poses["right"]["map"],
                          K["left"], dist["left"], K["right"], dist["right"], R_cal, T_cal)
    base_meas = float(np.linalg.norm(T_meas) * 1000)
    base_cal = float(np.linalg.norm(T_cal) * 1000)
    t_dir = float(np.degrees(np.arccos(np.clip(
        np.dot(T_meas, T_cal) / (np.linalg.norm(T_meas) * np.linalg.norm(T_cal) + 1e-12), -1, 1))))
    print(f"[stereo] shared corners={len(common)}")
    if errs:
        print(f"[stereo] epipolar error median={np.median(errs):.2f}px  "
              f"p90={np.percentile(errs, 90):.2f}px  max={np.max(errs):.2f}px")
    print(f"[stereo] rotation off the stored calibration by {rot_deg(R_meas, R_cal):.2f} deg")
    print(f"[stereo] baseline now {base_meas:.1f} mm, stored {base_cal:.1f} mm, "
          f"direction off {t_dir:.1f} deg")


if __name__ == "__main__":
    main()
