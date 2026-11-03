"""Board pose in each camera from one saved pair. No robot-base measurement."""
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_head_cameras as cal

DATA = os.path.expanduser("~/workspaces/FXLerobot/FXlerobot/calib/head")
PAIR = os.path.expanduser("~/workspaces/FXLerobot/FXlerobot/calib/head_20261027")


def main():
    with open(os.path.join(DATA, "head_cameras.yaml")) as f:
        calib = yaml.safe_load(f)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    board = cv2.aruco.CharucoBoard((7, 5), 0.040, 0.030, dictionary)
    params = cv2.aruco.CharucoParameters()
    params.minMarkers = 1
    detector = cv2.aruco.CharucoDetector(board, params)
    print("board: 7x5 squares, black square 40 mm, marker 30 mm")
    print("camera frame: x right, y down, z forward, origin at the optical center")
    print("board frame: origin at the checker corner, x/y in the board plane, z out of the front")
    for name in ("left", "right"):
        img = cv2.imread(os.path.join(PAIR, f"{name}_000.png"))
        if img is None:
            print(f"[{name}] no image")
            continue
        corners, ids, mc, mi = detector.detectBoard(img)
        n = 0 if ids is None else len(ids)
        if ids is None or n < cal.MIN_CORNERS:
            print(f"[{name}] corners={n}")
            continue
        K = np.array(calib[name]["K"])
        dist = np.array(calib[name]["dist"])
        obj, imgp = board.matchImagePoints(corners, ids)
        ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            print(f"[{name}] solvePnP failed")
            continue
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))
        x_dir = cal.upright_axis_dir(R[:, 0])
        y_dir = cal.upright_axis_dir(R[:, 1])
        print(f"[{name}] corners={n} reproj={err:.2f}px")
        print(f"  board origin in this camera: x={t[0]*1000:.0f} mm right, "
              f"y={t[1]*1000:.0f} mm down, z={t[2]*1000:.0f} mm forward, "
              f"distance={np.linalg.norm(t)*1000:.0f} mm")
        print(f"  board red x points {x_dir}, green y points {y_dir}")
        print(f"  measure the base from the {cal.upright_origin_corner(x_dir, y_dir)} corner")


if __name__ == "__main__":
    main()
