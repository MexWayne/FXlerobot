#!/usr/bin/env python
"""Rescale the head stereo baseline so triangulated board squares match the real square size.

    python stereo_rescale.py                 # uses calib/board_grasp.json from board_view.py

K, dist and R_right_left do not depend on the board size; only T_right_left does. The old
calib/head/head_cameras.yaml was calibrated with a board whose square size was off, so stereo
distances came out ~5% short. The result goes to llm_vla/calib/head_cameras.yaml and the original
file under calib/head is left alone (vln uses it).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "calib/head/head_cameras.yaml"
DST = HERE / "calib/head_cameras.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=str(HERE / "calib/board_grasp.json"))
    args = ap.parse_args()

    board = json.loads(Path(args.board).read_text())
    if Path(board.get("cameras", "")).resolve() != SRC.resolve():
        raise SystemExit(f"{args.board} 不是用 {SRC} 测的，先删掉 {DST} 再跑一次 board_view.py")
    measured, real = board["stereo_square_mm"], board["square_mm"]
    calib = yaml.safe_load(SRC.read_text())
    scale = real / measured
    T = np.array(calib["stereo"]["T_right_left_m"]) * scale
    calib["stereo"]["T_right_left_m"] = T.tolist()
    calib["stereo"]["baseline_mm"] = float(np.linalg.norm(T) * 1000)
    calib["stereo_scale"] = {"factor": scale, "from": str(SRC), "board": args.board,
                             "board_square_mm": real, "triangulated_square_mm": measured}
    calib["board"] = {"squares": [7, 5], "dict": "DICT_5X5_50", "square_mm": real,
                      "marker_mm": board.get("marker_mm", 30.0)}
    DST.parent.mkdir(exist_ok=True)
    DST.write_text(yaml.safe_dump(calib, sort_keys=False))
    print(f"缩放系数 {scale:.4f}，双目间距 {calib['stereo']['baseline_mm']:.1f} mm")
    print(DST)


if __name__ == "__main__":
    main()
