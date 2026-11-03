#!/usr/bin/env python
"""Move the neck to a yaw/pitch (degrees) and save one frame from each head eye.

    python neck_look.py 0 20            # yaw 0, pitch 20, save look/y0_p20_*.jpg
    python neck_look.py 0 20 --no-shot  # move only

Only neck IDs 10/11 on /dev/base_neck are written; wheel IDs 7/8/9 are never
touched. Torque stays on afterwards so the neck holds the pose.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "project_tool"))
from save_pose import (  # noqa: E402
    FULL_TURN,
    NECK_MOTORS,
    POSITION_MODE,
    load_neck_calibration,
    open_neck,
    read_neck,
)

LOOK_DIR = Path(__file__).resolve().parent / "look"
SPEED_DEG = 20.0
CONTROL_HZ = 50
ARRIVE_TICKS = 15
APPROACH_DEG = 3.0
EYES = {"left": "/dev/cam_front_left", "right": "/dev/cam_front_right"}


def deg_to_ticks(deg, cal):
    mid = (cal["range_min"] + cal["range_max"]) / 2
    return int(round(mid + deg * FULL_TURN / 360.0))


def ticks_to_deg(ticks, cal):
    mid = (cal["range_min"] + cal["range_max"]) / 2
    return (ticks - mid) * 360.0 / FULL_TURN


def ramp(bus, commanded, target):
    max_step = max(1, int(round(SPEED_DEG * FULL_TURN / 360.0 / CONTROL_HZ)))
    while commanded != target:
        for sid, name in NECK_MOTORS.items():
            delta = target[name] - commanded[name]
            commanded[name] += max(-max_step, min(max_step, delta))
            bus.write(sid, "Goal_Position", commanded[name])
        time.sleep(1.0 / CONTROL_HZ)
    return commanded


def move_neck(yaw, pitch):
    cal = load_neck_calibration()
    approach = int(round(APPROACH_DEG * FULL_TURN / 360.0))
    target = {"neck_yaw": deg_to_ticks(yaw, cal["neck_yaw"]),
              "neck_pitch": deg_to_ticks(pitch, cal["neck_pitch"])}
    for name, ticks in target.items():
        lo, hi = cal[name]["range_min"], cal[name]["range_max"]
        if not lo <= ticks <= hi:
            raise SystemExit(f"{name} 目标刻度 {ticks} 超出标定范围 {lo}..{hi}")

    bus = open_neck()
    try:
        present = read_neck(bus)
        for sid, name in NECK_MOTORS.items():
            if bus.read(sid, "Operating_Mode") != POSITION_MODE:
                raise RuntimeError(f"{name} 不在位置模式，没有改力矩")
            bus.write(sid, "Goal_Position", present[name])
            if not bus.write(sid, "Torque_Enable", 1):
                raise RuntimeError(f"{name} 打不开力矩")

        # The neck has ~2 deg of backlash: always finish the move from the same side
        # (from lower yaw and higher pitch) so a given command lands on the same pose.
        before = {"neck_yaw": target["neck_yaw"] - approach,
                  "neck_pitch": target["neck_pitch"] + approach}
        commanded = ramp(bus, dict(present), before)
        ramp(bus, commanded, target)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            now = read_neck(bus)
            if all(abs(now[n] - target[n]) <= ARRIVE_TICKS for n in target):
                break
            time.sleep(0.05)
        now = read_neck(bus)
    finally:
        bus.close()
    return {n: ticks_to_deg(now[n], cal[n]) for n in now}


def shoot(tag):
    LOOK_DIR.mkdir(exist_ok=True)
    paths = []
    for eye, dev in EYES.items():
        subprocess.run(["v4l2-ctl", "-d", dev, "-c", "focus_automatic_continuous=0",
                        "-c", "focus_absolute=0"], capture_output=True)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame = None
        for _ in range(15):
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            print(f"{eye} 眼没有拿到画面")
            continue
        path = LOOK_DIR / f"{tag}_{eye}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaw", type=float)
    ap.add_argument("pitch", type=float)
    ap.add_argument("--no-shot", action="store_true")
    args = ap.parse_args()

    now = move_neck(args.yaw, args.pitch)
    print(f"neck yaw={now['neck_yaw']:.2f}° pitch={now['neck_pitch']:.2f}°")
    if not args.no_shot:
        for p in shoot(f"y{args.yaw:g}_p{args.pitch:g}"):
            print(p)


if __name__ == "__main__":
    main()
