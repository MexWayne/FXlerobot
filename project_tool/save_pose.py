#!/usr/bin/env python
"""Save the two follower arms under a name, or move them back to that pose.

    python save_pose.py save home
    python save_pose.py restore home

Left is /dev/arm_B_follower, right is /dev/arm_A_follower. Leader arms are not
opened. Both commands print each servo's angle. save only reads. restore turns
torque on and leaves it on; release_arms.py lets go.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import scservo_sdk as scs

ARMS = {
    "left": "/dev/arm_B_follower",
    "right": "/dev/arm_A_follower",
}
CALIBRATION = {
    "left": Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so_follower/bimanual_follower_left.json",
    "right": Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so_follower/bimanual_follower_right.json",
}
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
POSE_DIR = Path(__file__).resolve().parent / "poses"
BAUDRATE = 1_000_000
FULL_TURN = 4095
REG = {
    "Operating_Mode": (33, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Velocity": (46, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
}
POSITION_MODE = 0
RETRY = 3
ARRIVE_TICKS = 30
CONTROL_HZ = 50
SPEED_DEG = 25.0


def port_holders(port):
    try:
        out = subprocess.run(["fuser", os.path.realpath(port)], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    holders = []
    for pid in out.split():
        pid = pid.strip().rstrip("emc")
        if pid.isdigit() and int(pid) != os.getpid():
            try:
                cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()
            except OSError:
                cmd = "?"
            holders.append((int(pid), cmd))
    return holders


class Servos:
    def __init__(self, port):
        self.port = scs.PortHandler(port)
        self.pk = scs.PacketHandler(0)
        if not self.port.openPort() or not self.port.setBaudRate(BAUDRATE):
            raise OSError(f"打不开串口 {port}")

    def close(self):
        self.port.closePort()

    def read(self, sid, name):
        addr, n = REG[name]
        fn = self.pk.read1ByteTxRx if n == 1 else self.pk.read2ByteTxRx
        for _ in range(RETRY):
            val, comm, _ = fn(self.port, sid, addr)
            if comm == scs.COMM_SUCCESS:
                return val
        return None

    def write(self, sid, name, value):
        addr, n = REG[name]
        fn = self.pk.write1ByteTxRx if n == 1 else self.pk.write2ByteTxRx
        for _ in range(RETRY):
            comm, _ = fn(self.port, sid, addr, int(value))
            if comm == scs.COMM_SUCCESS:
                return True
        return False


def load_calibration(side):
    path = CALIBRATION[side]
    if not path.is_file():
        raise RuntimeError(f"没有标定文件 {path}")
    return json.loads(path.read_text())


def degrees(ticks, cal):
    mid = (cal["range_min"] + cal["range_max"]) / 2
    return (ticks - mid) * 360.0 / FULL_TURN


def print_angles(arms):
    for side, port in ARMS.items():
        calibration = load_calibration(side)
        print(f"{side}  {port}")
        for name in MOTORS:
            angle = degrees(arms[side][name], calibration[name])
            print(f"  {name:13s} {angle:8.2f}°")


def open_arms():
    buses = {}
    try:
        for side, port in ARMS.items():
            if not os.path.exists(port):
                raise RuntimeError(f"{port} 不存在，先看 ls -l /dev/arm_*")
            holders = port_holders(port)
            if holders:
                lines = "\n".join(f"    PID {pid}: {cmd[:120]}" for pid, cmd in holders)
                raise RuntimeError(f"{port} 被其他进程占用，先停掉它:\n{lines}")
            buses[side] = Servos(port)
        return buses
    except Exception:
        for bus in buses.values():
            bus.close()
        raise


def read_positions(buses):
    arms = {}
    for side, bus in buses.items():
        arms[side] = {}
        for sid, name in enumerate(MOTORS, start=1):
            ticks = bus.read(sid, "Present_Position")
            if ticks is None:
                raise RuntimeError(f"{ARMS[side]} 的 {name} (ID {sid}) 没有读到位置")
            arms[side][name] = int(ticks)
    return arms


def pose_path(pose_name):
    if not pose_name or pose_name in (".", "..") or any(c in pose_name for c in "/\\"):
        raise SystemExit("姿势名不能为空，也不能包含 / 或 \\")
    return POSE_DIR / f"{pose_name}.json"


def save_pose(pose_name):
    buses = open_arms()
    try:
        arms = read_positions(buses)
    finally:
        for bus in buses.values():
            bus.close()
    path = pose_path(pose_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": pose_name,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "arms": arms,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(pose_name)
    print_angles(arms)


def _step_towards(commanded, target, max_step):
    delta = target - commanded
    if abs(delta) <= max_step:
        return target
    return commanded + max_step * (1 if delta > 0 else -1)


def restore_pose(pose_name):
    path = pose_path(pose_name)
    if not path.is_file():
        raise SystemExit(f"没有这个姿势: {pose_name}")
    saved = json.loads(path.read_text())["arms"]
    for side in ARMS:
        for name in MOTORS:
            ticks = int(saved[side][name])
            if not 0 <= ticks <= FULL_TURN:
                raise SystemExit(f"{side} {name} 的刻度 {ticks} 超出 0..{FULL_TURN}")
            saved[side][name] = ticks

    buses = open_arms()
    try:
        present = {}
        for side, bus in buses.items():
            present[side] = {}
            for sid, name in enumerate(MOTORS, start=1):
                mode = bus.read(sid, "Operating_Mode")
                if mode != POSITION_MODE:
                    raise RuntimeError(f"{ARMS[side]} {name} 不在位置模式，没有改力矩")
                ticks = bus.read(sid, "Present_Position")
                if ticks is None:
                    raise RuntimeError(f"{ARMS[side]} {name} 没有读到当前位置，没有改力矩")
                present[side][name] = int(ticks)

        speed_ticks = SPEED_DEG * FULL_TURN / 360.0
        max_step = max(1, int(round(speed_ticks / CONTROL_HZ)))
        goal_velocity = max(1, min(3400, int(round(speed_ticks * 1.3))))
        farthest = max(
            abs(saved[side][name] - present[side][name]) for side in ARMS for name in MOTORS
        )
        deadline = time.monotonic() + farthest / speed_ticks * 1.8 + 2.0

        for side, bus in buses.items():
            for sid, name in enumerate(MOTORS, start=1):
                if not bus.write(sid, "Goal_Position", present[side][name]):
                    raise RuntimeError(f"{ARMS[side]} {name} 写当前位置失败，没有打开力矩")
                bus.write(sid, "Acceleration", 40)
                bus.write(sid, "Goal_Velocity", goal_velocity)
                if not bus.write(sid, "Torque_Enable", 1) or not bus.write(sid, "Lock", 1):
                    raise RuntimeError(f"{ARMS[side]} {name} 打不开力矩")

        commanded = {side: dict(positions) for side, positions in present.items()}
        period = 1.0 / CONTROL_HZ
        try:
            while True:
                arrived = True
                started = time.monotonic()
                for side, bus in buses.items():
                    for sid, name in enumerate(MOTORS, start=1):
                        nxt = _step_towards(commanded[side][name], saved[side][name], max_step)
                        commanded[side][name] = nxt
                        if nxt != saved[side][name]:
                            arrived = False
                        if not bus.write(sid, "Goal_Position", nxt):
                            raise RuntimeError(f"{ARMS[side]} {name} 写目标失败，力矩仍开着")
                if arrived or time.monotonic() >= deadline:
                    break
                time.sleep(max(0.0, period - (time.monotonic() - started)))
        except KeyboardInterrupt:
            for side, bus in buses.items():
                for sid, name in enumerate(MOTORS, start=1):
                    here = bus.read(sid, "Present_Position")
                    if here is not None:
                        bus.write(sid, "Goal_Position", here)
            print()
            print(pose_name)
            print_angles(read_positions(buses))
            raise SystemExit(130)

        settle_until = time.monotonic() + 2.0
        while time.monotonic() < settle_until:
            now = read_positions(buses)
            pending = any(
                abs(now[side][name] - saved[side][name]) > ARRIVE_TICKS
                for side in ARMS
                for name in MOTORS
            )
            if not pending:
                break
            time.sleep(0.05)
        print(pose_name)
        print_angles(read_positions(buses))
    finally:
        for bus in buses.values():
            bus.close()


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("save", "restore"):
        raise SystemExit("用法: python save_pose.py save 名字 | restore 名字")
    if sys.argv[1] == "save":
        save_pose(sys.argv[2])
    else:
        restore_pose(sys.argv[2])


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        sys.exit(str(exc))
