"""The left SO-101 follower (/dev/arm_B_follower) and nothing else: read joints, slow joint moves,
a gripper close that stops when it meets the object.

The right arm's port is never opened. The first move turns torque on and it stays on, so the arm
holds where it stopped; relax() turns it off and the arm falls under its own weight.
Angles are LeRobot degrees, (ticks - range mid) * 360 / 4095, as everywhere else in llm_vla.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE.parent), str(ROOT / "project_tool")]
from save_pose import ARMS, FULL_TURN, MOTORS, POSITION_MODE, Servos, degrees, load_calibration, load_range  # noqa: E402

SPEED_DEG = 50.0  # peak joint speed of a move through free space, deg/s
GRIP_SPEED_DEG = 30.0
CONTROL_HZ = 50
ARRIVE_TICKS = 15
SETTLE_TICKS = 2  # a joint that moved less than this over SETTLE_S has stopped (e.g. held by a load)
SETTLE_S = 0.1
SETTLE_MAX_S = 1.0
LIMIT_MARGIN_DEG = 3.0  # stay this far inside the calibrated range
# Closing: the jaw is blocked when it lags the command by this much for this many cycles; it is
# then told to squeeze only this far past where it stopped.
STALL_LAG_TICKS = 50
STALL_CYCLES = 8
SQUEEZE_TICKS = 100  # ~9 deg: the servo pushes with what its torque limit allows (25 held bottles loosely)
# The gripper's strength, as LeRobot leaves it (so_follower.py) and the right arm's gripper has it: 50 %
# torque, current protection 250, Overload_Torque 25 (user, 2026-11-07: it was too strong). From
# 2026-11-04 it had 1000 / 500 / 80, after a cap held for Protection_Time (2 s) counted as an overload,
# the jaw dropped to 20 % torque and the bottle slipped out on the way; the jaws have rubber strips
# since 2026-11-07. Written on torque-on where the EEPROM differs.
GRIP_EEPROM = {"Max_Torque_Limit": (16, 2, 500), "Protection_Current": (28, 2, 250), "Overload_Torque": (36, 1, 25)}
GRIP_TORQUE_LIMIT = (48, 500)  # RAM Torque_Limit, loaded from Max_Torque_Limit only at power-up
# Squeezed SQUEEZE_TICKS past the cap the jaw draws ~325 mA (36 % load) for as long as it holds: carried
# 13 min to the can it reached its 70 C limit, unloaded and did not open over the can (2026-11-04).
# Carrying it is eased to HOLD_TICKS past where it stopped: ~70 mA, 16 % load. Held at 60 ticks (~130 mA)
# it still warmed 55 -> 60 C in 12 min; at 30 it cooled 60 -> 54 C in 7 min, and the cap stayed put.
HOLD_TICKS = 40
TEMP_ADDR, STATUS_ADDR = 63, 65  # Present_Temperature (C), Status (bit 2: too hot, torque unloaded)


def joint_limits(cal=None):
    """joint -> (lo, hi) LeRobot degrees, LIMIT_MARGIN_DEG inside the range the arm may use
    (save_pose.load_range: the calibrated one, widened where the body no longer stops it, e.g. the
    shoulder turning out to -117 deg once the side tubes were sawn off, 2026-11-10)."""
    cal = cal or load_calibration("left")
    usable = load_range("left")
    return {n: (degrees(usable[n][0], cal[n]) + LIMIT_MARGIN_DEG,
                degrees(usable[n][1], cal[n]) - LIMIT_MARGIN_DEG) for n in MOTORS}


READ_TRIES = 3  # a position read is tried this often before the arm counts as not answering


class LeftArm:
    def __init__(self):
        self.cal = load_calibration("left")
        self.limits = joint_limits(self.cal)
        self.bus = Servos(ARMS["left"])
        self.ids = {name: i for i, name in enumerate(MOTORS, start=1)}
        self.torque = False

    def close(self):
        self.bus.close()

    def _mid(self, name):
        return (self.cal[name]["range_min"] + self.cal[name]["range_max"]) / 2

    def to_ticks(self, name, deg):
        lo, hi = self.limits[name]
        if not lo <= deg <= hi:
            raise ValueError(f"{name} {deg:.1f}° 超出范围 {lo:.1f}..{hi:.1f}°")
        return int(round(self._mid(name) + deg * FULL_TURN / 360.0))

    def ticks(self):
        out = {n: self.bus.read(i, "Present_Position") for n, i in self.ids.items()}
        # A servo that missed one reply is asked again: the wrist and gripper each missed one while the
        # arm moved and the run ended with the arm over a scrap (2026-11-07, twice in 10 min).
        for _ in range(READ_TRIES - 1):
            if None not in out.values():
                break
            time.sleep(0.02)
            out.update({n: self.bus.read(self.ids[n], "Present_Position") for n, v in out.items() if v is None})
        if None in out.values():
            raise RuntimeError(f"左臂没有读全: {out}")
        return out

    def deg(self):
        return {n: degrees(t, self.cal[n]) for n, t in self.ticks().items()}

    def _raw(self, sid, addr, n, value=None):
        """Read (value None) or write a register by address."""
        pk, port = self.bus.pk, self.bus.port
        if value is None:
            v, comm, _ = (pk.read1ByteTxRx if n == 1 else pk.read2ByteTxRx)(port, sid, addr)
            return v if comm == 0 else None
        comm, _ = (pk.write1ByteTxRx if n == 1 else pk.write2ByteTxRx)(port, sid, addr, int(value))
        return comm == 0

    def strong_gripper(self):
        """The gripper's EEPROM protection to GRIP_EEPROM where it differs (torque off meanwhile, as
        LeRobot does), its torque limit to GRIP_TORQUE_LIMIT. Returns the names changed."""
        sid = self.ids["gripper"]
        todo = {k: v for k, v in GRIP_EEPROM.items() if self._raw(sid, v[0], v[1]) != v[2]}
        if todo:
            self.bus.write(sid, "Torque_Enable", 0)
            self.bus.write(sid, "Lock", 0)
            try:
                for name, (addr, n, value) in todo.items():
                    # The status reply of an EEPROM write gets lost now and then: read it back instead.
                    for _ in range(3):
                        self._raw(sid, addr, n, value)
                        time.sleep(0.02)
                        if self._raw(sid, addr, n) == value:
                            break
                    else:
                        raise RuntimeError(f"左臂夹爪写 {name} 失败")
            finally:
                self.bus.write(sid, "Lock", 1)
        return list(todo)

    def enable(self):
        if self.torque:
            return
        changed = self.strong_gripper()
        if changed:
            print(f"  夹爪力矩保护改成和其他关节一样：{', '.join(changed)}", flush=True)
        present = self.ticks()
        for name, sid in self.ids.items():
            if self.bus.read(sid, "Operating_Mode") != POSITION_MODE:
                raise RuntimeError(f"左臂 {name} 不在位置模式，没有改力矩")
        for name, sid in self.ids.items():
            if not self.bus.write(sid, "Goal_Position", present[name]):
                raise RuntimeError(f"左臂 {name} 写当前位置失败，没有打开力矩")
            self.bus.write(sid, "Acceleration", 40)
            if not self.bus.write(sid, "Torque_Enable", 1):
                raise RuntimeError(f"左臂 {name} 打不开力矩")
        self._raw(self.ids["gripper"], GRIP_TORQUE_LIMIT[0], 2, GRIP_TORQUE_LIMIT[1])
        self.torque = True

    def hold(self):
        """Stop where the arm is now."""
        present = self.ticks()
        for name, sid in self.ids.items():
            self.bus.write(sid, "Goal_Position", present[name])

    def move(self, target_deg, speed_deg=SPEED_DEG):
        """Move the named joints to target_deg (dict) together, the others hold.

        All joints start and stop at the same time on a cosine speed profile peaking at speed_deg,
        so the tool runs close to a straight line in joint space without a jolt at either end.
        Ctrl-C stops and holds."""
        goal = {n: self.to_ticks(n, d) for n, d in target_deg.items()}
        self.enable()
        start = self.ticks()
        span = max(abs(goal[n] - start[n]) for n in goal) * 360.0 / FULL_TURN
        duration = span * np.pi / 2 / speed_deg
        try:
            t0 = time.monotonic()
            while True:
                tau = min(1.0, (time.monotonic() - t0) / duration) if duration > 0 else 1.0
                s = 0.5 - 0.5 * np.cos(np.pi * tau)
                for n in goal:
                    self.bus.write(self.ids[n], "Goal_Position", int(round(start[n] + (goal[n] - start[n]) * s)))
                if tau >= 1.0:
                    break
                time.sleep(1.0 / CONTROL_HZ)
            self._settle(goal)
        except KeyboardInterrupt:
            self.hold()
            raise
        return self.deg()

    def move_path(self, path, speed_deg=SPEED_DEG):
        """Through the joint targets in `path` (dicts of the same joints) without stopping at the ones
        between: straight in joint space from each to the next, one cosine speed profile over the whole
        way peaking at speed_deg. Moved one move() per waypoint, the arm stopped at every one and the
        drop over the can went in jerks (user, 2026-11-07). Ctrl-C stops and holds."""
        names = list(path[0])
        self.enable()
        start = self.ticks()
        pts = [np.array([start[n] for n in names], float)]
        pts += [np.array([self.to_ticks(n, q[n]) for n in names], float) for q in path]
        seg = [float(np.max(np.abs(b - a))) for a, b in zip(pts, pts[1:])]
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        duration = cum[-1] * 360.0 / FULL_TURN * np.pi / 2 / speed_deg
        try:
            t0 = time.monotonic()
            while True:
                tau = min(1.0, (time.monotonic() - t0) / duration) if duration > 0 else 1.0
                s = (0.5 - 0.5 * np.cos(np.pi * tau)) * cum[-1]
                k = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
                p = pts[k] + (pts[k + 1] - pts[k]) * ((s - cum[k]) / seg[k] if seg[k] > 0 else 1.0)
                for n, v in zip(names, p):
                    self.bus.write(self.ids[n], "Goal_Position", int(round(v)))
                if tau >= 1.0:
                    break
                time.sleep(1.0 / CONTROL_HZ)
            self._settle({n: int(round(v)) for n, v in zip(names, pts[-1])})
        except KeyboardInterrupt:
            self.hold()
            raise
        return self.deg()

    def _settle(self, goal):
        """Wait until every joint is at its goal or has stopped short of it."""
        deadline = time.monotonic() + SETTLE_MAX_S
        last = self.ticks()
        while time.monotonic() < deadline:
            time.sleep(SETTLE_S)
            now = self.ticks()
            if all(abs(now[n] - goal[n]) <= ARRIVE_TICKS or abs(now[n] - last[n]) < SETTLE_TICKS for n in goal):
                return
            last = now

    def grip(self, closed_deg):
        """Close the gripper towards closed_deg until it is blocked by the object; returns its angle."""
        self.enable()
        sid = self.ids["gripper"]
        goal = self.to_ticks("gripper", closed_deg)
        commanded = self.ticks()["gripper"]
        sign = 1 if goal > commanded else -1
        step = max(1, int(round(GRIP_SPEED_DEG * FULL_TURN / 360.0 / CONTROL_HZ)))
        lagging = 0
        try:
            while commanded != goal:
                commanded += sign * min(step, abs(goal - commanded))
                self.bus.write(sid, "Goal_Position", commanded)
                time.sleep(1.0 / CONTROL_HZ)
                present = self.bus.read(sid, "Present_Position")
                if present is not None and (commanded - present) * sign > STALL_LAG_TICKS:
                    lagging += 1
                    if lagging >= STALL_CYCLES:
                        self.bus.write(sid, "Goal_Position", present + sign * SQUEEZE_TICKS)
                        break
                else:
                    lagging = 0
        except KeyboardInterrupt:
            self.hold()
            raise
        time.sleep(0.3)
        return self.deg()["gripper"]

    def ease_grip(self, closed_deg):
        """Holding something: squeeze only HOLD_TICKS past where the jaw stands (towards closed_deg)
        instead of SQUEEZE_TICKS, so the gripper does not overheat on a long carry."""
        sid = self.ids["gripper"]
        present = self.bus.read(sid, "Present_Position")
        sign = 1 if self.to_ticks("gripper", closed_deg) > present else -1
        self.bus.write(sid, "Goal_Position", present + sign * HOLD_TICKS)

    def gripper_heat(self):
        """(temperature C, overheated) of the gripper servo."""
        sid = self.ids["gripper"]
        status = self._raw(sid, STATUS_ADDR, 1) or 0
        return self._raw(sid, TEMP_ADDR, 1), bool(status & 4)

    def relax(self):
        for sid in self.ids.values():
            self.bus.write(sid, "Torque_Enable", 0)
        self.torque = False
