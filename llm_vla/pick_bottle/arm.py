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
from save_pose import ARMS, FULL_TURN, MOTORS, POSITION_MODE, Servos, degrees, load_calibration  # noqa: E402

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
SQUEEZE_TICKS = 25


def joint_limits(cal=None):
    """joint -> (lo, hi) LeRobot degrees, LIMIT_MARGIN_DEG inside the calibrated range."""
    cal = cal or load_calibration("left")
    return {n: (degrees(cal[n]["range_min"], cal[n]) + LIMIT_MARGIN_DEG,
                degrees(cal[n]["range_max"], cal[n]) - LIMIT_MARGIN_DEG) for n in MOTORS}


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
        if None in out.values():
            raise RuntimeError(f"左臂没有读全: {out}")
        return out

    def deg(self):
        return {n: degrees(t, self.cal[n]) for n, t in self.ticks().items()}

    def enable(self):
        if self.torque:
            return
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

    def relax(self):
        for sid in self.ids.values():
            self.bus.write(sid, "Torque_Enable", 0)
        self.torque = False
