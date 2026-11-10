#!/usr/bin/env python3
"""雷达建图、深度优先探索、清扫。

  conda activate lerobot_env
  cd ~/workspaces/FXLerobot/FXlerobot/vln
  python coverage_run.py      # 把车放进场地里，直接开始

状态页：浏览器打开 http://<jetson>:8765 。Ctrl-C 停车。

1) 建图：原地慢转两圈，只用雷达建占据栅格；再分 6 个方向各停一次看一眼，用视觉认出雷达点云是什么
   （桌子、凳子……），补上雷达看不到的东西（桌面、贴地的电线和垃圾、落地玻璃）。
2) 探索：深度优先，车身要到每一片能到的空地附近（1 m 内），也去已知空地和未知区域的交界。
   每到一个新地方自转一圈：雷达补图，视觉给物体建模。
3) 清扫：按地图让车身扫过所有能到的空地，到 90% 结束。
定位：IMU 管转向；每次停车用雷达点云和已建的地图匹配，校正位置和航向。
行驶：每走 0.35 m 停车看一次：抬头看桌面这类伸在上方的东西，低头看地上的东西、电线、插线板和障碍。
VLM 只认出是什么（桌子就是桌子，柜子就是柜子）；位置和大小由双目和雷达量。每件东西按量到的点拟合
成矩形（长、宽、走向），加上量到的高度，放进地图；要夹起来的小东西单独列出：纸屑、纸团、纸巾车直接压过去（探索要走遍、清扫要扫到），瓶子和别的车绕开。
"""

import argparse
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from coverage_plan import ban_goal, explore_goal, route_to  # noqa: E402
from live_map import MapView  # noqa: E402
from occmap import BODY_CLEARANCE_M, BODY_LENGTH_M, BODY_WIDTH_M, OccMap, disk  # noqa: E402
from vln_scene import (DRIVE_OVER_KEYS, DRIVE_ROUND_KEYS, LANE_PITCHES, PICK_MAX_M, TAG_PICK,  # noqa: E402
                       TAG_KEEP, TAG_PICK_OVER, any_size, is_bin, label_tag, to_pick)

MAP_DIR = os.path.join(HERE, "map")

LIN = 0.08            # m/s, straight moves
ROT = 25.0            # deg/s, fastest turn in place
TURN_MIN_RATE = 8.0   # deg/s, slowest turn command that still moves the wheels
TURN_KP = 1.5         # deg/s of turn rate per degree still to go
TURN_TOL_DEG = 1.0    # a turn ends once the IMU says it is this close and has settled
TURN_SETTLE_S = 0.25
HOLD_KP = 3.0         # deg/s of correction per degree of heading error while driving straight
HOLD_KD = 0.5         # deg/s of correction per deg/s the gyro already shows
HOLD_ABORT_DEG = 60.0  # a heading error this big while driving is a sensor fault, not drift
HOLD_MAX_RATE = 15.0
WIGGLE_RATE = 20.0    # deg/s and seconds of the first turn, only used to find which way the IMU counts
WIGGLE_S = 0.9
SPIN_DPS = 12.0       # deg/s of a mapping turn; slow enough for several lidar turns per degree step
SPIN_SAMPLE_S = 0.25
WARMUP_TURNS = 2
SURVEY_STOPS = 6      # a survey turn stops this often to look (the camera sees about 65° across)
SURVEY_GAP_M = 1.0    # a new survey turn only this far from the last one
SLIP0 = 1.0           # lidar metres per metre the wheels report, before the lidar has measured it
SLIP_MIN_M = 0.06     # moves shorter than this do not update it
# Only a move whose lidar metres per wheel metre lie in this range teaches the slip, and the slip stays
# in it: pushing against a cable and a zigzag out (0.06 m for 0.09-0.16 m of wheels) took it from 0.92
# to 0.56, the next 0.90 m of wheels was taken for 0.50 m, the lidar no longer matched the map within
# its window and the body drove on by dead reckoning to the edge of the room (2026-11-07).
SLIP_RANGE = (0.75, 1.10)
# Straight moves stop by the wheels' own speeds, read back and added up as the body goes, not by time:
# timed (dist / speed) they overshot short legs by 4-13 cm at 0.12 m/s, and the demo turned round and
# round a stage spot 6-14 cm off for minutes (2026-11-06). The wheels roll on a little after the stop
# command; that is measured after every move (COAST_S: metres rolled per m/s at the stop) and the
# next move stops that much earlier.
COAST_S0 = 0.3        # s
COAST_S_MAX = 1.5
SLOW_M = 0.05         # the last this many metres slow down ...
SLOW_LIN = 0.03       # ... to no slower than this (m/s)
ROLL_MAX_S = 1.0      # after the stop command the wheels are watched this long at most ...
ROLL_STILL = 0.005    # ... or until they turn slower than this (m/s)
DRIVE_TIME_FACTOR = 2.0  # a move taking this many times dist / speed (plus 2 s) stops anyway
# An in-place turn moves the body too: 5-13 cm over 130-170 deg (2026-11-06). Not measured, that went
# down to the next straight move, which then overshot its goal, turned round and overshot again, and
# the demo went back and forth round a stage spot for minutes. Turns this big are followed by a lidar
# match (window TURN_MATCH_WIN_M) once the body is on a map (mover.on_map).
TURN_LOCATE_DEG = 10.0
TURN_MATCH_WIN_M = 0.15
DRAG_FRAC = 0.75     # the body going less than this share of the wheels' way over a bottle: dragging it
STUCK_FRAC = 0.25     # a move covering less than this share of the command did not move
STUCK_MOVES_MAX = 3   # stop the whole run after this many stuck moves in a row
PASS_OVER_M = 0.05    # things to pick up no higher than this are driven over (paper scraps measured 0.04 m)
UNSTICK_LIN = 0.16    # m/s: after a slip the zig-zag goes this fast (twice LIN) to get the wheels over it
ZIG_DEG = 30.0        # zig-zag pieces this far either side of the way
ZIG_PIECE_M = 0.10
ZIG_PIECES = 6
LOCATE_TURNS = 2      # lidar turns matched against the map at each stop
MATCH_WIN_M = 0.10    # pose search window: this much plus MATCH_WIN_PER_M of the distance driven
MATCH_WIN_PER_M = 0.5
MATCH_YAW_DEG = 3.0
MATCH_OK_M = 0.04     # mean distance of scan points to the map of a fit that is trusted
# A match that fails in its window is tried again in these wider ones (m, deg): the pose may have run
# off further than the window allows. On a known map (mover.follow_no_look), after LOST_MAX failures in
# a row the lidar is matched against the whole map (saved_map.locate_anywhere); found nowhere, the run
# stops rather than driving on by dead reckoning.
LOST_WINDOWS = ((0.4, 8.0), (0.8, 12.0))
LOST_MAX = 2
LOOK_M = 0.35         # stop and look after this much straight driving
LOOK_FRESH_S = 3.0    # a look older than this cannot veto a turn
LOOK_REUSE_S = 30.0   # follow: a look this recent, from the same place, is not taken again
NO_LOOK_STEP_M = 0.60  # follow without looks (mover.follow_no_look): steps this long, the lidar watching
STOP_MARGIN = 0.08    # keep this much floor between the body and anything seen ahead
TURN_SPARE = 0.05     # anything inside the turning circle plus this stops a turn
TURN_LOOK_DEG = 45.0  # turn in steps the head (NECK_SIDE_MAX plus half the camera view) can look ahead of
TURN_CHECK_DEG = 15.0  # smaller turns barely move the corners
# Head turned further, the arms fill the bottom corners (learn_self masks them); at 45 deg they take a
# third of the picture. The side poles that kept it to 15 deg are off the body (2026-11-06).
NECK_SIDE_MAX = 30.0
SLIDE_MAX_DEG = 75.0  # unable to turn, the body slides towards legs this far off its heading (the lidar sees about +-80 deg)
BLIND_SLIDE_M = 0.30  # farther round, where the lidar does not see, only this far and only as far as the map is free
STEP_OUT_DEG = 15     # directions tried when the planned way is closed
STEP_OUT_MIN_M = 0.10  # a direction is open when the body can move at least this far
PERSON_LIFE_S = 20.0  # people walk away; their marks expire
STUCK_LIFE_S = 120.0  # a spot the wheels could not cross blocks this long
CLEAR_GAP_M = 0.10    # floor the stereo saw this close to something it saw does not clear marks
OBJECT_KEEP = 1500    # points kept per modelled thing (3 cm apart: a few square metres of furniture)
# A thing is the cells (CORE_CELL_M) that at least CORE_SHARE of its looks measured (two or more once
# it has three looks): put together whole, 82 looks each a few cm off made a 0.14 m trash can
# 0.76 x 0.42 m and shut the way past it (2026-11-04). Its height is the looks' median.
CORE_CELL_M = 0.03
CORE_SHARE = 0.2
LOOKS_KEEP = 60       # the newest looks of a thing kept for its core
FIT_NOISE_M = 0.02    # a point this close to a side of its rectangle counts as on it (lidar range noise)
GLASS_JOIN_M = 0.30   # neighbouring points of a glass foot this close are one stretch of it
GLASS_BEHIND_M = 0.08  # seen this far past the foot of glass or a mirror: behind the glass or in the mirror
GLASS_KEEP = 300      # glass feet remembered
# An overhang (table top, drawer) goes on the map only once a second viewpoint (OVERHANG_VIEW_M away or
# OVERHANG_VIEW_DEG turned) measured it within OVERHANG_CONFIRM_M of the first. Its range is the
# nearest support in its bearings; a far table top ranged onto a chair in front of it lands somewhere
# else from every other viewpoint (2026-11-03: twice such an edge, seen once, closed the only way on).
OVERHANG_CONFIRM_M = 0.10
OVERHANG_VIEW_M = 0.05
OVERHANG_VIEW_DEG = 5.0
OVERHANG_PENDING = 200
RECHECK_CUT_M2 = 1.0  # exploring ends with this much it cannot reach: look again round the body first
# One place, one thing. FOOT_M: a thing's footprint is its points grown by the error of placing them (a box
# bottom projected 1.5 m out is about 0.09 m off, vln_scene._bottom_error). A look covering NMS_OVERLAP of
# the smaller footprint is the same thing; under another name only if neither is NMS_SIZE_RATIO times bigger.
FOOT_M = 0.10
NMS_OVERLAP = 0.5
NMS_SIZE_RATIO = 4.0
OVERHANG_KEYS = ("桌", "柜", "书架", "货架", "隔板")  # things whose top reaches over the floor
# A thing this long (its model) is no small thing, whatever its looks' median length: a 1.95 m
# cardboard model of 71 looks (median 0.11 m) took in paper balls and a white plastic bag beside it,
# outvoted them and they were never picked (2026-11-07). Looser than PICK_MAX_M: looks a few cm
# apart spread a small thing's model.
SMALL_SPREAD_M = 0.30
ROUND_VETO = 3  # this much vote (one stereo look) for a thing with bulk keeps it driven round (_settle)
HOW_WEIGHT = {"双目": 3, "雷达+双目": 3, "双目（贴地）": 2, "框底投到地面，雷达对上": 2, "框底投到地面，按雷达": 2,
              "雷达（框伸出画面下沿）": 2, "画面里的线投到地面": 2, "框底投到地面": 1}  # a name's vote by how the look measured the thing
EXPLORE_VIEW_M = 1.0  # unknown or unvisited floor this close to a spot counts for going there (stereo range)
OPEN_FLOOR_M = 0.5    # free floor narrower than this is lidar rays through gaps, not floor to go to
EXPLORE_MIN_GAIN_M2 = 0.015  # a goal worth going to first has this much to do within EXPLORE_VIEW_M
SWEEP_MIN_GAIN_M2 = 0.01  # likewise for a sweep goal, within half a brush
SURVEY_BAN_M = 0.5    # after a survey turn no new goal this close to it
GOAL_TRIES = 2
STILL_LEGS_MAX = 4    # legs in a row that leave the body where it was: look again at what pins it
RECHECK_M = 0.8
MAX_LEGS = 400
GOAL_SWEEP = 0.90
TRAIL_KEEP = 60
# Something that moves with the body (a thing left in the jaws, a bag caught on the bumper, a cable
# hanging in view) is no obstacle. A bottle in the jaws, not known to be there, was mapped 0.18 m ahead
# at every look: every turn refused, the body backed and slid to and fro until the user cut the power
# (2026-11-07; the user: any thing may hang in view, not only bottles). Each point of a hit near the body
# is compared with an earlier look from where a thing on the floor would have shifted at least
# ATTACH_SHIFT_M: seen there in the same base-frame place, and bare floor (within ATTACH_FLOOR_M, the
# stereo's floor grid) where a thing on the floor would have been, it votes "on the body"; not seen in
# the same place but something where it would have been, "on the floor"; else (not in that look's view,
# hidden) no vote. On the body: at least ATTACH_MIN_PTS votes and ATTACH_RATIO times the floor's, in
# ATTACH_CONFIRM earlier looks. Bare floor, not just nothing: a wall driven along, or a place the earlier
# look did not see, passed for "nothing there" in simulation. What is on the body is left out, its marks
# go, and the cameras leave its place (ATTACH_ZONE_M round its points) alone until the arm moves
# (body_changed).
ATTACH_NEAR_M = 0.6
ATTACH_SHIFT_M = 0.06
ATTACH_TOL_M = 0.03
ATTACH_FLOOR_M = 0.04
ATTACH_RATIO = 2.0
ATTACH_MIN_PTS = 4
ATTACH_CONFIRM = 2
ATTACH_LOOKS = 8
ATTACH_ZONE_M = 0.04
RETREAT_STEP_M = 0.10
RETREAT_MAX_M = 1.5
BACK_SLANT_DEG = 30.0  # reverse slanted: on the three omni wheels a straight reverse stalls on one wheel
BACK_PIECE_M = 0.08
BACK_OFF_MAX_M = 0.60
ESCAPE_MAX_M = 1.0
LOG_KEEP = 5


def passable(label, height):
    """A thing to pick up the body drives over: as the VLM judged it (vln_scene TAG_PICK_OVER); for a
    label without a verdict, flat paper (vln_scene.DRIVE_OVER_KEYS), or anything flat enough, but
    never bottles or cartons (DRIVE_ROUND_KEYS), however low they measure, whatever the VLM said."""
    if any(k in label for k in DRIVE_ROUND_KEYS):
        return False
    tag = label_tag(label)
    if tag in (TAG_PICK, TAG_PICK_OVER):
        return tag == TAG_PICK_OVER
    return any(k in label for k in DRIVE_OVER_KEYS) or (height is not None and height <= PASS_OVER_M)


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def _neck(d_yaw):
    """Head yaw: straight or NECK_SIDE_MAX to the side."""
    if abs(d_yaw) < NECK_SIDE_MAX / 2:
        return 0.0
    return math.copysign(NECK_SIDE_MAX, d_yaw)


def to_map(pose, p):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pose[0] + c * p[0] - s * p[1], pose[1] + s * p[0] + c * p[1]


def to_robot(pose, p):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = p[0] - pose[0], p[1] - pose[1]
    return c * dx + s * dy, -s * dx + c * dy


def _extent(rel):
    """Half the body along and across the direction `rel` (rad from the heading)."""
    hl, hw = BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0
    c, s = abs(math.cos(rel)), abs(math.sin(rel))
    return hl * c + hw * s, hl * s + hw * c


class Mover:
    """The base: IMU turns, straight moves held on heading, and the pose from lidar matching."""

    def __init__(self, base, imu, lidar, cmap, view):
        self.base, self.imu, self.lidar, self.cmap, self.view = base, imu, lidar, cmap, view
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.yaw0 = 0.0         # IMU degrees at map heading 0; lidar matches move it
        self.hold = None        # heading (rad) straight moves keep; None: the heading each move starts with
        self.slip = SLIP0
        self.coast_s = COAST_S0
        self.on_map = False     # the pose is on a map (built or relocated): turns are followed by a lidar match
        self.carry_zone = None  # base-frame box (x0, x1, y0, y1) the cameras leave alone (outside_carry)
        self.lost = 0           # lidar matches failed in a row (locate)
        self.stuck = False
        self.stuck_moves = 0
        self.trail = []         # straight moves made, oldest first: (x0, y0, yaw, rel)
        self.turn_refused = False
        self.stage = ""
        self.look = None        # (pose, time, base-frame points) of the last look
        self.recent_looks = []  # (pose, base-frame xy of hits, of bare floor near the body) of the last looks (on_body)
        self.body_things = None  # base-frame xy (N x 2) of what was found to move with the body
        self.objects = []       # things the cameras named and measured, map frame (remember)
        self.glass = []         # feet of windows and mirrors seen, map-frame polylines
        self.next_id = 0
        self.overhang_pending = []  # (viewpoint, map points) of overhangs seen from one viewpoint only
        self.survey_at = None
        self.still = 0
        self._shown_at = 0.0

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    def show(self, status=None):
        with self.view.lock:
            self.view.robot = self.pose
            if status is not None:
                self.view.status = status

    def set_stage(self, name):
        self.stage = name
        with self.view.lock:
            self.view.stage = name

    def imu_yaw(self):
        return math.radians(self.imu.yaw_deg() - self.yaw0)

    def _need_imu(self):
        if not self.imu.fresh():
            self.base.stop()
            raise SystemExit(f"IMU 数据中断 {self.imu.age():.1f} s，停车")

    def attach_imu(self):
        """A short first turn shows which way the IMU yaw counts."""
        if not self.imu.wait_ready():
            raise SystemExit("IMU 没有数据，停车")
        a0 = self.imu.yaw_deg()
        t_end = time.time() + WIGGLE_S
        try:
            while time.time() < t_end:
                self.base.base.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": WIGGLE_RATE})
                time.sleep(0.05)
        finally:
            self.base.stop()
        time.sleep(0.4)
        moved = self.imu.yaw_deg() - a0
        expect = WIGGLE_RATE * WIGGLE_S
        if abs(moved) < 0.25 * expect:
            raise SystemExit(f"车没有转起来：命令左转约 {expect:.0f}°，IMU 只变了 {moved:+.1f}°。"
                             "车可能被卡住、轮子打滑或电机没出力，也可能 IMU 没固定在车身上")
        self.imu.sign = 1.0 if moved > 0 else -1.0
        self.yaw0 = self.imu.sign * a0
        self.yaw = self.imu_yaw()
        print(f"IMU 方向：车向左转时模块 yaw {'增大' if self.imu.sign > 0 else '减小'}，"
              f"命令约 {expect:.0f}°，实测 {abs(moved):.1f}°")
        self.turn_to(0.0, "回到出发方向")

    # --- turning ---------------------------------------------------------------------------
    def turn_to(self, heading, status=None):
        self.turn_by(wrap_deg(math.degrees(heading - self.yaw)), status)

    def turn_by(self, deg, status=None):
        """Turn until the IMU heading is within TURN_TOL_DEG of the goal and has settled."""
        if abs(deg) < 1.0:
            return
        start = self.yaw = self.imu_yaw()
        target = start + math.radians(deg)
        t0 = time.time()
        limit = abs(deg) / ROT * 3.0 + 4.0
        try:
            while True:
                self._need_imu()
                self.yaw = self.imu_yaw()
                err = wrap_deg(math.degrees(target - self.yaw))
                self.show(status)
                if abs(err) <= TURN_TOL_DEG:
                    self.base.stop()
                    time.sleep(TURN_SETTLE_S)
                    self.yaw = self.imu_yaw()
                    if abs(wrap_deg(math.degrees(target - self.yaw))) <= TURN_TOL_DEG:
                        break
                if time.time() - t0 > limit:
                    print(f"  转弯超时 {limit:.0f} s，还差 {err:+.1f}°，停下")
                    break
                rate = max(-ROT, min(ROT, TURN_KP * err))
                if abs(rate) < TURN_MIN_RATE:
                    rate = math.copysign(TURN_MIN_RATE, err)
                self.base.base.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": rate})
                time.sleep(0.02)
        finally:
            self.base.stop()
        self.yaw = self.imu_yaw()
        self.hold = None
        self._passed_turn(start, self.yaw)
        if self.on_map and abs(deg) >= TURN_LOCATE_DEG:
            self.locate(TURN_MATCH_WIN_M)

    def spin(self, degrees, rate, status, on_sample):
        """Turn `degrees` in place no faster than `rate` deg/s, calling on_sample along the way.

        The rotation is counted as it happens: a goal of start + 360° wraps back onto the start.
        """
        start = prev = self.imu_yaw()
        turned = 0.0
        t0 = time.time()
        limit = abs(degrees) / rate * 1.5 + 10.0  # on carpet 12°/s commanded turned about 10°/s
        next_sample = 0.0

        def follow_imu():
            nonlocal prev, turned
            self._need_imu()
            self.yaw = self.imu_yaw()
            turned += wrap_deg(math.degrees(self.yaw - prev))
            prev = self.yaw

        try:
            while True:
                follow_imu()
                err = degrees - turned
                self.show(status)
                if time.time() >= next_sample:
                    on_sample()
                    next_sample = time.time() + SPIN_SAMPLE_S
                if abs(err) <= TURN_TOL_DEG:
                    self.base.stop()
                    time.sleep(TURN_SETTLE_S)
                    follow_imu()
                    if abs(degrees - turned) <= TURN_TOL_DEG:
                        break
                if time.time() - t0 > limit:
                    print(f"  转圈超时，还差 {err:+.1f}°")
                    break
                cmd = max(-rate, min(rate, TURN_KP * err))
                if abs(cmd) < TURN_MIN_RATE:
                    cmd = math.copysign(TURN_MIN_RATE, err)
                self.base.base.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": cmd})
                time.sleep(0.02)
        finally:
            self.base.stop()
        self.yaw = self.imu_yaw()
        self.hold = None
        on_sample()
        self._passed_turn(start, start + math.radians(turned))
        self.cmap.rebuild()

    def sample_lidar(self):
        """Put the newest lidar turn on the map at the current pose (the body turns in place)."""
        xy = self.lidar.latest_xy()
        if xy is not None and len(xy):
            self.cmap.insert_scan((self.x, self.y, self.imu_yaw()), xy)
            self._show_scan(xy)
            if time.time() - self._shown_at > 1.0:  # the page shows the map growing during a turn
                self.cmap.rebuild()
                self._shown_at = time.time()

    def _show_scan(self, xy):
        with self.view.lock:
            self.view.scan_points = [to_map(self.pose, p) for p in xy[::2]]

    # --- straight moves --------------------------------------------------------------------
    def drive(self, rel, dist, status, record=True, guard=True, speed=LIN, count_stuck=True):
        """Move `dist` m towards `rel` rad from the heading, holding the heading (omni wheels).

        How far it has gone is measured as it goes: the wheels' speeds read back (_wheel_speed), added
        up and times self.slip (lidar metres per wheel metre). It stops when that and what the wheels
        will still roll (coast_s times their speed now) make `dist`, slowing down for the last SLOW_M.
        The lidar watches the strip the body is about to cover while it moves (only in its view, not
        behind). Afterwards the pose comes from matching the lidar against the map, which says how far
        the body really went. Returns those metres.
        """
        if dist < 0.01:
            return 0.0
        if record:
            self.trail.append((self.x, self.y, self.yaw, rel))
            del self.trail[:-TRAIL_KEEP]
        x0, y0 = self.x, self.y
        target = self.hold if self.hold is not None else self.imu_yaw()
        c, s = math.cos(rel), math.sin(rel)
        wheels = commanded = 0.0  # metres along `rel`: the wheels reported, the commands asked for
        v = along = 0.0
        t0 = last = time.time()
        limit = dist / speed * DRIVE_TIME_FACTOR + 2.0
        try:
            while True:
                now = time.time()
                dt, last = now - last, now
                self._need_imu()
                self.yaw = self.imu_yaw()
                err = wrap_deg(math.degrees(target - self.yaw))
                if abs(err) > HOLD_ABORT_DEG:
                    self.base.stop()
                    raise SystemExit(f"直行时航向突然偏了 {err:+.0f}°（IMU 跳变 {self.imu.jumps} 次），停车")
                got = self._wheel_speed(c, s)
                along = v if got is None else got  # a missed read: as commanded
                wheels += along * dt
                commanded += v * dt
                self.x += self.slip * along * dt * math.cos(self.yaw + rel)
                self.y += self.slip * along * dt * math.sin(self.yaw + rel)
                left = dist - self.slip * (wheels + self.coast_s * max(along, 0.0))
                if left <= 0.0:
                    break
                if now - t0 > limit:
                    print(f"  走了 {limit:.0f} s，轮子只报 {self.slip * wheels:.2f} / {dist:.2f} m，停车", flush=True)
                    break
                if guard and self._lidar_cut(rel, left):
                    break
                v = speed if left >= SLOW_M else max(SLOW_LIN, speed * left / SLOW_M)
                # The gyro term brakes a swing before the angle error has grown.
                theta = max(-HOLD_MAX_RATE, min(HOLD_MAX_RATE, HOLD_KP * err - HOLD_KD * self.imu.rate_dps()))
                self.base.base.send_action({"x.vel": v * c, "y.vel": v * s, "theta.vel": theta})
                self.show(status)
                time.sleep(0.05)
        finally:
            self.base.stop()
        rolled = self._roll_out(c, s, rel)
        wheels += rolled
        if along > 2 * SLOW_LIN / 3:
            self.coast_s = min(COAST_S_MAX, max(0.0, 0.5 * self.coast_s + 0.5 * rolled / along))
        located = self.locate(MATCH_WIN_M + MATCH_WIN_PER_M * self.slip * wheels)
        a = target + rel
        moved = (self.x - x0) * math.cos(a) + (self.y - y0) * math.sin(a)
        self.stuck = located and commanded >= SLIP_MIN_M and moved < STUCK_FRAC * commanded
        ratio = moved / wheels if wheels > 0 else 0.0
        if located and wheels >= SLIP_MIN_M and not self.stuck and SLIP_RANGE[0] <= ratio <= SLIP_RANGE[1]:
            self.slip = min(SLIP_RANGE[1], max(SLIP_RANGE[0], 0.5 * self.slip + 0.5 * ratio))
        print(f"  [行驶] 要走 {dist:.2f} m：轮子报 {wheels:.2f} m（停车命令后又滚 {rolled:.3f}），"
              f"雷达量到 {moved:.2f} m；slip {self.slip:.2f} coast {self.coast_s:.2f} s", flush=True)
        if self.stuck and count_stuck:
            self.stuck_moves += 1
            print(f"  命令 {commanded:.2f} m，雷达量到只走了 {moved:.2f} m，车卡住了", flush=True)
            if self.stuck_moves >= STUCK_MOVES_MAX:
                raise SystemExit(f"车连着 {self.stuck_moves} 次动不了：轮子在转，雷达看车几乎没动。"
                                 "检查电池电量、轮子是否打滑、底盘有没有被线或家具卡住")
        elif moved > 0.03 and not self.stuck:
            self.stuck_moves = 0
        over = self._ran_over(x0, y0)
        if over:
            print(f"  车身开过了地图上要绕开的{'、'.join(over)}", flush=True)
            # The wheels going much further than the body, over a bottle or a carton: it is caught under
            # the body and pushed along (2026-11-07: 0.71 m of wheel for 0.37 m, both under the body).
            # Clearing its marks as floor the body stood on would hide it: stop instead.
            if located and wheels >= SLIP_MIN_M and moved < DRAG_FRAC * self.slip * wheels:
                raise SystemExit(f"车身开过{'、'.join(over)}，轮子转了 {wheels:.2f} m 车只走了 {moved:.2f} m："
                                 "可能卷到车底了，停车，请检查车底")
        self._passed_line(x0, y0)
        return max(0.0, moved)

    def _ran_over(self, x0, y0):
        """Names of the things driven round (not passable) the body covered on its way from (x0, y0)
        to where it is."""
        cmap = self.cmap
        n = max(1, int(math.hypot(self.x - x0, self.y - y0) / 0.05))
        sweep = np.zeros((cmap.h, cmap.w), bool)
        for k in range(n + 1):
            sweep |= cmap.body_mask(x0 + (self.x - x0) * k / n, y0 + (self.y - y0) * k / n, self.yaw)
        sweep &= ~cmap.body_mask(x0, y0, self.yaw)  # what it stood on before is not run over
        out = []
        for o in self.objects:
            if not o.get("trash") or passable(o["label"], o.get("height")):
                continue
            # Its core, as the routes keep off it (_trim_marks): a carton model that had taken in a
            # snack bag and scraps 20-40 cm round it was "driven over" on every route by its outer
            # points and the run stopped as if it were caught under the body (2026-11-07).
            cells = [cmap.world_to_cell(x, y) for x, y in (o.get("core") or o["points"])]
            if any(c is not None and sweep[c] for c in cells):
                out.append(o["label"])
        return out

    def _wheel_speed(self, c, s):
        """Speed (m/s) along the body-frame direction (c, s) the wheels report now, or None."""
        try:
            obs = self.base.base.get_observation()
        except (ConnectionError, RuntimeError, OSError):  # a dropped status packet
            return None
        return float(obs["x.vel"]) * c + float(obs["y.vel"]) * s

    def _roll_out(self, c, s, rel):
        """Wheel metres along (c, s) rolled after the stop command, watched until they stand still
        (ROLL_STILL) or ROLL_MAX_S; the pose goes along."""
        rolled, still = 0.0, 0
        t0 = last = time.time()
        while time.time() - t0 < ROLL_MAX_S and still < 2:
            time.sleep(0.03)
            now = time.time()
            dt, last = now - last, now
            along = self._wheel_speed(c, s)
            if along is None:
                continue
            rolled += along * dt
            self.x += self.slip * along * dt * math.cos(self.yaw + rel)
            self.y += self.slip * along * dt * math.sin(self.yaw + rel)
            still = still + 1 if abs(along) < ROLL_STILL else 0
        return rolled

    def _lidar_cut(self, rel, left):
        """True when a lidar return lies in the strip the body covers in the next `left` m."""
        if abs(rel) > math.radians(100.0):
            return False  # the lidar does not see behind the body
        xy = self.lidar.latest_xy()
        if xy is None:
            return False
        ext_a, ext_l = _extent(rel)
        c, s = math.cos(rel), math.sin(rel)
        along = xy[:, 0] * c + xy[:, 1] * s
        lat = -xy[:, 0] * s + xy[:, 1] * c
        hit = (along > ext_a - 0.02) & (along < ext_a + max(left, 0.0) + STOP_MARGIN) & (np.abs(lat) <= ext_l + 0.02)
        if hit.any():
            print(f"  行驶中雷达看到前方 {float(along[hit].min()) - ext_a:.2f} m 有东西，停车", flush=True)
            return True
        return False

    def locate(self, win):
        """Match the next lidar turns against the map; on a good fit correct the pose and add them.

        Returns True when the pose was measured.
        """
        scans = self.lidar.scans(LOCATE_TURNS)
        if not scans:
            print("  雷达没有新的一圈，位置按推算", flush=True)
            return False
        xy = np.vstack(scans)
        got = self.cmap.match(xy, self.pose, win, MATCH_YAW_DEG)
        if got is None:
            return False
        pose, fit = got
        if fit > MATCH_OK_M and self.on_map:
            for wide, wide_yaw in LOST_WINDOWS:
                if wide <= win:
                    continue
                again = self.cmap.match(xy, self.pose, wide, wide_yaw)
                if again is not None and again[1] <= MATCH_OK_M:
                    print(f"  雷达和地图在 {win:.2f} m 内对不上（平均差 {fit * 100:.0f} cm），放宽到 {wide:.1f} m 找到了",
                          flush=True)
                    pose, fit = again
                    break
        if fit > MATCH_OK_M:
            self.lost += 1
            print(f"  雷达和地图对不上（平均差 {fit * 100:.0f} cm），位置按推算（连着 {self.lost} 次）", flush=True)
            if self.on_map and getattr(self, "follow_no_look", False) and self.lost >= LOST_MAX:
                self.base.stop()
                import saved_map  # here: saved_map imports this module
                found = saved_map.locate_anywhere(self, xy)
                if found is None:
                    raise SystemExit(f"雷达连着 {self.lost} 次和地图对不上，全图也找不到车在哪，停车。"
                                     "把车放到空的地方，重新启动会原地转一圈重新定位")
                pose = found
            else:
                return False
        self.lost = 0
        dx, dy = pose[0] - self.x, pose[1] - self.y
        dyaw = wrap_deg(math.degrees(pose[2] - self.yaw))
        if math.hypot(dx, dy) > 0.03 or abs(dyaw) > 1.0:
            print(f"  雷达定位校正 {dx:+.2f}, {dy:+.2f} m, {dyaw:+.1f}°（平均差 {fit * 100:.1f} cm）", flush=True)
        self.yaw0 = self.imu.yaw_deg() - math.degrees(pose[2])
        self.x, self.y, self.yaw = pose
        for scan in scans:
            self.cmap.insert_scan(pose, scan)
        self.cmap.rebuild()
        self._show_scan(xy)
        self.show()
        return True

    # --- what the body itself tells the map ---------------------------------------------------
    def _passed(self, poses):
        """The body was at these poses: nothing stands there, and that floor is swept."""
        cmap = self.cmap
        under = np.zeros((cmap.h, cmap.w), bool)
        for x, y, yaw in poses:
            under |= cmap.body_mask(x, y, yaw)
            cmap.stamp_body(x, y, yaw)
            code = {"warmup": 1, "explore": 2, "sweep": 3}.get(self.stage, 0)
            cmap.mark_pass(x, y, code, cmap.body_brush * 0.5 if code == 3 else 0.07)
        had_trash = bool((under & cmap.trash).any())
        n = cmap.unblock(under)
        if n:
            print(f"  车身刚经过的地方不可能有东西：清掉 {n} 格障碍{'（含那里标的小东西）' if had_trash else ''}", flush=True)
        cmap.rebuild()
        if had_trash:
            show_objects(self)

    def _passed_line(self, x0, y0):
        n = max(1, int(math.hypot(self.x - x0, self.y - y0) / 0.05))
        self._passed([(x0 + (self.x - x0) * k / n, y0 + (self.y - y0) * k / n, self.yaw) for k in range(n + 1)])

    def _passed_turn(self, yaw0, yaw1):
        n = max(1, int(abs(math.degrees(yaw1 - yaw0)) / 5.0))
        self._passed([(self.x, self.y, yaw0 + (yaw1 - yaw0) * k / n) for k in range(n + 1)])


# --- looking -------------------------------------------------------------------------------------
def look(mover, head, d_yaw=0.0, near_only=False, pitches=None):
    """One stop's look ahead. The map takes what it measured; returns the hits that block the body
    (base frame): what it drives over (passable(): paper, flat things) is marked but not returned.
    An overhang seen from this viewpoint only (confirmed_overhang) stays off the map but is kept
    clear of now: a real table edge first seen 0.5 m ahead must not be driven into on this step.

    Floor the stereo measured clears older camera marks there. Floor-standing glass stops lidar
    rays; people block for PERSON_LIFE_S; small things to pick up are marked apart and driven round;
    everything else blocks until the cameras see the floor free again. Named things are modelled
    (remember) and what their looks measured blocks as well. What lies behind a window or mirror
    seen now or before is left out: it is seen through the glass or in the mirror.
    """
    if mover.carry_zone is not None and d_yaw <= 0:
        # Carrying, the left arm's load fills the head's left and lower middle (neck yaw + is right):
        # the head looks right, where nothing hangs in view (the user, 2026-11-07); the lidar sees left.
        d_yaw = NECK_SIDE_MAX
    pose = mover.pose
    mover.show("停车看前方")
    head.taken = lambda pts: taken(mover, pts)
    hits = head.lane(d_yaw=d_yaw, near_only=near_only, where=pose, pitches=pitches)
    new = [[to_map(pose, p) for p in line] for line in head.last_glass]
    new += [[to_map(pose, p) for p in h.points] for h in hits if h.kind == "glass"]
    if new:
        mover.glass = (mover.glass + new)[-GLASS_KEEP:]
        drop_behind_glass(mover, pose, new)
    hits = through_glass(mover, pose, hits)
    hits = outside_carry(mover, hits)
    hits = on_body(mover, pose, hits, head.last_floor)
    now = time.monotonic()
    view = (pose[0], pose[1], math.degrees(pose[2]) - head.pose[0])  # neck yaw + is right
    accepted = []
    for h in hits:
        if h.kind == "overhead" and not confirmed_overhang(mover, view, [to_map(pose, p) for p in h.sure]):
            print(f"  {h.label}伸在上方的边只从这里量到过，换个位置再量到同一处才标进地图", flush=True)
            continue
        accepted.append(h)
    # What the body drives over stops neither a move (free_ahead) nor a turn (turn_blocker); an
    # unconfirmed overhang does, for this stop.
    blocking = [h for h in hits if not (h.kind == "trash" and passable(h.label, h.height))]
    mover.look = (pose, now, [p for h in blocking for p in h.points])
    cmap = mover.cmap
    marks = [to_map(pose, p) for h in accepted for p in h.points]
    floor = [to_map(pose, p) for p in head.last_floor]
    floor = [q for q, b in zip(floor, behind_glass(mover.glass, pose, floor)) if not b]
    if floor:
        seen = cmap.mask_of(floor, 0.0)
        if marks:
            seen &= ~cmap.mask_of(marks, CLEAR_GAP_M)
        n = cmap.clear_vis(seen)
        if n:
            print(f"  双目看到这些地方是空地，清掉 {n} 格以前标的障碍", flush=True)
    for h in accepted:
        pts = [to_map(pose, p) for p in h.sure]
        if h.kind == "glass":
            cmap.mark_glass(pts)
        elif h.kind == "trash":
            cmap.mark_trash(pts, passable=passable(h.label, h.height))
        elif h.kind == "person":
            cmap.block(pts, 0.05, now + PERSON_LIFE_S)
        else:
            cmap.block(pts, 0.03, math.inf)
        if h.kind in ("floor", "overhead", "trash") and pts:
            remember(mover, h, pts, [[to_map(pose, p) for p in band] for band in h.area or []])
    block_objects(mover)
    show_objects(mover)
    return blocking


def outside_carry(mover, hits):
    """The hits without their points in mover.carry_zone (base-frame box (x0, x1, y0, y1), set while the
    parked left arm carries something). The cameras saw the bottle hanging from the jaws in front of
    the bumper, where the lidar is blind, as a thing 0.2 m off: every turn on the way to the can was
    refused, and the body stayed too far from the can to drop into it (2026-11-06)."""
    if mover.carry_zone is None:
        return hits
    x0, x1, y0, y1 = mover.carry_zone

    def keep(pts):
        return [p for p in pts if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1)]

    return [h._replace(points=keep(h.points), sure=keep(h.sure)) for h in hits if keep(h.points)]


def body_changed(mover):
    """The arm moved (a grasp, a drop): what hung from the body may be gone, and earlier looks no
    longer show it where it hangs now."""
    mover.recent_looks = []
    mover.body_things = None


def _xy(pts):
    return np.array([(float(p[0]), float(p[1])) for p in pts]).reshape(-1, 2)


def _nearest(a, b):
    """Distance from each point of a to the nearest of b (both N x 2)."""
    if not len(b):
        return np.full(len(a), np.inf)
    return np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2), axis=1)


def moves_with_body(mover, pose, pts):
    """True when the base-frame points `pts` seen from `pose` are on the body (see ATTACH_*)."""
    xy = _xy(pts)
    xy = xy[np.hypot(xy[:, 0], xy[:, 1]) <= ATTACH_NEAR_M]
    if len(xy) < ATTACH_MIN_PTS:
        return False
    confirmed = 0
    for pose0, seen, floor in mover.recent_looks:
        # Where each point would have been in that look, were it lying on the floor.
        was = np.array([to_robot(pose0, to_map(pose, p)) for p in xy])
        moved = np.linalg.norm(was - xy, axis=1) >= ATTACH_SHIFT_M
        if moved.sum() < ATTACH_MIN_PTS:
            continue
        same = _nearest(xy[moved], seen) <= ATTACH_TOL_M
        there = _nearest(was[moved], seen) <= ATTACH_TOL_M
        bare = (_nearest(was[moved], floor) <= ATTACH_FLOOR_M) & ~there
        body, on_floor = int((same & bare).sum()), int((~same & there).sum())
        if body >= ATTACH_MIN_PTS and body >= ATTACH_RATIO * on_floor:
            confirmed += 1
    return confirmed >= ATTACH_CONFIRM


def on_body(mover, pose, hits, floor):
    """The hits without what moves with the body (moves_with_body, or within ATTACH_ZONE_M of what was
    found to); what is found now loses its marks on the map and its models. floor: base-frame points
    the look measured as bare floor."""
    zone = mover.body_things

    def off_body(pts):
        if zone is None or not len(pts):
            return list(pts)
        return [p for p, d in zip(pts, _nearest(_xy(pts), zone)) if d > ATTACH_ZONE_M]

    kept, found = [], []
    for h in hits:
        h = h._replace(points=off_body(h.points), sure=off_body(h.sure))
        if not h.points:
            continue
        (found if h.kind != "glass" and moves_with_body(mover, pose, h.points) else kept).append(h)
    near = _xy([p for h in kept + found for p in h.points])
    near = near[np.hypot(near[:, 0], near[:, 1]) <= ATTACH_NEAR_M]
    bare = _xy(floor)
    bare = bare[np.hypot(bare[:, 0], bare[:, 1]) <= ATTACH_NEAR_M + ATTACH_SHIFT_M * 3]
    for h in found:
        xy = _xy(h.points)
        print(f"  {h.label or '这个东西'}跟着车一起动（车挪了，它在车身上的位置没变）：是挂在车上的，不算障碍", flush=True)
        cmap = mover.cmap
        marks = np.zeros((cmap.h, cmap.w), bool)
        for pose0, _, _ in mover.recent_looks:
            marks |= cmap.mask_of([to_map(pose0, p) for p in xy], ATTACH_TOL_M)
        cmap.clear_vis(marks)
        mover.objects = [o for o in mover.objects
                         if not (o["foot"].any() and (o["foot"] & marks).sum() >= 0.5 * o["foot"].sum())]
        cmap.rebuild()
        mover.body_things = xy if mover.body_things is None else np.vstack([mover.body_things, xy])
    mover.recent_looks = (mover.recent_looks + [(pose, near, bare)])[-ATTACH_LOOKS:]
    return kept


def confirmed_overhang(mover, view, pts):
    """True when this overhang (map points) was measured in the same place from another viewpoint, or
    lies on a table, cabinet or shelf already measured more than once. Otherwise it waits in
    mover.overhang_pending and blocks nothing."""
    xy = np.asarray(pts, np.float64).reshape(-1, 2)
    if not len(xy):
        return False
    foot = mover.cmap.mask_of(xy, mover.cmap.res)
    if any(o["seen"] >= 2 and any(k in o["label"] for k in OVERHANG_KEYS) and (o["foot"] & foot).any()
           for o in mover.objects):
        return True
    for k, (v, other) in enumerate(mover.overhang_pending):
        if math.hypot(view[0] - v[0], view[1] - v[1]) < OVERHANG_VIEW_M and abs(wrap_deg(view[2] - v[2])) < OVERHANG_VIEW_DEG:
            continue
        d = np.sqrt(((xy[:, None, :] - other[None, :, :]) ** 2).sum(-1)).min(axis=1)
        if (d <= OVERHANG_CONFIRM_M).mean() >= 0.5:
            del mover.overhang_pending[k]
            return True
    mover.overhang_pending = (mover.overhang_pending + [(view, xy)])[-OVERHANG_PENDING:]
    return False


def behind_glass(lines, pose, pts):
    """For map points `pts`: True where the sight line from the body (pose) crosses the foot of a known
    window or mirror and the point lies more than GLASS_BEHIND_M past it."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    out = np.zeros(len(pts), bool)
    seg = [(a, b) for line in lines for a, b in zip(line, line[1:])
           if math.hypot(b[0] - a[0], b[1] - a[1]) <= GLASS_JOIN_M]
    if not seg or not len(pts):
        return out
    a = np.array([s[0] for s in seg], np.float64)
    e = np.array([s[1] for s in seg], np.float64) - a
    o = np.array(pose[:2], np.float64)
    d = pts - o                                                   # N x 2 sight lines
    ao = a - o                                                    # M x 2
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ao[None, :, 0] * e[None, :, 1] - ao[None, :, 1] * e[None, :, 0]) / den
        u = (ao[None, :, 0] * d[:, None, 1] - ao[None, :, 1] * d[:, None, 0]) / den
    past = (1.0 - t) * np.hypot(d[:, 0], d[:, 1])[:, None]
    cross = (np.abs(den) > 1e-12) & (t > 0.0) & (t < 1.0) & (u >= 0.0) & (u <= 1.0) & (past > GLASS_BEHIND_M)
    return cross.any(axis=1)


def through_glass(mover, pose, hits):
    """The hits less what lies behind a known window or mirror: seen through it or in it, not here."""
    out = []
    for h in hits:
        if h.kind == "glass" or not h.points:
            out.append(h)
            continue
        here = [p for p, b in zip(h.points, behind_glass(mover.glass, pose, [to_map(pose, p) for p in h.points])) if not b]
        if len(here) < len(h.points):
            print(f"  {h.label}有 {len(h.points) - len(here)} 个点在玻璃或镜子后面，是透过去或映出来的，不算", flush=True)
        if here:
            keep = set(here)
            out.append(h._replace(points=here, sure=[p for p in h.sure if p in keep]))
    return out


def drop_behind_glass(mover, pose, lines):
    """Things modelled earlier that lie behind newly seen glass, seen from here: not in the room. Their
    model goes and the camera marks under them are cleared (the lidar marks stay)."""
    for obj in list(mover.objects):
        behind = behind_glass(lines, pose, obj["points"])
        if behind.mean() > 0.5:
            n = mover.cmap.clear_vis(mover.cmap.mask_of(obj["points"], 0.03))
            mover.objects.remove(obj)
            print(f"  {obj['id']} 号{obj['label']}在玻璃或镜子后面，不是房间里的东西，删掉（清掉 {n} 格视觉标记）",
                  flush=True)


def taken(mover, pts):
    """For map points: True on the footprint of a known thing that stands on the floor (a chair, a
    box, a bag), so a lidar return there is not a leg of a desk or shelf seen beyond it."""
    pts = list(pts)
    feet = [o["foot"] for o in mover.objects if not any(k in o["label"] for k in OVERHANG_KEYS)]
    if not feet or not pts:
        return [False] * len(pts)
    cells = [mover.cmap.world_to_cell(x, y) for x, y in pts]
    return [c is not None and any(f[c] for f in feet) for c in cells]


def _thin(points, cell=0.03, keep=OBJECT_KEEP):
    seen = {}
    for x, y in points:
        seen.setdefault((round(x / cell), round(y / cell)), (x, y))
    return list(seen.values())[-keep:]


def _rect(points):
    """Rectangle round map points, turned so the points lie on its sides: (corners, length, width,
    heading of the long side in degrees 0-180, map frame).

    The smallest rectangle is not enough: two faces of a cabinet (an L) fit as well along the line
    between their ends. Of the headings in 1° steps the one whose rectangle has the points closest to
    its sides wins (each point counts 1 / its distance to the nearest side, at least FIT_NOISE_M).
    """
    xy = np.asarray(points, np.float64).reshape(-1, 2)
    a = np.radians(np.arange(0.0, 90.0, 1.0))
    c1 = xy[:, 0][None, :] * np.cos(a)[:, None] + xy[:, 1][None, :] * np.sin(a)[:, None]
    c2 = -xy[:, 0][None, :] * np.sin(a)[:, None] + xy[:, 1][None, :] * np.cos(a)[:, None]
    lo1, hi1, lo2, hi2 = c1.min(1), c1.max(1), c2.min(1), c2.max(1)
    d = np.minimum(np.minimum(c1 - lo1[:, None], hi1[:, None] - c1), np.minimum(c2 - lo2[:, None], hi2[:, None] - c2))
    k = int(np.argmax((1.0 / np.maximum(d, FIT_NOISE_M)).sum(1)))
    e1, e2 = np.array([math.cos(a[k]), math.sin(a[k])]), np.array([-math.sin(a[k]), math.cos(a[k])])
    corners = [u * e1 + v * e2 for u, v in ((lo1[k], lo2[k]), (hi1[k], lo2[k]), (hi1[k], hi2[k]), (lo1[k], hi2[k]))]
    span1, span2 = float(hi1[k] - lo1[k]), float(hi2[k] - lo2[k])
    heading = (math.degrees(a[k]) + (0.0 if span1 >= span2 else 90.0)) % 180.0
    return [(float(x), float(y)) for x, y in corners], max(span1, span2), min(span1, span2), heading


def _fill(cmap, poly):
    """Cells of a polygon (map frame); a point or a line still takes the cells it touches."""
    mask = np.zeros((cmap.h, cmap.w), np.uint8)
    px = np.round(np.array([[(x - cmap.origin[0]) / cmap.res - 0.5, (y - cmap.origin[1]) / cmap.res - 0.5]
                            for x, y in poly])).astype(np.int32)
    if len(px) >= 3:
        cv2.fillPoly(mask, [px], 1)
    cv2.polylines(mask, [px], len(px) >= 3, 1, 1)
    return mask > 0


def _footprint(cmap, points):
    """Map cells within FOOT_M of a thing's points: where it is, give or take the measuring error."""
    return cmap.mask_of(points, FOOT_M)


def _covers_stood(cmap, rect):
    """True when a rectangle lies over floor the body stood on: nothing stands there, not even the
    top of a table, so the points in it are not one thing. Its edge cells do not count."""
    inner = cv2.erode(_fill(cmap, rect).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return bool((inner & cmap.stood).any())


def _same(cmap, obj, label, foot, points):
    """True when a footprint (of `points`) is the thing `obj`.

    Non-maximum suppression on the floor plan: most of the smaller of the two lies in the other. Under
    another name a much smaller one is a thing of its own lying by or under the bigger one (a power
    strip under a desk). Under the same name footprints that only touch are faces of one thing (the
    front and the side of a cabinet). Never when together they would lie over floor the body stood
    on: two desks with the robot between them stay two desks. Never a trash can and anything else: a
    can beside a desk merged into it was named desk by the votes and lost (2026-11-03). Nor a thing to
    pick up and one not: a bottle lying under a tripod went into the tripod and was never picked, and
    one called 细杆 a few times was not on the list of bottles (2026-11-04). Two small things to pick
    up are one by overlap alone, as boxes in object detection: the body had driven over a bottle, and
    the stood check made 26 bottles within 20 cm of it, one per look (2026-11-04).
    """
    if is_bin(label) != is_bin(obj["label"]):
        return False
    # A small thing judged trash in one look and not in another is still one thing, and its votes
    # decide (_settle); a small thing beside a big one that is no trash (a bottle under a tripod)
    # stays apart.
    small = (obj.get("size", obj["length"]) <= PICK_MAX_M and obj["length"] <= SMALL_SPREAD_M
             and _rect(list(points))[1] <= PICK_MAX_M)
    if to_pick(label) != to_pick(obj["label"]) and not small:
        return False
    a, b = int(obj["foot"].sum()), int(foot.sum())
    inter = int((obj["foot"] & foot).sum()) if a and b else 0
    if not inter:
        return False
    if inter >= NMS_OVERLAP * min(a, b):
        same = label == obj["label"] or max(a, b) <= NMS_SIZE_RATIO * min(a, b)
    else:
        same = label == obj["label"]
    if (to_pick(label) and to_pick(obj["label"])) or (small and (to_pick(label) or to_pick(obj["label"]))):
        return same
    return same and not _covers_stood(cmap, _rect(obj["points"] + list(points))[0])


def _absorb(obj, other):
    for name, score in other["names"].items():
        obj["names"][name] = obj["names"].get(name, 0) + score
    obj["points"] = _thin(obj["points"] + other["points"])
    obj["sizes"] = obj.get("sizes", []) + other.get("sizes", [])
    obj["parts"] += other["parts"]
    obj["seen"] += other["seen"]
    # Older things (saved maps) have no per-look record: all they had counts as one look.
    for o in (obj, other):
        o.setdefault("looks", [o["points"]])
        o.setdefault("heights", [o["height"]] if o["height"] is not None else [])
        o.setdefault("cuts", [bool(o["top_cut"])])
    obj["looks"] = (obj["looks"] + other["looks"])[-LOOKS_KEEP:]
    obj["heights"] = (obj["heights"] + other["heights"])[-LOOKS_KEEP:]
    obj["cuts"] = (obj["cuts"] + other["cuts"])[-LOOKS_KEEP:]


def _core(obj):
    """The points of a thing most of its looks agree on (CORE_SHARE), one per CORE_CELL_M cell."""
    looks = obj.get("looks") or [obj["points"]]
    count, where = {}, {}
    for pts in looks:
        for key, p in {(round(x / CORE_CELL_M), round(y / CORE_CELL_M)): (x, y) for x, y in pts}.items():
            count[key] = count.get(key, 0) + 1
            where.setdefault(key, p)
    need = max(2 if len(looks) >= 3 else 1, math.ceil(CORE_SHARE * len(looks)))
    core = [where[k] for k, c in count.items() if c >= need]
    return core or list(obj["points"])


def _settle(obj, cmap):
    """Name, measures and kind of a thing after its looks changed: the VLM's verdict (trash or not, may
    be driven over) with the most evidence over all names, then of the names with it the best."""
    votes = {}
    for name, score in obj["names"].items():
        votes[label_tag(name)] = votes.get(label_tag(name), 0) + score
    tag = max(votes, key=votes.get)
    # Driven over only if nothing says it has bulk: scraps lying by a bottle and a milk carton merged
    # into them and outvoted them (7 bottle against 13 for five paper names), the thing was driven
    # over and both caught under the body (2026-11-07).
    bulk = sum(s for n, s in obj["names"].items()
               if label_tag(n) in (TAG_PICK, TAG_KEEP) or any(k in n for k in DRIVE_ROUND_KEYS))
    if tag == TAG_PICK_OVER and bulk >= ROUND_VETO:
        # Still trash, only driven round: a paper ball named 纸箱/纸板【留】 in some looks (7 against 12)
        # went to 【留】 that way and was off the list (2026-11-07).
        if TAG_PICK not in votes:
            best = max((n for n in obj["names"] if label_tag(n) == tag), key=obj["names"].get)
            obj["label"] = best.replace(TAG_PICK_OVER, TAG_PICK)
            tag = None
        else:
            tag = TAG_PICK
    if tag is not None:
        obj["label"] = max((n for n in obj["names"] if label_tag(n) == tag), key=obj["names"].get)
    obj["core"] = _core(obj)
    obj["foot"] = _footprint(cmap, obj["core"])
    obj["rect"], obj["length"], obj["width"], obj["heading"] = _rect(obj["core"])
    heights = obj.get("heights") or []
    if heights:
        obj["height"] = float(np.median(heights))
        obj["top_cut"] = 2 * sum(obj.get("cuts") or []) > len(obj.get("cuts") or [])
    # Each look's own length: put together, looks placed a few cm apart spread a small thing out.
    obj["size"] = float(np.median(obj["sizes"])) if obj.get("sizes") else obj["length"]
    obj["trash"] = to_pick(obj["label"]) and (obj["size"] <= PICK_MAX_M or any_size(obj["label"]))


def remember(mover, hit, pts, area=()):
    """Model a named thing from what one look measured: its points (map frame), the floor under it
    the look measured (area: the bands under an overhang, map frame), how high it is, and from the
    points its rectangle, length, width and heading.

    One place holds one thing. A look whose footprint mostly lies on a known thing's (or the other
    way round) is that thing, whatever name the VLM gave this time (_same); the names are votes
    weighted by how well the look measured it (HOW_WEIGHT) and the thing goes by the strongest. A
    thing that grows over another is merged with it the same way. Only measured points shape a thing:
    of a cabinet seen from the front the rectangle is its front until another look measures its side.
    """
    cmap = mover.cmap
    look = {"names": {hit.label: HOW_WEIGHT.get(hit.how, 1)}, "points": _thin(list(pts)),
            "sizes": [_rect(list(pts))[1] if len(pts) >= 2 else 0.0],
            "parts": [list(band) for band in area], "seen": 1, "height": hit.height, "top_cut": hit.top_cut}
    look["looks"] = [look["points"]]
    look["heights"] = [hit.height] if hit.height is not None else []
    look["cuts"] = [bool(hit.top_cut)]
    foot = _footprint(cmap, look["points"])
    joins = [o for o in mover.objects if _same(cmap, o, hit.label, foot, look["points"])]
    if joins:
        joins.sort(key=lambda o: -sum(o["names"].values()))  # the thing keeps the number of its best-known part
        obj = joins[0]
        for other in joins[1:]:
            _absorb(obj, other)
            mover.objects.remove(other)
        _absorb(obj, look)
    else:
        mover.next_id += 1
        obj = dict(look, id=mover.next_id)
        mover.objects.append(obj)
    _settle(obj, cmap)
    while True:  # grown, it may now cover others
        more = [o for o in mover.objects
                if o is not obj and _same(cmap, o, obj["label"], obj["foot"], obj["core"])]
        if not more:
            break
        best = max(more, key=lambda o: sum(o["names"].values()))
        if sum(best["names"].values()) > sum(obj["names"].values()):  # keep the better-known number
            obj["id"], best["id"] = best["id"], obj["id"]
        for other in more:
            _absorb(obj, other)
            mover.objects.remove(other)
        joins += more
        _settle(obj, cmap)
    _trim_marks(mover, obj)
    if joins:
        others = sorted({o["id"] for o in joins} - {obj["id"]})
        if len(obj["names"]) > 1 or others:
            print(f"  这次量到的{hit.label}是 {obj['id']} 号{'（并进了 ' + '、'.join(map(str, others)) + ' 号）' if others else ''}，"
                  f"各名字的分数 {obj['names']}，按 {obj['label']} 算", flush=True)


def _trim_marks(mover, obj):
    """The camera marks this thing's looks left outside its core go (the core and other things'
    cores keep theirs; the lidar's are not touched): its marks shrink with its model."""
    cmap = mover.cmap
    extra = cmap.mask_of(obj["points"], 0.03) & ~cmap.mask_of(obj["core"], 0.03)
    for o in mover.objects:
        if o is not obj and o.get("core"):
            extra &= ~cmap.mask_of(o["core"], 0.03)
    if extra.any():
        cmap.vis_end[extra] = 0.0
        if obj["trash"]:
            cmap.trash[extra] = False
            cmap.trash_low[extra] = False


def block_objects(mover):
    """Rebuild the map layer of the modelled things: the floor under their overhangs, as measured.

    What a thing stands on is marked where it was measured (look). On top of that a table or shelf
    blocks the bands of floor each look measured under its edge, back to its supports in the same
    bearing, and a chair or stool the floor between the legs one look measured: the body cannot go
    under its top or seat though nothing stands there. No outline round a whole look or round the
    thing, which would take in free floor; the rectangle is only its measures for the page. The
    cells the body stood on never block (OccMap.rebuild).
    """
    cmap = mover.cmap
    layer = np.zeros((cmap.h, cmap.w), bool)
    for obj in mover.objects:
        if obj["trash"]:
            continue
        for part in obj["parts"]:
            layer |= _fill(cmap, part)
    cmap.objects = layer
    cmap.rebuild()


def _blocked_outlines(cmap, obj):
    """Outlines (map frame) of the cells a thing blocks: where it was measured and the floor under
    its top or seat. The page draws these, not the rectangle round all its points."""
    cells = cmap.mask_of(obj["points"], 0.03)
    for part in obj["parts"]:
        cells |= _fill(cmap, part)
    contours, _ = cv2.findContours(cells.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [[cmap.cell_center((int(r), int(c))) for c, r in cnt.reshape(-1, 2)] for cnt in contours]


def show_objects(mover):
    # A thing to pick up that is gone from the map (the stereo saw bare floor there) is gone from the page.
    mover.objects = [o for o in mover.objects
                     if not o["trash"] or any(mover.cmap.is_trash(x, y) for x, y in o["points"])]
    for o in mover.objects:
        o["outlines"] = _blocked_outlines(mover.cmap, o)
    with mover.view.lock:
        mover.view.objects = [{"id": o["id"], "label": o["label"], "outlines": o["outlines"],
                               "x": float(np.mean([p[0] for p in o["points"]])),
                               "y": float(np.mean([p[1] for p in o["points"]])),
                               "length": o["length"], "width": o["width"], "heading": o["heading"],
                               "height": o["height"], "top_cut": o["top_cut"], "trash": o["trash"],
                               "size": o.get("size", o["length"]),
                               "passable": o["trash"] and passable(o["label"], o["height"]),
                               "seen": o["seen"]}
                              for o in mover.objects]


# --- safety checks ---------------------------------------------------------------------------
def _occupied_near(mover, limit, skip_body=True):
    """Map points (robot frame) of occupied cells within `limit` of the body center, nearest first."""
    cmap = mover.cmap
    here = cmap.world_to_cell(mover.x, mover.y)
    if here is None:
        return []
    r = int(math.ceil(limit / cmap.res))
    y0, y1 = max(0, here[0] - r), min(cmap.h, here[0] + r + 1)
    x0, x1 = max(0, here[1] - r), min(cmap.w, here[1] + r + 1)
    occ = cmap.occupied()[y0:y1, x0:x1].copy()
    if skip_body:
        occ &= ~cmap.body_mask(*mover.pose)[y0:y1, x0:x1]
    ys, xs = np.nonzero(occ)
    pts = [to_robot(mover.pose, cmap.cell_center((y0 + int(a), x0 + int(b)))) for a, b in zip(ys, xs)]
    return sorted((p for p in pts if math.hypot(*p) <= limit), key=lambda p: math.hypot(*p))


def turn_blocker(mover):
    """Something inside the circle the corners sweep: the lidar now, a fresh look, or the map."""
    limit = BODY_CLEARANCE_M + TURN_SPARE
    near = []  # (point, where from): the where only for the diagnostic line below
    xy = mover.lidar.latest_xy()
    if xy is not None:
        near += [((float(x), float(y)), "雷达") for x, y in xy if math.hypot(float(x), float(y)) <= limit]
    if mover.look is not None:
        pose, t, pts = mover.look
        if (time.monotonic() - t < LOOK_FRESH_S and math.hypot(mover.x - pose[0], mover.y - pose[1]) < 0.03
                and abs(wrap_deg(math.degrees(mover.yaw - pose[2]))) < 3.0):
            near += [(p, "相机") for p in pts if math.hypot(*p) <= limit]
    near += [(p, "地图") for p in _occupied_near(mover, BODY_CLEARANCE_M + 0.02)[:1]]
    if not near:
        return None
    p, why = min(near, key=lambda e: math.hypot(*e[0]))
    # Diagnostics (2026-11-06): which layer of the map, and how many lidar returns, made a turn refused.
    cmap = mover.cmap
    cell = cmap.world_to_cell(*to_map(mover.pose, p))
    if why == "地图" and cell is not None:
        now = time.monotonic()
        layers = {"雷达层": cmap._lidar, "相机层": cmap.vis_end > now, "垃圾": cmap.trash, "贴地垃圾": cmap.trash_low,
                  "物体": cmap.objects & ~cmap.stood, "玻璃": cmap.glass}
        why += "/" + ",".join(k for k, v in layers.items() if v[cell])
    lidar_n = sum(1 for _, w in near if w == "雷达")
    print(f"  [转身挡] {why} 点 ({p[0]:+.2f}, {p[1]:+.2f}) 离车 {math.hypot(*p):.2f} m "
          f"方位 {math.degrees(math.atan2(p[1], p[0])):+.0f}°；雷达近点 {lidar_n} 个", flush=True)
    return p


def map_room(mover, rel, limit, ignore=None):
    """Metres the body can move towards `rel` before it covers an occupied map cell it is not on now
    (nor in the mask `ignore`: the thing it goes up to)."""
    cmap = mover.cmap
    occ = cmap.occupied() & ~cmap.body_mask(*mover.pose)
    if ignore is not None:
        occ &= ~ignore
    a = mover.yaw + rel
    d = 0.0
    while d < limit:
        nxt = min(limit, d + 0.03)
        if occ[cmap.body_mask(mover.x + nxt * math.cos(a), mover.y + nxt * math.sin(a), mover.yaw)].any():
            return d
        d = nxt
    return limit


def free_ahead(mover, hits, rel=0.0, limit=LOOK_M):
    """Metres the body may move towards `rel` now: the look's points, the lidar and the map all agree."""
    ext_a, ext_l = _extent(rel)
    c, s = math.cos(rel), math.sin(rel)
    pts = [p for h in hits for p in h.points]
    xy = mover.lidar.latest_xy()
    if xy is not None and abs(rel) <= math.radians(100.0):
        pts += [(float(x), float(y)) for x, y in xy]
    best = limit
    for px, py in pts:
        along, lat = px * c + py * s, -px * s + py * c
        if along > ext_a - 0.05 and abs(lat) <= ext_l + 0.02:
            gap = along - ext_a - STOP_MARGIN
            if gap < best:
                best = gap
    room = map_room(mover, rel, best if best > 0 else 0.0)
    return max(0.0, min(best, room))


def swept_hit(local, rel, dist, margin=0.05):
    """True if sliding the body `dist` towards `rel` (no turning) runs into any point (robot frame)."""
    hl, hw = BODY_LENGTH_M / 2.0 + margin, BODY_WIDTH_M / 2.0 + margin
    c, s = math.cos(rel), math.sin(rel)
    for px, py in local:
        lo, hi = 0.0, dist
        for p, h, v in ((px, hl, c), (py, hw, s)):
            if abs(v) < 1e-9:
                if abs(p) > h:
                    lo, hi = 1.0, 0.0
                continue
            a, b = sorted(((p - h) / v, (p + h) / v))
            lo, hi = max(lo, a), min(hi, b)
        if lo <= hi:
            return True
    return False


def can_turn(mover):
    cell = mover.cmap.world_to_cell(mover.x, mover.y)
    return cell is not None and bool(mover.cmap.drive_mask[cell]) and turn_blocker(mover) is None


# --- moving safely -----------------------------------------------------------------------------
def safe_turn(mover, head, heading, status):
    """Turn in place towards `heading` in steps, looking first where the front corners will swing.

    Returns False (the turn refused) when something is inside the turning circle.
    """
    for _ in range(12):
        left = wrap_deg(math.degrees(heading - mover.yaw))
        if abs(left) < TURN_TOL_DEG + 0.5:
            return True
        step = max(-TURN_LOOK_DEG, min(TURN_LOOK_DEG, left))
        if abs(step) >= TURN_CHECK_DEG:
            look(mover, head, d_yaw=_neck(-step), near_only=True)
        stand = turn_blocker(mover)
        if stand is not None:
            print(f"  转身半径内有东西（离车 {math.hypot(*stand):.2f} m，"
                  f"方位 {math.degrees(math.atan2(stand[1], stand[0])):+.0f}°），不转", flush=True)
            mover.turn_refused = True
            return False
        mover.turn_by(step, status)
    return True


def follow(mover, head, route, status):
    """Drive the corner points of `route`, stopping to look every LOOK_M.

    The body turns to face each leg. Where it cannot turn (something inside the turning circle) it
    keeps its heading and slides towards the leg on its omni wheels: within SLIDE_MAX_DEG of the
    heading, where the lidar watches the way, or farther round for a short leg (BLIND_SLIDE_M, out of
    a tight spot onto drivable floor) as far as the map is free. Returns "done", or why it stopped:
    "turn" (no room to turn and the leg out of sliding reach), "blocked" (something ahead), "stuck"
    (the wheels did not move the body) or "replan" (the map closed the way).
    """
    for tx, ty in route[1:]:
        for _ in range(200):
            d = math.hypot(tx - mover.x, ty - mover.y)
            if d < 0.04:
                break
            heading = math.atan2(ty - mover.y, tx - mover.x)
            rel = 0.0
            if abs(wrap_deg(math.degrees(heading - mover.yaw))) > 4.0:
                if not safe_turn(mover, head, heading, status):
                    rel = math.radians(wrap_deg(math.degrees(heading - mover.yaw)))
                    if abs(math.degrees(rel)) <= SLIDE_MAX_DEG:
                        print(f"  转不了身，车头不动，朝 {math.degrees(rel):+.0f}° 平移过去", flush=True)
                    elif d <= BLIND_SLIDE_M:
                        # The lidar does not see that way; the map the turns at the start built does.
                        print(f"  转不了身，雷达看不到 {math.degrees(rel):+.0f}° 那边，按地图平移 {d:.2f} m 到能走的地方",
                              flush=True)
                    else:
                        return "turn"
            mover.hold = heading if rel == 0.0 else mover.yaw
            # mover.follow_near_only (set by a caller on a known map): only the near floor look,
            # not three, per step; demo/bottle_to_bin crept 1 m in 20 min with three (2026-11-05).
            # Not moved since the last look (same place, same head yaw, LOOK_REUSE_S ago at most):
            # that look again. Hemmed in where it could not turn, it looked up and down at the same
            # bottle, strip and wire for every next spot it tried (2026-11-05).
            # mover.follow_no_look (a caller on a known map): no camera looks at all, the lidar and the
            # map keep the way, in NO_LOOK_STEP_M steps. A trip to the trash can with looks took 27 of
            # a 52 min run, 412 VLM calls ~21 min in all (2026-11-05).
            d_yaw = _neck(-math.degrees(rel))
            last = getattr(mover, "last_hits", None)
            no_look = getattr(mover, "follow_no_look", False)
            if no_look:
                hits = []
            elif (last is not None and last[0] == d_yaw and time.monotonic() - last[2] < LOOK_REUSE_S
                    and math.hypot(mover.x - last[1][0], mover.y - last[1][1]) < 0.03
                    and abs(wrap_deg(math.degrees(mover.yaw - last[1][2]))) < 3.0):
                hits = last[3]
            else:
                near = getattr(mover, "follow_near_only", False)  # the near floor only, no look up
                hits = look(mover, head, d_yaw=d_yaw, pitches=(LANE_PITCHES[-1],) if near else None)
                mover.last_hits = (d_yaw, mover.pose, time.monotonic(), hits)
            limit = NO_LOOK_STEP_M if no_look else LOOK_M
            free = free_ahead(mover, hits, rel, limit)
            step = min(limit, d, free)
            if step < 0.03:
                print(f"  {'前方' if rel == 0.0 else '平移方向上'}只剩 {free:.2f} m，换路", flush=True)
                return "blocked"
            mover.drive(rel, step, status, count_stuck=False)
            if mover.stuck and not zigzag(mover, rel, step, status):
                mover.stuck_moves += 1
                if mover.stuck_moves >= STUCK_MOVES_MAX:
                    raise SystemExit(f"车连着 {mover.stuck_moves} 次动不了，之字形加速也没脱困：轮子在转，雷达看车几乎没动。"
                                     "检查电池电量、轮子是否打滑、底盘有没有被线或家具卡住")
                reach = _extent(rel)[0] + 0.08  # just past the body's edge in the way it moved
                spot = to_map(mover.pose, (reach * math.cos(rel), reach * math.sin(rel)))
                mover.cmap.block([spot], 0.06, time.monotonic() + STUCK_LIFE_S)
                mover.cmap.rebuild()
                return "stuck"
            if rel != 0.0:
                mover.turn_refused = False  # it got away from what kept it from turning
            cell = mover.cmap.world_to_cell(tx, ty)
            if cell is None or not mover.cmap.drive_mask[cell]:
                return "replan"
    return "done"


def zigzag(mover, rel, need, status):
    """The wheels slipped on a move towards `rel`: zig-zag out, faster. Short pieces ZIG_DEG either side
    of the way alternately at UNSTICK_LIN, each only where the lidar and the map leave room, until the
    body has made `need` m along the way, at most ZIG_PIECES pieces. Returns True when it got through.
    """
    a = mover.yaw + rel
    x0, y0 = mover.x, mover.y
    side = 1.0
    print(f"  打滑了：加速到 {UNSTICK_LIN:.2f} m/s，走之字形脱困", flush=True)
    for _ in range(ZIG_PIECES):
        way = rel + side * math.radians(ZIG_DEG)
        room = free_ahead(mover, [], way, ZIG_PIECE_M)
        if room < 0.03:
            side = -side
            way = rel + side * math.radians(ZIG_DEG)
            room = free_ahead(mover, [], way, ZIG_PIECE_M)
            if room < 0.03:
                print("  之字形两边都没地方，停下", flush=True)
                break
        mover.drive(way, room, status, record=False, speed=UNSTICK_LIN, count_stuck=False)
        side = -side
        made = (mover.x - x0) * math.cos(a) + (mover.y - y0) * math.sin(a)
        if made >= need - 0.02:
            print(f"  之字形脱困了：沿原方向走了 {made:.2f} m", flush=True)
            mover.stuck, mover.stuck_moves = False, 0
            return True
    made = (mover.x - x0) * math.cos(a) + (mover.y - y0) * math.sin(a)
    print(f"  之字形没脱困：沿原方向只走了 {made:.2f} m", flush=True)
    mover.stuck = True
    return False


def back_up(mover, dist, status):
    """Reverse `dist` m, net straight back, in slanted pieces that alternate left and right.

    Straight back, one of the three omni wheels stalls and the body barely moves. Returns the net
    metres backed.
    """
    slant = math.radians(BACK_SLANT_DEG)
    backed = 0.0
    side = 1.0
    while backed < dist - 0.02:
        piece = min(BACK_PIECE_M, dist - backed)
        x0, y0 = mover.x, mover.y
        mover.drive(math.pi - side * slant, piece / math.cos(slant), status, record=False, guard=False)
        side = -side
        a = mover.yaw + math.pi
        got = (mover.x - x0) * math.cos(a) + (mover.y - y0) * math.sin(a)
        backed += max(0.0, got)
        if got < 0.01 or mover.stuck:
            break
    return backed


def retreat(mover):
    """Reverse along the driven path until the body can turn, then straight back if the map allows.

    The driven floor was swept by the body moments ago, so reversing over it needs no camera; the
    lidar does not see behind the body either. Returns the metres backed.
    """
    backed = 0.0
    refused = mover.pose if mover.turn_refused else None
    mover.turn_refused = False

    def clear():
        if not can_turn(mover):
            return False
        return refused is None or math.hypot(mover.x - refused[0], mover.y - refused[1]) >= mover.cmap.turn_radius

    while mover.trail and backed < RETREAT_MAX_M and not clear():
        x0, y0, yaw, rel = mover.trail[-1]
        a = yaw + rel
        along = (mover.x - x0) * math.cos(a) + (mover.y - y0) * math.sin(a)
        off = -(mover.x - x0) * math.sin(a) + (mover.y - y0) * math.cos(a)
        if abs(off) > 0.07 or along < -0.02:
            mover.trail.clear()  # the pose has left the trail
            break
        if along < 0.03:
            mover.trail.pop()
            continue
        if abs(wrap_deg(math.degrees(yaw - mover.yaw))) >= 1.0:
            mover.turn_to(yaw, "退回走过的路")
        step = min(along, RETREAT_STEP_M)
        if rel == 0.0:
            got = back_up(mover, step, "退回走过的路")
        else:
            got = mover.drive(rel + math.pi, step, "退回走过的路", record=False, guard=False)
        backed += got
        if got < 0.01 or mover.stuck:
            break
    if backed:
        print(f"  沿走过的路退回 {backed:.2f} m")
    extra = 0.0
    while extra < BACK_OFF_MAX_M - 0.02 and not clear():
        behind = [p for p in _occupied_near(mover, 1.0) if p[0] < 0.0]
        slant = math.radians(BACK_SLANT_DEG)
        reach = (RETREAT_STEP_M + 0.03) / math.cos(slant)
        if swept_hit(behind, math.pi - slant, reach, 0.03) or swept_hit(behind, math.pi + slant, reach, 0.03):
            print("  车尾后面地图上有东西，不能再往后退")
            break
        got = back_up(mover, RETREAT_STEP_M, "后退腾出转身的地方")
        extra += got
        if got < 0.01 or mover.stuck:
            break
    if extra:
        print(f"  又斜着往后退了 {extra:.2f} m 好转身")
    return backed + extra


def make_turn_room(mover):
    """Slide to the nearest spot where the body can turn in place, if it cannot here.

    Only directions the lidar sees (not straight back) and only as far as the lidar and the map
    leave free. Returns True when the body can turn now.
    """
    if turn_blocker(mover) is None:
        return True
    xy = mover.lidar.latest_xy()
    local = [] if xy is None else [(float(x), float(y)) for x, y in xy]
    local += _occupied_near(mover, 1.5)
    need = BODY_CLEARANCE_M + TURN_SPARE
    for k in range(4, int(ESCAPE_MAX_M / 0.05) + 1):
        d = k * 0.05
        for deg in (0, 30, -30, 60, -60, 90, -90):
            rel = math.radians(deg)
            if swept_hit(local, rel, d):
                continue
            px, py = d * math.cos(rel), d * math.sin(rel)
            if all(math.hypot(px - x, py - y) >= need for x, y in local):
                print(f"  这里转不了身，往 {deg:+d}° 平移 {d:.2f} m", flush=True)
                mover.drive(rel, d, "平移到能转身的地方")
                return turn_blocker(mover) is None
    return False


# --- stages --------------------------------------------------------------------------------------
def survey(mover, head, status):
    """A full circle in SURVEY_STOPS steps: the lidar maps all the way round and the cameras look at
    every stop, naming what the lidar points are and adding what the lidar cannot see."""
    if not make_turn_room(mover):
        print("  这里转不开一圈，先不自转", flush=True)
        return
    step = 360.0 / SURVEY_STOPS
    for k in range(SURVEY_STOPS):
        mover.show(f"{status} {k + 1}/{SURVEY_STOPS}")
        mover.locate(MATCH_WIN_M)
        look(mover, head)
        stand = turn_blocker(mover)
        if stand is not None:
            print(f"  自转到一半，转身半径内有东西（离车 {math.hypot(*stand):.2f} m），停下", flush=True)
            break
        mover.spin(step, SPIN_DPS, f"{status} {k + 1}/{SURVEY_STOPS}", mover.sample_lidar)
    mover.survey_at = (mover.x, mover.y)


def learn_self(mover, head):
    """Turn a little left and right so the head can tell its own body from the floor."""
    mover.show("左右转一下，认出画面里的车身")
    h0 = mover.yaw

    def turn(deg):
        goal = h0 + math.radians(deg)
        mover.turn_to(goal, "认车身")
        return abs(wrap_deg(math.degrees(mover.yaw - goal))) < 5.0

    if not head.learn_self(turn, NECK_SIDE_MAX):
        print("  没认出车身，只按画面底部的规则排除车身")


def warm_up(mover, head):
    """Two slow lidar turns build the first map; then a survey turn names what is in it."""
    mover.set_stage("warmup")
    learn_self(mover, head)
    if not make_turn_room(mover):
        raise SystemExit("出发的地方转不开身：把车放到四周 0.4 m 内没有东西的地方")
    print(f"阶段 1/3 建图：原地以 {SPIN_DPS:.0f}°/s 慢转 {WARMUP_TURNS} 圈，只用雷达")
    mover.spin(360.0 * WARMUP_TURNS, SPIN_DPS, "建图：雷达转圈", mover.sample_lidar)
    mover.on_map = True
    print(f"  雷达建图：已知空地 {mover.cmap.stats()['free_m2']:.1f} m²")
    survey(mover, head, "建图：视觉看一圈")


def go(mover, head, goal, tries, status):
    """Drive to a goal cell. True when it got there; a goal missed GOAL_TRIES times is given up."""
    cmap = mover.cmap
    route = route_to(cmap, mover.x, mover.y, goal)
    if route is None:
        ban_goal(cmap, goal, 0.30)
        return False
    start = (mover.x, mover.y)
    result = follow(mover, head, route, status)
    if result == "done":
        mover.still = 0
        return True
    tries[goal] = tries.get(goal, 0) + 1
    if tries[goal] >= GOAL_TRIES:
        print(f"  这个目标 {tries[goal]} 次都没到（{result}），不再去", flush=True)
        ban_goal(cmap, goal, 0.15)
    if result in ("turn", "blocked"):
        retreat(mover)
    if math.hypot(mover.x - start[0], mover.y - start[1]) < 0.05:
        mover.still += 1
        if mover.still >= STILL_LEGS_MAX:
            recheck(mover, head)
            mover.still = 0
    else:
        mover.still = 0
    return False


def recheck(mover, head, why=None):
    """The body has not got away for several legs (or `why`): drop the camera marks around it and look again.

    Lidar marks stay (the lidar is measuring them on every stop); camera marks may be old or from a
    mismatch, and the looks put back what is still there.
    """
    cmap = mover.cmap
    near = cmap.mask_of([(mover.x, mover.y)], RECHECK_M) & ~cmap.glass
    n = cmap.clear_vis(near)
    print(f"  {why or f'连续 {STILL_LEGS_MAX} 段没离开这里'}：清掉周围 {n} 格视觉标记，重新看", flush=True)
    for d_yaw in (-NECK_SIDE_MAX, 0.0, NECK_SIDE_MAX):
        look(mover, head, d_yaw=d_yaw, near_only=True)


def unvisited(cmap):
    """Free floor the body has not been within EXPLORE_VIEW_M of (the stereo ranges well that close).
    Only open floor counts: lidar rays reaching out through gaps (free less than OPEN_FLOOR_M wide)
    are no floor to go to."""
    k = max(1, int(round(OPEN_FLOOR_M / cmap.res)))
    free = cmap.seen & ~cmap.occupied()
    open_floor = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0
    near = cv2.dilate(cmap.stood.astype(np.uint8), disk(EXPLORE_VIEW_M / cmap.res)) > 0
    return open_floor & ~near


def step_out(mover, head, target):
    """The planned way is closed and the body has not moved: pick a direction itself. Of the
    directions the lidar watches (every STEP_OUT_DEG within SLIDE_MAX_DEG of the heading), the one
    where the body can move at least STEP_OUT_MIN_M (lidar, the look and the map agree) and lands
    nearest to the most `target` floor (within EXPLORE_VIEW_M) gets one step, the heading kept.
    Returns True when the body moved."""
    hits = look(mover, head)
    cmap = mover.cmap
    rad = max(1, int(math.ceil(EXPLORE_VIEW_M / cmap.res)))
    gain = cv2.filter2D(target.astype(np.float32), -1, disk(rad).astype(np.float32), borderType=cv2.BORDER_CONSTANT)
    best = None
    for deg in range(-int(SLIDE_MAX_DEG), int(SLIDE_MAX_DEG) + 1, STEP_OUT_DEG):
        rel = math.radians(deg)
        room = free_ahead(mover, hits, rel, LOOK_M)
        if room < STEP_OUT_MIN_M:
            continue
        cell = cmap.world_to_cell(*to_map(mover.pose, (room * math.cos(rel), room * math.sin(rel))))
        score = (float(gain[cell]) if cell is not None else 0.0, -abs(deg))
        if best is None or score > best[0]:
            best = (score, rel, room)
    if best is None or best[0][0] <= 0:
        print("  自己找方向：雷达看得到的方向都走不出去，或那边没有没去过的地方", flush=True)
        return False
    _, rel, room = best
    print(f"  规划的路走不通，自己找方向：车头不动，朝 {math.degrees(rel):+.0f}° 平移 {room:.2f} m，那边没去过的地方最多",
          flush=True)
    x0, y0 = mover.x, mover.y
    mover.hold = mover.yaw
    mover.drive(rel, room, "探索：自己找方向")
    return math.hypot(mover.x - x0, mover.y - y0) > 0.03


def explore(mover, head):
    """Depth first until the body has been near all the floor it can reach: to the frontier between
    known free floor and unknown, and to free floor it has not been within EXPLORE_VIEW_M of (the
    lidar may have seen it from afar; the body has to go there). A survey turn at every new place.
    Ends when none of those can be reached."""
    mover.set_stage("explore")
    cmap = mover.cmap
    branches, tries, rechecked = [], {}, []
    said = {"ahead": "接着往前方没去过的地方走", "branch": "前面走到头了，退回最近的岔路",
            "nearest": "岔路都走完了，去最近的没去过的地方"}
    for leg in range(MAX_LEGS):
        target = cmap.frontier() | unvisited(cmap)
        goal = None
        for gain in (max(1, round(EXPLORE_MIN_GAIN_M2 / cmap.res ** 2)), 1):
            goal, how, reach, cut = explore_goal(cmap, mover.x, mover.y, mover.yaw, branches, target,
                                                 EXPLORE_VIEW_M, gain)
            if goal is not None:
                break
        if goal is None:
            # Camera marks round the body may have shut it in (an edge seen wrong, a person gone):
            # once per place, clear them, look again and see whether the way opens.
            if cut * cmap.res ** 2 >= RECHECK_CUT_M2 and all(
                    math.hypot(mover.x - x, mover.y - y) >= RECHECK_M for x, y in rechecked):
                rechecked.append((mover.x, mover.y))
                recheck(mover, head, f"没有能去的地方了，可车身到不了的还有 {cut * cmap.res ** 2:.1f} m²")
                continue
            print(f"阶段 2/3 探索完：车身能到的地方都去过了（车身到不了的还有 {cut * cmap.res ** 2:.1f} m²）")
            return
        gx, gy = cmap.cell_center(goal)
        print(f"探索第 {leg + 1} 段：深度优先，{said[how]} ({gx:+.2f}, {gy:+.2f})，"
              f"还有 {reach * cmap.res ** 2:.1f} m² 没去过的地方能去", flush=True)
        x0, y0 = mover.x, mover.y
        if not go(mover, head, goal, tries, "探索"):
            if math.hypot(mover.x - x0, mover.y - y0) < 0.05:
                step_out(mover, head, target)
            continue
        last = mover.survey_at
        if last is None or math.hypot(mover.x - last[0], mover.y - last[1]) >= SURVEY_GAP_M:
            survey(mover, head, "探索：自转一圈补图")
        ban_goal(cmap, goal, SURVEY_BAN_M)
    print(f"探索走了 {MAX_LEGS} 段，进入清扫")


def sweep(mover, head):
    """The body passes over every free cell it can reach, depth first, until GOAL_SWEEP."""
    mover.set_stage("sweep")
    cmap = mover.cmap
    cmap.reset_swept()
    mover._passed([mover.pose])
    branches, tries = [], {}
    for leg in range(MAX_LEGS):
        done = cmap.stats()["sweep"]
        if done >= GOAL_SWEEP:
            print(f"清扫完成：扫过 {done:.0%}")
            return "done"
        todo = cmap.sweepable() & ~cmap.swept
        goal = None
        for gain in (max(1, round(SWEEP_MIN_GAIN_M2 / cmap.res ** 2)), 1):
            goal, how, reach, cut = explore_goal(cmap, mover.x, mover.y, mover.yaw, branches, todo,
                                                 cmap.body_brush * 0.5, gain)
            if goal is not None:
                break
        if goal is None:
            print(f"能到的地方都扫过了：{done:.0%}，到不了的还有 {cut} 格")
            return "incomplete"
        gx, gy = cmap.cell_center(goal)
        print(f"清扫第 {leg + 1} 段：去 ({gx:+.2f}, {gy:+.2f})，已扫 {done:.0%}", flush=True)
        if go(mover, head, goal, tries, "清扫"):
            # Arriving within the follow tolerance may need no move at all, and only moves paint the
            # floor: paint where the body stands and do not pick this goal again.
            mover._passed([mover.pose])
            ban_goal(cmap, goal, cmap.res)
    print(f"清扫走了 {MAX_LEGS} 段，还没到 {GOAL_SWEEP:.0%}")
    return "incomplete"


# --- program ---------------------------------------------------------------------------------------
def ensure_map_server(port):
    """Start map_server.py in its own session unless something already serves the port.

    It outlives this program and the terminal, so the page keeps the last result.
    """
    import socket
    import subprocess

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            return
    os.makedirs(MAP_DIR, exist_ok=True)
    log = open(os.path.join(MAP_DIR, "map_server.log"), "a")
    subprocess.Popen([sys.executable, os.path.join(HERE, "map_server.py"), "--port", str(port)],
                     cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    print("二维图网页已在后台开启，程序退出后也会保留")


def local_ip():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class _Tee:
    """Write what the run prints to the terminal and to a log file."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno: the terminal's
        return getattr(self.stream, name)


def start_log():
    """Copy stdout and stderr to map/logs/<time>.log, keeping the newest LOG_KEEP logs."""
    folder = os.path.join(MAP_DIR, "logs")
    os.makedirs(folder, exist_ok=True)
    for name in sorted(os.listdir(folder))[:-(LOG_KEEP - 1)]:
        os.remove(os.path.join(folder, name))
    path = os.path.join(folder, time.strftime("%Y%m%d_%H%M%S") + ".log")
    log = open(path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imu-port", default=None, help="IMU 串口，默认自动找")
    p.add_argument("--lidar-port", default=None, help="雷达串口，默认自动找 CP2102")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default="qwen3.8-max")
    p.add_argument("--timeout-s", type=float, default=20.0)
    args = p.parse_args()

    def _interrupt(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, _interrupt)

    print(f"运行日志：{start_log()}")
    view = MapView(OccMap())
    view.write(MAP_DIR)
    ensure_map_server(args.port)
    writer = view.start_writer(MAP_DIR)
    print(f"状态页：http://{local_ip()}:{args.port}")
    robot = {}
    try:
        work(args, view, robot)
    except SystemExit as stop:
        msg = str(stop.code) if stop.code not in (None, 0) else "停下"
        print(msg)
        with view.lock:
            view.status = f"已停车：{msg}"
    except KeyboardInterrupt:
        print("\nCtrl-C，停车")
        with view.lock:
            view.status = "Ctrl-C 停车"
    finally:
        # Each device is closed on its own: a second Ctrl-C while one closes must not leave the
        # lidar streaming (its CP2102 then hangs until the cable is replugged).
        steps = [("底盘停车", lambda: robot["base"].stop()), ("雷达", lambda: robot["lidar"].close()),
                 ("相机", lambda: robot["head"].close()), ("IMU", lambda: robot["imu"].close()),
                 ("底盘", lambda: robot["base"].close())]
        for name, fn in steps:
            try:
                fn()
            except KeyError:
                pass
            except BaseException as err:  # noqa: B902 - a second Ctrl-C included
                print(f"  关{name}时出错：{err!r}")
        writer.set()
        view.write(MAP_DIR, running=False)
        if "mover" in robot:  # the next run can start from this map (saved_map.py)
            from saved_map import save
            print(f"地图存到 {save(view.cmap, robot['mover'])}")
        print(f"底盘已停。状态页仍开着：http://{local_ip()}:{args.port}")


def work(args, view, robot):
    from imu import Imu
    from lidar_scan import Lidar
    from vln_scene import Head, nav

    head = Head(args.model, args.timeout_s, os.path.join(MAP_DIR, "look", time.strftime("%Y%m%d_%H%M%S")))
    robot["head"] = head
    base = nav.Base(*head.neck_home)
    robot["base"] = base
    head.attach(base)
    try:
        imu = Imu(args.imu_port)
    except RuntimeError as err:
        raise SystemExit(f"{err}。检查 IMU 的 USB 线。没有 IMU 不能跑")
    robot["imu"] = imu
    print(f"IMU：{imu.port}")
    lidar = Lidar(args.lidar_port)
    robot["lidar"] = lidar
    try:
        lidar.start()
    except Exception as err:
        raise SystemExit(f"雷达没打开：{err}。检查雷达 USB 和电机；定位靠雷达，没有雷达不能跑")
    print(f"雷达：{lidar.port}")
    head.lidar_xy = lidar.latest_xy
    cmap = view.cmap
    mover = Mover(base, imu, lidar, cmap, view)
    robot["mover"] = mover
    mover.attach_imu()
    warm_up(mover, head)
    print("阶段 2/3 探索：深度优先去未知区域，每到新地方自转一圈补图、给物体建模")
    explore(mover, head)
    print("阶段 3/3 清扫：按地图让车身扫过能到的空地")
    result = sweep(mover, head)
    msg = "清扫完成" if result == "done" else f"清扫未达到 {GOAL_SWEEP:.0%}，需要检查到不了的区域或障碍"
    mover.show(msg)
    print(msg)


if __name__ == "__main__":
    main()
