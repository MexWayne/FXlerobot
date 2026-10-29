#!/usr/bin/env python3
"""雷达建图、深度优先探索、清扫。

  conda activate lerobot_env
  cd ~/workspaces/FXLerobot/FXlerobot/vln
  python coverage_run.py      # 把车放进场地里，直接开始

状态页：浏览器打开 http://<jetson>:8765 。Ctrl-C 停车。

1) 建图：原地慢转两圈，只用雷达建占据栅格；再分 6 个方向各停一次看一眼，用视觉认出雷达点云是什么
   （桌子、凳子……），补上雷达看不到的东西（桌面、贴地的电线和垃圾、落地玻璃）。
2) 探索：深度优先，去已知空地和未知区域的交界。每到一个新地方自转一圈：雷达补图，视觉给物体建模。
3) 清扫：按地图让车身扫过所有能到的空地，到 90% 结束。
定位：IMU 管转向；每次停车用雷达点云和已建的地图匹配，校正位置和航向。
行驶：每走 0.35 m 停车看一次：抬头看桌面这类伸在上方的东西，低头看地上的东西、电线、插线板和障碍。
VLM 只认出是什么（桌子就是桌子，柜子就是柜子）；位置和大小由双目和雷达量。每件东西按量到的点拟合
成矩形（长、宽、走向），加上量到的高度，放进地图；要夹起来的小东西单独列出，车绕开它们走。
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
from occmap import BODY_CLEARANCE_M, BODY_LENGTH_M, BODY_WIDTH_M, OccMap  # noqa: E402
from vln_scene import CLUSTER_GAP_M, spans  # noqa: E402

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
SLIP0 = 0.85          # real / commanded distance on this carpet before the lidar has measured it
SLIP_MIN_M = 0.06     # moves shorter than this do not update it
STUCK_FRAC = 0.25     # a move covering less than this share of the command did not move
STUCK_MOVES_MAX = 3   # stop the whole run after this many stuck moves in a row
LOCATE_TURNS = 2      # lidar turns matched against the map at each stop
MATCH_WIN_M = 0.10    # pose search window: this much plus MATCH_WIN_PER_M of the distance driven
MATCH_WIN_PER_M = 0.5
MATCH_YAW_DEG = 3.0
MATCH_OK_M = 0.04     # mean distance of scan points to the map of a fit that is trusted
LOOK_M = 0.35         # stop and look after this much straight driving
LOOK_FRESH_S = 3.0    # a look older than this cannot veto a turn
STOP_MARGIN = 0.08    # keep this much floor between the body and anything seen ahead
TURN_SPARE = 0.05     # anything inside the turning circle plus this stops a turn
TURN_LOOK_DEG = 30.0  # turn in steps the head (NECK_SIDE_MAX plus half the camera view) can look ahead of
TURN_CHECK_DEG = 15.0  # smaller turns barely move the corners
NECK_SIDE_MAX = 15.0  # wider head turns bring the body's own uprights into view
PERSON_LIFE_S = 20.0  # people walk away; their marks expire
STUCK_LIFE_S = 120.0  # a spot the wheels could not cross blocks this long
CLEAR_GAP_M = 0.10    # floor the stereo saw this close to something it saw does not clear marks
THIN_KEYS = ("线", "绳", "插线板", "充电器")  # flat and long: block where measured, no rectangle
OBJECT_KEEP = 1500    # points kept per modelled thing (3 cm apart: a few square metres of furniture)
FIT_NOISE_M = 0.02    # a point this close to a side of its rectangle counts as on it (lidar range noise)
EXPLORE_VIEW_M = 1.0  # unknown floor this close to a spot counts for going there
EXPLORE_MIN_GAIN = 6
SURVEY_BAN_M = 0.5    # after a survey turn no new goal this close to it
GOAL_TRIES = 2
STILL_LEGS_MAX = 4    # legs in a row that leave the body where it was: look again at what pins it
RECHECK_M = 0.8
MAX_LEGS = 400
GOAL_SWEEP = 0.90
TRAIL_KEEP = 60
RETREAT_STEP_M = 0.10
RETREAT_MAX_M = 1.5
BACK_SLANT_DEG = 30.0  # reverse slanted: on the three omni wheels a straight reverse stalls on one wheel
BACK_PIECE_M = 0.08
BACK_OFF_MAX_M = 0.60
ESCAPE_MAX_M = 1.0
LOG_KEEP = 5


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def _neck(d_yaw):
    """Head yaw kept where the body's own uprights stay out of the picture (straight or NECK_SIDE_MAX)."""
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
        self.stuck = False
        self.stuck_moves = 0
        self.trail = []         # straight moves made, oldest first: (x0, y0, yaw, rel)
        self.turn_refused = False
        self.stage = ""
        self.look = None        # (pose, time, base-frame points) of the last look
        self.objects = []       # things the cameras named and measured, map frame (remember)
        self.next_id = 0
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
    def drive(self, rel, dist, status, record=True, guard=True):
        """Move `dist` m towards `rel` rad from the heading, holding the heading (omni wheels).

        The lidar watches the strip the body is about to cover while it moves (only in its view,
        not behind). Afterwards the pose comes from matching the lidar against the map, which says
        how far the body really went. Returns those metres.
        """
        if dist < 0.01:
            return 0.0
        if record:
            self.trail.append((self.x, self.y, self.yaw, rel))
            del self.trail[:-TRAIL_KEEP]
        x0, y0 = self.x, self.y
        target = self.hold if self.hold is not None else self.imu_yaw()
        vx, vy = LIN * math.cos(rel), LIN * math.sin(rel)
        t0 = last = time.time()
        span = dist / LIN
        try:
            while True:
                now = time.time()
                if now - t0 >= span:
                    break
                self._need_imu()
                self.yaw = self.imu_yaw()
                err = wrap_deg(math.degrees(target - self.yaw))
                if abs(err) > HOLD_ABORT_DEG:
                    self.base.stop()
                    raise SystemExit(f"直行时航向突然偏了 {err:+.0f}°（IMU 跳变 {self.imu.jumps} 次），停车")
                ahead = self.slip * LIN * (now - last)
                last = now
                self.x += ahead * math.cos(self.yaw + rel)
                self.y += ahead * math.sin(self.yaw + rel)
                if guard and self._lidar_cut(rel, dist * self.slip - math.hypot(self.x - x0, self.y - y0)):
                    break
                # The gyro term brakes a swing before the angle error has grown.
                theta = max(-HOLD_MAX_RATE, min(HOLD_MAX_RATE, HOLD_KP * err - HOLD_KD * self.imu.rate_dps()))
                self.base.base.send_action({"x.vel": vx, "y.vel": vy, "theta.vel": theta})
                self.show(status)
                time.sleep(0.05)
        finally:
            self.base.stop()
        commanded = LIN * min(time.time() - t0, span)
        time.sleep(0.1)
        located = self.locate(MATCH_WIN_M + MATCH_WIN_PER_M * commanded)
        a = target + rel
        moved = (self.x - x0) * math.cos(a) + (self.y - y0) * math.sin(a)
        if located and commanded >= SLIP_MIN_M:
            self.slip = min(1.2, max(0.3, 0.5 * self.slip + 0.5 * moved / commanded))
        self.stuck = located and commanded >= SLIP_MIN_M and moved < STUCK_FRAC * commanded
        if self.stuck:
            self.stuck_moves += 1
            print(f"  命令 {commanded:.2f} m，雷达量到只走了 {moved:.2f} m，车卡住了", flush=True)
            if self.stuck_moves >= STUCK_MOVES_MAX:
                raise SystemExit(f"车连着 {self.stuck_moves} 次动不了：轮子在转，雷达看车几乎没动。"
                                 "检查电池电量、轮子是否打滑、底盘有没有被线或家具卡住")
        elif moved > 0.03:
            self.stuck_moves = 0
        self._passed_line(x0, y0)
        return max(0.0, moved)

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
        if fit > MATCH_OK_M:
            print(f"  雷达和地图对不上（平均差 {fit * 100:.0f} cm），位置按推算", flush=True)
            return False
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
def look(mover, head, d_yaw=0.0, near_only=False):
    """One stop's look ahead. The map takes what it measured; returns the hits (base frame).

    Floor the stereo measured clears older camera marks there. Floor-standing glass stops lidar
    rays; people block for PERSON_LIFE_S; small things to pick up are marked apart and driven round;
    everything else blocks until the cameras see the floor free again. Named things are modelled
    (remember) and their rectangles block as well.
    """
    pose = mover.pose
    mover.show("停车看前方")
    hits = head.lane(d_yaw=d_yaw, near_only=near_only, where=pose)
    now = time.monotonic()
    mover.look = (pose, now, [p for h in hits for p in h.points])
    cmap = mover.cmap
    marks = [to_map(pose, p) for h in hits for p in h.points]
    floor = [to_map(pose, p) for p in head.last_floor]
    if floor:
        cmap.mark_floor(floor)
        seen = cmap.mask_of(floor, 0.0)
        if marks:
            seen &= ~cmap.mask_of(marks, CLEAR_GAP_M)
        n = cmap.clear_vis(seen)
        if n:
            print(f"  双目看到这些地方是空地，清掉 {n} 格以前标的障碍", flush=True)
    for h in hits:
        pts = [to_map(pose, p) for p in h.sure]
        if h.kind == "glass":
            cmap.mark_glass(pts)
        elif h.kind == "trash":
            cmap.mark_trash(pts)
        elif h.kind == "person":
            cmap.block(pts, 0.05, now + PERSON_LIFE_S)
        else:
            cmap.block(pts, 0.03, math.inf)
        if h.kind in ("floor", "overhead", "trash") and pts:
            remember(mover, h, pts)
    block_objects(mover)
    show_objects(mover)
    return hits


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


def _joins(obj, arr):
    """True when look points `arr` belong to modelled thing `obj` (same name checked by the caller):
    they come within CLUSTER_GAP_M of its points, or, for a thing on legs, fall inside its rectangle
    (looks from under a table see other legs of it)."""
    old = np.asarray(obj["points"])
    if np.sqrt(((old[:, None, :] - arr[None, :, :]) ** 2).sum(-1)).min() <= CLUSTER_GAP_M:
        return True
    if obj["spans"] and len(obj["rect"]) == 4:
        rect = np.asarray(obj["rect"], np.float32).reshape(-1, 1, 2)
        return any(cv2.pointPolygonTest(rect, (float(x), float(y)), True) >= -CLUSTER_GAP_M for x, y in arr)
    return False


def remember(mover, hit, pts):
    """Model a named thing from what one look measured: its points (map frame), how high it is, and
    from those its rectangle, length, width and heading.

    A look joins a known thing of the same name it belongs to (_joins); one that belongs to several
    makes them one. Only measured points shape a thing: of a cabinet seen from the front the
    rectangle is its front until another look measures its side.
    """
    arr = np.asarray(pts, np.float64)
    joins = [o for o in mover.objects if o["label"] == hit.label and _joins(o, arr)]
    if joins:
        obj = joins[0]
        for other in joins[1:]:
            obj["points"] += other["points"]
            obj["seen"] += other["seen"]
            if other["height"] is not None and (obj["height"] is None or other["height"] > obj["height"]):
                obj["height"], obj["top_cut"] = other["height"], other["top_cut"]
            mover.objects.remove(other)
        if len(joins) > 1:
            print(f"  这次量到的{hit.label}连上了已知的 {len(joins)} 个，是同一个", flush=True)
    else:
        mover.next_id += 1
        obj = {"id": mover.next_id, "label": hit.label, "points": [], "seen": 0, "height": None,
               "top_cut": False, "trash": hit.kind == "trash", "spans": spans(hit.label),
               "thin": any(k in hit.label for k in THIN_KEYS)}
        mover.objects.append(obj)
    obj["points"] = _thin(obj["points"] + list(pts))
    obj["seen"] += 1
    if hit.height is not None and (obj["height"] is None or hit.height > obj["height"]):
        obj["height"], obj["top_cut"] = hit.height, hit.top_cut
    obj["rect"], obj["length"], obj["width"], obj["heading"] = _rect(obj["points"])


def block_objects(mover):
    """Rebuild the map layer of the modelled things from their rectangles.

    A thing on legs blocks its whole rectangle: the body cannot go under its top, though the floor
    there is bare. Of any other body, floor the stereo measured bare inside its rectangle, away from
    its own points, is not part of it. Thin things and things to pick up block only where measured
    (look marks those). The cells the body stood on never block (OccMap.rebuild).
    """
    cmap = mover.cmap
    layer = np.zeros((cmap.h, cmap.w), bool)
    for obj in mover.objects:
        if obj["trash"] or obj["thin"]:
            continue
        one = np.zeros((cmap.h, cmap.w), np.uint8)
        px = np.array([[(x - cmap.origin[0]) / cmap.res - 0.5, (y - cmap.origin[1]) / cmap.res - 0.5]
                       for x, y in obj["rect"]])
        cv2.fillConvexPoly(one, np.round(px).astype(np.int32), 1)
        one = one > 0
        if not obj["spans"]:
            one &= ~(cmap.floor_seen & ~cmap.mask_of(obj["points"], CLUSTER_GAP_M))
        layer |= one
    cmap.objects = layer
    cmap.rebuild()


def show_objects(mover):
    # A thing to pick up that is gone from the map (the stereo saw bare floor there) is gone from the page.
    mover.objects = [o for o in mover.objects
                     if not o["trash"] or any(mover.cmap.is_trash(x, y) for x, y in o["points"])]
    with mover.view.lock:
        mover.view.objects = [{"id": o["id"], "label": o["label"], "rect": o["rect"],
                               "x": float(np.mean([p[0] for p in o["points"]])),
                               "y": float(np.mean([p[1] for p in o["points"]])),
                               "length": o["length"], "width": o["width"], "heading": o["heading"],
                               "height": o["height"], "top_cut": o["top_cut"], "trash": o["trash"],
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
    near = []
    xy = mover.lidar.latest_xy()
    if xy is not None:
        near += [(float(x), float(y)) for x, y in xy if math.hypot(float(x), float(y)) <= limit]
    if mover.look is not None:
        pose, t, pts = mover.look
        if (time.monotonic() - t < LOOK_FRESH_S and math.hypot(mover.x - pose[0], mover.y - pose[1]) < 0.03
                and abs(wrap_deg(math.degrees(mover.yaw - pose[2]))) < 3.0):
            near += [p for p in pts if math.hypot(*p) <= limit]
    near += _occupied_near(mover, BODY_CLEARANCE_M + 0.02)[:1]
    return min(near, key=lambda p: math.hypot(*p)) if near else None


def map_room(mover, rel, limit):
    """Metres the body can move towards `rel` before it covers an occupied map cell it is not on now."""
    cmap = mover.cmap
    occ = cmap.occupied() & ~cmap.body_mask(*mover.pose)
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

    Returns "done", or why it stopped: "turn" (no room to turn), "blocked" (something ahead),
    "stuck" (the wheels did not move the body) or "replan" (the map closed the way).
    """
    for tx, ty in route[1:]:
        for _ in range(200):
            d = math.hypot(tx - mover.x, ty - mover.y)
            if d < 0.04:
                break
            heading = math.atan2(ty - mover.y, tx - mover.x)
            if abs(wrap_deg(math.degrees(heading - mover.yaw))) > 4.0:
                if not safe_turn(mover, head, heading, status):
                    return "turn"
            mover.hold = heading
            hits = look(mover, head)
            free = free_ahead(mover, hits)
            step = min(LOOK_M, d, free)
            if step < 0.03:
                print(f"  前方只剩 {free:.2f} m，换路", flush=True)
                return "blocked"
            mover.drive(0.0, step, status)
            if mover.stuck:
                spot = to_map(mover.pose, (BODY_LENGTH_M / 2.0 + 0.08, 0.0))
                mover.cmap.block([spot], 0.06, time.monotonic() + STUCK_LIFE_S)
                mover.cmap.rebuild()
                return "stuck"
            cell = mover.cmap.world_to_cell(tx, ty)
            if cell is None or not mover.cmap.drive_mask[cell]:
                return "replan"
    return "done"


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


def recheck(mover, head):
    """The body has not got away for several legs: drop the camera marks around it and look again.

    Lidar marks stay (the lidar is measuring them on every stop); camera marks may be old or from a
    mismatch, and the looks put back what is still there.
    """
    cmap = mover.cmap
    near = cmap.mask_of([(mover.x, mover.y)], RECHECK_M) & ~cmap.glass
    n = cmap.clear_vis(near)
    print(f"  连续 {STILL_LEGS_MAX} 段没离开这里：清掉周围 {n} 格视觉标记，重新看", flush=True)
    for d_yaw in (-NECK_SIDE_MAX, 0.0, NECK_SIDE_MAX):
        look(mover, head, d_yaw=d_yaw, near_only=True)


def explore(mover, head):
    """Depth first to the frontier between known free floor and unknown, with a survey turn at
    every new place. Ends when no frontier can be reached."""
    mover.set_stage("explore")
    cmap = mover.cmap
    branches, tries = [], {}
    said = {"ahead": "接着往前方的新地方走", "branch": "前面走到头了，退回最近的岔路",
            "nearest": "岔路都走完了，去最近的未知边缘"}
    for leg in range(MAX_LEGS):
        frontier = cmap.frontier()
        goal = None
        for gain in (EXPLORE_MIN_GAIN, 1):
            goal, how, reach, cut = explore_goal(cmap, mover.x, mover.y, mover.yaw, branches, frontier,
                                                 EXPLORE_VIEW_M, gain)
            if goal is not None:
                break
        if goal is None:
            print(f"阶段 2/3 探索完：能到的未知边缘都看过了（到不了的还有 {cut} 格）")
            return
        gx, gy = cmap.cell_center(goal)
        print(f"探索第 {leg + 1} 段：深度优先，{said[how]} ({gx:+.2f}, {gy:+.2f})，"
              f"还有 {reach} 格未知边缘能去", flush=True)
        if not go(mover, head, goal, tries, "探索"):
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
        for gain in (4, 1):
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
