#!/usr/bin/env python3
"""预热、探索、清扫。

  conda activate lerobot_env
  cd ~/workspaces/FXLerobot/FXlerobot/vln
  python coverage_run.py            # 把车放进场地里，直接开始
  python coverage_run.py --dry-run  # 电机和相机都不用，在一个假围栏里演示

状态页：浏览器打开 http://<jetson>:8765 ，画围栏、已清扫的地面和看到的障碍。Ctrl-C 停车。

1) 预热：原地先雷达转一圈、再视觉转一圈，得到初始边界。然后像探索一样把场地走一遍，
尽量把图补全。每到一个新地方，先缓慢转一圈用雷达核对，再缓慢转一圈用视觉改玻璃和镜子。
2) 探索：预热边界只是起点，走的过程中会改。看到空地或车开过去，边界往外推；
实体墙一旦被视觉确认，就用雷达把该方向改到实测距离；玻璃和镜子改到视觉底边。
这一遍走过的格子不算清扫。
3) 清扫：按探索修正后的地图，遍历可达的可清洁地面。还有没去过的地方就去最近的一块，
去过的占比达到 90% 才算完成。转角和直线用 IMU 保持。
"""

import argparse
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from coverage_map import (BLOCKED, BODY_CLEARANCE_M, BODY_DONE, BODY_REACH, BODY_HEIGHT_M, BODY_LENGTH_M, BODY_WIDTH_M,  # noqa: E402
                          PLAN_MARGIN_M, UNKNOWN, CleanMap, dist_to_boundary, point_in_poly)
from coverage_plan import ban_goal, cleanup_goal, drive_dist, line_clear, route_to, todo_rows  # noqa: E402
from live_map import MapView  # noqa: E402

MAP_DIR = os.path.join(HERE, "map")

LIN = 0.08          # m/s, straight moves
ROT = 25.0          # deg/s, fastest turn in place
TURN_MIN_RATE = 8.0   # deg/s, slowest turn command that still moves the wheels
TURN_KP = 1.5         # deg/s of turn rate per degree still to go
TURN_TOL_DEG = 1.0    # a turn ends once the IMU says it is this close and has settled
TURN_SETTLE_S = 0.25  # wait after stopping, then read the IMU again
HOLD_KP = 3.0         # deg/s of correction per degree of heading error while driving straight
HOLD_KD = 0.5         # deg/s of correction per deg/s the gyro already shows
HOLD_ABORT_DEG = 60.0  # a heading error this big while driving is a sensor fault, not drift
HOLD_MAX_RATE = 15.0  # deg/s, most the heading hold may rotate while driving
WIGGLE_RATE = 20.0    # deg/s and seconds of the first turn, only used to find which way the IMU counts
WIGGLE_S = 0.9        # the wheels need about half a second to get going, so this is longer than it looks
SEG_M = 0.15        # commanded metres between two floor-texture measurements of the real motion
SEG_MIN_M = 0.08    # shorter segments are too small for the texture to measure
WALL_RANGE_M = 1.5  # walls further away than this are not used to move the outline: the floor estimate is rough
WALL_GAIN_M = 0.10  # a wall this far from the outline moves it; closer than that is just noise
SURVEY_GAP_M = 1.5          # warm-up spins again after the robot has moved at least this far
WARMUP_LIDAR_DPS = 12.0      # deg/s of a warm-up turn; slow enough for several lidar revolutions
WARMUP_LIDAR_SAMPLE_S = 0.25
OPTICAL_RANGE_M = 6.0        # a seen glass/mirror bottom edge may set the boundary out to here
EXPLORE_WALL_M = 12.0        # once vision names a solid wall, lidar may move that bearing out to here
FENCE_GROW_M = 2.0   # the map leaves this much room around the scanned outline for the fence to grow into
FENCE_BIN_DEG = 5.0  # the fence is kept as one distance per bearing this wide
FENCE_PAST_M = 0.10  # floor seen free puts the fence at least this far past it
CLEAR_FLOOR_M = 1.0  # a clear lane check vouches for the floor this far ahead
NEAR_FLOOR_M = 0.6   # the same for the steep single-angle checks before turns and slides
STUCK_MOVES_MAX = 3  # stop the whole run after this many stuck moves in a row
STUCK_FRAC = 0.25   # a segment that moves the body less than this share of the command is "not moving"
LOOK_M = 0.35       # stop and look ahead after this much straight driving
SCAN_TURNS = 8      # initial scan: 8 x 45 deg
STOP_MARGIN = 0.08  # keep this much floor between the body front and anything seen ahead
OBSTACLE_R = 0.10   # assumed half size of an obstacle around its floor contact point
MAX_LEGS = 300
GOAL_SWEEP = 0.90   # report success from here
LIDAR_FUSED_HALF_FOV_DEG = 40.0  # only map lidar returns in the sector the camera just checked
# The black uprights on both sides of the body come into view once the head turns much past this,
# so wider side views turn the body instead.
NECK_SIDE_MAX = 15.0
NECK_SCAN = (-NECK_SIDE_MAX, 0.0, NECK_SIDE_MAX)  # head yaw offsets before the base may turn
ESCAPE_DIRS = (0.0, 30.0, -30.0, 60.0, -60.0)  # slide directions the head can see, deg, + is left
ESCAPE_MAX_M = 1.0
NECK_PEEK = NECK_SIDE_MAX  # head yaw for a second look when a scan direction finds no fence
MAX_GAP_DEG = 100.0  # unseen bearings this wide are filled from the nearer neighbour
OPEN_RANGE_M = 1.5   # a direction with no fence and nothing in the way is bounded here for now
MARK_RANGE = 1.0     # obstacles further than this are not drawn: the floor estimate near the image top is poor
FENCE_MARK_RANGE = 0.6  # fence contacts seen closer than this are drawn; further ones are already in the map
LOCATE_EVERY_M = 1.2    # look at the fence and correct the pose after this much driving
LOCATE_WIN = (0.25, 10.0)   # metres, degrees searched while driving
START_WIN = (0.60, 30.0)    # where the robot may stand compared with the saved start
LOCATE_OK_M = 0.05      # mean fence distance of a fit that is trusted
LOCATE_GAIN_M = 0.01    # a correction has to improve the fit by this much
LOCATE_PRIOR = (0.12, 0.003)  # cost per metre / degree of correction while driving: 0.25 m or 10 deg = 3 cm
START_PRIOR = (0.02, 0.0005)  # weak: the robot may really have been put down 0.5 m off
LOCATE_MIN_PTS = 15
TURN_LOOK_DEG = 30.0   # turn in steps the head (NECK_SIDE_MAX plus half the camera view) can look ahead of
TURN_CHECK_DEG = 15.0  # smaller turns barely move the corners
TURN_SPARE = 0.05      # anything inside the turning circle plus this stops the turn
BLIND_SLIDE_M = 0.10   # slides up to this long are not looked at first
SLIDE_LOOK_DEG = 35.0  # the head (neck yaw +-NECK_SIDE_MAX) sees slides this close to straight ahead
GOAL_TRIES = 2         # fill-in attempts per goal before it is given up
ROW_TRIES = 2          # blocked attempts per sweep row before it is given up
TRAIL_KEEP = 60        # straight moves remembered for backing out
RETREAT_STEP_M = 0.10  # back out in steps this long, checking after each whether the body can turn
RETREAT_MAX_M = 1.5
TURN_BACK_M = 0.20  # least back-out after the camera refused a turn
LANE_M = 0.30       # after a right-angle turn, drive this far to the next row
ROW_LIMIT_M = 14.0  # one row stops here even if the camera still says the lane is open
MAX_LANES = 30


def area_spec(boundary):
    return {
        "resolution_m": 0.05,
        "body_length_m": BODY_LENGTH_M,
        "body_width_m": BODY_WIDTH_M,
        "body_height_m": BODY_HEIGHT_M,
        "body_brush_m": BODY_WIDTH_M,
        "arm_brush_m": 0.16,
        "start": [0.0, 0.0, 0.0],
        "boundary": boundary,
        "furniture": [],
    }


def placeholder_map():
    """Grey 3 m x 3 m square around the robot, shown while the scan is running."""
    half = 12.0
    cmap = CleanMap(area_spec([[-half, -half], [half, -half], [half, half], [-half, half]]))
    cmap.grid[:] = UNKNOWN
    cmap.plan_mask[:] = False
    cmap.drive_mask[:] = False
    cmap.boundary = []
    return cmap


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def _neck(d_yaw):
    """Head yaw kept where the body's own uprights stay out of the picture.

    Only straight or NECK_SIDE_MAX to a side, the poses Head.learn_self knows the body in.
    """
    if abs(d_yaw) < NECK_SIDE_MAX / 2:
        return 0.0
    return math.copysign(NECK_SIDE_MAX, d_yaw)


class Mover:
    """Dead reckoning from commanded velocity and time, the way vln_nav.Base.turn works."""

    def __init__(self, cmap, view, base, speedup=1.0):
        self.cmap = cmap
        self.view = view
        self.base = base
        self.speedup = speedup
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.travel = 0.0       # metres driven or slid, which is what makes the pose drift
        self.located_at = 0.0   # self.travel at the last fence fit
        # Straight moves already made, oldest first: (x0, y0, yaw, rel). The body swept that floor,
        # so backing out along them is safe even where the camera cannot look.
        self.trail = []
        self.turn_refused = False
        self.imu = None         # set by attach_imu; None means turns and lines are open loop
        self.yaw0_deg = 0.0     # IMU yaw at the start, so self.yaw begins at 0
        self.hold = None        # heading (rad) the straight moves keep; None: the heading each move starts with
        self.head = None        # the camera; with it, distances come from the floor texture, not wheel time
        self.eff = 1.0          # recent ratio of real to commanded distance
        self.stuck = False      # the last move barely moved the body
        self.stuck_moves = 0    # such moves in a row

    def _imu_yaw(self):
        return math.radians(self.imu.yaw_deg() - self.yaw0_deg)

    def _need_imu(self):
        if not self.imu.fresh():
            self.base.stop()
            raise SystemExit(f"IMU 数据中断 {self.imu.age():.1f} s，停车")

    def attach_imu(self, imu):
        """Use the IMU for the heading. A short first turn shows which way its yaw counts."""
        if not imu.wait_ready():
            raise SystemExit("IMU 没有数据，停车")
        a0 = imu.yaw_deg()
        t_end = time.time() + WIGGLE_S
        try:
            while time.time() < t_end:
                self.base.base.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": WIGGLE_RATE})
                time.sleep(0.05)
        finally:
            self.base.stop()
        time.sleep(0.4)
        moved = imu.yaw_deg() - a0
        expect = WIGGLE_RATE * WIGGLE_S
        if abs(moved) < 0.25 * expect:
            raise SystemExit(f"车没有转起来：命令左转约 {expect:.0f}°，IMU 只变了 {moved:+.1f}°。"
                             "车可能被卡住、轮子打滑或电机没出力，也可能 IMU 没固定在车身上")
        imu.sign = 1.0 if moved > 0 else -1.0
        self.yaw0_deg = imu.sign * a0
        self.imu = imu
        self.yaw = self._imu_yaw()
        print(f"IMU 方向：车向左转时模块 yaw {'增大' if imu.sign > 0 else '减小'}（已按车的左为正处理），"
              f"命令约 {expect:.0f}°，实测 {abs(moved):.1f}°")
        self.turn_to(0.0, "回到出发方向")

    def _record(self, x0, y0, rel):
        self.trail.append((x0, y0, self.yaw, rel))
        del self.trail[:-TRAIL_KEEP]

    def move_frame(self, old, new):
        """Re-express the trail after the pose was corrected from `old` to `new`."""
        d = new[2] - old[2]
        c, s = math.cos(d), math.sin(d)
        self.trail = [(new[0] + c * (x - old[0]) - s * (y - old[1]),
                       new[1] + s * (x - old[0]) + c * (y - old[1]), yaw + d, rel)
                      for x, y, yaw, rel in self.trail]

    def show(self, status=None, remaining=None):
        with self.view.lock:
            self.view.robot = (self.x, self.y, self.yaw)
            if status is not None:
                self.view.status = status
            if remaining is not None:
                self.view.remaining = remaining

    def _run(self, duration, x_vel, y_vel, on_tick):
        """Drive at a fixed body velocity for `duration` s. With the IMU the turn rate is
        corrected every tick so the heading stays on `self.hold` (or where the move began);
        self.yaw is the measured heading when on_tick is called."""
        t0 = time.time()
        span = duration / self.speedup
        live = self.imu is not None and self.base is not None
        target = None
        if live:
            self.yaw = self._imu_yaw()
            target = self.hold if self.hold is not None else self.yaw
        try:
            while True:
                el = time.time() - t0
                if el >= span:
                    break
                theta = 0.0
                if live:
                    self._need_imu()
                    self.yaw = self._imu_yaw()
                    err = wrap_deg(math.degrees(target - self.yaw))
                    if abs(err) > HOLD_ABORT_DEG:
                        self.base.stop()
                        raise SystemExit(f"直行时航向突然偏了 {err:+.0f}°（IMU 跳变 {self.imu.jumps} 次），停车")
                    # The gyro term brakes a swing before the angle error has grown.
                    theta = max(-HOLD_MAX_RATE, min(HOLD_MAX_RATE,
                                                    HOLD_KP * err - HOLD_KD * self.imu.rate_dps()))
                if self.base is not None:
                    self.base.base.send_action({"x.vel": x_vel, "y.vel": y_vel, "theta.vel": theta})
                on_tick(el * self.speedup)
                time.sleep(0.05)
        finally:
            if self.base is not None:
                self.base.stop()
        if live:
            time.sleep(0.1)
            self.yaw = self._imu_yaw()
            drift = wrap_deg(math.degrees(target - self.yaw))
            if abs(drift) > 2.0:
                print(f"  直行结束航向偏 {drift:+.1f}°")
        on_tick(duration)

    def turn_by(self, deg, status=None):
        if abs(deg) < 1.0:
            return
        if self.imu is not None and self.base is not None:
            self._turn_imu(deg, status)
            return
        yaw0 = self.yaw
        rate = math.copysign(ROT, deg)

        def tick(el):
            self.yaw = yaw0 + math.radians(rate * min(el, abs(deg) / ROT))
            self.show(status)

        self._run_turn_open(abs(deg) / ROT, rate, tick)
        self.yaw = yaw0 + math.radians(deg)
        self.hold = None

    def _run_turn_open(self, duration, rate, on_tick):
        t0 = time.time()
        span = duration / self.speedup
        try:
            while True:
                el = time.time() - t0
                if el >= span:
                    break
                if self.base is not None:
                    self.base.base.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": rate})
                on_tick(el * self.speedup)
                time.sleep(0.05)
        finally:
            if self.base is not None:
                self.base.stop()
        on_tick(duration)

    def _turn_imu(self, deg, status):
        """Turn until the IMU heading is within TURN_TOL_DEG of the goal and has settled."""
        start = self._imu_yaw()
        target = start + math.radians(deg)
        t0 = time.time()
        limit = abs(deg) / ROT * 3.0 + 4.0
        try:
            while True:
                self._need_imu()
                self.yaw = self._imu_yaw()
                err = wrap_deg(math.degrees(target - self.yaw))
                self.show(status)
                if abs(err) <= TURN_TOL_DEG:
                    self.base.stop()
                    time.sleep(TURN_SETTLE_S)
                    self.yaw = self._imu_yaw()
                    err = wrap_deg(math.degrees(target - self.yaw))
                    if abs(err) <= TURN_TOL_DEG:
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
        self.yaw = self._imu_yaw()
        self.hold = None
        print(f"  转 {deg:+.0f}°：IMU 实测 {math.degrees(self.yaw - start):+.1f}°，"
              f"离目标 {wrap_deg(math.degrees(target - self.yaw)):+.1f}°")

    def turn_to(self, heading, status=None):
        self.turn_by(wrap_deg(math.degrees(heading - self.yaw)), status)

    def _advance(self, rel, d):
        """Move the pose `d` m towards `rel` from the heading and paint the floor the body crossed."""
        a = self.yaw + rel
        steps = max(1, int(d / 0.03))
        code = {"warmup": 1, "explore": 2, "sweep": 3}.get(getattr(self, "stage", ""), 0)
        radius = self.cmap.body_brush * 0.5 if code == 3 else 0.07
        for k in range(1, steps + 1):
            self.x += d / steps * math.cos(a)
            self.y += d / steps * math.sin(a)
            self.cmap.stamp_body(self.x, self.y)
            if code:
                self.cmap.mark_pass(self.x, self.y, code, radius)
        observe_body(self, self.cmap)

    def set_stage(self, name):
        """Which pass is driving: warmup, explore, or sweep. The page colours the trail from this."""
        self.stage = name
        with self.view.lock:
            self.view.stage = name

    def _translate(self, rel, dist, status):
        """Move `dist` m towards `rel` radians from the heading, holding the heading. Returns the
        metres the body really moved.

        The wheels turn at the commanded speed even when the robot slips or is held, so wheel
        time says little about distance. With a camera the move is cut into short segments; after
        each one the floor texture in the frames before and after gives the real displacement,
        and only that moves the pose and paints the map. Two segments in a row that barely move
        the body mean it is stuck (self.stuck).
        """
        self.stuck = False
        vx, vy = LIN * math.cos(rel), LIN * math.sin(rel)
        if self.head is None or self.base is None:
            last = [0.0]

            def tick(el):
                d = min(el * LIN, dist)
                step, last[0] = d - last[0], d
                self._advance(rel, step)
                self.show(status)

            self._run(dist / LIN, vx, vy, tick)
            self.travel += dist
            return dist
        moved, driven, slow = 0.0, 0.0, 0
        while moved < dist - 0.02 and driven < 3.0 * dist + 0.20:
            seg = min(dist - moved, SEG_M)
            snap0 = self.head.snapshot()
            self._run(seg / LIN, vx, vy, lambda el: self.show(status))
            driven += seg
            if seg < SEG_MIN_M:
                got = seg * self.eff   # too short for the texture to measure: use the recent ratio
            else:
                found = self.head.floor_motion(snap0, self.head.snapshot())
                if found is None:
                    got = seg * self.eff
                    print(f"  地面纹理太少，这一小段 {seg:.2f} m 只能按以往的 {self.eff:.0%} 估计", flush=True)
                else:
                    got = max(0.0, found[0] * math.cos(rel) + found[1] * math.sin(rel))
                    self.eff = 0.5 * self.eff + 0.5 * min(1.5, got / seg)
                    slow = slow + 1 if got < STUCK_FRAC * seg else 0
                    print(f"  这一小段命令 {seg:.2f} m，地面纹理量到 {got:.2f} m（{got / seg:.0%}）", flush=True)
            got = min(got, 1.5 * seg)
            self._advance(rel, got)
            moved += got
            self.show(status)
            if slow >= 2:
                self.stuck = True
                print("  连着两小段几乎没动，车卡住了", flush=True)
                break
        if moved < dist - 0.02 and not self.stuck and driven >= 3.0 * dist + 0.20:
            self.stuck = True
            print(f"  命令了 {driven:.2f} m 只走了 {moved:.2f} m，车走不动", flush=True)
        self.travel += moved
        if self.stuck:
            self.stuck_moves += 1
            if self.stuck_moves >= STUCK_MOVES_MAX:
                raise SystemExit(f"车连着 {self.stuck_moves} 次动不了：轮子在转，地面纹理几乎没有位移。"
                                 "检查电池电量、轮子是否打滑、底盘有没有被线或家具卡住")
        elif moved > 0.05:
            self.stuck_moves = 0
        return moved

    def forward(self, dist, status=None, record=True):
        """Drive `dist` m (negative: reverse). Returns the metres really moved."""
        if abs(dist) < 0.01:
            return 0.0
        if record and dist > 0:
            self._record(self.x, self.y, 0.0)
        return self._translate(0.0 if dist > 0 else math.pi, abs(dist), status)

    def slide(self, rel, dist, status=None, record=True):
        """Move `dist` metres towards `rel` radians from the heading without turning (omni wheels)."""
        if dist < 0.01:
            return 0.0
        if record:
            self._record(self.x, self.y, rel)
        return self._translate(rel, dist, status)


def to_map(mover, fx, fy):
    c, s = math.cos(mover.yaw), math.sin(mover.yaw)
    return mover.x + c * fx - s * fy, mover.y + s * fx + c * fy


def to_robot(mover, mx, my):
    c, s = math.cos(mover.yaw), math.sin(mover.yaw)
    dx, dy = mx - mover.x, my - mover.y
    return c * dx + s * dy, -s * dx + c * dy


def look_without_turning(mover, head, view):
    """Fence points ahead and to both sides, turning only the head."""
    points = []
    for d_yaw in NECK_SCAN:
        mover.show(f"不转身，转头看 {d_yaw:+.0f}°")
        seen = head.fence_points(f"start{int(d_yaw):+d}", d_yaw=d_yaw)
        if seen is None:
            seen = head.fence_points(f"start{int(d_yaw):+d}b", d_yaw=d_yaw)
        if seen is None:
            print(f"  头偏 {d_yaw:+.0f}° 没看清，跳过")
            continue
        points.extend(to_map(mover, fx, fy) for fx, fy in seen)
        with view.lock:
            view.scan_points = list(points)
    return points


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


def find_turn_spot(mover, points):
    """Shortest slide, within what the head saw, to a spot where the body can turn in place.

    Moving at least 0.20 m away from the start also clears the floor behind, which the head
    cannot see: the body was standing on it.
    """
    local = [to_robot(mover, x, y) for x, y in points]
    need = BODY_CLEARANCE_M + PLAN_MARGIN_M
    best = None
    for deg in ESCAPE_DIRS:
        rel = math.radians(deg)
        for k in range(4, int(ESCAPE_MAX_M / 0.05) + 1):
            d = k * 0.05
            if swept_hit(local, rel, d):
                break
            px, py = d * math.cos(rel), d * math.sin(rel)
            if all(math.hypot(px - x, py - y) >= need for x, y in local):
                if best is None or d < best[1]:
                    best = (rel, d)
                break
    return best


def ensure_turn_room(mover, head, cmap):
    """If the body is too close to something to turn, slide to the nearest spot where it can.

    The slide goes where the camera has not necessarily looked, so beyond a few centimetres the
    head looks along it first, and directions the head cannot face are not slid into.
    """
    here = cmap.world_to_cell(mover.x, mover.y)
    if here is not None and cmap.plan_mask[here]:
        return
    move = escape_move(mover, cmap)
    if move is None:
        print("  附近没有能平移过去转身的地方，原地小心转身")
        return
    rel, dist = move
    print(f"  这里转不了身，先往 {math.degrees(rel):+.0f}° 平移 {dist:.2f} m")
    if head is not None and abs(math.degrees(rel)) <= SLIDE_LOOK_DEG and dist > BLIND_SLIDE_M:
        mover.show("平移前先看一眼")
        seen = head.lane(d_yaw=_neck(-math.degrees(rel)), near_only=True)
        if seen is None:
            return
        if seen[0] == "hit":
            points = seen[4] or [(seen[1], seen[2])]
            if swept_hit(points, rel, dist + STOP_MARGIN):
                print(f"  平移方向上有{seen[3]}，不平移")
                mark_seen(mover, cmap, (seen[1], seen[2]), points)
                return
    mover.slide(rel, dist, "平移到能转身的地方")


def known_points(mover, cmap, reach=1.2):
    """Marked obstacle cells and the fence line within `reach`, in the robot frame."""
    if getattr(cmap, "_fence_pts", None) is None:
        pts = []
        poly = cmap.boundary
        for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
            n = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / 0.04)))
            pts.extend((ax + (bx - ax) * k / n, ay + (by - ay) * k / n) for k in range(n))
        cmap._fence_pts = pts
    rows, cols = np.where(cmap.grid == BLOCKED)
    pts = list(cmap._fence_pts) + [cmap.cell_center((int(r), int(c))) for r, c in zip(rows, cols)]
    return [to_robot(mover, x, y) for x, y in pts if math.hypot(x - mover.x, y - mover.y) <= reach]


def escape_move(mover, cmap):
    """Shortest straight slide that keeps the body off everything on the map and ends on a turning spot.

    Forward-ish directions are ones the head can check; straight back is where the body came from.
    """
    local = known_points(mover, cmap)
    dirs = [math.radians(d) for d in (0, 15, -15, 30, -30, 45, -45, 180, 165, -165)]
    dead = set()
    for k in range(1, int(ESCAPE_MAX_M / 0.05) + 1):
        d = k * 0.05
        for rel in dirs:
            if rel in dead:
                continue
            if swept_hit(local, rel, d, margin=PLAN_MARGIN_M):
                dead.add(rel)
                continue
            a = mover.yaw + rel
            cell = cmap.world_to_cell(mover.x + d * math.cos(a), mover.y + d * math.sin(a))
            if cell is not None and cmap.plan_mask[cell]:
                return rel, d
    return None


def open_side(head):
    """A direction with no fence: the nearest thing in the way (chair, desk leg, legs...) bounds
    the area there; with nothing in the way, open floor up to OPEN_RANGE_M does."""
    print("  这个方向没有围栏，改找挡路的最近东西")
    got = head.lane()
    if got is None:
        print("  没看清，这个方向靠左右两个方向补")
        return []
    if got[0] == "hit":
        _, fx, fy, label, points = got
        points = points or [(fx, fy)]
        if min(math.hypot(px, py) for px, py in points) <= OPEN_RANGE_M:
            print(f"  这个方向用{label}当边界，离车 {fx:.2f} m")
            return points
    print(f"  这个方向 {OPEN_RANGE_M:.1f} m 内没东西挡路，边界先放在 {OPEN_RANGE_M:.1f} m，走的时候相机再看")
    return [(OPEN_RANGE_M * math.cos(math.radians(a)), OPEN_RANGE_M * math.sin(math.radians(a)))
            for a in range(-20, 21, 5)]


def _lidar_map_points(mover):
    """One lidar revolution in the map frame, or [] when the lidar has no fresh turn."""
    lidar = getattr(mover, "lidar", None)
    if lidar is None:
        return []
    xy = lidar.latest_xy()
    if xy is None:
        return []
    return [to_map(mover, float(x), float(y)) for x, y in xy]


def _bearing_bins(center, pts, bin_deg=FENCE_BIN_DEG):
    """Bin indices covered by `pts` in the map frame, including the span between them."""
    cx, cy = center
    n = int(round(360.0 / bin_deg))
    angs = []
    for px, py in pts:
        x, y = px - cx, py - cy
        if math.hypot(x, y) < 0.05:
            continue
        angs.append(math.atan2(y, x))
    if not angs:
        return set()
    mid = math.atan2(sum(math.sin(a) for a in angs), sum(math.cos(a) for a in angs))
    rel = [math.atan2(math.sin(a - mid), math.cos(a - mid)) for a in angs]
    lo, hi = min(rel) - math.radians(bin_deg), max(rel) + math.radians(bin_deg)
    bins = set()
    a = lo
    while a <= hi + 1e-9:
        bins.add(int((math.degrees(mid + a) % 360.0) // bin_deg) % n)
        a += math.radians(bin_deg * 0.5)
    return bins


def _optical_hits(head):
    """Glass, mirror and kickboard contacts from the look that just finished."""
    hits = list(getattr(head, "last_hits", []) or [])
    return [hit for hit in hits if hit[0] == "hit" and is_optical_wall(str(hit[3]))]


def _spin_slow(mover, degrees, rate, status, on_sample):
    """Turn `degrees` in place no faster than `rate` deg/s, calling on_sample along the way."""
    mover.show(status)
    if mover.base is None:
        on_sample()
        return
    if mover.imu is None:
        duration = abs(degrees) / rate
        yaw0 = mover.yaw
        signed = math.copysign(rate, degrees)

        def tick(el):
            mover.yaw = yaw0 + math.radians(signed * min(el, duration))
            on_sample()

        mover._run_turn_open(duration, signed, tick)
        mover.yaw = yaw0 + math.radians(degrees)
        mover.hold = None
        on_sample()
        return
    start = mover._imu_yaw()
    goal = start + math.radians(degrees)
    t0 = time.time()
    limit = abs(degrees) / rate + 8.0
    next_sample = 0.0
    try:
        while True:
            mover._need_imu()
            mover.yaw = mover._imu_yaw()
            err = wrap_deg(math.degrees(goal - mover.yaw))
            mover.show(status)
            now = time.time()
            if now >= next_sample:
                on_sample()
                next_sample = now + WARMUP_LIDAR_SAMPLE_S
            if abs(err) <= TURN_TOL_DEG:
                mover.base.stop()
                time.sleep(TURN_SETTLE_S)
                mover.yaw = mover._imu_yaw()
                if abs(wrap_deg(math.degrees(goal - mover.yaw))) <= TURN_TOL_DEG:
                    break
            if time.time() - t0 > limit:
                print(f"  雷达转圈超时，还差 {err:+.1f}°")
                break
            cmd = max(-rate, min(rate, TURN_KP * err))
            if abs(cmd) < TURN_MIN_RATE:
                cmd = math.copysign(TURN_MIN_RATE, err)
            mover.base.base.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": cmd})
            time.sleep(0.02)
    finally:
        if mover.base is not None:
            mover.base.stop()
    mover.yaw = mover._imu_yaw()
    mover.hold = None
    on_sample()


def scan_lidar(mover, view):
    """First warm-up turn: rotate slowly and keep only lidar returns."""
    if getattr(mover, "lidar", None) is None:
        print("没有雷达，这一圈跳过")
        return []
    pts = []
    next_sample = [0.0]

    def sample():
        now = time.time()
        if now < next_sample[0]:
            return
        next_sample[0] = now + WARMUP_LIDAR_SAMPLE_S
        cloud = _lidar_map_points(mover)
        if not cloud:
            return
        pts.extend(cloud)
        with view.lock:
            step = max(1, len(pts) // 2000)
            view.scan_points = pts[::step]
        mover.show(f"预热：雷达建图，已采 {len(pts)} 个点")

    print(f"预热第一圈：以 {WARMUP_LIDAR_DPS:.0f}°/s 慢慢转，只用雷达")
    _spin_slow(mover, 360.0, WARMUP_LIDAR_DPS, "预热：雷达慢慢转圈", sample)
    print(f"雷达转完：{len(pts)} 个点")
    show_bounds(mover, lidar_pts=pts, center=(mover.x, mover.y))
    return pts


def _edge_points(mover, pairs):
    return [to_map(mover, fx, fy) for fx, fy in pairs if not _on_body(fx, fy)]


def scan_glass(mover, head, view, full_fence=False):
    """Second warm-up turn. Vision replaces lidar only where it sees glass or a mirror.

    full_fence is the fallback when the lidar turn produced nothing: every direction's visual
    fence is kept, and glass bearings are still marked so a later lidar point cannot override them.
    """
    points = []
    optical = set()
    step = 360.0 / SCAN_TURNS
    h0 = mover.yaw
    for k in range(SCAN_TURNS):
        mover.show(f"预热：视觉看玻璃 {k + 1}/{SCAN_TURNS}")
        head.lane()
        hits = _optical_hits(head)
        edge = []
        for hit in hits:
            edge.extend(_edge_points(mover, hit[4] or [(hit[1], hit[2])]))
        kickboard = False
        if hits or full_fence:
            seen = head.fence_points(f"glass{k}")
            kickboard = bool(getattr(head, "saw_kickboard", False))
            if seen is None and full_fence:
                seen = head.fence_points(f"glass{k}b")
                kickboard = kickboard or bool(getattr(head, "saw_kickboard", False))
            edge.extend(_edge_points(mover, seen or []))
        if hits or kickboard:
            labels = list(dict.fromkeys(str(hit[3]) for hit in hits)) or ["踢脚线"]
            optical |= _bearing_bins((mover.x, mover.y), edge)
            print(f"  对面是{'、'.join(labels)}，这个方向改用视觉底边")
        if edge and (hits or kickboard or full_fence):
            points.extend(edge)
        with view.lock:
            view.scan_points = list(points)
        print(f"  方向 {k + 1}/{SCAN_TURNS}：视觉底边 {len(edge)} 个点")
        if edge:
            show_bounds(mover, vision_pts=edge, center=(mover.x, mover.y))
        mover.turn_to(h0 + math.radians(step * (k + 1)), f"预热：视觉看玻璃 {k + 1}/{SCAN_TURNS}")
    return points, optical


def fuse_warmup_points(vision, lidar, center, optical=(), bin_deg=FENCE_BIN_DEG):
    """Lidar outline, with glass and mirror bearings replaced by the visual bottom edge.

    Bearings in `optical` never keep a lidar range. Every other bearing keeps lidar when it
    has one. Vision fills a bearing only when lidar has no return there.
    Returns (points, lidar_bearings, vision_bearings).
    """
    cx, cy = center
    n = int(round(360.0 / bin_deg))
    optical = set(optical)
    vision_r = [math.inf] * n
    lidar_r = [math.inf] * n

    def nearest(store, pts):
        for px, py in pts:
            x, y = px - cx, py - cy
            dist = math.hypot(x, y)
            if dist < 0.05:
                continue
            i = int((math.degrees(math.atan2(y, x)) % 360.0) // bin_deg) % n
            store[i] = min(store[i], dist)

    nearest(vision_r, vision)
    nearest(lidar_r, lidar)
    points = []
    from_lidar = from_vision = 0
    for i in range(n):
        lv, vv = lidar_r[i], vision_r[i]
        if i in optical:
            if not math.isfinite(vv):
                continue
            dist = vv
            from_vision += 1
        elif math.isfinite(lv):
            dist = lv
            from_lidar += 1
        elif math.isfinite(vv):
            dist = vv
            from_vision += 1
        else:
            continue
        ang = math.radians((i + 0.5) * bin_deg)
        points.append((cx + dist * math.cos(ang), cy + dist * math.sin(ang)))
    return points, from_lidar, from_vision


def bearing_arcs(points, center, farthest=False, bin_deg=FENCE_BIN_DEG):
    """Ordered boundary arcs. Lidar keeps the far return; vision keeps the near edge."""
    cx, cy = center
    n = int(round(360.0 / bin_deg))
    dist = [None] * n
    for x, y in points:
        d = math.hypot(x - cx, y - cy)
        if d < 0.05:
            continue
        i = int((math.degrees(math.atan2(y - cy, x - cx)) % 360.0) // bin_deg) % n
        if dist[i] is None or (d > dist[i] if farthest else d < dist[i]):
            dist[i] = d
    if farthest:
        known = [d for d in dist if d is not None]
        if known:
            fill = sorted(known)[len(known) // 2]
            closed = _close_narrow_radii([d if d is not None else fill for d in dist])
            if closed is not None:
                for i, d in enumerate(dist):
                    if d is not None:
                        dist[i] = closed[i]
    gap = next((i for i in range(n) if dist[i] is None), 0)
    order = list(range(gap, n)) + list(range(0, gap))
    arcs, cur = [], []

    def flush():
        if len(cur) == 1:
            x, y = cur[0]
            arcs.append([(x, y), (x + 0.04, y)])
        elif len(cur) >= 2:
            arcs.append(cur + ([cur[0]] if len(cur) == n else []))

    for i in order:
        if dist[i] is None:
            flush()
            cur = []
            continue
        ang = math.radians((i + 0.5) * bin_deg)
        cur.append((cx + dist[i] * math.cos(ang), cy + dist[i] * math.sin(ang)))
    flush()
    return arcs


def show_bounds(mover, lidar_pts=(), vision_pts=(), center=None):
    """Remember lidar and vision outlines separately and publish both to the map page."""
    view = getattr(mover, "view", None)
    if view is None:
        return
    with view.lock:
        if center is not None and not hasattr(view, "bound_center"):
            view.bound_center = center
        origin = getattr(view, "bound_center", None) or center or (mover.x, mover.y)
        if lidar_pts:
            view.lidar_cloud = (getattr(view, "lidar_cloud", []) + list(lidar_pts))[-80000:]
        if vision_pts:
            view.vision_cloud = (getattr(view, "vision_cloud", []) + list(vision_pts))[-20000:]
        if getattr(view, "lidar_cloud", None):
            view.lidar_bounds = bearing_arcs(view.lidar_cloud, origin, farthest=True)
        if getattr(view, "vision_cloud", None):
            view.vision_bounds = bearing_arcs(view.vision_cloud, origin, farthest=False)


def _close_narrow_radii(radii, min_width=BODY_WIDTH_M, jut=0.30):
    """Return radii with impassable spikes closed, or None when nothing changed.

    A run of bearings much farther than both shoulders is an opening. Its width at the
    entrance is the shoulder range times the angle. Narrower than `min_width` means the
    0.50 m body cannot pass, so those bearings stop at the entrance.
    """
    n = len(radii)
    if n < 3 or all(radii[i] > radii[(i + 1) % n] + jut for i in range(n)):
        return None
    r = list(radii)
    ang = 2.0 * math.pi / n
    window = 6
    bg = []
    for i in range(n):
        vals = sorted(r[(i + k) % n] for k in range(-window, window + 1) if k)
        bg.append(vals[len(vals) // 2])
    spike = [r[i] > bg[i] + jut for i in range(n)]
    if not any(spike) or all(spike):
        return None
    start = spike.index(False)
    changed = False
    i = 0
    while i < n:
        idx = (start + i) % n
        if not spike[idx]:
            i += 1
            continue
        run = []
        while i < n and spike[(start + i) % n]:
            run.append((start + i) % n)
            i += 1
        left = r[(run[0] - 1) % n]
        right = r[(run[-1] + 1) % n]
        entrance = min(left, right)
        width = entrance * ang * len(run)
        if width < min_width:
            for j in run:
                r[j] = entrance
            changed = True
    return r if changed else None


def _bin_index(cx, cy, x, y, bin_deg=FENCE_BIN_DEG):
    n = int(round(360.0 / bin_deg))
    return int((math.degrees(math.atan2(y - cy, x - cx)) % 360.0) // bin_deg) % n


def absorb_survey(mover, cmap, lidar_pts, vision_pts):
    """Fold one stationary survey into the fence that already exists.

    Lidar only pushes a bearing outward, to the farthest return on that ray, and marks a
    return standing well inside as an obstacle. Vision points are the glass and mirror
    bottom edge and may move those bearings either way.
    """
    fence = getattr(cmap, "fence", None)
    if fence is None:
        return
    farthest = {}
    for x, y in lidar_pts:
        dist = math.hypot(x - fence.cx, y - fence.cy)
        if dist < 0.30:
            continue
        i = _bin_index(fence.cx, fence.cy, x, y)
        if i not in farthest or dist > farthest[i][0]:
            farthest[i] = (dist, x, y)
    extend = []
    here = (mover.x, mover.y)
    for i, (dist, x, y) in farthest.items():
        if dist > fence.r[i] + WALL_GAIN_M:
            extend.append((x, y))
        elif fence.r[i] - dist > 0.40 and point_in_poly(x, y, cmap.boundary):
            cmap.block(x, y, 0.06, protect_xy=here)
    nearest = {}
    for x, y in vision_pts:
        dist = math.hypot(x - fence.cx, y - fence.cy)
        if dist < 0.30:
            continue
        i = _bin_index(fence.cx, fence.cy, x, y)
        if i not in nearest or dist < nearest[i][0]:
            nearest[i] = (dist, x, y)
    edge = [(x, y) for _dist, x, y in nearest.values()]
    moved = fence.wall(extend) if extend else set()
    rewritten = fence.wall(edge) if edge else set()
    if moved or rewritten:
        parts = []
        if moved:
            parts.append(f"雷达把 {len(moved)} 个方向往外扩")
        if rewritten:
            parts.append(f"视觉把 {len(rewritten)} 个方向改到玻璃或镜子的底边")
        _apply_fence(cmap, "，".join(parts))


def survey_here(mover, head, view, cmap):
    """Slow lidar turn, then a slow vision turn, both folded into the current map."""
    print(f"到了 ({mover.x:+.2f}, {mover.y:+.2f})：先以 {WARMUP_LIDAR_DPS:.0f}°/s 转一圈，雷达核对")
    lidar_pts = scan_lidar(mover, view)
    print("再缓慢转一圈，视觉建玻璃和镜子")
    vision_pts, _optical = scan_glass(
        mover, head, view, full_fence=not lidar_pts and getattr(mover, "lidar", None) is None)
    absorb_survey(mover, cmap, lidar_pts, vision_pts)
    show_bounds(mover, lidar_pts=lidar_pts, vision_pts=vision_pts)
    mover.survey_at = (mover.x, mover.y)


def survey_if_far(mover, head, view, cmap):
    """Survey again only after the robot has travelled to a new place."""
    last = getattr(mover, "survey_at", None)
    if last is not None and math.hypot(mover.x - last[0], mover.y - last[1]) < SURVEY_GAP_M:
        return
    survey_here(mover, head, view, cmap)


def _on_body(px, py):
    """A floor point on the chassis cannot be an obstacle: the robot is standing there."""
    return abs(px) <= BODY_LENGTH_M / 2.0 + 0.02 and abs(py) <= BODY_WIDTH_M / 2.0 + 0.02


def mark_seen(mover, cmap, center, points):
    """Put what the camera boxed ahead on the map. Returns how many marks were obstacles inside the fence.

    `center` is the contact under the box middle: the object extends away from the robot behind
    it, so a disk of OBSTACLE_R is put OBSTACLE_R further along the ray. `points` along the box
    bottom only say how wide the object is and are marked thin where they are. A mark near the
    fence is the fence itself standing closer than the scan measured.
    """
    here = (mover.x, mover.y)
    points = [p for p in points if not _on_body(*p)]
    fx, fy = center
    if _on_body(fx, fy):
        if not points:
            return 0
        fx, fy = points[len(points) // 2]

    def pushed(px, py, by):
        r = math.hypot(px, py) or 1.0
        return to_map(mover, px + by * px / r, py + by * py / r)

    # The box corners give the width at the front contact's depth; the widest part of a round
    # object sits behind it, so they move back by half the half-width.
    half = max((math.hypot(px - fx, py - fy) for px, py in points), default=0.0)
    marks = [(pushed(fx, fy, OBSTACLE_R), OBSTACLE_R)]
    marks += [(pushed(px, py, min(OBSTACLE_R, half / 2.0)), 0.03) for px, py in points]
    inner = 0
    for (mx, my), foot in marks:
        rng = math.hypot(mx - mover.x, my - mover.y)
        if rng > MARK_RANGE or not point_in_poly(mx, my, cmap.boundary):
            continue
        if dist_to_boundary(mx, my, cmap.boundary) > cmap.robot_radius:
            cmap.block(mx, my, foot, protect_xy=here)
            inner += 1
        elif rng <= FENCE_MARK_RANGE:
            cmap.block(mx, my, 0.03, protect_xy=here)
    return inner


def fence_field(cmap):
    if getattr(cmap, "_fence_field", None) is None:
        from relocalize import fence_field as build
        cmap._fence_field = build(cmap)
    return cmap._fence_field


def locate(mover, cmap, local, win, tag, prior=LOCATE_PRIOR):
    """Shift the pose so robot-frame fence points `local` lie on the saved fence. True if trusted."""
    from relocalize import locate as fit

    if len(local) < LOCATE_MIN_PTS:
        print(f"  {tag}：看到的围栏点太少（{len(local)}），不校正")
        return False
    pose, cost, start, gain = fit(fence_field(cmap), cmap, (mover.x, mover.y, mover.yaw), local, *win, *prior)
    dx, dy = pose[0] - mover.x, pose[1] - mover.y
    dyaw = wrap_deg(math.degrees(pose[2] - mover.yaw))
    if cost > LOCATE_OK_M:
        print(f"  {tag}：和保存的围栏对不上（平均差 {cost * 100:.0f} cm），不校正")
        return False
    if gain < LOCATE_GAIN_M:
        print(f"  {tag}：位置和围栏对得上（平均差 {start * 100:.0f} cm），不用校正")
        mover.located_at = mover.travel
        return True
    print(f"  {tag}：对照围栏校正 {dx:+.2f} m, {dy:+.2f} m, {dyaw:+.0f}°，"
          f"平均差 {start * 100:.0f} → {cost * 100:.0f} cm")
    mover.move_frame((mover.x, mover.y, mover.yaw), pose)
    mover.x, mover.y, mover.yaw = pose
    mover.located_at = mover.travel
    mover.show()
    return True


def locate_on_the_way(mover, head, cmap):
    """Every LOCATE_EVERY_M of driving: look at the fence turning only the head, and correct the pose."""
    if head is None or mover.travel - mover.located_at < LOCATE_EVERY_M:
        return
    local = []
    for d_yaw in NECK_SCAN:
        mover.show(f"转头看围栏校正位置 {d_yaw:+.0f}°")
        local.extend(head.fence_points(f"locate{int(d_yaw):+d}", d_yaw=d_yaw) or [])
    mover.located_at = mover.travel
    locate(mover, cmap, local, LOCATE_WIN, "途中定位")


def check_lane(mover, head, cmap, step, target):
    """How far the body may drive straight towards `target` now: a distance, "replan", or None to stop."""
    seen = head.lane()
    if seen is None:
        return None
    observe(mover, cmap, seen)
    lidar_pt = _lidar_ahead(mover, cmap, seen)
    if lidar_pt is not None and not (seen[0] == "hit" and is_optical_wall(str(seen[3]))):
        lidar_free = lidar_pt[0] - BODY_LENGTH_M / 2.0 - STOP_MARGIN
        if lidar_free < step:
            print(f"  视觉没有挡住路线，但雷达在前方 {math.hypot(*lidar_pt):.2f} m 测到实体，重新规划")
            return "replan"
    if seen[0] == "clear":
        return step
    _, fx, fy, label, points = seen
    points = points or [(fx, fy)]
    here = (mover.x, mover.y)
    was_clear = line_clear(cmap, cmap.drive_mask, here, target)
    inner = mark_seen(mover, cmap, (fx, fy), points)
    blocked = was_clear and not line_clear(cmap, cmap.drive_mask, here, target)
    ahead = [px for px, py in points if abs(py) <= BODY_WIDTH_M / 2.0 + 0.05]
    free = (min(ahead) if ahead else math.inf) - BODY_LENGTH_M / 2.0 - STOP_MARGIN
    if inner:
        if free < step or blocked:
            print(f"  前方 {min(px for px, _ in points):.2f} m 有{label}，标红，重新规划")
            return "replan"
        return step
    if free >= step:
        return "replan" if blocked else step
    print(f"  前方 {fx:.2f} m 是围栏（{label}），这一段只走 {max(free, 0.0):.2f} m")
    return max(free, 0.0)


def can_turn_at(cmap, x, y):
    cell = cmap.world_to_cell(x, y)
    return cell is not None and bool(cmap.plan_mask[cell])


def retreat(mover, cmap):
    """Back out along the floor just driven until the body can turn again.

    That floor was swept by the body moments ago, so reversing over it needs no camera. Each
    trail move is undone in RETREAT_STEP_M pieces: first turn back to the heading it was driven
    with, then drive (or slide) it backwards. After a refused turn it backs at least
    TURN_BACK_M, since the map may still call that spot turnable. Returns the metres backed.
    """
    backed = 0.0
    least = TURN_BACK_M if mover.turn_refused else 0.0
    mover.turn_refused = False
    while mover.trail and backed < RETREAT_MAX_M and (
            backed < least or not can_turn_at(cmap, mover.x, mover.y)):
        x0, y0, yaw, rel = mover.trail[-1]
        a = yaw + rel
        along = (mover.x - x0) * math.cos(a) + (mover.y - y0) * math.sin(a)
        off = -(mover.x - x0) * math.sin(a) + (mover.y - y0) * math.cos(a)
        if abs(off) > 0.05 or along < -0.02:
            mover.trail.clear()  # the pose has left the trail (a correction or an unrecorded move)
            break
        if along < 0.01:
            mover.trail.pop()
            continue
        if abs(wrap_deg(math.degrees(yaw - mover.yaw))) >= 1.0:
            mover.turn_to(yaw, "退回走过的路")
        step = min(along, RETREAT_STEP_M)
        if rel == 0.0:
            mover.forward(-step, "退回走过的路", record=False)
        else:
            mover.slide(rel + math.pi, step, "退回走过的路", record=False)
        backed += step
    if backed:
        print(f"  沿走过的路退回 {backed:.2f} m")
    return backed


def safe_turn(mover, head, cmap, heading, status):
    """Turn in place towards `heading`, looking first where the front corners will swing.

    The camera only checks the lane before driving, so something beside the body is unseen until
    a turn sweeps into it. Returns False (after marking it) if something is inside the turning circle.
    """
    for _ in range(8):
        # Measured from the real heading, so a turn that ended 1° short is not carried forward.
        left = wrap_deg(math.degrees(heading - mover.yaw))
        if abs(left) < TURN_TOL_DEG + 0.5:
            break
        step = max(-TURN_LOOK_DEG, min(TURN_LOOK_DEG, left))
        if head is not None and abs(step) >= TURN_CHECK_DEG:
            mover.show("转身前先看一眼")
            look = _neck(-step)
            seen = head.lane(d_yaw=look, near_only=True)
            observe(mover, cmap, seen, look_deg=-look, reach=NEAR_FLOOR_M)
            lidar_pt = None if seen is None else _lidar_ahead(mover, cmap, seen, look_deg=-look)
            if lidar_pt is not None and math.hypot(*lidar_pt) < cmap.robot_radius + TURN_SPARE:
                print("  雷达测到转身扫过范围内有实体，标红，重新规划")
                mover.turn_refused = True
                return False
            if seen is not None and seen[0] == "hit":
                _, fx, fy, label, points = seen
                points = points or [(fx, fy)]
                near = [p for p in points if math.hypot(*p) < cmap.robot_radius + TURN_SPARE]
                inner = [p for p in near
                         if dist_to_boundary(*to_map(mover, *p), cmap.boundary) > cmap.robot_radius]
                if inner:
                    print(f"  转身会碰到旁边的{label}，标红，重新规划")
                    mark_seen(mover, cmap, (fx, fy), points)
                    mover.turn_refused = True
                    return False
        mover.turn_by(step, status)
    return True


def follow(route, mover, head, cmap, status):
    """Drive a list of corner points. Returns "done", "short", "replan" or "stop"."""
    for tx, ty in route[1:]:
        dist = math.hypot(tx - mover.x, ty - mover.y)
        if dist < 0.03:
            continue
        # No fence fit on the way: open sides of the outline are placeholders, not a real fence.
        heading = math.atan2(ty - mover.y, tx - mover.x)
        if not safe_turn(mover, head, cmap, heading, status):
            return "replan"
        mover.hold = heading
        left = dist
        while left > 0.02:
            step = min(LOOK_M, left)
            if head is not None:
                print(f"  前进 {step:.2f} m 之前，VLN 看前方", flush=True)
                mover.show("停车看前方")
                allowed = check_lane(mover, head, cmap, step, (tx, ty))
                if allowed is None:
                    print("前方看不清，停车")
                    return "stop"
                if allowed == "replan":
                    return "replan"
                if allowed < step:
                    mover.forward(allowed, status)
                    return "short"
            got = mover.forward(step, status)
            if mover.stuck:
                _blocked_ahead(mover, cmap)
                return "replan"
            left -= got
            if not point_in_poly(mover.x, mover.y, cmap.boundary):
                print("  位置到了围栏外，重新规划")
                return "replan"
    return "done"


def nearest_row(cmap, mover, rows):
    """(key, first_cell, last_cell) of the row whose nearer end is fewest drivable steps away."""
    dist = drive_dist(cmap, mover.x, mover.y)
    best = None
    for key, a, b in rows:
        for first, last in ((a, b), (b, a)):
            d = int(dist[first])
            if d >= 0 and (best is None or d < best[0]):
                best = (d, key, first, last)
    return None if best is None else best[1:]


def _inside(mover, cmap):
    return not cmap.boundary or point_in_poly(mover.x, mover.y, cmap.boundary)


def body_samples(x, y, yaw):
    """Points on the 0.30 m by 0.50 m base of the body. The cuboid stands 0.80 m tall on this base."""
    hl, hw = BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    lx = -hl
    while lx <= hl + 1e-9:
        ly = -hw
        while ly <= hw + 1e-9:
            pts.append((x + c * lx - s * ly, y + s * lx + c * ly))
            ly += 0.05
        lx += 0.05
    return pts


def footprint_clear(x, y, yaw, cmap):
    """True when the whole base rectangle is inside the fence and off blocked cells."""
    for px, py in body_samples(x, y, yaw):
        if cmap.boundary and not point_in_poly(px, py, cmap.boundary):
            return False
        cell = cmap.world_to_cell(px, py)
        if cell is not None and cmap.grid[cell] == BLOCKED:
            return False
    return True


def travel_clear(x, y, yaw, rel, dist, cmap):
    """How far the rectangle can move along `rel` from `yaw` without entering the fence or an obstacle."""
    if dist <= 0.0 or not footprint_clear(x, y, yaw, cmap):
        return 0.0
    a = yaw + rel
    c, s = math.cos(a), math.sin(a)
    ok = 0.0
    d = 0.05
    while d <= dist + 1e-9:
        if not footprint_clear(x + c * d, y + s * d, yaw, cmap):
            return ok
        ok = min(d, dist)
        d += 0.05
    return dist


def _fence_ahead(mover, cmap, dist):
    """How far the 0.30 x 0.50 m base can go before it would cross the fence."""
    return travel_clear(mover.x, mover.y, mover.yaw, 0.0, dist, cmap)


def _inside_xy(x, y, cmap):
    return not cmap.boundary or point_in_poly(x, y, cmap.boundary)


def _hit_info(mover, cmap, seen):
    """Mark what the camera boxed and say how far the body may still go."""
    _, fx, fy, label, points = seen
    points = points or [(fx, fy)]
    inner = mark_seen(mover, cmap, (fx, fy), points)
    ahead = [px for px, py in points if abs(py) <= BODY_WIDTH_M / 2.0 + 0.05]
    free = (min(ahead) if ahead else math.inf) - BODY_LENGTH_M / 2.0 - STOP_MARGIN
    return str(label), points, free, inner


def is_wall(label):
    """Anything that defines the room boundary, including transparent and reflective walls."""
    return any(k in label for k in ("墙", "围", "栏", "挡", "踢脚", "基座", "玻璃", "镜"))


def is_optical_wall(label):
    """A transparent/reflective boundary whose lidar returns must never set its distance."""
    return any(k in label for k in ("玻璃", "镜", "踢脚", "基座"))


def _optical_edge(mover, reach=OPTICAL_RANGE_M):
    """Map points on the glass or mirror bottom edge from the look that just finished.

    The edge is the kickboard or base where it meets the floor. That line is the boundary.
    """
    head = getattr(mover, "head", None)
    pts, labels = [], []
    for hit in _optical_hits(head):
        mapped = []
        for x, y in (hit[4] or [(hit[1], hit[2])]):
            if _on_body(x, y) or math.hypot(x, y) > reach:
                continue
            mapped.append(to_map(mover, x, y))
        if mapped:
            pts.extend(mapped)
            labels.append(str(hit[3]))
    return pts, list(dict.fromkeys(labels))


class Fence:
    """The fence as a distance per bearing (FENCE_BIN_DEG wide) from the spot where it was scanned.

    Every look updates only the bearings it covers: a wall seen moves those bearings to the
    wall, floor seen (or driven on) pushes them out past it. Nothing else of the outline moves,
    so one wall never cuts away floor in other directions.
    """

    def __init__(self, center, poly):
        self.cx, self.cy = center
        self.n = int(round(360.0 / FENCE_BIN_DEG))
        self.r = [self._ray(poly, math.radians((i + 0.5) * FENCE_BIN_DEG)) for i in range(self.n)]

    def _ray(self, poly, a):
        """Distance from the center to `poly` along bearing a."""
        dx, dy = math.cos(a), math.sin(a)
        best = math.inf
        for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
            ex, ey = bx - ax, by - ay
            den = dx * ey - dy * ex
            if abs(den) < 1e-12:
                continue
            t = ((ax - self.cx) * ey - (ay - self.cy) * ex) / den
            u = ((ax - self.cx) * dy - (ay - self.cy) * dx) / den
            if t > 0 and 0.0 <= u <= 1.0:
                best = min(best, t)
        return best if math.isfinite(best) else 0.5

    def _bin(self, x, y):
        d = math.hypot(x - self.cx, y - self.cy)
        i = int((math.degrees(math.atan2(y - self.cy, x - self.cx)) % 360.0) // FENCE_BIN_DEG) % self.n
        return i, d

    def close_narrow(self, min_width=BODY_WIDTH_M, jut=0.30):
        """Pull back rays that only continue through an opening narrower than the body.

        The body is 0.50 m wide. A far lidar return behind a thinner gap cannot be driven
        into, so that direction is closed at the entrance instead of left as a spike.
        """
        closed = _close_narrow_radii(self.r, min_width, jut)
        if closed is None:
            return False
        self.r = closed
        return True

    def polygon(self):
        self.close_narrow()
        return [(self.cx + d * math.cos(math.radians((i + 0.5) * FENCE_BIN_DEG)),
                 self.cy + d * math.sin(math.radians((i + 0.5) * FENCE_BIN_DEG))) for i, d in enumerate(self.r)]

    def wall(self, pts):
        """Wall contacts (map frame) put their bearings onto the wall. Returns the bearings moved."""
        moved = set()
        for x, y in pts:
            i, d = self._bin(x, y)
            if d > 0.30 and abs(d - self.r[i]) > WALL_GAIN_M:
                self.r[i] = d
                moved.add(i)
        return moved

    def floor(self, pts, past=FENCE_PAST_M):
        """Floor points (map frame) seen free or driven on: the fence is at least `past` beyond them."""
        moved = set()
        for x, y in pts:
            i, d = self._bin(x, y)
            if d + past > self.r[i] + 0.02:
                self.r[i] = d + past
                moved.add(i)
        return moved


def _apply_fence(cmap, why):
    cmap.set_boundary(cmap.fence.polygon())
    print(f"  围栏实时更新：{why}", flush=True)


def observe(mover, cmap, seen, look_deg=0.0, reach=None):
    """Update the fence from one lane check.

    The free floor in front of the camera (up to the hit, or up to `reach` when clear) pushes the
    fence out; a wall pulls the bearings it covers in. look_deg is where the head pointed, robot
    frame (+ is left).
    """
    fence = getattr(cmap, "fence", None)
    if fence is None or seen is None:
        return
    if reach is None:
        reach = CLEAR_FLOOR_M
    label = ""
    if seen[0] == "hit":
        _, fx, fy, label, points = seen
        points = points or [(fx, fy)]
        reach = min(reach, math.hypot(fx, fy) - 0.10)
    a = math.radians(look_deg)
    c, s = math.cos(a), math.sin(a)
    floor = []
    along = BODY_LENGTH_M / 2.0 + 0.05
    while along <= reach:
        for k in range(-5, 6):  # 5 cm apart: at 1 m that is under one fence bearing
            lat = 0.05 * k
            floor.append(to_map(mover, c * along - s * lat, s * along + c * lat))
        along += 0.05
    grown = fence.floor(floor)
    pulled = set()
    edge_note = ""
    if seen[0] == "hit" and is_wall(label) and not is_optical_wall(label):
        seen_wall = [to_map(mover, px, py) for px, py in points
                     if not _on_body(px, py) and math.hypot(px, py) <= WALL_RANGE_M]
        pulled = fence.wall(seen_wall)
        view = getattr(mover, "view", None)
        if view is not None and seen_wall:
            with view.lock:
                view.scan_points.extend(seen_wall)
                del view.scan_points[:-3000]
    edge, edge_labels = _optical_edge(mover)
    if edge:
        moved = fence.wall(edge)
        pulled |= moved
        if moved:
            edge_note = f"看到的{'、'.join(edge_labels)}底边把 {len(moved)} 个方向定在边缘上"
            show_bounds(mover, vision_pts=edge)
            view = getattr(mover, "view", None)
            if view is not None:
                with view.lock:
                    view.scan_points.extend(edge)
                    del view.scan_points[:-3000]
    if grown or pulled:
        parts = []
        if grown:
            parts.append(f"看到的空地把 {len(grown)} 个方向往外推")
        if edge_note:
            parts.append(edge_note)
        elif pulled:
            parts.append(f"看到的{label}把 {len(pulled)} 个方向移到墙上")
        _apply_fence(cmap, "，".join(parts))


def observe_body(mover, cmap):
    """The body really stands here, so the fence is past its corners."""
    fence = getattr(cmap, "fence", None)
    if fence is None:
        return
    hl, hw = BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0
    body = [to_map(mover, hl * i / 3.0, hw * j / 5.0) for i in range(-3, 4) for j in range(-5, 6)]
    if fence.floor(body, past=0.02):
        _apply_fence(cmap, "车开到了原来的围栏外，往外推")


def _blocked_ahead(mover, cmap):
    """Mark the spot just in front of the body as impassable after it could not get there."""
    fx, fy = to_map(mover, BODY_LENGTH_M / 2.0 + 0.12, 0.0)
    if _inside_xy(fx, fy, cmap):
        cmap.block(fx, fy, 0.08, protect_xy=(mover.x, mover.y))


def _forward_checked(mover, cmap, step, status):
    """Drive `step` m. Returns ("go", metres really moved) or ("stuck", label, points, metres)."""
    got = mover.forward(step, status)
    if mover.stuck:
        _blocked_ahead(mover, cmap)
        return ("stuck", "走不动的地方", [], got)
    return ("go", got)


def _glass_mask_for_hit(angs, seen):
    """Angular span covered by one visual box/contact line."""
    drop = np.zeros(len(angs), dtype=bool)
    pts = seen[4] or [(seen[1], seen[2])]
    bears = [math.atan2(py, px) for px, py in pts]
    lo, hi = min(bears), max(bears)
    if hi - lo < 2.2:
        drop |= (angs >= lo - 0.12) & (angs <= hi + 0.12)
    for bear in bears:
        drop |= np.abs(np.arctan2(np.sin(angs - bear), np.cos(angs - bear))) < 0.25
    return drop


def _glass_mask(angs, observations):
    """Bearings vision called glass or mirror; lidar is untrustworthy on all of them."""
    drop = np.zeros(len(angs), dtype=bool)
    for seen in observations:
        if seen[0] != "hit" or not is_optical_wall(str(seen[3])):
            continue
        drop |= _glass_mask_for_hit(angs, seen)
    return drop


def _lidar_ahead(mover, cmap, seen, look_deg=0.0):
    """Fuse the camera sector with lidar and return its closest point in that corridor.

    `look_deg` is the camera direction in the base frame (+ left). Returns in a direction the
    camera called glass are dropped: the lidar mirrors them or sees through.
    """
    lidar = getattr(mover, "lidar", None)
    if lidar is None:
        return None
    pts = lidar.latest_xy()
    if pts is None or len(pts) < 30:
        return None
    angs = np.arctan2(pts[:, 1], pts[:, 0])
    # Pair every mapped lidar return with the camera's current view. Otherwise a reflected return
    # from glass seen in another direction can be painted as a solid object before vision checks it.
    look = math.radians(look_deg)
    rel_angs = np.arctan2(np.sin(angs - look), np.cos(angs - look))
    in_view = np.abs(rel_angs) <= math.radians(LIDAR_FUSED_HALF_FOV_DEG)
    # Head.lane returns the nearest object for motion, but preserves every box it saw. A small
    # object in front of glass must not hide the glass mask behind it.
    observations = list(getattr(getattr(mover, "head", None), "last_hits", []) or [])
    if seen not in observations:
        observations.append(seen)
    solid = pts[in_view & ~_glass_mask(angs, observations)]
    # Vision says which bearings are an ordinary opaque wall; lidar supplies the more accurate
    # range on those bearings. Optical walls were removed above and remain vision-only.
    opaque = [hit for hit in observations
              if hit[0] == "hit" and is_wall(str(hit[3])) and not is_optical_wall(str(hit[3]))]
    if opaque and len(solid):
        wall_mask = np.zeros(len(solid), dtype=bool)
        solid_angs = np.arctan2(solid[:, 1], solid[:, 0])
        for hit in opaque:
            wall_mask |= _glass_mask_for_hit(solid_angs, hit)
        wall_points = [to_map(mover, float(x), float(y)) for x, y in solid[wall_mask]
                       if 0.30 < math.hypot(float(x), float(y)) <= EXPLORE_WALL_M]
        fence = getattr(cmap, "fence", None)
        if fence is not None and fence.wall(wall_points):
            _apply_fence(cmap, "视觉确认实体墙，雷达校正墙距")
            show_bounds(mover, lidar_pts=wall_points)
    here = (mover.x, mover.y)
    for x, y in solid[::5]:
        if not 0.28 <= math.hypot(float(x), float(y)) <= 3.5:
            continue
        mx, my = to_map(mover, float(x), float(y))
        # A return on the outline is the wall already measured. Only things standing inside get marked.
        if not _inside_xy(mx, my, cmap) or dist_to_boundary(mx, my, cmap.boundary) < 0.30:
            continue
        cmap.block(mx, my, 0.06, protect_xy=here)
    c, s = math.cos(look), math.sin(look)
    along = solid[:, 0] * c + solid[:, 1] * s
    lateral = -solid[:, 0] * s + solid[:, 1] * c
    in_corridor = (along > 0.28) & (np.abs(lateral) <= BODY_WIDTH_M / 2.0 + 0.05)
    if not np.any(in_corridor):
        return None
    corridor = solid[in_corridor]
    i = int(np.argmin(along[in_corridor]))
    return float(corridor[i, 0]), float(corridor[i, 1])


def vln_step(mover, head, cmap, limit, status):
    """One straight step. The camera names what is ahead; the lidar ranges everything that is not glass.

    Returns ("go", metres) or ("chair"|"obstacle"|"fence"|"stuck"|"stop", label, points[, metres]).
    """
    step = min(LOOK_M, limit)
    if step < 0.02:
        return ("go", 0.0)
    room = travel_clear(mover.x, mover.y, mover.yaw, 0.0, step, cmap)
    if room < step - 0.02:
        print("  车身 30×50 cm 的底面再往前会压进围栏或障碍")
        if room > 0.03:
            return _forward_checked(mover, cmap, room, status)
        return ("fence", "车身碰到边界", [])
    step = room
    if head is None:
        free = _fence_ahead(mover, cmap, step + BODY_LENGTH_M)
        if free >= step:
            return _forward_checked(mover, cmap, step, status)
        if free > 0.03:
            mover.forward(free, status)
        return ("fence", "围栏", [])
    print(f"  前进 {step:.2f} m 之前，VLN 看前方", flush=True)
    mover.show("VLN 看前方")
    seen = head.lane()
    if seen is None:
        return ("stop", "", [])
    observe(mover, cmap, seen)
    lidar_pt = _lidar_ahead(mover, cmap, seen)
    glass = seen[0] == "hit" and is_optical_wall(str(seen[3]))
    lidar_free = None
    if lidar_pt is not None and not glass:
        lidar_free = lidar_pt[0] - BODY_LENGTH_M / 2.0 - STOP_MARGIN
        print(f"  雷达测到前方 {math.hypot(*lidar_pt):.2f} m", flush=True)
    if seen[0] == "clear":
        dist = step if lidar_free is None else min(step, max(0.0, lidar_free))
        if dist >= step - 0.02:
            out = _forward_checked(mover, cmap, step, status)
            if out[0] == "go" and not _inside(mover, cmap):
                return ("fence", "围栏", [])
            return out
        if dist > 0.03:
            out = _forward_checked(mover, cmap, dist, status)
            if out[0] == "stuck":
                return out
        return ("obstacle", "雷达障碍", [lidar_pt])
    label, points, free, inner = _hit_info(mover, cmap, seen)
    if lidar_free is not None and lidar_free < free:
        print(f"  视觉说前方是{label}，距离用雷达的 {lidar_pt[0]:.2f} m")
        free = lidar_free
        points = [lidar_pt]
    if free >= step:
        # The camera says this step is open. A red cell on the drifted map does not cancel it.
        out = _forward_checked(mover, cmap, step, status)
        if out[0] == "go" and not _inside(mover, cmap):
            return ("fence", "围栏", [])
        return out
    if free > 0.03:
        out = _forward_checked(mover, cmap, free, status)
        if out[0] == "stuck":
            return out
    if not inner or is_wall(label):
        print(f"  前方是围栏（{label}），这一行到头")
        return ("fence", label, points)
    if "椅" in label:
        print(f"  前方是{label}，先沿椅子打扫，再把它当成墙")
        return ("chair", label, points)
    print(f"  前方是{label}，当成墙，换下一行")
    return ("obstacle", label, points)


def side_free(mover, head, cmap, rel, status):
    """Turn the body to face `rel`, look, turn back. Metres the body can slide that way (0 if unknown)."""
    origin = mover.yaw
    print(f"  侧移方向 {math.degrees(rel):+.0f}° 脖子看不到，先转身看", flush=True)
    if not safe_turn(mover, head, cmap, origin + rel, "转身看侧移方向"):
        mover.turn_to(origin, "转回原来的方向")
        return 0.0
    seen = head.lane()
    free = 0.0
    if seen is not None:
        observe(mover, cmap, seen)
        if seen[0] == "clear":
            free = CLEAR_FLOOR_M - BODY_WIDTH_M / 2.0
        else:
            _, points, _, _ = _hit_info(mover, cmap, seen)
            # Facing the slide, the body's extent across it is its length.
            ahead = [px for px, py in points if abs(py) <= BODY_LENGTH_M / 2.0 + 0.05]
            free = min(CLEAR_FLOOR_M, min(ahead) if ahead else math.inf) - BODY_WIDTH_M / 2.0 - STOP_MARGIN
    if not safe_turn(mover, head, cmap, origin, "转回原来的方向"):
        mover.turn_to(origin, "转回原来的方向")
    print(f"  侧移方向能走 {max(free, 0.0):.2f} m", flush=True)
    return max(free, 0.0)


def vln_slide(mover, head, cmap, rel, dist, status):
    """Slide without turning. A cut-short slide is reversed.

    Slides close to straight ahead are looked at with the head before each piece. For a slide
    further to the side the head would have to turn so far that the body's own uprights fill the
    view, so the body turns to face the slide once, looks, turns back, and slides as far as was free.
    """
    room = travel_clear(mover.x, mover.y, mover.yaw, rel, dist, cmap)
    if room < dist - 0.02:
        print("  车身 30×50 cm 的底面侧向会压进围栏或障碍")
    dist = room
    moved = 0.0
    look = _neck(-math.degrees(rel))
    side_view = head is not None and abs(math.degrees(rel)) > SLIDE_LOOK_DEG
    if side_view:
        dist = min(dist, side_free(mover, head, cmap, rel, status))
        if dist < 0.03:
            return 0.0
    while moved < dist - 0.02:
        step = min(0.15, dist - moved)
        if head is not None and not side_view:
            mover.show(status)
            seen = head.lane(d_yaw=look, near_only=True)
            if seen is None:
                break
            observe(mover, cmap, seen, look_deg=-look, reach=NEAR_FLOOR_M)
            lidar_pt = _lidar_ahead(mover, cmap, seen, look_deg=-look)
            if lidar_pt is not None and swept_hit([lidar_pt], rel, step + STOP_MARGIN):
                print("  雷达测到平移扫过范围内有实体，不平移")
                break
            if seen[0] == "hit":
                label, points, _, _ = _hit_info(mover, cmap, seen)
                along = []
                for px, py in points:
                    side = -px * math.sin(rel) + py * math.cos(rel)
                    if abs(side) <= BODY_LENGTH_M / 2.0 + 0.05:
                        along.append(px * math.cos(rel) + py * math.sin(rel))
                free = (min(along) if along else math.inf) - BODY_WIDTH_M / 2.0 - STOP_MARGIN
                if free < step:
                    break
        got = mover.slide(rel, step, status)
        moved += got
        if mover.stuck or not _inside(mover, cmap):
            break
    if moved < dist - 0.05 and moved > 0.03:
        mover.slide(rel + math.pi, moved, "退回这一行", record=False)
        return 0.0
    return moved


def vln_drive(mover, head, cmap, limit, status):
    """Drive up to `limit` metres. Returns (reason, label, points, driven)."""
    driven = 0.0
    while driven < limit - 0.02:
        step = vln_step(mover, head, cmap, limit - driven, status)
        if step[0] == "stop":
            return step[0], step[1], step[2], driven
        if step[0] == "go":
            driven += step[1]
            if step[1] < 0.02:
                break
            continue
        if step[0] == "stuck":
            driven += step[3]
        return step[0], step[1], step[2], driven
    return "ok", "", [], driven


def face(mover, head, cmap, heading, status):
    if head is None:
        mover.turn_to(heading, status)
        return True
    return safe_turn(mover, head, cmap, heading, status)


def clean_chair(mover, head, cmap, points):
    """Slide along the chair's near side without turning into it, then back onto the row."""
    ys = [py for _, py in points]
    span = max(0.30, min(1.0, (max(ys) - min(ys)) if len(ys) >= 2 else BODY_WIDTH_M))
    print(f"  沿椅子扫 {span:.2f} m")
    if vln_slide(mover, head, cmap, math.pi / 2.0, span, "沿椅子打扫") < span - 0.05:
        return
    vln_slide(mover, head, cmap, -math.pi / 2.0, span, "沿椅子返回")


def known_back(mover):
    """Metres straight behind the body that it drove along its current heading a moment ago."""
    reach = 0.0
    x, y = mover.x, mover.y
    for x0, y0, yaw, rel in reversed(mover.trail):
        a = yaw + rel
        if rel != 0.0 or abs(wrap_deg(math.degrees(a - mover.yaw))) > 5.0:
            break
        along = (mover.x - x0) * math.cos(a) + (mover.y - y0) * math.sin(a)
        off = -(mover.x - x0) * math.sin(a) + (mover.y - y0) * math.cos(a)
        if abs(off) > 0.05:
            break
        reach = max(reach, along)
    return reach


def turn_with_room(mover, head, cmap, heading, status):
    """Turn to the absolute `heading`. If the spot is too tight, back up the way it came and retry."""
    for _ in range(4):
        origin = mover.yaw
        if face(mover, head, cmap, heading, status):
            return True
        mover.turn_to(origin, "转回再退后")
        if known_back(mover) < TURN_BACK_M:
            print("  转身空间不够，后面也没有走过的路可退")
            return False
        print("  转身空间不够，退后一点")
        mover.hold = origin
        mover.forward(-TURN_BACK_M, "退后一点再转身", record=False)
    return False


def next_row(mover, head, cmap, side, row_heading):
    """Turn 90° towards `side` (+1 left, -1 right), drive 30 cm, turn 90° again: the next row
    then runs the opposite way. Alternating the side keeps the rows from doubling back."""
    perp = row_heading + side * math.pi / 2.0
    back = row_heading + side * math.pi
    word = "左" if side > 0 else "右"
    if not turn_with_room(mover, head, cmap, perp, f"{word}转 90°"):
        print("转不过身，遍历结束")
        return False
    mover.hold = perp
    _, _, _, driven = vln_drive(mover, head, cmap, LANE_M, "换行 30 cm")
    if driven < LANE_M - 0.05:
        print("侧面走不满 30 cm，这一侧到头")
        return False
    if not turn_with_room(mover, head, cmap, back, f"{word}转 90°"):
        print("转不过身，遍历结束")
        return False
    return True


def swept_ahead(mover, cmap, dist):
    """True when the floor the body would cover in the next `dist` m is already swept."""
    total = done = 0
    for k in range(int(dist / 0.05) + 1):
        fx = BODY_WIDTH_M / 2.0 + 0.05 + k * 0.05   # beyond the disk the body has just painted
        for j in range(-4, 5):
            cell = cmap.world_to_cell(*to_map(mover, fx, j * BODY_WIDTH_M / 9.0))
            if cell is None:
                continue
            state = cmap.grid[cell]
            if state == BODY_REACH:
                total += 1
            elif state == BODY_DONE:
                total += 1
                done += 1
    return total > 0 and done >= 0.85 * total


def fresh_side(mover, cmap, heading):
    """+1 if more unswept floor lies left of `heading` near the robot, else -1."""
    c, s = math.cos(heading), math.sin(heading)
    count = {1.0: 0, -1.0: 0}
    for side in count:
        for lat in np.arange(0.30, 1.21, 0.10):
            for along in np.arange(-1.0, 1.01, 0.10):
                x = mover.x + c * along - s * side * lat
                y = mover.y + s * along + c * side * lat
                cell = cmap.world_to_cell(x, y)
                if cell is not None and cmap.grid[cell] == BODY_REACH:
                    count[side] += 1
    return 1.0 if count[1.0] > count[-1.0] else -1.0


def cover(mover, head, cmap, view, start=None, first_side=-1.0, survey=None):
    """Straight rows, each at an absolute heading from the IMU. At the end of a row: turn, 30 cm,
    turn the same way, come back. The turns alternate sides, starting with `first_side`
    (-1 right, +1 left). A row also ends where the floor ahead is already swept; two such
    empty rows in a row end the sweep, so fill_in can take the robot somewhere new.

    `survey`, when given, runs at the start of a row so a warm-up pass can spin in place there.
    """
    del view
    if start is None:
        start = mover.yaw
    empty = 0
    for lane in range(MAX_LANES):
        if survey is not None:
            survey()
        heading = start + (math.pi if lane % 2 else 0.0)
        print(f"S 形第 {lane + 1} 行，航向 {wrap_deg(math.degrees(heading)):+.0f}°")
        mover.show(f"S 形第 {lane + 1} 行")
        if abs(wrap_deg(math.degrees(heading - mover.yaw))) > TURN_TOL_DEG + 0.5:
            if not turn_with_room(mover, head, cmap, heading, "转到这一行的方向"):
                return "done"
        mover.hold = heading
        driven = 0.0
        while driven < ROW_LIMIT_M - 0.02:
            if swept_ahead(mover, cmap, LOOK_M):
                print("  前面已经扫过了，这一行到头")
                break
            step = vln_step(mover, head, cmap, min(LOOK_M, ROW_LIMIT_M - driven), "S形直行")
            if step[0] == "stop":
                return "stop"
            if step[0] == "go":
                driven += step[1]
                if step[1] < 0.02:
                    break
                continue
            if step[0] == "stuck":
                print("  这一行走不动，当成走到头，换下一行")
                break
            if step[0] == "chair":
                clean_chair(mover, head, cmap, step[2])
            break
        empty = empty + 1 if driven < 0.05 else 0
        if empty >= 2:
            print("  连着两行都是扫过的地面，这一片扫完了")
            return "done"
        if not next_row(mover, head, cmap, first_side if lane % 2 == 0 else -first_side, heading):
            return "done"
    print(f"走了 {MAX_LANES} 行，停下")
    return "cut_off"


def fill_in(mover, head, cmap, axis, purpose="清扫", survey=None):
    """Breadth-first fill-in after the S sweep stops.

    cleanup_goal floods the drivable cells outward from the robot (breadth first) and returns
    the nearest place that still has unswept floor. The robot drives there along an A* route,
    checking ahead with the camera and lidar, and sweeps a short S there with rows along `axis`.
    Places it cannot reach after GOAL_TRIES are given up. Only returns "done" after 90% was
    physically visited; other convergence limits return "incomplete".
    """
    tries = {}

    def miss(cell, why):
        tries[cell] = tries.get(cell, 0) + 1
        if tries[cell] >= GOAL_TRIES:
            print(f"  {why}，连续 {tries[cell]} 次没到达，只排除这个目标点")
            ban_goal(cmap, cell, max(cmap.res, 0.10))

    for leg in range(MAX_LEGS):
        sweep = cmap.stats()["sweep"]
        if sweep >= GOAL_SWEEP:
            print(f"已访问 {sweep:.0%}，达到 {GOAL_SWEEP:.0%} 目标")
            return "done"
        # Prefer a centre that visits several new cells, instead of inching along an edge one
        # 5 cm cell at a time. When only small islands remain, lower the threshold to one so no
        # reachable unvisited patch is silently ignored.
        goal, via, left, unreachable = cleanup_goal(cmap, mover.x, mover.y, min_gain=4)
        if goal is None:
            goal, via, left, unreachable = cleanup_goal(cmap, mover.x, mover.y, min_gain=1)
        if goal is None:
            print(f"没有可达的未访问目标（不可达还有 {unreachable} 格），"
                  f"当前 {sweep:.0%}，未达到 {GOAL_SWEEP:.0%}")
            return "incomplete"
        target = via if via is not None else goal
        route = route_to(cmap, mover.x, mover.y, target)
        gx, gy = cmap.cell_center(goal)
        if route is None:
            ban_goal(cmap, goal, 0.30)
            continue
        print(f"{purpose}第 {leg + 1} 段：广度优先找到最近的未访问区域 ({gx:+.2f}, {gy:+.2f})，"
              f"能到达的还剩 {left} 格，已访问 {sweep:.0%}", flush=True)
        mover.show(f"{purpose}第 {leg + 1} 段：去未访问区域")
        result = follow(route, mover, head, cmap, f"{purpose}：去未访问区域")
        if result == "stop":
            return "stop"
        if result != "done":
            miss(goal, "这个地方过不去")
            if mover.turn_refused or result == "short":
                retreat(mover, cmap)
            continue
        if via is not None:
            # A dead end too tight to turn in: drive straight in, then reverse out the same way.
            d = math.hypot(gx - mover.x, gy - mover.y)
            if face(mover, head, cmap, math.atan2(gy - mover.y, gx - mover.x), "对准窄处"):
                mover.hold = mover.yaw
                _, _, _, got = vln_drive(mover, head, cmap, d, "开进窄处")
                if got > 0.02:
                    mover.forward(-got, "从窄处原路退出", record=False)
            # cleanup_goal returns a drive/approach candidate, not necessarily an unvisited cell.
            # Retire only this exact candidate so a neighbouring centre can cover any tiny edge gap.
            ban_goal(cmap, goal, 0.5 * cmap.res)
            continue
        if survey is not None:
            survey()
        heading = axis if math.cos(mover.yaw - axis) >= 0.0 else axis + math.pi
        result = cover(mover, head, cmap, None, start=heading,
                       first_side=fresh_side(mover, cmap, heading), survey=survey)
        if result == "stop":
            return "stop"
        ban_goal(cmap, goal, 0.5 * cmap.res)
    print(f"{purpose}进行了 {MAX_LEGS} 段仍未达到 {GOAL_SWEEP:.0%}，停车检查")
    return "incomplete"



def ensure_scan_room(mover, head, view, points):
    """Slide to the nearest spot where the body can turn a full circle, if the fence is too close here."""
    local = [to_robot(mover, x, y) for x, y in points]
    need = BODY_CLEARANCE_M + PLAN_MARGIN_M
    if all(math.hypot(px, py) >= need for px, py in local):
        return
    move = find_turn_spot(mover, points)
    if move is None:
        print("  附近没有看得到的转身空间，原地小心转圈")
        return
    rel, dist = move
    print(f"  这里转不开一整圈，先往 {math.degrees(rel):+.0f}° 平移 {dist:.2f} m")
    vln_slide(mover, head, view.cmap, rel, dist, "平移到能转圈的地方")


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


def build_area(mover, head, view):
    """Warm-up: a slow lidar turn, then a vision turn that rewrites glass and mirror bearings.

    The outline is measured again on every run. The robot may be put down somewhere else next time.
    """
    from vln_scene import boundary_from_points

    mover.show("预热：先看能否转圈")
    points = look_without_turning(mover, head, view)
    ensure_scan_room(mover, head, view, points)
    learn_self(mover, head)
    lidar_pts = scan_lidar(mover, view)
    print("预热第二圈：用视觉找雷达可能看错的玻璃和镜子")
    vision_pts, optical = scan_glass(mover, head, view, full_fence=not lidar_pts)
    fused, from_lidar, from_vision = fuse_warmup_points(
        vision_pts, lidar_pts, (mover.x, mover.y), optical)
    print(f"预热：{from_lidar} 个方向用雷达，{from_vision} 个方向用视觉，"
          f"{len(optical)} 个方向是玻璃或镜子")
    boundary, why = boundary_from_points(fused, (mover.x, mover.y), max_gap_deg=MAX_GAP_DEG)
    if boundary is None:
        raise SystemExit(f"预热没测出边界：{why}。把车放进场地里，四周别挡住")
    xs, ys = [p[0] for p in boundary], [p[1] for p in boundary]
    print(f"预热边界：约 {max(xs) - min(xs):.1f} m × {max(ys) - min(ys):.1f} m")
    mover.survey_at = (mover.x, mover.y)
    return [list(p) for p in boundary]


def make_map(boundary, mover):
    """Grid for the scanned outline, with FENCE_GROW_M of room around it for the fence to move out into.

    The robot stands where it scanned, which is the center the fence bearings are measured from.
    """
    xs, ys = [p[0] for p in boundary], [p[1] for p in boundary]
    g = FENCE_GROW_M
    room = [[min(xs) - g, min(ys) - g], [max(xs) + g, min(ys) - g],
            [max(xs) + g, max(ys) + g], [min(xs) - g, max(ys) + g]]
    spec = area_spec(room)
    spec["start"] = [mover.x, mover.y, math.degrees(mover.yaw)]
    cmap = CleanMap(spec)
    cmap.fence = Fence((mover.x, mover.y), [tuple(p) for p in boundary])
    cmap.set_boundary(cmap.fence.polygon())
    cmap.stamp_body(mover.x, mover.y)
    return cmap


def go_to_corner(mover, head, cmap):
    """Drive to the left fence, then the back fence, and face the first row.

    The rows then start at a corner and sweep the whole area to the right, not just the part
    ahead of wherever the robot happened to stand.
    """
    h0 = mover.yaw
    for rel, name in ((math.pi / 2.0, "左"), (math.pi, "后")):
        heading = h0 + rel
        if not turn_with_room(mover, head, cmap, heading, f"转向{name}边"):
            print(f"转不向{name}边，就从这里开始")
            break
        mover.hold = heading
        reason, label, _, driven = vln_drive(mover, head, cmap, ROW_LIMIT_M, f"去{name}边的围栏")
        print(f"  去{name}边走了 {driven:.2f} m（{reason}{'：' + label if label else ''}）")
        if reason == "stop":
            return False
    return turn_with_room(mover, head, cmap, h0, "转回第一行的方向")


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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="电机和相机都不用，在一个假围栏里演示")
    p.add_argument("--no-imu", action="store_true", help="不用 IMU（转角和直线只靠速度×时间）")
    p.add_argument("--imu-port", default=None, help="IMU 串口，默认自动找")
    p.add_argument("--no-lidar", action="store_true", help="明确允许不用雷达，只靠视觉（默认雷达失败会停车）")
    p.add_argument("--lidar-port", default=None, help="雷达串口，默认自动找 CP2102")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default="qwen3-vl-plus")
    p.add_argument("--timeout-s", type=float, default=20.0)
    args = p.parse_args()

    def _interrupt(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, _interrupt)

    view = MapView(placeholder_map())
    view.write(MAP_DIR)
    ensure_map_server(args.port)
    writer = view.start_writer(MAP_DIR)
    print(f"状态页：http://{local_ip()}:{args.port}")

    robot = {"head": None, "base": None, "imu": None, "lidar": None, "cmap": view.cmap}
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
        if robot["base"] is not None:
            robot["base"].close()
        if robot["head"] is not None:
            robot["head"].close()
        if robot["imu"] is not None:
            robot["imu"].close()
        if robot["lidar"] is not None:
            robot["lidar"].close()
        writer.set()
        view.write(MAP_DIR, running=False)
        print(f"底盘已停。状态页仍开着：http://{local_ip()}:{args.port}")


def work(args, view, robot):
    cmap = view.cmap
    if args.dry_run:
        mover = Mover(cmap, view, None, speedup=6.0)
        head = None
        mover.lidar = None
    else:
        from vln_scene import Head, nav

        head = Head(args.model, args.timeout_s, os.path.join(MAP_DIR, "look", time.strftime("%Y%m%d_%H%M%S")))
        robot["head"] = head
        base = nav.Base(*head.neck_home)
        robot["base"] = base
        head.attach(base)
        mover = Mover(cmap, view, base)
        mover.head = head
        if not args.no_imu:
            from imu import Imu

            try:
                imu = Imu(args.imu_port)
            except RuntimeError as err:
                raise SystemExit(f"{err}。检查 IMU 的 USB 线，或加 --no-imu（转角和直线会不准）")
            robot["imu"] = imu
            print(f"IMU：{imu.port}")
            mover.attach_imu(imu)
        else:
            print("没有用 IMU：转角和直线只靠速度×时间，不准")
        mover.lidar = None
        if args.no_lidar:
            print("明确不用雷达：精建图只用视觉")
        else:
            from lidar_scan import Lidar

            lidar = None
            try:
                lidar = Lidar(args.lidar_port)
                lidar.start()
            except Exception as err:
                if lidar is not None:
                    lidar.close()
                raise SystemExit(f"雷达没打开：{err}。检查雷达 USB 和电机；"
                                 "如果确认只用视觉，加 --no-lidar")
            mover.lidar = lidar
            robot["lidar"] = lidar
            print(f"雷达：{lidar.port}")

    if args.dry_run:
        boundary = [[-1.2, -1.4], [2.0, -1.4], [2.0, 1.4], [-1.2, 1.4]]
        print("阶段 1/3 预热：干跑用假围栏，不转圈")
    else:
        mover.set_stage("warmup")
        boundary = build_area(mover, head, view)
        print("原地两圈完成，接着把预热图走全")
    cmap = make_map(boundary, mover)
    mover.cmap = cmap
    view.set_map(cmap)
    robot["cmap"] = cmap
    result = None
    if head is not None:
        print("预热走查：和探索一样覆盖场地。每到新地方，先缓慢转圈用雷达核对，再缓慢转圈用视觉建图")
        mover.show("预热：先开到角上")

        def survey():
            survey_if_far(mover, head, view, cmap)

        if not go_to_corner(mover, head, cmap):
            result = "stop"
        else:
            survey()
            axis = mover.yaw
            result = cover(mover, head, cmap, view, survey=survey)
            if result != "stop":
                warm = fill_in(mover, head, cmap, axis, purpose="预热", survey=survey)
                if warm == "stop":
                    result = "stop"
                else:
                    print(f"阶段 1/3 预热完成：走查覆盖 {cmap.stats()['sweep'] * 100:.0f}%，"
                          "边界已按各处的转圈更新")
                    cmap.reset_visited(mover.x, mover.y)
                    mover.trail.clear()
                    result = None
    if result != "stop":
        mover.set_stage("explore")
        print("阶段 2/3 探索：预热边界会随实测改，空地往外推，墙和玻璃往实测边上收")
        mover.show("探索：先开到角上")
        if not go_to_corner(mover, head, cmap):
            result = "stop"
        else:
            print("从角上开始探索，边走边修正边界和障碍")
            mover.show("探索：视觉和雷达修正地图")
            axis = mover.yaw
            result = cover(mover, head, cmap, view)
            if result != "stop":
                print(f"这一片看完了，当前走过 {cmap.stats()['sweep'] * 100:.0f}%；"
                      f"继续去还没看过的地方，至少达到 {GOAL_SWEEP:.0%}")
                exploration = fill_in(mover, head, cmap, axis, purpose="探索")
                if exploration == "done":
                    explored = cmap.stats()["sweep"]
                    print(f"阶段 2/3 探索完成：走过 {explored * 100:.0f}%，障碍和边界已按实测修正")
                    # Exploration travel is not cleaning. Keep the corrected fence and obstacles,
                    # clear only the visited marks, then count the sweep from the current footprint.
                    cmap.reset_visited(mover.x, mover.y)
                    mover.trail.clear()
                    print(f"阶段 3/3 清扫：按探索后的地图遍历可清洁区域，"
                          f"从 {cmap.stats()['sweep'] * 100:.0f}% 重新统计，目标 {GOAL_SWEEP:.0%}")
                    mover.set_stage("sweep")
                    mover.show("清扫未清洁区域")
                    result = fill_in(mover, head, cmap, axis, purpose="清扫")
                else:
                    result = "stop" if exploration == "stop" else "explore_incomplete"
                    print("阶段 2/3 探索没看完可达区域，不开始清扫")
    visited = cmap.stats()["sweep"]
    print(f"已访问约 {visited * 100:.0f}% 的可达地面")
    msg = {"done": "清扫完成",
           "incomplete": f"清扫未达到 {GOAL_SWEEP:.0%}，需要检查到不了的区域或障碍",
           "explore_incomplete": f"探索未达到 {GOAL_SWEEP:.0%}，地图还没修正完，没有开始清扫",
           "stop": "中途停下"}[result]
    mover.show(msg, [])
    print(msg)


if __name__ == "__main__":
    main()
