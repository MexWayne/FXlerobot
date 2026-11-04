#!/usr/bin/env python3
"""建图、探索，找到地上的瓶子，用左臂夹起来扔进垃圾桶。

  conda activate lerobot_env
  cd ~/workspaces/FXLerobot/FXlerobot/demo
  python bottle_to_bin.py                    # 桶口高度用双目量的
  python bottle_to_bin.py --bin-height 0.32  # 桶口离地 0.32 m（量不准或桶顶出了画面时给）
  python bottle_to_bin.py --no-explore --no-bin  # 测试：建完图不探索，去捡已看到的瓶子，拿着就停
  python bottle_to_bin.py --use-map --no-explore  # 用上次存的地图：原地转一圈定位，不建图
  python bottle_to_bin.py --rescan                # 用上次存的地图定位，到几个观察点低头看一圈找瓶子，捡起扔掉
  python bottle_to_bin.py --rescan --holding 0.16 # 左臂上还夹着上次没扔掉的 16 cm 瓶子：定位后先扔掉它

每次运行结束都把地图存到 vln/map/saved/；--use-map 读它（或 --use-map 某个 live*.npz：只有状态页文件的旧地图）。
读进来的地图里不留要捡的东西（瓶子、纸屑会被挪走），家具、墙和垃圾桶留着；定位后先原地用相机看一圈。

状态页同 coverage_run.py：http://<jetson>:8765 。Ctrl-C 停车，左臂停在原地保持力矩。

1) 建图：vln 的预热（雷达转两圈 + 视觉看一圈），地图、物体建模、定位都用 vln 的。
2) 探索：vln 的深度优先探索，把能到的地方都走完；探索完再看地图上有没有垃圾桶、有哪些瓶子。
   没有垃圾桶就停下说明，一个瓶子也不捡。
3) 一个瓶子：按地图开到离瓶子 0.45-1 m、车能到的最近处，转身朝向它，头转到抓取视角（yaw -15, pitch 35），
   VLM 框出瓶子和瓶盖、GrabCut 抠出瓶身（detect_any.py，什么颜色的瓶子都行），量出瓶盖、方向和瓶长，
   一点点平移到瓶盖在保险杠前约 10 cm（--cap-gap）、夹爪能竖直朝下夹到为止（不斜着夹）；然后按 pick.py 的
   流程夹起、抬起，再合一次夹爪仍被瓶盖挡住才算拿起来，拿着瓶子慢慢收臂。
4) 垃圾桶：捡第一个瓶子之前先空手去量桶：雷达拟合圆心和半径，双目量桶口高度，左臂够不着就停下。
   扔的时候按地图开到桶前约 0.6 m，雷达再量一次圆心，平移到桶在车头左前方，
   左臂把瓶子举到桶口上方（桶口 + 瓶子挂下来的长度）松开，收臂，接着探索找下一个。
只抓躺在地上的瓶子。瓶子挂下来的长度用量到的瓶长：左臂举到约 0.6 m，所以 0.21 m 长的瓶子桶口
最高约 0.36 m；扔不进的桶或瓶子，夹之前就停下说明。
"""

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "vln"), str(ROOT / "project_tool"), str(ROOT / "llm_vla" / "pick_bottle"),
                str(ROOT / "llm_vla")]
import arm_drop  # noqa: E402
import coverage_run as cr  # noqa: E402
import detect_any  # noqa: E402
import pick  # noqa: E402
import ik  # noqa: E402
from arm import LeftArm, joint_limits  # noqa: E402
from coverage_plan import drive_dist, route_to  # noqa: E402
from cup_compare import fit_circle  # noqa: E402
from live_map import MapView  # noqa: E402
from occmap import OccMap, disk  # noqa: E402
import saved_map  # noqa: E402
from vln_scene import is_bin  # noqa: E402

# Only upright grasps: leaning, the jaws hold the cap by a corner (two lost bottles at 20 deg) and miss
# a cap they did not come straight down onto. Out of upright reach the body moves closer instead.
pick.TILT_STEPS_DEG = (0.0,)

RUNS = HERE / "runs"
GRASP_NECK = (-15.0, 35.0)    # pick.ROBOT_MASK is drawn for this head pose (and the arm in PARK)
NECK_APPROACH_DEG = 3.0       # neck backlash: finish from lower yaw and higher pitch, like neck_look.py
NECK_TOL_DEG = 1.0
# Staging: the body stops where the bottle's middle lies here in the base frame, inside the grasp
# view (it sees the floor 0.25-0.8 m ahead) and far enough that the body can still turn.
BOTTLE_AT = (0.58, 0.08)
STAGE_HEADINGS = 24
STAGE_TRIES = 3
# Near a bottle the body stops anywhere it can reach this far from it, and faces it: in clutter (a
# box beside it, the lidar on the bottle itself) a fixed spot in front of it is often not drivable
# (2026-11-03: three bottles given up). The grasp view and lidar-checked slides do the rest.
NEAR_M = (0.45, 1.0)
NEAR_SPREAD_M = 0.20          # spots tried after a failed one are at least this far from it
FIND_SLIDES = 3               # slides towards the map's bottle while the grasp view does not see it
# On a known map: between places the body drives on the map and the lidar alone (no VLM on the way),
# in steps of FAST_STEP_M; where that does not go it falls back to coverage_run.follow, which looks.
FAST_LIN = 0.12
FAST_STEP_M = 0.6
FAST_TURN_TOL_DEG = 5.0
# Looking for bottles: a turn in place in QUICK_STOPS steps, each one look at the floor at QUICK_PITCHES
# (pitch 10 sees the floor 0.4-1.7 m out). The places are picked so that all open floor lies within
# VIEW_R_M of one (greedy, the best new cover first), at most VIEW_MAX of them.
QUICK_STOPS = 6
QUICK_PITCHES = (10.0,)
VIEW_R_M = 1.5
VIEW_MIN_M2 = 0.3             # a place that would show less new floor than this is not worth going to
VIEW_MAX = 12
BUMPER_X = 0.165              # the bumper is this far ahead of the base centre
# The cap goes this far ahead of the bumper, CAP_Y to the left, for the grasp: as close as the bottle
# allows. At 8-12 cm every direction of a bottle lying ahead of the bumper has an upright plan
# (checked 2026-11-03); the picks that worked had caps 10-20 cm ahead.
CAP_GAP_M = 0.10
CAP_Y = 0.06
# The grasp is aimed this far off the cap the cameras found, along the bottle towards its body: aimed
# at the cap itself the jaws came down beyond its end and closed on nothing; 17.5 mm along caught
# its edge, and the user put it 1 cm further (2026-11-04, the wrist half round).
GRASP_ALONG_M = 0.027
# Grasp with the wrist half round: the left wrist camera then looks at the cap and the jaws.
FLIP_GRASP = True


# Lifted, the bottle hangs from its cap: the grasp point rises its length and this much more, so its
# bottom leaves the floor (a fixed 20 cm left a 22 cm bottle standing on its end).
LIFT_CLEAR_M = 0.06


def lift_heights(length, cut):
    """pick.LIFT_HEIGHTS_M for a bottle `length` long (cut: partly out of view, length not known)."""
    need = (max(length, HANG_CUT_M) if cut else length) + LIFT_CLEAR_M
    return (need, need - 0.03, 0.20)


def grasp_nudge_mm(axis):
    """pick.py's nudge (base frame, mm) for a bottle along `axis` (cap -> body, unit)."""
    return (np.asarray(axis, float) * GRASP_ALONG_M * 1000.0).tolist()
BOTTLE_CLEAR_X = 0.20         # no part of the bottle nearer than this when the body has stopped
TURN_CLEAR_M = 0.38           # the body corners sweep 0.29 m: turn only with the bottle beyond this
POINTS_AT_BODY = -0.6         # bottle axis this much along -x points at the body: turn it across
APPROACH_TOL_M = 0.02         # the cap this close to where it should be: no more sliding
APPROACH_STEPS = 7
FINE_LIN = 0.05               # m/s for the last few centimetres
SEEN_VS_MAP_M = 0.30          # the cameras' bottle this far from the map's is another thing
PICK_TRIES = 2
HANG_CUT_M = 0.22             # a bottle partly out of view hangs at least this far (its length is not known)
MIN_HANG_M = 0.10             # the shortest bottle: a can too tall even for it is not worth a pick
BOTTLE_MAX_M = 0.35           # a "bottle" in the map longer than this is not one
SKIP_NEAR_M = 0.25            # bottles done or given up are skipped within this of where they were
# A "bottle" this close to a trash can is one already thrown in, seen over the rim: going for it the body
# slid on into the can and knocked over the stand beside it (2026-11-04). Measured can: within its
# radius plus this of its centre; not yet measured: within this of any point of a can in the map.
IN_BIN_M = 0.06
# Trash can: staged with its centre here, measured by the lidar, then slid to BIN_Y beside the bumper.
BIN_AT = (0.62, 0.15)
BIN_Y = 0.15
BIN_GAP_M = 0.04              # between the bumper and the can
BIN_R_DEFAULT = 0.12
BIN_R_RANGE = (0.06, 0.25)
BIN_LIDAR_M = 0.15            # lidar returns this far round the can's expected wall are the can
BIN_TOL_M = 0.02
# Rim: the highest stereo points over the can's circle, head level (the can 0.6 m off fills the
# lower half of the picture, rim included), median of a few pairs. vln's stereo calibration reads
# distances ~5% short, which puts a 0.3 m rim ~1.5 cm high: on the safe side.
RIM_PITCH = 0.0
RIM_SHOTS = 3
RIM_MIN_PTS = 30
# Gripper heat: carried eased (LeftArm.ease_grip) it warms slowly, but each grasp squeezes hard. Hotter
# than GRIP_MAX_C before a grasp, the body waits until it is down to GRIP_COOL_C (~2.5 C a minute):
# at 70 C it unloads and cannot open (2026-11-04).
GRIP_MAX_C = 50
GRIP_COOL_C = 45
GRIP_COOL_MAX_S = 600.0


def flipped(plan):
    """hover, grasp and lift joints of the plan with the wrist half round: the same grasp points and
    closing line, the jaws on the other sides."""
    offset = ik.tcp_offset(pick.CAP_RADIUS_M + pick.JAW_CLEARANCE_M)
    limits = joint_limits()
    _, _, close = ik.pose(plan["grasp"]["joints"], offset)
    axis = np.cross(close, [0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    out, prev = {}, None
    for step in ("grasp", "hover", "lift"):
        q0 = plan[step]["joints"]
        roll = q0["wrist_roll"] + 180.0 if q0["wrist_roll"] + 180.0 <= limits["wrist_roll"][1] else q0["wrist_roll"] - 180.0
        q, err = ik.solve(ik.pose(q0, offset)[0], axis, offset, limits, [prev or {**q0, "wrist_roll": roll}])
        if err > 3:
            raise SystemExit(f"手腕转半圈后 {step} 解不出 IK（{err:.1f}）")
        out[step], prev = {"joints": q}, q
    if max(abs(out[a]["joints"]["wrist_roll"] - out["grasp"]["joints"]["wrist_roll"]) for a in ("hover", "lift")) > 30:
        raise SystemExit("手腕转半圈后上方和下面的解差半圈，不动")
    return out


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Fail(Exception):
    """This bottle or can cannot be done now; the run goes on."""


class Fetcher:
    def __init__(self, mover, head, base, arm, args):
        self.mover, self.head, self.base, self.arm, self.args = mover, head, base, arm, args
        self.skip = []      # map points of bottles done or given up
        self.done = 0
        self.bin_at = None  # (centre in the map frame, radius, rim height) once measured

    # --- what the map knows ---------------------------------------------------------------
    @staticmethod
    def _centre(obj):
        return tuple(np.mean(np.asarray(obj["points"], float), axis=0))

    def in_bin(self, c):
        """Map point `c` lies in a trash can."""
        if self.bin_at is not None:
            (bx, by), r, _ = self.bin_at
            return math.hypot(c[0] - bx, c[1] - by) <= r + IN_BIN_M
        return any(math.hypot(c[0] - x, c[1] - y) <= IN_BIN_M + BIN_R_DEFAULT
                   for o in self.mover.objects if is_bin(o["label"]) for x, y in [self._centre(o)])

    def bottles(self):
        out = []
        for o in self.mover.objects:
            if "瓶" not in o["label"] or "瓶盖" in o["label"] or o.get("size", o["length"]) > BOTTLE_MAX_M:
                continue
            c = self._centre(o)
            if self.in_bin(c):
                continue
            if all(math.hypot(c[0] - s[0], c[1] - s[1]) > SKIP_NEAR_M for s in self.skip):
                out.append(o)
        return sorted(out, key=lambda o: math.hypot(self._centre(o)[0] - self.mover.x,
                                                     self._centre(o)[1] - self.mover.y))

    def bin(self):
        cans = [o for o in self.mover.objects if is_bin(o["label"])]
        return max(cans, key=lambda o: o["seen"]) if cans else None

    # --- driving --------------------------------------------------------------------------
    def go_stage(self, target, at, axis_deg=None, status=""):
        """Drive to where map point `target` lies at base-frame `at`, facing it. With the target's
        long side known (axis_deg, map frame) headings across it come first."""
        m, cmap = self.mover, self.mover.cmap
        dist = drive_dist(cmap, m.x, m.y)
        cands = []
        for k in range(STAGE_HEADINGS):
            h = 2 * math.pi * k / STAGE_HEADINGS
            c, s = math.cos(h), math.sin(h)
            x, y = target[0] - (c * at[0] - s * at[1]), target[1] - (s * at[0] + c * at[1])
            cell = cmap.world_to_cell(x, y)
            if cell is None or dist[cell] < 0:
                continue
            score = dist[cell] * cmap.res + 0.3 * abs(wrap(h - m.yaw)) / math.pi
            if axis_deg is not None:
                score += 0.5 * abs(math.cos(h - math.radians(axis_deg)))
            cands.append((score, cell, h))
        for _, cell, h in sorted(cands)[:STAGE_TRIES]:
            route = route_to(cmap, m.x, m.y, cell)
            if route is None:
                continue
            result = fast_follow(m, self.head, route, status)
            if result == "done" and cr.safe_turn(m, self.head, h, status):
                return
            print(f"  没开到停靠点（{result}），换一个", flush=True)
        raise Fail("找不到能开过去的停靠点")

    def go_near(self, target, status):
        """To a spot NEAR_M from map point `target` the body can reach, nearest by road, facing it so
        that it lies where BOTTLE_AT is. Raises Fail when no such spot can be reached."""
        m, cmap = self.mover, self.mover.cmap
        def near():
            return NEAR_M[0] <= math.hypot(target[0] - m.x, target[1] - m.y) <= NEAR_M[1]
        if not near():
            dist = drive_dist(cmap, m.x, m.y)
            ys, xs = np.nonzero(dist >= 0)
            cx, cy = cmap.origin[0] + (xs + 0.5) * cmap.res, cmap.origin[1] + (ys + 0.5) * cmap.res
            off = np.hypot(cx - target[0], cy - target[1])
            ok = (off >= NEAR_M[0]) & (off <= NEAR_M[1])
            order = np.argsort(dist[ys, xs][ok] * cmap.res + 0.5 * np.abs(off[ok] - BOTTLE_AT[0]))
            cells = list(zip(ys[ok][order], xs[ok][order]))
            tried = []
            for cell in cells:
                if len(tried) == STAGE_TRIES or near():
                    break
                p = cmap.cell_center(cell)
                if any(math.hypot(p[0] - q[0], p[1] - q[1]) < NEAR_SPREAD_M for q in tried):
                    continue
                tried.append(p)
                route = route_to(cmap, m.x, m.y, (int(cell[0]), int(cell[1])))
                if route is None:
                    continue
                result = fast_follow(m, self.head, route, status)
                if result != "done" and not near():
                    print(f"  没开到瓶子附近（{result}），换一个地方", flush=True)
            if not near():
                raise Fail(f"开不到离瓶子 {NEAR_M[0]}-{NEAR_M[1]} m 的地方")
        bearing = math.atan2(target[1] - m.y, target[0] - m.x) - math.atan2(BOTTLE_AT[1], BOTTLE_AT[0])
        if not cr.safe_turn(m, self.head, bearing, status):
            print("  转不过去正对瓶子，先这样，靠平移对准", flush=True)

    def slide(self, rel, dist, status, ignore=(), search=False):
        """Move `dist` towards `rel` (rad from the heading), heading kept, slowly. The lidar returns
        near `ignore` (base frame: the thing being approached) do not stop it, any other in the way does.
        search: towards where the map says a thing is that the cameras do not see; nothing is being
        approached, so the map limits the way and the lidar watches while it moves."""
        m = self.mover
        if search:
            room = cr.map_room(m, rel, dist)
            if room < 0.03:
                raise Fail(f"朝 {math.degrees(rel):+.0f}° 平移，地图上紧挨着就有东西")
            dist = min(dist, room)
        if dist < 0.005:
            return
        xy = m.lidar.latest_xy()
        if abs(rel) <= math.radians(100.0) and xy is not None:
            local = [(float(x), float(y)) for x, y in xy
                     if all(math.hypot(x - a, y - b) > 0.10 for a, b in ignore)]
            if cr.swept_hit(local, rel, dist, 0.03):
                raise Fail(f"朝 {math.degrees(rel):+.0f}° 平移 {dist:.2f} m 的路上雷达看到别的东西")
        m.hold = m.yaw
        m.drive(rel, dist, status, record=False, guard=search, speed=FINE_LIN)

    # --- the head -------------------------------------------------------------------------
    def grasp_view(self):
        yaw, pitch = GRASP_NECK
        for _ in range(2):
            self.base.look(yaw - NECK_APPROACH_DEG, pitch + NECK_APPROACH_DEG, settle_s=0.6)
            got = self.base.look(yaw, pitch, settle_s=0.8)
            if max(abs(got[0] - yaw), abs(got[1] - pitch)) <= NECK_TOL_DEG:
                break
        else:
            raise Fail(f"脖子到不了抓取视角：yaw {got[0]:.1f} pitch {got[1]:.1f}")
        self.head.neck = None  # the next vln look points the head again

    def neck_deg(self):
        obs = self.base.neck.get_observation()
        return {"neck_yaw": obs["neck_yaw.pos"], "neck_pitch": obs["neck_pitch.pos"]}

    def shoot(self, d, tag):
        left, right = self.head._grab_pair()
        cv2.imwrite(str(d / f"{tag}_left.jpg"), left)
        cv2.imwrite(str(d / f"{tag}_right.jpg"), right)

    def look_bottle(self, folder, k, expect):
        """Photos in the grasp view, the bottle nearest `expect` (base frame) found (detect_any) and an
        upright grasp planned: (dir, run, planned). run has no cap_axis_base_m when none was found."""
        d = folder / f"{k:02d}"
        d.mkdir(parents=True)
        run = {"neck_deg": self.neck_deg(), "nudge_mm": [0.0, 0.0, 0.0]}
        self.shoot(d, "look")
        img = cv2.imread(str(d / "look_left.jpg"))
        cam = pick.head_cam(run["neck_deg"])
        stereo = (self.head.stereo.points(img, cv2.imread(str(d / "look_right.jpg")), cam["T_base_cam"]),
                  self.head.stereo.uv)
        got = detect_any.locate(img, cam, self.args.model, self.args.timeout_s, pick.ROBOT_MASK["left"], expect, stereo)
        cv2.imwrite(str(d / "look_left_detect.jpg"), img if got is None else got["vis"])
        if got is None:
            print("  抓取视角里没找到瓶子", flush=True)
            pick.save_run(d, run)
            return d, run, False
        # The fixed jaw's tip stays MIN_TIP_Z_M off the floor: a cap lower than that is taken that high.
        got["cap"][2] = max(got["cap"][2], pick.MIN_TIP_Z_M + pick.ik.GRASP_INSET + 0.002)
        run["nudge_mm"] = grasp_nudge_mm(got["axis"])
        run.update({"cap_axis_base_m": got["cap"].tolist(), "label_base_m": got["mid"].tolist(),
                    "bottle_axis_base": got["axis"].tolist(), "bottle_length_m": got["length"],
                    "bottle_radius_m": got["radius"], "bottle_cut": got["cut"]})
        print(f"  瓶盖 (mm) {np.round(got['cap'] * 1000).astype(int).tolist()}（高度按{got['z_from']}），"
              f"瓶长 {got['length'] * 100:.0f} cm" + ("，瓶子有一截在画面外" if got["cut"] else ""), flush=True)
        pick.LIFT_HEIGHTS_M = lift_heights(got["length"], got["cut"])
        try:
            run["plan"] = pick.solve_steps(run, self.arm.deg())
            if FLIP_GRASP:
                lift_h = run["plan"]["lift"]["height_m"]
                run["plan"] = flipped(run["plan"])
                run["plan"]["lift"]["height_m"] = lift_h
            planned = True
        except SystemExit as err:
            print(f"  {err}", flush=True)
            planned = False
        pick.save_run(d, run)
        return d, run, planned

    # --- one bottle -----------------------------------------------------------------------
    def approach_bottle(self, obj, folder):
        """From the staging spot to where the arm can take the cap. Returns (dir, run) with a plan."""
        m = self.mover
        self.grasp_view()
        misses = 0
        for k in range(APPROACH_STEPS):
            m.show(f"对准瓶子 {k + 1}/{APPROACH_STEPS}")
            expect = cr.to_robot(m.pose, self._centre(obj))
            d, run, planned = self.look_bottle(folder, k, expect)
            if "cap_axis_base_m" not in run:
                misses += 1
                gap = math.hypot(expect[0] - BOTTLE_AT[0], expect[1] - BOTTLE_AT[1])
                if misses > FIND_SLIDES or gap < 0.05:
                    raise Fail("抓取视角里找不到瓶子")
                # Not in view yet: bring the map's bottle to the middle of the view, a step at a time.
                self.slide(math.atan2(expect[1] - BOTTLE_AT[1], expect[0] - BOTTLE_AT[0]), min(0.30, gap),
                           "找瓶子", search=True)
                continue
            cap = np.array(run["cap_axis_base_m"][:2])
            label = np.array(run["label_base_m"][:2])
            axis = np.array(run["bottle_axis_base"][:2])
            pts = [cap, label, cap + axis * run["bottle_length_m"]]  # cap, middle, bottom
            if math.hypot(*(label - np.array(expect))) > SEEN_VS_MAP_M:
                raise Fail(f"抓取视角里的瓶子离地图上的瓶子 {math.hypot(*(label - np.array(expect))):.2f} m，不是同一个")
            if axis[0] < POINTS_AT_BODY:
                self.turn_across(pts, axis)
                continue
            # Move so the cap comes to its place, but never so far that the bottle gets under the bumper.
            near_x = min(p[0] for p in pts)
            move = cap - np.array([BUMPER_X + self.args.cap_gap, CAP_Y])
            move[0] = min(move[0], near_x - BOTTLE_CLEAR_X)
            if np.linalg.norm(move) < APPROACH_TOL_M:
                if planned:
                    print(f"  瓶盖在保险杠前 {(cap[0] - BUMPER_X) * 100:.0f} cm，竖直夹得到", flush=True)
                    return d, run
                raise Fail("瓶盖已经在抓取位置附近，IK 还是规划不出来")
            self.slide(math.atan2(move[1], move[0]), float(np.linalg.norm(move)), "靠近瓶子",
                       [tuple(p) for p in pts])
        raise Fail(f"{APPROACH_STEPS} 步还没对准瓶子")

    def turn_across(self, pts, axis):
        """The bottle points at the body: turn so it lies across, then bring it back into view."""
        m = self.mover
        near = min(math.hypot(*p) for p in pts)
        if near < TURN_CLEAR_M:
            cr.back_up(m, TURN_CLEAR_M - near + 0.02, "后退一点好转身")
        stand = cr.turn_blocker(m)
        if stand is not None:
            raise Fail(f"瓶子正对着车身，要转身，但转身半径内有东西（离车 {math.hypot(*stand):.2f} m）")
        a = math.atan2(axis[1], axis[0])
        turn = a - math.copysign(math.pi / 2, a)
        print(f"  瓶子正对着车身（朝 {math.degrees(a):+.0f}°），原地转 {math.degrees(turn):+.0f}° 让它横过来", flush=True)
        before = [cr.to_map(m.pose, p) for p in pts]
        m.turn_by(math.degrees(turn), "转身让瓶子横过来")
        now = [cr.to_robot(m.pose, p) for p in before]
        mid = np.mean(now, axis=0)
        move = mid - np.array(BOTTLE_AT)
        self.slide(math.atan2(move[1], move[0]), float(np.linalg.norm(move)), "把瓶子放回视野", now)

    def cool_gripper(self):
        """Wait, with the arm parked, until the gripper is cool enough for a grasp and a carry."""
        temp, hot = self.arm.gripper_heat()
        if temp is not None and temp <= GRIP_MAX_C and not hot:
            return
        t0 = time.monotonic()
        while temp is None or temp > GRIP_COOL_C or hot:
            if time.monotonic() - t0 > GRIP_COOL_MAX_S:
                raise Fail(f"夹爪 {GRIP_COOL_MAX_S / 60:.0f} 分钟还没凉下来（{temp}°C）")
            self.mover.show(f"夹爪 {temp}°C，等它凉到 {GRIP_COOL_C}°C")
            time.sleep(10.0)
            temp, hot = self.arm.gripper_heat()
        print(f"  夹爪凉到 {temp}°C，等了 {time.monotonic() - t0:.0f} s", flush=True)

    def pick_up(self, d, run):
        """pick.py's grasp: hover, down, grip, lift, check it left the floor. True when held up."""
        arm, p = self.arm, run["plan"]
        self.cool_gripper()
        pick.move(arm, {**p["hover"]["joints"], "gripper": pick.OPEN_DEG}, pick.FREE_DEG_S, "到瓶盖上方")
        pick.move(arm, p["grasp"]["joints"], pick.NEAR_DEG_S, "下降")
        self.shoot(d, "grasp")  # the open jaws round the cap: where they really are against it
        closed = arm.limits["gripper"][0]
        g = arm.grip(closed)
        self.shoot(d, "closed")
        run["result"] = {"grip_deg": g, "held": bool(g > closed + 3)}
        pick.save_run(d, run)
        if not run["result"]["held"]:
            print(f"  夹爪合到 {g:.1f}°，没夹到", flush=True)
            pick.move(arm, {**p["hover"]["joints"], "gripper": pick.OPEN_DEG}, pick.NEAR_DEG_S, "退回上方")
            pick.park(arm)
            return False
        pick.move(arm, p["lift"]["joints"], pick.NEAR_DEG_S, "抬起")
        self.shoot(d, "lift")
        # Squeeze again: a cap still between the jaws stops them where they were, an empty grip closes.
        # (Seen from above, a bottle lifted straight up covers where it lay, and the VLM asked whether
        # one hangs from the gripper said no on 5 of 8 photos of bottles that did.)
        g = arm.grip(closed)
        run["result"]["after_lift_deg"] = g
        pick.save_run(d, run)
        if g <= closed + 3:
            print(f"  抬起后夹爪合到 {g:.1f}°，瓶子掉了", flush=True)
            pick.move(arm, {"gripper": pick.OPEN_DEG}, pick.FREE_DEG_S, "松开")
            pick.park(arm)
            return False
        print(f"  拿起来了：抬起后夹爪仍停在 {g:.1f}°", flush=True)
        pick.park(arm, keep_grip=True)
        arm.ease_grip(closed)
        return True

    def fetch_bottle(self, obj):
        """Pick the bottle up, if it can go into the measured can (bin_at); returns where it lay
        (map frame) and how far it hangs below the grasp point."""
        m = self.mover
        folder = RUNS / time.strftime("%Y%m%d_%H%M%S")
        centre = self._centre(obj)
        print(f"去捡 {obj['id']} 号{obj['label']} ({centre[0]:+.2f}, {centre[1]:+.2f})，图片在 {folder}", flush=True)
        self.go_near(centre, "去瓶子旁边")
        for attempt in range(PICK_TRIES):
            d, run = self.approach_bottle(obj, folder / f"try{attempt + 1}")
            where = cr.to_map(m.pose, run["label_base_m"][:2])
            hang = max(run["bottle_length_m"], HANG_CUT_M) if run["bottle_cut"] else run["bottle_length_m"]
            if self.bin_at is not None:
                try:
                    _, r, rim = self.bin_at
                    arm_drop.plan_drop(pick.PARK, (BUMPER_X + r + BIN_GAP_M, BIN_Y), r, rim, hang)
                except SystemExit as err:
                    raise Fail(f"瓶子长 {hang * 100:.0f} cm，扔不进这个垃圾桶：{err}")
            try:
                if self.pick_up(d, run):
                    return where, hang
            except SystemExit as err:  # pick.move refused a move: nothing moved
                raise Fail(str(err))
            m.show("没夹起来，再试一次")
        raise Fail(f"{PICK_TRIES} 次都没夹起来")

    # --- the trash can --------------------------------------------------------------------
    def measure_bin(self, expect):
        """Centre (map frame) and radius of the can near map point `expect`, from the lidar arc of its
        near wall."""
        m = self.mover
        expect = np.array(cr.to_robot(m.pose, expect))
        scans = m.lidar.scans(n=3, timeout=4.0)
        if not scans:
            raise Fail("雷达没有新数据，量不了垃圾桶")
        pts = np.vstack(scans)
        near = pts[np.linalg.norm(pts - expect, axis=1) < BIN_R_RANGE[1] + BIN_LIDAR_M]
        if len(near) < 5:
            raise Fail(f"垃圾桶附近只有 {len(near)} 个雷达点，量不出位置")
        centre, r, rms = fit_circle(near.astype(float))
        if not BIN_R_RANGE[0] <= r <= BIN_R_RANGE[1]:
            # Only a short arc: take the nearest wall point and the usual radius behind it.
            wall = near[np.argmin(np.linalg.norm(near, axis=1))]
            r = BIN_R_DEFAULT
            centre = wall + r * wall / np.linalg.norm(wall)
        print(f"  雷达量到垃圾桶：中心离车 ({centre[0]:.2f}, {centre[1]:+.2f}) m，半径 {r * 100:.0f} cm"
              f"（{len(near)} 点）", flush=True)
        return cr.to_map(m.pose, centre), float(r)

    def measure_rim(self, centre, r):
        """Rim height (m) of the can at `centre` (map frame), radius r, by stereo from where the body is."""
        m, head = self.mover, self.head
        c = np.array(cr.to_robot(m.pose, centre))
        tops = []
        for _ in range(RIM_SHOTS):
            cam = head._look(0.0, RIM_PITCH)
            left, right = head._grab_pair()
            pts = head.stereo.points(left, right, cam["T_base_cam"])
            pts = pts[np.isfinite(pts).all(axis=-1)]
            near = pts[(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) <= r + 0.02) & (pts[:, 2] > 0.10) & (pts[:, 2] < 1.0)]
            if len(near) >= RIM_MIN_PTS:
                tops.append(float(np.percentile(near[:, 2], 95)))
        if not tops:
            raise SystemExit("双目没量到垃圾桶口：用 --bin-height 给出桶口离地高度")
        rim = float(np.median(tops))
        print(f"  双目量到垃圾桶口离地 {rim:.2f} m（{', '.join(f'{t:.2f}' for t in tops)}）", flush=True)
        return rim

    def survey_bin(self, obj):
        """Before any bottle is picked: to the can, its centre and radius by lidar, its rim by stereo
        (or --bin-height), and whether the arm reaches over it. Raises SystemExit when it cannot."""
        print(f"先空手去量 {obj['id']} 号垃圾桶", flush=True)
        try:
            self.go_stage(self._centre(obj), BIN_AT, None, "先去量垃圾桶")
            centre, r = self.measure_bin(self._centre(obj))
        except Fail as err:
            raise SystemExit(f"量不了垃圾桶：{err}")
        rim = self.args.bin_height if self.args.bin_height is not None else self.measure_rim(centre, r)
        try:
            arm_drop.plan_drop(pick.PARK, (BUMPER_X + r + BIN_GAP_M, BIN_Y), r, rim, MIN_HANG_M)
        except SystemExit as err:
            raise SystemExit(f"这个垃圾桶扔不进去：{err}")
        self.bin_at = (centre, r, rim)
        self.head.neck = None

    def to_bin(self, hang):
        """Into the can; False when the bottle was lost on the way."""
        m = self.mover
        where, _, rim = self.bin_at
        print(f"拿着瓶子去垃圾桶，桶口离地 {rim:.2f} m", flush=True)
        self.go_stage(where, BIN_AT, None, "拿着瓶子去垃圾桶")
        centre, r = self.measure_bin(where)
        goal = (BUMPER_X + r + BIN_GAP_M, BIN_Y)
        for _ in range(3):
            c = cr.to_robot(m.pose, centre)
            move = np.array(c) - goal
            if np.linalg.norm(move) < BIN_TOL_M:
                break
            self.slide(math.atan2(move[1], move[0]), float(np.linalg.norm(move)), "靠近垃圾桶", [c])
        c = cr.to_robot(m.pose, centre)
        # Squeeze hard again for the swing out over the can; a jaw that closes right up holds nothing.
        closed = self.arm.limits["gripper"][0]
        g = self.arm.grip(closed)
        if g <= closed + 3:
            print(f"  到了桶边夹爪合到 {g:.1f}°：瓶子路上掉了", flush=True)
            pick.park(self.arm)
            return False
        m.show("扔进垃圾桶")
        try:
            arm_drop.drop(self.arm, c, r, rim, hang)
        except SystemExit as err:
            raise Fail(str(err))
        print("  扔进去了", flush=True)
        return True

    # --- all of it ------------------------------------------------------------------------
    def run(self):
        """Every bottle the map knows, while it knows a trash can. False when there was nothing to do."""
        did = False
        while True:
            can = self.bin()
            todo = self.bottles()
            if (can is None and not self.args.no_bin) or not todo:
                return did
            if self.args.max_bottles and self.done >= self.args.max_bottles:
                return did
            did = True
            if self.bin_at is None and not self.args.no_bin:  # only pick up what can be dropped in
                self.survey_bin(can)
            obj = todo[0]
            self.skip.append(self._centre(obj))
            try:
                where, hang = self.fetch_bottle(obj)
            except Fail as err:
                print(f"  这个瓶子先不捡了：{err}", flush=True)
                self.end_fetch()
                continue
            self.skip.append(where)
            self.forget(where)
            if self.args.no_bin:
                self.done += 1
                print("  --no-bin：拿着瓶子停下（python ../llm_vla/pick_bottle/pick.py open 松开）", flush=True)
                return did
            try:
                if self.to_bin(hang):
                    self.done += 1
            except Fail as err:
                raise SystemExit(f"瓶子拿在手里，但扔不进垃圾桶：{err}")
            self.end_fetch()

    def drop_held(self, hang):
        """The bottle the gripper already holds (a run that stopped) into the can, before anything
        else; then the head learns its body again without the bottle hanging in view."""
        can = self.bin()
        if can is None:
            raise SystemExit("左臂上夹着瓶子，但地图上没有垃圾桶")
        self.survey_bin(can)
        try:
            if self.to_bin(hang):
                self.done += 1
        except Fail as err:
            raise SystemExit(f"瓶子拿在手里，但扔不进垃圾桶：{err}")
        self.end_fetch()
        if cr.turn_blocker(self.mover) is None:
            cr.learn_self(self.mover, self.head)
        else:
            print("  转身半径内有东西，不重新认车身（画面左下角瓶子挂过的地方还当车身）", flush=True)

    def end_fetch(self):
        try:
            pick.park(self.arm)  # nothing in hand here; does nothing when already parked
        except SystemExit as err:
            raise SystemExit(f"左臂收不回来：{err}")
        self.head.neck = None
        self.mover.cmap.rebuild()

    def forget(self, where):
        """The bottle that lay at `where` (map frame) is in the gripper: whatever the map modelled
        there and its marks go. Looks on the way there may have merged it into another model, so it
        is found by place, not by the model the fetch started from."""
        m, cmap = self.mover, self.mover.cmap
        cmap.clear_vis(cmap.mask_of([where], SKIP_NEAR_M / 2))
        m.objects = [o for o in m.objects
                     if o.get("size", o["length"]) > BOTTLE_MAX_M
                     or math.hypot(*(np.subtract(self._centre(o), where))) > SKIP_NEAR_M / 2]
        cmap.rebuild()
        cr.show_objects(m)


def forget_movable(mover):
    """A loaded map without the things that get moved about: what the arms pick up (bottles, paper)
    and their marks. Furniture, walls and the trash can stay."""
    cmap = mover.cmap
    gone = [o for o in mover.objects if o["trash"] or ("瓶" in o["label"] and "瓶盖" not in o["label"])]
    for o in gone:
        cmap.clear_vis(cmap.mask_of(o["points"], 0.05))
    cmap.trash[:] = False
    cmap.trash_low[:] = False
    mover.objects = [o for o in mover.objects if all(o is not g for g in gone)]
    cmap.rebuild()
    cr.show_objects(mover)
    print(f"  地图里的 {len(gone)} 个瓶子、纸屑等不算数了（会被挪走），重新看", flush=True)


def fast_follow(mover, head, route, status):
    """coverage_run.follow on a known map without the looks: turn to each leg (when nothing is inside the
    turning circle), then straight on in FAST_STEP_M steps as far as the lidar and the map leave room,
    the lidar watching while it drives. Where that cannot go on, the rest is coverage_run.follow."""
    for i, (tx, ty) in enumerate(route[1:], start=1):
        for _ in range(100):
            d = math.hypot(tx - mover.x, ty - mover.y)
            if d < 0.04:
                break
            heading = math.atan2(ty - mover.y, tx - mover.x)
            if abs(cr.wrap_deg(math.degrees(heading - mover.yaw))) > FAST_TURN_TOL_DEG:
                if cr.turn_blocker(mover) is not None:
                    return cr.follow(mover, head, [(mover.x, mover.y)] + list(route[i:]), status)
                mover.turn_to(heading, status)
            step = min(FAST_STEP_M, d, cr.free_ahead(mover, [], 0.0, FAST_STEP_M))
            if step < 0.05:
                return cr.follow(mover, head, [(mover.x, mover.y)] + list(route[i:]), status)
            mover.hold = heading
            mover.drive(0.0, step, status, speed=FAST_LIN)
            if mover.stuck:
                return cr.follow(mover, head, [(mover.x, mover.y)] + list(route[i:]), status)
            cell = mover.cmap.world_to_cell(tx, ty)
            if cell is None or not mover.cmap.drive_mask[cell]:
                return "replan"
    return "done"


def quick_survey(mover, head, status):
    """A turn in place in QUICK_STOPS steps, one look at the floor (QUICK_PITCHES) at each: what lies
    there now. No look up and no lidar mapping turn, the map is known."""
    for k in range(QUICK_STOPS):
        mover.show(f"{status} {k + 1}/{QUICK_STOPS}")
        mover.locate(cr.MATCH_WIN_M)
        cr.look(mover, head, pitches=QUICK_PITCHES)
        if k == QUICK_STOPS - 1:
            break
        if cr.turn_blocker(mover) is not None:
            print("  转身半径内有东西，这里只看到这些方向", flush=True)
            break
        mover.turn_by(360.0 / QUICK_STOPS, status)
    mover.survey_at = (mover.x, mover.y)


def viewpoints(cmap, x, y):
    """Places (map cells) the body can reach from which all open floor lies within VIEW_R_M of one,
    the one showing the most new floor first, nearest by road among near-equals."""
    k = max(1, int(round(cr.OPEN_FLOOR_M / cmap.res)))
    free = cmap.seen & ~cmap.occupied()
    todo = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0
    dist = drive_dist(cmap, x, y)
    reach = dist >= 0
    kern = disk(VIEW_R_M / cmap.res).astype(np.float32)
    todo &= ~(cv2.dilate(cmap.mask_of([(x, y)], 0.0).astype(np.uint8), kern.astype(np.uint8)) > 0)
    out = []
    while len(out) < VIEW_MAX:
        gain = cv2.filter2D(todo.astype(np.float32), -1, kern, borderType=cv2.BORDER_CONSTANT)
        gain[~reach] = 0
        best = float(gain.max())
        if best * cmap.res ** 2 < VIEW_MIN_M2:
            break
        ys, xs = np.nonzero(gain >= 0.8 * best)
        j = int(np.argmin(dist[ys, xs]))
        cell = (int(ys[j]), int(xs[j]))
        out.append(cell)
        todo &= ~(cv2.dilate(cmap.mask_of([cmap.cell_center(cell)], 0.0).astype(np.uint8), kern.astype(np.uint8)) > 0)
    return out


def rescan(mover, head):
    """What lies on the floor now, on a known map: a quick look round here, then to the viewpoints
    (nearest by road next) driving on the map and the lidar, a quick look round at each."""
    cmap = mover.cmap
    places = viewpoints(cmap, mover.x, mover.y)
    print(f"重新巡视：除了这里还有 {len(places)} 个观察点（每个点周围 {VIEW_R_M} m 的地面都看得到）", flush=True)
    tries = {}
    while places:
        dist = drive_dist(cmap, mover.x, mover.y)
        places.sort(key=lambda c: dist[c] if dist[c] >= 0 else 1 << 30)
        cell = places.pop(0)
        if dist[cell] < 0:
            print(f"  观察点 {cmap.cell_center(cell)} 现在开不过去，跳过", flush=True)
            continue
        gx, gy = cmap.cell_center(cell)
        print(f"去观察点 ({gx:+.2f}, {gy:+.2f})，还剩 {len(places)} 个", flush=True)
        route = route_to(cmap, mover.x, mover.y, cell)
        if route is None or fast_follow(mover, head, route, "重新巡视") != "done":
            tries[cell] = tries.get(cell, 0) + 1
            if tries[cell] < 2:
                places.append(cell)
            continue
        quick_survey(mover, head, "重新巡视：低头看一圈找瓶子")
    print("重新巡视完了", flush=True)


def work(args, view, robot):
    from imu import Imu
    from lidar_scan import Lidar
    from vln_scene import Head, nav

    head = Head(args.model, args.timeout_s, os.path.join(cr.MAP_DIR, "look", time.strftime("%Y%m%d_%H%M%S")))
    robot["head"] = head
    base = nav.Base(*head.neck_home)
    robot["base"] = base
    head.attach(base)
    arm = LeftArm()
    robot["arm"] = arm
    if args.holding:
        # A bottle still in the gripper: the jaw stays on it, eased, until it goes into the can.
        arm.enable()
        arm.ease_grip(arm.limits["gripper"][0])
        pick.park(arm, keep_grip=True)
    else:
        pick.park(arm)  # out of the floor in front, and where the grasp view's mask expects it
    try:
        imu = Imu(args.imu_port)
    except RuntimeError as err:
        raise SystemExit(f"{err}。检查 IMU 的 USB 线。没有 IMU 不能跑")
    robot["imu"] = imu
    lidar = Lidar(args.lidar_port)
    robot["lidar"] = lidar
    try:
        lidar.start()
    except Exception as err:
        raise SystemExit(f"雷达没打开：{err}。检查雷达 USB 和电机；定位靠雷达，没有雷达不能跑")
    head.lidar_xy = lidar.latest_xy
    mover = cr.Mover(base, imu, lidar, view.cmap, view)
    robot["mover"] = mover
    if args.use_map:
        # Off the map until it is found on it: the first turns must not stamp the body at (0, 0).
        mover.x = mover.y = saved_map.OFF_MAP
        if os.path.isdir(args.use_map):
            print(f"用 {args.use_map} 存的地图（{saved_map.load(mover, args.use_map)}）")
        else:
            saved_map.from_live(mover, args.use_map)
            print(f"用状态页文件 {args.use_map} 里的地图")
        # Not on the map yet, so only the lidar can say whether the body may turn here.
        xy = lidar.latest_xy()
        near = [] if xy is None else [math.hypot(float(x), float(y)) for x, y in xy
                                      if math.hypot(float(x), float(y)) <= cr.BODY_CLEARANCE_M + cr.TURN_SPARE]
        if near:
            raise SystemExit(f"转身半径内有东西（离车 {min(near):.2f} m）：把车挪到四周 0.4 m 内没东西的地方再定位")
        forget_movable(mover)
        mover.attach_imu()
        cr.learn_self(mover, head)
        saved_map.relocate(mover)
        mover.set_stage("explore")
        quick_survey(mover, head, "定位好了：低头看一圈找瓶子")
    else:
        mover.attach_imu()
        cr.warm_up(mover, head)
    fetch = Fetcher(mover, head, base, arm, args)
    if args.holding:
        print(f"--holding：先把左臂上的瓶子（长 {args.holding * 100:.0f} cm）扔进垃圾桶")
        fetch.drop_held(args.holding)
    if args.rescan:
        print("--rescan：在已有地图上去几个观察点低头看，找现在的瓶子")
        rescan(mover, head)
    elif args.no_explore:
        print("--no-explore：不探索，去捡已经看到的瓶子")
    else:
        print("探索：深度优先，把能到的地方都走完，再去捡瓶子")
        cr.explore(mover, head)
    # Only the whole map says whether there is a trash can at all.
    bottles = fetch.bottles()
    print(f"地图上有 {len(bottles)} 个瓶子：" + "，".join(
        f"{o['id']} 号 ({fetch._centre(o)[0]:+.2f}, {fetch._centre(o)[1]:+.2f})" for o in bottles))
    if fetch.bin() is None and not args.no_bin:
        msg = "地图上没有垃圾桶：瓶子一个也不捡（只想捡起来试试用 --no-bin）"
    elif not bottles:
        msg = "地图上没有瓶子"
    else:
        mover.set_stage("explore")
        fetch.run()
        left = len(fetch.bottles())
        did = "捡起" if args.no_bin else "扔进垃圾桶"
        msg = f"{did} {fetch.done} 个瓶子" + (f"，还有 {left} 个没捡成" if left else "")
    mover.show(msg)
    print(msg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin-height", type=float, default=None, help="垃圾桶口离地高度 (m)，默认用双目量的")
    p.add_argument("--max-bottles", type=int, default=0, help="扔够这么多个就停，0 = 不限")
    p.add_argument("--no-explore", action="store_true", help="建完图不探索，只去捡建图时看到的瓶子")
    p.add_argument("--no-bin", action="store_true", help="不去垃圾桶：捡起一个瓶子拿着就停")
    p.add_argument("--cap-gap", type=float, default=CAP_GAP_M, help="夹之前瓶盖中心离保险杠多远 (m)，默认 0.10")
    p.add_argument("--rescan", action="store_true", help="用存的地图（同 --use-map），到几个观察点低头看一圈找瓶子")
    p.add_argument("--use-map", nargs="?", const=saved_map.SAVED_DIR, default=None,
                   help="不建图：用存的地图（默认 vln/map/saved，或给一个 live*.npz），原地转一圈定位")
    p.add_argument("--holding", type=float, default=None, metavar="LENGTH_M",
                   help="左臂上已经夹着一个这么长的瓶子：地图好了先把它扔进垃圾桶")
    p.add_argument("--imu-port", default=None)
    p.add_argument("--lidar-port", default=None)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default="qwen3.8-max")
    p.add_argument("--timeout-s", type=float, default=20.0)
    args = p.parse_args()

    def _interrupt(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, _interrupt)

    if args.rescan and not args.use_map:
        args.use_map = saved_map.SAVED_DIR
    if args.use_map and not os.path.exists(args.use_map):
        raise SystemExit(f"没有 {args.use_map}：先跑一次建图，结束时会存下地图")
    if args.use_map and os.path.abspath(args.use_map) == os.path.join(cr.MAP_DIR, "live.npz"):
        raise SystemExit("live.npz 一开始就会被新地图覆盖：先复制一份再用（cp live.npz live_old.npz）")
    print(f"运行日志：{cr.start_log()}")
    view = MapView(OccMap())
    view.write(cr.MAP_DIR)
    cr.ensure_map_server(args.port)
    writer = view.start_writer(cr.MAP_DIR)
    print(f"状态页：http://{cr.local_ip()}:{args.port}")
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
        # The left arm keeps its torque (it holds where it is, a bottle too); its port is closed.
        steps = [("底盘停车", lambda: robot["base"].stop()), ("左臂", lambda: robot["arm"].close()),
                 ("雷达", lambda: robot["lidar"].close()), ("相机", lambda: robot["head"].close()),
                 ("IMU", lambda: robot["imu"].close()), ("底盘", lambda: robot["base"].close())]
        for name, fn in steps:
            try:
                fn()
            except KeyError:
                pass
            except BaseException as err:  # noqa: B902 - a second Ctrl-C included
                print(f"  关{name}时出错：{err!r}")
        writer.set()
        view.write(cr.MAP_DIR, running=False)
        m = robot.get("mover")
        if m is not None and abs(m.x) < saved_map.OFF_MAP / 2:  # only a map the body was found on
            print(f"地图存到 {saved_map.save(view.cmap, m)}")
        print(f"底盘已停。状态页仍开着：http://{cr.local_ip()}:{args.port}")


if __name__ == "__main__":
    main()
