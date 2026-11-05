#!/usr/bin/env python3
"""建图、探索，把区域里地上的垃圾都统计下来，用左臂夹起来扔进垃圾桶；--patrol 时之后定时巡视，看到新扔的垃圾再去捡。

  conda activate lerobot_env
  cd ~/workspaces/FXLerobot/FXlerobot/demo
  python bottle_to_bin.py                    # 桶口高度用双目量的
  python bottle_to_bin.py --bin-height 0.32  # 桶口离地 0.32 m（量不准或桶顶出了画面时给）
  python bottle_to_bin.py --no-explore --no-bin  # 测试：建完图不探索，去捡已看到的垃圾，拿着就停
  python bottle_to_bin.py --use-map --no-explore  # 用上次存的地图：原地转一圈定位，不建图
  python bottle_to_bin.py --rescan                # 用上次存的地图定位，到几个观察点低头看一圈找垃圾，捡起扔掉
  python bottle_to_bin.py --rescan --patrol 3     # 同上，捡完后每 3 分钟再巡视一圈，看到新垃圾就去捡（Ctrl-C 停）
  python bottle_to_bin.py --rescan --holding 0.16 # 左臂上还夹着上次没扔掉的 16 cm 瓶子：定位后先扔掉它

每次运行结束都把地图存到 vln/map/saved/；--use-map 读它（或 --use-map 某个 live*.npz：只有状态页文件的旧地图）。
读进来的地图里不留要捡的东西（瓶子、纸屑会被挪走），家具、墙和垃圾桶留着；定位后先原地用相机看一圈。

状态页同 coverage_run.py：http://<jetson>:8765 。Ctrl-C 停车，左臂停在原地保持力矩。

1) 建图：vln 的预热（雷达转两圈 + 视觉看一圈），地图、物体建模、定位都用 vln 的。
2) 探索：vln 的深度优先探索，把能到的地方都走完；探索完再看地图上有没有垃圾桶、有哪些垃圾。
   没有垃圾桶就停下说明，一个也不捡。垃圾怎么走（vln_scene）：瓶子、纸团绕着走，纸屑、纸片、纸巾压过去也行。
3) 统计：地图上认作垃圾的东西（名字、瓶子还是小件、位置、第一次看到的时间、结果）记在台账里，
   每轮捡完打印，存到 runs/trash_ledger.json。
4) 瓶子：按地图开到瓶子在车头前方约 0.6 m 的地方（停车位置和朝向一起选，转不了身就换个地方），头转到抓取视角（yaw -15, pitch 35），
   VLM 框出瓶子和瓶盖、GrabCut 抠出瓶身（detect_any.py，什么颜色的瓶子都行），量出瓶盖、方向和瓶长，
   一点点平移到瓶盖在保险杠前约 10 cm（--cap-gap）、夹爪能竖直朝下夹到为止（不斜着夹）；然后按 pick.py 的
   流程夹起、抬起，再合一次夹爪仍被瓶盖挡住才算拿起来，拿着瓶子慢慢收臂。
5) 小件（纸片、纸团、塑料片……）：同样开到旁边，平移到它在车头左前方 SMALL_AT，然后按
   llm_vla/pick_small 的办法夹：检测、手腕相机对准，固定爪顺着它平直的边贴近，合爪，收臂后看原处没了才算拿起来。
6) 垃圾桶：捡第一个之前先空手去量桶：雷达拟合圆心和半径，双目量桶口高度，左臂够不着就停下。
   扔的时候按地图开到桶前约 0.6 m，雷达再量一次圆心，平移到桶在车头左前方，
   左臂把东西举到桶口上方（桶口 + 挂下来的长度）松开，收臂，接着找下一个。
7) --patrol N：都捡完后每 N 分钟去观察点低头看一圈，看到新的垃圾就去捡、扔，再打印台账，直到 Ctrl-C。
只抓躺在地上的。瓶子挂下来的长度用量到的瓶长：左臂举到约 0.6 m，所以 0.21 m 长的瓶子桶口
最高约 0.36 m；扔不进的桶或瓶子，夹之前就停下说明。
"""

import argparse
import fcntl
import json
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
                str(ROOT / "llm_vla"), str(ROOT / "llm_vla" / "pick_small")]
import arm_drop  # noqa: E402
import coverage_run as cr  # noqa: E402
import detect_any  # noqa: E402
import pick  # noqa: E402
import pick_small  # noqa: E402
import ik  # noqa: E402
from arm import LeftArm, joint_limits  # noqa: E402
from coverage_plan import drive_dist, route_to  # noqa: E402
from live_map import MapView  # noqa: E402
from occmap import OccMap, disk  # noqa: E402
import saved_map  # noqa: E402
from detect_any import to_plane  # noqa: E402
from vln_scene import _numbers, _px, ask_json, cal, is_bin, to_pick  # noqa: E402

# Only upright grasps: leaning, the jaws hold the cap by a corner (two lost bottles at 20 deg) and miss
# a cap they did not come straight down onto. Out of upright reach the body moves closer instead.
pick.TILT_STEPS_DEG = (0.0,)

RUNS = HERE / "runs"
GRASP_NECK = (-15.0, 35.0)    # pick.ROBOT_MASK is drawn for this head pose (and the arm in PARK)
NECK_APPROACH_DEG = 3.0       # neck backlash: finish from lower yaw and higher pitch, like neck_look.py
NECK_TOL_DEG = 1.0
NECK_SAME_DEG = 0.6           # look_neck: the neck this near where it should be is not moved
# Staging: the body stops where the bottle's middle lies here in the base frame, inside the grasp
# view (it sees the floor 0.25-0.8 m ahead) and far enough that the body can still turn.
BOTTLE_AT = (0.58, 0.08)
STAGE_HEADINGS = 24
STAGE_TRIES = 5
STAGE_FARTHER_M = (0.10, 0.20, 0.30)  # a stage this much further from the target when nearer ones fail
STAGE_FAR_COST = 2.0                   # score per metre further (a metre of driving costs 1)
STAGE_SPREAD_M = 0.15                  # a spot tried is not tried again, nor any this near it
BIN_TRIES = 3                           # the can measured from up to this many places before giving up
HEM_TRIES = 2                           # spots missed in a row for want of room to turn: back away ...
HEM_BACK_M = 0.12                       # ... this far, once
# Near a thing the body stops where it lies about NEAR_M ahead (go_near: go_stage at BOTTLE_AT, a little
# further when that does not go); already that near and facing it, it stays.
NEAR_M = (0.45, 1.0)
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
# At FINE_LIN, sliding sideways over the carpet, the wheels turned and the body stayed (0.02 of 0.13 m,
# three times, with room ahead; the run then stopped, 2026-11-05): a slide that hardly moves goes once
# more at KICK_LIN for what is left, and if that does not move either only this piece is given up.
KICK_LIN = 0.10
SEEN_VS_MAP_M = 0.30          # the cameras' bottle this far from the map's is another thing
PICK_TRIES = 2
HANG_CUT_M = 0.22             # a bottle partly out of view hangs at least this far (its length is not known)
MIN_HANG_M = 0.10             # the shortest bottle: a can too tall even for it is not worth a pick
BOTTLE_MAX_M = 0.35           # a "bottle" in the map longer than this is not one
SKIP_NEAR_M = 0.25            # things done or given up are skipped within this of where they were
# A small thing (not a bottle) is picked from here (base frame): in the grasp view, ~19 cm from the left
# arm's base, where pick_small reaches it upright (things 25 cm out needed the tool leaning and failed).
SMALL_AT = (0.26, 0.10)
SMALL_TOL_M = 0.03            # slides until the thing is this close to SMALL_AT
SMALL_STEPS = 4
SMALL_HANG_M = 0.04           # a small thing reaches this far below the grasp point, over the can
LEDGER = RUNS / "trash_ledger.json"
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
# The can is found by detection first: the head turned to it, the VLM's box gives its bearings, and
# only lidar returns within them count. A black wire-mesh can let most beams through (13 returns) and
# the arc fit took the tripod's legs behind it for the can: 0.21 m right of the body where the can stood
# left, and the stereo then found no rim there (2026-11-05).
BIN_PITCH = 15.0              # head pitch for the can's photo: from 0.5 m its foot stays in the picture
BIN_PAD_DEG = 2.0             # lidar returns this far outside the box's bearings still count
BIN_DEPTH_M = 0.06            # returns this far behind the nearest one in the box are the can too
BIN_MIN_PTS = 3               # fewer than this: the box's bottom on the floor gives the near wall
BIN_PROMPT = """你是机器人的视觉模块。框出图中的垃圾桶（垃圾箱、纸篓），框要紧贴它，包括桶口和桶底。
只输出 JSON，不要其他文字：{"objects": [{"bbox": [x1, y1, x2, y2], "label": "垃圾桶"}, ...]}；没有就输出 {"objects": []}。
坐标用 0-1000 归一化，图像左上角为 (0,0)，右下角为 (1000,1000)。"""
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
        self.skip = []      # map points of things done or given up
        self.done = 0
        self.bin_at = None  # (centre in the map frame, radius, rim height) once measured
        self.ledger = []    # every piece of trash seen: label, kind, map x/y, first seen, result
        self.later = []     # where pieces given up this round lay: one more go at the end
        self.second_go = False
        # pick_small on this run's devices: the base owns the neck's port, the head the cameras.
        self.rig = pick_small.Rig(arm=arm, look=self.look_neck, neck_deg=self.neck_deg,
                                  frames=lambda: dict(zip(("left", "right"), head._grab_pair())))

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

    @staticmethod
    def kind(o):
        """"bottle" (picked by its cap, pick.py) or "small" (pick_small)."""
        return "bottle" if "瓶" in o["label"] and "瓶盖" not in o["label"] else "small"

    def targets(self):
        """The trash on the map still to fetch, nearest first: what vln marked as trash (vln_scene
        to_pick, in size), not in a can, not done or given up."""
        out = []
        for o in self.mover.objects:
            if not (o.get("trash") and to_pick(o["label"])):
                continue
            if self.kind(o) == "bottle" and o.get("size", o["length"]) > BOTTLE_MAX_M:
                continue
            c = self._centre(o)
            if self.in_bin(c):
                continue
            if all(math.hypot(c[0] - s[0], c[1] - s[1]) > SKIP_NEAR_M for s in self.skip):
                out.append(o)
        return sorted(out, key=lambda o: math.hypot(self._centre(o)[0] - self.mover.x,
                                                     self._centre(o)[1] - self.mover.y))

    # --- the tally --------------------------------------------------------------------------
    def entry(self, c):
        """The ledger's entry for trash at map point c, or None."""
        return next((e for e in self.ledger if math.hypot(e["x"] - c[0], e["y"] - c[1]) <= SKIP_NEAR_M), None)

    def count(self):
        """Every piece of trash the map knows goes into the ledger; returns how many were new."""
        new = 0
        for o in self.targets():
            c = self._centre(o)
            if self.entry(c) is None:
                self.ledger.append({"n": len(self.ledger) + 1, "label": o["label"], "kind": self.kind(o),
                                    "x": round(float(c[0]), 3), "y": round(float(c[1]), 3),
                                    "seen": time.strftime("%H:%M:%S"), "result": "待捡"})
                new += 1
        self.save_ledger()
        return new

    def settle(self, c, result):
        e = self.entry(c)
        if e is not None:
            e["result"], e["at"] = result, time.strftime("%H:%M:%S")
            self.save_ledger()

    def save_ledger(self):
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(self.ledger, indent=1, ensure_ascii=False) + "\n")

    def print_ledger(self, title):
        print(f"{title}：一共见过 {len(self.ledger)} 个垃圾（{LEDGER}）", flush=True)
        for e in self.ledger:
            kind = "瓶子" if e["kind"] == "bottle" else "小件"
            print(f"  {e['n']:2d}. {e['label']}（{kind}）({e['x']:+.2f}, {e['y']:+.2f}) {e['seen']} 看到：{e['result']}",
                  flush=True)
        tally = {}
        for e in self.ledger:
            tally[e["result"]] = tally.get(e["result"], 0) + 1
        print("  " + "，".join(f"{k} {v}" for k, v in tally.items()), flush=True)
        self.mover.show(f"垃圾 {len(self.ledger)} 个：" + "，".join(f"{k} {v}" for k, v in tally.items()))

    def bin(self):
        cans = [o for o in self.mover.objects if is_bin(o["label"])]
        return max(cans, key=lambda o: o["seen"]) if cans else None

    # --- driving --------------------------------------------------------------------------
    def stage_spots(self, target, at, axis_deg, tried):
        """[(score, cell, heading, place, how far)] where map point `target` lies at base-frame `at`, or
        STAGE_FARTHER_M further ahead, on cells the body can drive to now, away from `tried` places."""
        m, cmap = self.mover, self.mover.cmap
        dist = drive_dist(cmap, m.x, m.y)
        out = []
        for extra in (0.0,) + STAGE_FARTHER_M:
            ax, ay = at[0] + extra, at[1]
            for k in range(STAGE_HEADINGS):
                h = 2 * math.pi * k / STAGE_HEADINGS
                c, s_ = math.cos(h), math.sin(h)
                x, y = target[0] - (c * ax - s_ * ay), target[1] - (s_ * ax + c * ay)
                cell = cmap.world_to_cell(x, y)
                if cell is None or dist[cell] < 0:
                    continue
                if any(math.hypot(x - tx, y - ty) < STAGE_SPREAD_M for tx, ty in tried):
                    continue
                score = dist[cell] * cmap.res + 0.3 * abs(wrap(h - m.yaw)) / math.pi + STAGE_FAR_COST * extra
                if axis_deg is not None:
                    score += 0.5 * abs(math.cos(h - math.radians(axis_deg)))
                out.append((score, cell, h, (x, y), ax))
        return sorted(out)

    def go_stage(self, target, at, axis_deg=None, status=""):
        """Drive to where map point `target` lies at base-frame `at` (or up to STAGE_FARTHER_M further
        ahead), facing it. With the target's long side known (axis_deg, map frame) headings across it
        come first. The spots are worked out again from the map after every miss: a can standing in
        clutter beyond the drivable floor had three spots, all on that floor's edge, and when looks on
        the way moved the edge the first went and the other two were lost unsaid (2026-11-05)."""
        m, cmap = self.mover, self.mover.cmap
        tried, hemmed = [], 0
        for attempt in range(STAGE_TRIES):
            spots = self.stage_spots(target, at, axis_deg, tried)
            if not spots:
                break
            _, cell, h, place, ax = spots[0]
            tried.append(place)
            route = route_to(cmap, m.x, m.y, cell)
            if route is None:
                print(f"  停靠点 ({place[0]:+.2f}, {place[1]:+.2f}) 规划不出路线，换一个", flush=True)
                continue
            if ax > at[0]:
                print(f"  近处停不了，停在离它 {ax:.2f} m 处", flush=True)
            result = fast_follow(m, self.head, route, status)
            if result == "done" and cr.safe_turn(m, self.head, h, status):
                return
            print(f"  没开到停靠点 ({place[0]:+.2f}, {place[1]:+.2f})（{result}），按现在的地图换一个", flush=True)
            hemmed = hemmed + 1 if result in ("turn", "done") else 0
            self.unhem(hemmed, status)
        raise Fail(f"找不到能开过去的停靠点（试了 {len(tried)} 个）")

    def go_near(self, target, status):
        """Beside map point `target`, facing it so that it lies at BOTTLE_AT (or a little further ahead):
        go_stage, a spot and the heading to have there chosen together. Picking any spot 0.45-1 m off
        by road and turning to face it once there left things beside or behind the body where it could
        not turn (6 of 11 given up, 2026-11-05); go_stage counts a spot it cannot turn at as missed and
        plans the next from the map as it is then. Raises Fail when none is reached."""
        m = self.mover
        e = np.array(cr.to_robot(m.pose, target))
        if NEAR_M[0] <= np.linalg.norm(e) <= NEAR_M[1] and abs(math.atan2(e[1], e[0])) < math.radians(30):
            return  # already beside it, facing it
        try:
            self.go_stage(target, BOTTLE_AT, None, status)
            return
        except Fail as err:
            # In clutter a spot right in front may not be drivable at all (three bottles given up so,
            # 2026-11-03): then any spot NEAR_M off it, and the slides make up for the heading.
            print(f"  {err}：改去离它 {NEAR_M[0]}-{NEAR_M[1]} m 的任意能到的地方", flush=True)
        self.go_any_near(target, status)

    def unhem(self, hemmed, status):
        """After spots missed in a row for want of room to turn here (`hemmed` of them), the next
        spot would start with the same turn: once, back away HEM_BACK_M (as far as the map behind
        allows) to get room; after that, give up here. Raises Fail when it cannot."""
        if hemmed < HEM_TRIES:
            return
        m = self.mover
        if hemmed == HEM_TRIES:
            room = cr.map_room(m, math.pi, HEM_BACK_M)
            if room >= 0.05:
                print(f"  这里转不了身：先往后退 {room:.2f} m 腾出地方", flush=True)
                cr.back_up(m, room, status)
                return
        raise Fail("这里转不了身，往后也退不开")

    def go_any_near(self, target, status):
        """To a spot NEAR_M from map point `target` the body can reach, nearest by road, then turned to
        face it as far as it can turn. Raises Fail when no such spot can be reached."""
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
            tried, hemmed = [], 0
            for cell in zip(ys[ok][order], xs[ok][order]):
                if len(tried) == STAGE_TRIES or near():
                    break
                p_ = cmap.cell_center(cell)
                if any(math.hypot(p_[0] - q[0], p_[1] - q[1]) < STAGE_SPREAD_M for q in tried):
                    continue
                tried.append(p_)
                route = route_to(cmap, m.x, m.y, (int(cell[0]), int(cell[1])))
                result = "noroute" if route is None else fast_follow(m, self.head, route, status)
                if result != "done" and not near():
                    print(f"  没开到它附近（{result}），换一个地方", flush=True)
                    hemmed = hemmed + 1 if result == "turn" else 0
                    self.unhem(hemmed, status)
            if not near():
                raise Fail(f"开不到离它 {NEAR_M[0]}-{NEAR_M[1]} m 的地方")
        bearing = math.atan2(target[1] - m.y, target[0] - m.x) - math.atan2(BOTTLE_AT[1], BOTTLE_AT[0])
        if not cr.safe_turn(m, self.head, bearing, status):
            print("  转不过去正对它，先这样，靠平移对准", flush=True)

    def slide(self, rel, dist, status, ignore=(), search=False):
        """Move `dist` towards `rel` (rad from the heading), heading kept, slowly. The lidar returns
        near `ignore` (base frame: the thing being approached) do not stop it, any other in the way does.
        search: the map limits the way and the lidar watches while it moves (towards where the map says
        a thing is that the cameras do not see, or up to a small thing on the floor)."""
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
        moved = m.drive(rel, dist, status, record=False, guard=search, speed=FINE_LIN, count_stuck=False)
        if m.stuck:
            print(f"  慢速平移只走了 {moved:.2f} / {dist:.2f} m，加快再推一次", flush=True)
            m.hold = m.yaw
            again = m.drive(rel, dist - moved, status, record=False, guard=search, speed=KICK_LIN, count_stuck=False)
            if m.stuck:
                raise Fail(f"朝 {math.degrees(rel):+.0f}° 平移推不动（走了 {moved + again:.2f} / {dist:.2f} m）")

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

    def look_neck(self, yaw, pitch):
        """pick_small's neck: to (yaw, pitch), finishing from lower yaw and higher pitch (backlash);
        the neck as read after. Already there it stays: the backlash wiggle at every look (several
        per piece of trash) kept the head nodding all the time (the user, 2026-11-05)."""
        now = self.neck_deg()
        if abs(now["neck_yaw"] - yaw) <= NECK_SAME_DEG and abs(now["neck_pitch"] - pitch) <= NECK_SAME_DEG:
            return now
        self.base.look(yaw - NECK_APPROACH_DEG, pitch + NECK_APPROACH_DEG, settle_s=0.6)
        self.base.look(yaw, pitch, settle_s=0.8)
        self.head.neck = None  # the next vln look points the head again
        return self.neck_deg()

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

    def fetch_small(self, obj):
        """A small thing (pick_small): beside it (go_near), then slid until it lies at SMALL_AT where
        the grasp view sees it (the map's place was 10 cm off: a scrap ended up 27 cm from the arm's
        base, out of upright reach, 2026-11-05), then aimed with the wrist camera and picked. Returns
        where it lay (map frame) and how far it hangs below the grasp point."""
        m = self.mover
        centre = self._centre(obj)
        print(f"去捡 {obj['id']} 号{obj['label']} ({centre[0]:+.2f}, {centre[1]:+.2f})", flush=True)
        self.go_near(centre, "去垃圾旁边")
        args = argparse.Namespace(model=self.args.model, dx=0.0, dy=0.0)
        for attempt in range(PICK_TRIES):
            for step in range(SMALL_STEPS):
                # Where it is: as the grasp view measures it, or as the map has it when not in view.
                try:
                    _, seen = pick_small.look(self.args.model, self.rig, expect=cr.to_robot(m.pose, centre),
                                              plan_it=False)
                    here = np.array(seen["mid_base_m"][:2])
                    centre = cr.to_map(m.pose, here)
                except SystemExit as err:
                    print(f"  {err}", flush=True)
                    here = np.array(cr.to_robot(m.pose, centre))
                move = here - np.array(SMALL_AT)
                if np.linalg.norm(move) < SMALL_TOL_M:
                    break
                # The map limits the way and the lidar watches (search): sliding blind into a can
                # knocked its stand over (2026-11-04); backwards only the map can see. Held short (a
                # paper ball is an obstacle on the map), the grasp view and the arm decide.
                try:
                    self.slide(math.atan2(move[1], move[0]), float(np.linalg.norm(move)), "靠近垃圾",
                               [tuple(here)], search=True)
                except Fail as err:
                    print(f"  {err}，就在这里夹", flush=True)
                    break
            self.cool_gripper()
            try:
                d, run = pick_small.look(self.args.model, self.rig, expect=cr.to_robot(m.pose, centre))
            except SystemExit as err:
                print(f"  {err}", flush=True)
            else:
                centre = cr.to_map(m.pose, run["mid_base_m"][:2])
                try:
                    if pick_small.cmd_grab(args, self.rig, d, run):
                        return centre, SMALL_HANG_M
                except SystemExit as err:  # aimed badly, stuck on the jaws, out of reach: nothing held
                    print(f"  {err}", flush=True)
                pick.park(self.arm)
            m.show("没夹起来，再试一次")
        raise Fail(f"{PICK_TRIES} 次都没夹起来")

    def fetch(self, obj):
        return self.fetch_bottle(obj) if self.kind(obj) == "bottle" else self.fetch_small(obj)

    # --- the trash can --------------------------------------------------------------------
    def bin_box(self, expect):
        """The head turned to map point `expect`, a photo, and the VLM's box round the can nearest that
        bearing: (box in pixels, camera model, photo), or None when it boxes no can."""
        m, head = self.mover, self.head
        e = np.array(cr.to_robot(m.pose, expect))
        bearing = math.degrees(math.atan2(e[1], e[0]))
        cam = head._look(float(np.clip(-bearing, -40.0, 40.0)), BIN_PITCH)  # neck yaw + is right
        left, _ = head._grab_pair()
        ans = ask_json(left, BIN_PROMPT, self.args.model, self.args.timeout_s) or {}
        boxes = []
        for o in ans.get("objects") or []:
            b = _numbers(o.get("bbox") or o.get("bbox_2d"), 4) if isinstance(o, dict) else None
            if b is not None:
                boxes.append((_px(b[0], cal.WIDTH), _px(b[1], cal.HEIGHT), _px(b[2], cal.WIDTH), _px(b[3], cal.HEIGHT)))
        if not boxes:
            return None
        box = min(boxes, key=lambda b: abs(self.px_bearing((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, cam) - bearing))
        vis = left.copy()
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        RUNS.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(RUNS / f"bin_{time.strftime('%H%M%S')}.jpg"), vis)
        return box, cam, left

    @staticmethod
    def px_bearing(u, v, cam):
        """Bearing (deg, base frame, + left) of the ray through pixel (u, v)."""
        n = cv2.undistortPoints(np.array([[[u, v]]], float), cam["K"], cam["dist"]).reshape(2)
        d = cam["T_base_cam"][:3, :3] @ np.array([n[0], n[1], 1.0])
        return math.degrees(math.atan2(d[1], d[0]))

    def measure_bin(self, expect):
        """Centre (map frame) and radius of the can near map point `expect`: detected (bin_box), then
        the lidar returns within its box's bearings, the nearest bunch, fitted with the can's radius
        (--bin-diameter, else BIN_R_DEFAULT); with too few of them, the box's bottom on the floor is the
        near foot of the can and the centre lies a radius behind it."""
        m = self.mover
        r = self.args.bin_diameter / 2 if self.args.bin_diameter else BIN_R_DEFAULT
        got = self.bin_box(expect)
        if got is None:
            raise Fail("抓不到垃圾桶：VLM 在画面里没框出垃圾桶")
        (x1, y1, x2, y2), cam, _ = got
        ym = (y1 + y2) / 2
        a1, a2 = sorted((self.px_bearing(x1, ym, cam), self.px_bearing(x2, ym, cam)))
        scans = m.lidar.scans(n=3, timeout=4.0)
        pts = np.vstack(scans) if scans else np.zeros((0, 2))
        az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) if len(pts) else np.zeros(0)
        inside = pts[(az >= a1 - BIN_PAD_DEG) & (az <= a2 + BIN_PAD_DEG)] if len(pts) else pts
        rng = np.linalg.norm(inside, axis=1) if len(inside) else np.zeros(0)
        wall = inside[rng <= rng.min() + BIN_DEPTH_M] if len(inside) else inside
        mid_dir = math.radians((a1 + a2) / 2)
        if len(wall) >= BIN_MIN_PTS:
            # The centre r behind the near wall, along the box's middle bearing, refined so that the
            # wall's returns lie r from it.
            near = float(np.median(np.linalg.norm(wall, axis=1)))
            c = np.array([math.cos(mid_dir), math.sin(mid_dir)]) * (near + r)
            for _ in range(20):
                dv = wall - c
                dist = np.linalg.norm(dv, axis=1)
                Jm = -dv / dist[:, None]
                c = c - np.linalg.lstsq(Jm, dist - r, rcond=None)[0]
            how = f"雷达 {len(wall)} 点"
        else:
            foot = to_plane((x1 + x2) / 2, min(y2, cal.HEIGHT - 1), cam, 0.0)
            if foot is None or y2 >= cal.HEIGHT - 3:
                raise Fail(f"垃圾桶框里只有 {len(wall)} 个雷达点，桶底也不在画面里，量不出位置")
            f = np.array(foot[:2])
            c = f + r * f / np.linalg.norm(f)
            how = f"框底投到地面（框里雷达只有 {len(wall)} 点）"
        print(f"  量到垃圾桶：中心离车 ({c[0]:.2f}, {c[1]:+.2f}) m，半径 {r * 100:.0f} cm（{how}，"
              f"框的方位 {a1:+.0f}~{a2:+.0f}°）", flush=True)
        self.bin_photo = got
        return cr.to_map(m.pose, c), float(r)

    def measure_rim(self, centre, r):
        """Rim height (m) of the can at `centre` (map frame): the top of its box (the far rim, the
        camera being above it) taken that far away (the can's far side); --bin-height wins over this."""
        m = self.mover
        box, cam, _ = self.bin_photo
        c = np.array(cr.to_robot(m.pose, centre))
        n = cv2.undistortPoints(np.array([[[(box[0] + box[2]) / 2, box[1]]]], float), cam["K"], cam["dist"]).reshape(2)
        R, t = cam["T_base_cam"][:3, :3], cam["T_base_cam"][:3, 3]
        d = R @ np.array([n[0], n[1], 1.0])
        # Along the ray, the point as far out (from the base centre, along the ray's bearing) as the
        # can's far side: its height is the rim's.
        u = d[:2] / np.linalg.norm(d[:2])
        far = float(np.linalg.norm(c)) + r
        rim = float(t[2] + (far - float(t[:2] @ u)) / np.linalg.norm(d[:2]) * d[2])
        if not 0.10 <= rim <= 0.60 or box[1] <= 3:
            raise SystemExit(f"从垃圾桶的框量出桶口离地 {rim:.2f} m，不可信：用 --bin-height 给出桶口离地高度")
        print(f"  从垃圾桶的框量到桶口离地 {rim:.2f} m", flush=True)
        return rim

    def survey_bin(self, obj):
        """Before any bottle is picked: to the can, its centre and radius by lidar, its rim by stereo
        (or --bin-height), and whether the arm reaches over it. Raises SystemExit when it cannot."""
        print(f"先空手去量 {obj['id']} 号垃圾桶", flush=True)
        for attempt in range(BIN_TRIES):
            try:
                self.reach_bin(self._centre(obj), "先去量垃圾桶")
                centre, r = self.measure_bin(self._centre(obj))
                break
            except Fail as err:
                print(f"  量不了垃圾桶（第 {attempt + 1} 次）：{err}", flush=True)
                if attempt + 1 == BIN_TRIES or not self.elsewhere(self._centre(obj)):
                    raise SystemExit(f"量不了垃圾桶：{err}。没有桶扔不了，停下")
        rim = self.args.bin_height if self.args.bin_height is not None else self.measure_rim(centre, r)
        try:
            arm_drop.plan_drop(pick.PARK, (BUMPER_X + r + BIN_GAP_M, BIN_Y), r, rim, MIN_HANG_M)
        except SystemExit as err:
            raise SystemExit(f"这个垃圾桶扔不进去：{err}")
        self.bin_at = (centre, r, rim)
        self.head.neck = None

    def reach_bin(self, where, status):
        """Beside the can at map point `where`, facing it: go_stage with it at BIN_AT (or a little
        further), else any spot NEAR_M from it (the drop's slides then bring it into place). Carrying a
        paper ball, no stage was left round a can in clutter after a long pick beside it, and the run
        stopped with the ball in the jaws (2026-11-05)."""
        try:
            self.go_stage(where, BIN_AT, None, status)
            return
        except Fail as err:
            print(f"  {err}：改去离桶 {NEAR_M[0]}-{NEAR_M[1]} m 的任意能到的地方", flush=True)
        self.go_any_near(where, status)

    def elsewhere(self, target):
        """To another viewpoint on the map, on another side of map point `target` from here, so the
        next try at it comes from there. False when there is none the body gets to."""
        m, cmap = self.mover, self.mover.cmap
        here = math.atan2(m.y - target[1], m.x - target[0])
        dist = drive_dist(cmap, m.x, m.y)
        places = []
        for cell in viewpoints(cmap, m.x, m.y) + [c for c in self.stage_cells(target)]:
            if dist[cell] < 0:
                continue
            x, y = cmap.cell_center(cell)
            side = abs(wrap(math.atan2(y - target[1], x - target[0]) - here))
            places.append((-side + 0.3 * dist[cell] * cmap.res, cell))
        for _, cell in sorted(places)[:3]:
            route = route_to(cmap, m.x, m.y, cell)
            if route is not None and fast_follow(m, self.head, route, "换个方向再试") == "done":
                print(f"  换到 {np.round(cmap.cell_center(cell), 2).tolist()}，从这边再试", flush=True)
                return True
        return False

    def stage_cells(self, target):
        """Cells round map point `target`, 0.8-1.2 m off, the body can drive to: places to look from."""
        m, cmap = self.mover, self.mover.cmap
        dist = drive_dist(cmap, m.x, m.y)
        out = []
        for k in range(STAGE_HEADINGS):
            for d in (0.8, 1.0, 1.2):
                a = 2 * math.pi * k / STAGE_HEADINGS
                cell = cmap.world_to_cell(target[0] + d * math.cos(a), target[1] + d * math.sin(a))
                if cell is not None and dist[cell] >= 0:
                    out.append(cell)
        return out

    def to_bin(self, hang, check_held=True):
        """Into the can; False when the bottle was lost on the way (check_held: a bottle's cap keeps
        the jaws apart; a paper scrap lets them close nearly all the way, so for it there is no check)."""
        m = self.mover
        where, _, rim = self.bin_at
        print(f"拿着它去垃圾桶，桶口离地 {rim:.2f} m", flush=True)
        for attempt in range(BIN_TRIES):
            try:
                self.reach_bin(where, "拿着垃圾去垃圾桶")
                centre, r = self.measure_bin(where)
                break
            except Fail as err:
                print(f"  到不了垃圾桶边（第 {attempt + 1} 次）：{err}", flush=True)
                if attempt + 1 == BIN_TRIES or not self.elsewhere(where):
                    raise
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
        if check_held and g <= closed + 3:
            print(f"  到了桶边夹爪合到 {g:.1f}°：瓶子路上掉了", flush=True)
            pick.park(self.arm)
            return False
        m.show("扔进垃圾桶")
        try:
            arm_drop.drop(self.arm, c, r, rim, hang)
        except SystemExit as err:
            raise Fail(str(err))
        print(f"  扔进去了（桶边照片 {self.drop_photo(centre)}）", flush=True)
        return True

    def drop_photo(self, centre):
        """A photo of the can (map point `centre`) after a drop, the head turned to it, to tell later
        whether the thing went in: the arm only knows it let go over where the can was measured."""
        try:
            e = np.array(cr.to_robot(self.mover.pose, centre))
            self.head._look(float(np.clip(-math.degrees(math.atan2(e[1], e[0])), -40.0, 40.0)), BIN_PITCH)
            left, _ = self.head._grab_pair()
            path = RUNS / f"drop_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(path), left)
            return path
        finally:
            self.head.neck = None

    # --- all of it ------------------------------------------------------------------------
    def run(self):
        """Every piece of trash the map knows, while it knows a trash can. False when there was nothing
        to do. Each goes into the ledger first, and its result after."""
        did = False
        self.later, self.second_go = [], False
        self.mover.follow_near_only = True  # the map is built: one near look per step on the way
        self.count()
        while True:
            can = self.bin()
            todo = self.targets()
            if not todo and self.later and not self.second_go:
                # The map has grown since: what could not be got to gets one more go.
                print(f"再试一次刚才没捡成的 {len(self.later)} 个", flush=True)
                self.skip = [s_ for s_ in self.skip if all(math.hypot(s_[0] - c[0], s_[1] - c[1]) > 1e-6 for c in self.later)]
                self.later, self.second_go = [], True
                todo = self.targets()
            if (can is None and not self.args.no_bin) or not todo:
                return did
            if self.args.max_items and self.done >= self.args.max_items:
                return did
            did = True
            if self.bin_at is None and not self.args.no_bin:  # only pick up what can be dropped in
                self.survey_bin(can)
            obj = todo[0]
            self.count()  # looks on the way may have found more
            c0 = self._centre(obj)
            self.skip.append(c0)
            try:
                where, hang = self.fetch(obj)
            except Fail as err:
                print(f"  这个先不捡了，先捡别的：{err}", flush=True)
                self.settle(c0, f"没捡成（{err}）")
                self.later.append(c0)
                self.end_fetch()
                continue
            self.skip.append(where)
            self.forget(where)
            if self.args.no_bin:
                self.done += 1
                self.settle(c0, "捡起来了（没扔）")
                print("  --no-bin：拿着它停下（python ../llm_vla/pick_bottle/pick.py open 松开）", flush=True)
                return did
            try:
                if self.to_bin(hang, check_held=self.kind(obj) == "bottle"):
                    self.done += 1
                    self.settle(c0, "扔进垃圾桶了")
                else:
                    self.settle(c0, "路上掉了")
            except Fail as err:
                raise SystemExit(f"垃圾拿在手里，但扔不进垃圾桶：{err}")
            self.end_fetch()

    def drop_held(self, hang):
        """The bottle the gripper already holds (a run that stopped) into the can, before anything
        else; then the head learns its body again without the bottle hanging in view."""
        can = self.bin()
        if can is None:
            raise SystemExit("左臂上夹着瓶子，但地图上没有垃圾桶")
        self.survey_bin(can)
        try:
            # A small thing (short hang) lets the jaws close nearly all the way: no held check.
            if self.to_bin(hang, check_held=hang >= MIN_HANG_M):
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
        quick_survey(mover, head, "重新巡视：低头看一圈找垃圾")
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
        quick_survey(mover, head, "定位好了：低头看一圈找垃圾")
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
    fetch.count()
    fetch.print_ledger("地图上的垃圾")
    if fetch.bin() is None and not args.no_bin:
        msg = "地图上没有垃圾桶：一个也不捡（只想捡起来试试用 --no-bin）"
        mover.show(msg)
        print(msg)
        return
    mover.set_stage("explore")
    fetch.run()
    fetch.print_ledger("这一轮捡完")
    while args.patrol:
        # The user throws new trash in now and then: look again every so often, fetch what is new.
        mover.show(f"等 {args.patrol:g} 分钟再巡视（Ctrl-C 停）")
        print(f"等 {args.patrol:g} 分钟再巡视（Ctrl-C 停）", flush=True)
        time.sleep(args.patrol * 60.0)
        rescan(mover, head)
        new = fetch.count()
        print(f"巡视看到 {new} 个新垃圾", flush=True)
        if new or fetch.targets():
            mover.set_stage("explore")
            fetch.run()
        fetch.print_ledger("巡视完")
    msg = f"扔进垃圾桶 {fetch.done} 个" if not args.no_bin else f"捡起 {fetch.done} 个"
    mover.show(msg)
    print(msg)

def main():
    # The left arm's port has one owner at a time: this and llm_vla/pick_small share the lock.
    lock = open(pick_small.LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("pick_small.py 或另一个 bottle_to_bin.py 正在用左臂，先等它结束或停掉它")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin-height", type=float, default=None, help="垃圾桶口离地高度 (m)，默认从垃圾桶的框量")
    p.add_argument("--bin-diameter", type=float, default=None, help="垃圾桶口直径 (m)，默认按 0.24 m 算")
    p.add_argument("--max-items", type=int, default=0, help="扔够这么多个就停，0 = 不限")
    p.add_argument("--patrol", type=float, default=0.0, metavar="MIN",
                   help="捡完后每隔这么多分钟巡视一圈，看到新扔的垃圾就去捡（0 = 不巡视）")
    p.add_argument("--no-explore", action="store_true", help="建完图不探索，只去捡建图时看到的垃圾")
    p.add_argument("--no-bin", action="store_true", help="不去垃圾桶：捡起一个拿着就停")
    p.add_argument("--cap-gap", type=float, default=CAP_GAP_M, help="夹之前瓶盖中心离保险杠多远 (m)，默认 0.10")
    p.add_argument("--rescan", action="store_true", help="用存的地图（同 --use-map），到几个观察点低头看一圈找垃圾")
    p.add_argument("--use-map", nargs="?", const=saved_map.SAVED_DIR, default=None,
                   help="不建图：用存的地图（默认 vln/map/saved，或给一个 live*.npz），原地转一圈定位")
    p.add_argument("--holding", type=float, default=None, metavar="LENGTH_M",
                   help="左臂上已经夹着东西、挂下来这么长（瓶子给瓶长，小件给 0.04）：地图好了先把它扔进垃圾桶")
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
