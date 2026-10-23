#!/usr/bin/env python3
"""围栏里走固定的 S 形。每行只做三件事：判断障碍，绕过障碍，把障碍背后走到这一行的尽头。

  conda activate lerobot_env
  cd ~/workspaces/FXLerobot/FXlerobot/vln
  python s_route.py               # 把车放进围栏里，直接开始
  python s_route.py --check-m 0.3 # 每走 0.3 m 看一次（默认 0.35）
  python s_route.py --dry-run     # 电机和相机都不用，假围栏里放两个假纸箱演示绕障

状态页：浏览器打开 http://<jetson>:8765 。Ctrl-C 停车。

流程：不先转圈测围栏，也不先开到墙角。车朝哪就从哪开始固定 S 形，行距 30 cm，每行往右边换。
每一行笔直向前，每 --check-m 米看一次：看到的围栏或踢脚线就换行；其他东西就是障碍，侧移绕开，
开到它背后，再回到原来这一行，然后把背后剩下的路笔直走到围栏。两边都绕不过才结束这一行。不做补扫。
状态页上的图随车走出来：扫过的地面、障碍，以及看到的踢脚线和围栏。一开始只有车身边的一小块。
"""

import argparse
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import coverage_run as cr  # noqa: E402
from coverage_map import BODY_LENGTH_M, BODY_WIDTH_M  # noqa: E402
from live_map import MapView  # noqa: E402

DETOUR_MIN_M = 0.30     # least sideways step around anything
DETOUR_MAX_M = 1.20     # an obstacle wider than this is treated as the row end
DETOUR_PASS_MAX_M = 4.0  # keep going beside an obstacle at most this far before just staying offset
DETOUR_PER_ROW = 3      # detours allowed in one row
# _hit_info treats a point within this of the centre line as standing in the body's path.
CORRIDOR_M = BODY_WIDTH_M / 2.0 + 0.05
PASS_CLEAR_M = 0.10     # keep the obstacle being passed at least this far outside the corridor
UNKNOWN_HALF_M = 0.15   # half width assumed for something the body ran into without seeing it
ROW_BACK_M = 0.30       # back along the row this much when an obstacle blocks the change to the next row
ROW_SHIFT_TRIES = 3


def to_map_at(pose, fx, fy):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * fx - s * fy, y + s * fx + c * fy


def _sweep_hits(px, py, side, dist):
    """A robot-frame point lies in the new ground a slide of `dist` toward `side` (+1 left) will cover."""
    hl = BODY_LENGTH_M / 2.0 + 0.05
    hw = BODY_WIDTH_M / 2.0
    if abs(px) > hl or dist < 0.02:
        return False
    if side > 0:
        return hw < py <= hw + dist
    return -(hw + dist) <= py < -hw


def _slide_room(mover, cmap, side, dist):
    """How far the body can slide toward `side` before the fence or a blocked cell."""
    hl, hw, step = BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0, 0.05
    lat = hw
    while lat < hw + dist - 1e-6:
        lat = min(hw + dist, lat + step)
        along = -hl
        while along <= hl + 1e-6:
            x, y = cr.to_map(mover, along, side * lat)
            if not cr._inside_xy(x, y, cmap):
                return max(0.0, lat - hw - step)
            cell = cmap.world_to_cell(x, y)
            if cell is not None and cmap.grid[cell] == cr.BLOCKED:
                return max(0.0, lat - hw - step)
            along += step
    return dist


def _something_else(head, side, dist, local):
    """True when the camera sees something other than `local` (robot frame) in the slide."""
    if head is None:
        return False
    look = cr._neck(-math.degrees(side * math.pi / 2.0))
    seen = head.lane(d_yaw=look, near_only=True)
    if seen is None or seen[0] != "hit":
        return False
    for px, py in seen[4] or [(seen[1], seen[2])]:
        if any(math.hypot(px - ax, py - ay) < 0.45 for ax, ay in local):
            continue
        if _sweep_hits(px, py, side, dist):
            return True
    return False


def _back_clear(mover, obstacle):
    """Back up until an obstacle overlapping the body is in front of it. False if there is no room."""
    hl = BODY_LENGTH_M / 2.0 + 0.06
    need = 0.0
    for x, y in obstacle:
        px, py = cr.to_robot(mover, x, y)
        if abs(py) <= BODY_WIDTH_M / 2.0 + 0.05 and px < hl:
            need = max(need, hl - px)
    if need < 0.02:
        return True
    if cr.known_back(mover) < need - 0.02:
        return False
    print(f"  障碍贴着车身，先退后 {need:.2f} m 再绕")
    mover.forward(-need, "贴太近，先退后一点", record=False)
    return True


def slide_all(mover, head, cmap, rel, dist, status, avoid=None):
    """Slide the whole `dist` or not at all. A slide that cannot finish is undone.

    `avoid` (map points) is the obstacle being passed. The slide then does not turn the body:
    turning to look sideways swings the front into the very obstacle the body is going around,
    and the turn is refused. The omni base slides without turning; the known obstacle, the fence
    and blocked cells say whether that slide is clear.
    """
    if avoid is None:
        moved = cr.vln_slide(mover, head, cmap, rel, dist, status)
    else:
        side = 1.0 if math.sin(rel) > 0 else -1.0
        local = [cr.to_robot(mover, x, y) for x, y in avoid]
        if any(_sweep_hits(px, py, side, dist) for px, py in local):
            return False
        if _slide_room(mover, cmap, side, dist) < dist - 0.05:
            return False
        if _something_else(head, side, dist, local):
            return False
        moved = mover.slide(rel, dist, status)
    if moved >= dist - 0.05:
        return True
    if moved > 0.02:
        mover.slide(rel + math.pi, moved, "退回原位", record=False)
    return False


def change_row(mover, head, cmap, side, heading):
    """Slide LANE_M towards `side` and turn round for the next row.

    An obstacle beside the body only moves the change back along the row just driven; only
    the fence on that side ends the sweep.
    """
    word = "左" if side > 0 else "右"
    for _ in range(ROW_SHIFT_TRIES):
        mover.hold = heading
        if slide_all(mover, head, cmap, side * math.pi / 2.0, cr.LANE_M, f"换行：往{word}平移 30 cm"):
            if cr.turn_with_room(mover, head, cmap, heading + math.pi, "掉头走下一行"):
                return True
            print("转不过身，遍历结束")
            return False
        beyond = cr.to_map(mover, 0.0, side * (cr.LANE_M + BODY_WIDTH_M / 2.0))
        if not cr._inside_xy(*beyond, cmap):
            print(f"{word}边就是围栏，这一侧到头")
            return False
        if cr.known_back(mover) < ROW_BACK_M + 0.02:
            print(f"{word}边有障碍，后面也没有走过的路可退，遍历结束")
            return False
        print(f"  {word}边有障碍，沿这一行退后 {ROW_BACK_M:.2f} m 再换行")
        mover.hold = heading
        mover.forward(-ROW_BACK_M, "退后再换行", record=False)
    print("换行试了几次都被挡住，遍历结束")
    return False


def _same_obstacle(points, side):
    """True when everything standing in the drive corridor is the obstacle just slid away from.

    `side` (+1 left) is the way the body slid. The obstacle then lies on the other side, so a hit
    that is only there means the body is still alongside it, not that a new thing blocks the way.
    """
    corridor = [(px, py) for px, py in points if abs(py) <= CORRIDOR_M]
    return bool(corridor) and all(-side * py > 0.02 for _, py in corridor)


def _clear_of(points, side):
    """Further slide toward `side` that puts the obstacle being passed outside the drive corridor."""
    need = 0.0
    for _, py in points:
        if -side * py <= 0.02:
            continue
        need = max(need, CORRIDOR_M + PASS_CLEAR_M + side * py)
    return max(0.0, need)


def _rejoin(mover, head, cmap, side, dist, obstacle):
    """Slide back onto the row. A slide that cannot finish is undone, so the body stays put."""
    if dist < 0.03:
        return True
    return slide_all(mover, head, cmap, -side * math.pi / 2.0, dist, "绕障：回到这一行", avoid=obstacle)


def detour(mover, head, cmap, heading, side, obstacle):
    """Go around `obstacle` (map points; empty if unseen) and rejoin the row past it.

    The row then carries on to the fence, so the floor behind the obstacle is swept too.
    `side` (+1 left, -1 right) is tried first. Returns ("ok", metres driven along the row),
    ("row_end", metres) or ("stop", metres).
    """
    covered = 0.0
    for s in (side, -side):
        word = "左" if s > 0 else "右"
        local = [cr.to_robot(mover, x, y) for x, y in obstacle]
        if local:
            reach = max(s * py for _, py in local)
            front = max(px for px, _ in local)
        else:
            reach, front = UNKNOWN_HALF_M, BODY_LENGTH_M / 2.0 + 0.12
        shift = max(DETOUR_MIN_M, reach + CORRIDOR_M + PASS_CLEAR_M)
        if shift > DETOUR_MAX_M:
            print(f"  障碍往{word}伸出 {reach:.2f} m，从{word}边绕太远")
            continue
        if local:
            ys = [py for _, py in local]
            print(f"  障碍在车前 {min(px for px, _ in local):.2f}~{front:.2f} m，"
                  f"横向 {min(ys):+.2f}~{max(ys):+.2f} m（左为正）")
        print(f"  绕障：往{word}平移 {shift:.2f} m，贴着它开到背后", flush=True)
        mover.hold = heading
        if not _back_clear(mover, obstacle):
            print(f"  障碍贴着车身，后面又退不了，{word}边绕不开")
            continue
        if not slide_all(mover, head, cmap, s * math.pi / 2.0, shift, f"绕障：往{word}让开", avoid=obstacle):
            print(f"  {word}边让不开")
            continue
        status, passed = _pass_to_the_back(mover, head, cmap, heading, s, shift, front, obstacle)
        covered += passed
        if status == "try_other":
            continue
        return status, covered
    return "row_end", covered


def _pass_to_the_back(mover, head, cmap, heading, side, shift, front, obstacle):
    """Drive beside the obstacle until the body is past it, then return to the original row.

    Returns ("ok", metres) once the body is behind the obstacle, so the row can keep going
    straight across that ground to the fence. ("try_other", metres) when this side cannot get
    there and the body is back on the row. ("stop", metres) when the run must end.
    """
    word = "左" if side > 0 else "右"
    offset, passed = shift, 0.0
    # The camera only saw the near face, so this far is still beside it, not behind it.
    min_pass = max(0.20, front + BODY_LENGTH_M / 2.0)
    pushes = 0
    while passed < DETOUR_PASS_MAX_M:
        mover.hold = heading
        if marked_ahead(mover, cmap, 0.35):
            pushes += 1
            if pushes > 2 or not back_off(mover):
                print("  绕的时候又顶到已看到的障碍")
                break
            continue
        step = cr.vln_step(mover, head, cmap, cr.LOOK_M, "绕障：往障碍背后开")
        kind = step[0]
        if kind == "stop":
            return "stop", passed
        if kind == "stuck":
            passed += step[3]
            break
        if kind == "fence":
            print("  绕到了这一行尽头的围栏")
            _rejoin(mover, head, cmap, side, offset, obstacle)
            return "ok", passed
        if kind == "go":
            passed += step[1]
            if step[1] < 0.02:
                break
        elif passed < min_pass or _same_obstacle(step[2], side):
            more = max(_clear_of(step[2], side), 0.05)
            if offset + more > DETOUR_MAX_M + 0.05:
                print(f"  再让就要超过 {DETOUR_MAX_M:.1f} m，换一边试")
                break
            print(f"  还在障碍旁边，再往{word}让 {more:.2f} m")
            if not slide_all(mover, head, cmap, side * math.pi / 2.0, more,
                             f"绕障：再往{word}让一点", avoid=obstacle):
                print(f"  {word}边让不开了")
                break
            offset += more
            continue
        else:
            print(f"  背后还有{step[1] or '障碍'}，先回到这一行再处理")
            break
        if passed < min_pass:
            print("  还在障碍旁边，继续往前开到它背后")
            continue
        if _rejoin(mover, head, cmap, side, offset, obstacle):
            print(f"  已到障碍背后，回到这一行，继续把背后走完", flush=True)
            return "ok", passed
        print("  背后还被挡着，再往前一点")
    if passed >= min_pass:
        if _rejoin(mover, head, cmap, side, offset, obstacle):
            print("  已到障碍背后，回到这一行，继续把背后走完", flush=True)
        else:
            print("  已到障碍背后，回不到原线，就从这里把背后走到围栏", flush=True)
        return "ok", passed
    if _rejoin(mover, head, cmap, side, offset, obstacle):
        return "try_other", passed
    print(f"  从{word}边没有绕到背后")
    return "row_end", passed


def marked_ahead(mover, cmap, reach=0.45):
    """Map cells already marked blocked, in front of the body and inside its width."""
    found = []
    along = 0.12
    while along <= reach:
        for lat in (-0.20, -0.10, 0.0, 0.10, 0.20):
            x, y = cr.to_map(mover, along, lat)
            cell = cmap.world_to_cell(x, y)
            if cell is not None and cmap.grid[cell] == cr.BLOCKED:
                found.append((x, y))
        along += 0.08
    return found


def back_off(mover):
    """Reverse a short way. False when the body does not actually move."""
    print("  已经顶到障碍，先后退再绕", flush=True)
    got = mover.forward(-0.20, "退出障碍", record=False)
    return got >= 0.05 and not mover.stuck


def drive_row(mover, head, cmap, heading, side, check_m):
    """One fixed row: look, go around anything that is not the fence, then finish the ground behind it.

    Returns ("ok"|"stop", metres). The row ends at the fence, or when an obstacle cannot be passed.
    """
    driven, detours = 0.0, 0
    while driven < cr.ROW_LIMIT_M - 0.02:
        pose = (mover.x, mover.y, mover.yaw)
        mover.hold = heading
        remembered = marked_ahead(mover, cmap)
        if remembered:
            # A chair seat leaves the downward view, and the carpet underneath looks clear.
            if detours >= DETOUR_PER_ROW or not back_off(mover):
                print("  退不出已看到的障碍，这一行停下")
                return "stop", driven
            detours += 1
            print("  前面是刚才看到的障碍，绕到它背后", flush=True)
            result, passed = detour(mover, head, cmap, heading, side, remembered)
            driven += passed
            if result == "stop":
                return "stop", driven
            if result == "row_end":
                break
            continue
        step = cr.vln_step(mover, head, cmap, min(check_m, cr.ROW_LIMIT_M - driven), "S 形直行")
        kind = step[0]
        if kind == "stop":
            return "stop", driven
        if kind == "go":
            driven += step[1]
            if step[1] < 0.02:
                break
            continue
        if kind == "fence":
            break
        if kind == "stuck":
            driven += step[3]
            if not back_off(mover):
                print("  顶住了，退不出来，停车")
                return "stop", driven
        if detours >= DETOUR_PER_ROW:
            print(f"  这一行已经绕了 {detours} 次，当成到头")
            break
        detours += 1
        obstacle = [to_map_at(pose, px, py) for px, py in step[2]]
        print(f"  判断到{step[1] or '障碍'}，绕到它背后再把这一行走完", flush=True)
        result, passed = detour(mover, head, cmap, heading, side, obstacle)
        driven += passed
        if result == "stop":
            return "stop", driven
        if result == "row_end":
            break
    return "ok", driven


def sweep(mover, head, cmap, check_m, first_side=-1.0):
    """S-shaped rows from where the robot stands, each new row `first_side` (-1 right) of the last."""
    start = mover.yaw
    empty = 0
    for lane in range(cr.MAX_LANES):
        heading = start + (math.pi if lane % 2 else 0.0)
        # The world side the rows advance to stays the same, so seen from the robot it alternates.
        side = first_side if lane % 2 == 0 else -first_side
        print(f"S 形第 {lane + 1} 行，航向 {cr.wrap_deg(math.degrees(heading)):+.0f}°")
        mover.show(f"S 形第 {lane + 1} 行")
        if abs(cr.wrap_deg(math.degrees(heading - mover.yaw))) > cr.TURN_TOL_DEG + 0.5:
            if not cr.turn_with_room(mover, head, cmap, heading, "转到这一行的方向"):
                return "done"
        result, driven = drive_row(mover, head, cmap, heading, side, check_m)
        if result == "stop":
            return "stop"
        print(f"  这一行走了 {driven:.2f} m，已清扫约 {cmap.stats()['sweep'] * 100:.0f}%")
        empty = empty + 1 if driven < 0.05 else 0
        if empty >= 2:
            print("  连着两行都走不动，停下")
            return "done"
        if not change_row(mover, head, cmap, side, heading):
            return "done"
    print(f"走了 {cr.MAX_LANES} 行，停下")
    return "done"


class FakeHead:
    """--dry-run camera: boxes the fence and a few cardboard boxes the way the VLM lane check would."""

    HALF_FOV = math.radians(35.0)

    def __init__(self, boundary, boxes):
        self.mover = None
        self.pts = []
        for poly, label in [(boundary, "围栏")] + [(b, "纸箱") for b in boxes]:
            for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
                n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.03))
                self.pts += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n, label) for k in range(n)]

    def lane(self, d_yaw=0.0, near_only=False):
        reach = cr.NEAR_FLOOR_M if near_only else cr.CLEAR_FLOOR_M + 0.2
        groups = {}
        for x, y, label in self.pts:
            fx, fy = cr.to_robot(self.mover, x, y)
            r = math.hypot(fx, fy)
            # Neck yaw + is right, so the camera looks at -d_yaw in the base frame.
            ang = math.atan2(fy, fx) + math.radians(d_yaw)
            if BODY_LENGTH_M / 2.0 < r <= reach and abs(math.atan2(math.sin(ang), math.cos(ang))) <= self.HALF_FOV:
                groups.setdefault(label, []).append((fx, fy))
        if not groups:
            return ("clear",)

        def key(item):
            pts = item[1]
            in_path = any(abs(py) <= BODY_WIDTH_M / 2.0 + 0.05 and px > 0 for px, py in pts)
            return (not in_path, min(math.hypot(px, py) for px, py in pts))

        label, pts = min(groups.items(), key=key)
        near = min(math.hypot(px, py) for px, py in pts)
        # Like a box bottom: the side facing the camera, a handful of points across it.
        pts = sorted((p for p in pts if math.hypot(*p) <= near + 0.25), key=lambda p: math.atan2(p[1], p[0]))
        pts = [pts[round(k * (len(pts) - 1) / 6)] for k in range(7)]
        c = min(pts, key=lambda p: math.hypot(*p))
        print(f"  [假相机] 看到{label}，离车 {math.hypot(*c):.2f} m")
        return ("hit", c[0], c[1], label, pts)


def growing_map(mover):
    """A wide grid whose fence starts at the body and is drawn out by driving and by walls seen later."""
    half = 6.0
    spec = cr.area_spec([[-half, -half], [half, -half], [half, half], [-half, half]])
    spec["start"] = [mover.x, mover.y, math.degrees(mover.yaw)]
    cmap = cr.CleanMap(spec)
    x, y, r = mover.x, mover.y, 0.45
    ring = [(x + r, y + r), (x + r, y - r), (x - r, y - r), (x - r, y + r)]
    cmap.fence = cr.Fence((x, y), ring)
    cmap.set_boundary(cmap.fence.polygon())
    cmap.stamp_body(x, y)
    return cmap


def work(args, view, robot):
    cmap = view.cmap
    if args.dry_run:
        mover = cr.Mover(cmap, view, None, speedup=6.0)
        boundary = [[-1.2, -1.4], [2.0, -1.4], [2.0, 1.4], [-1.2, 1.4]]
        head = FakeHead(boundary, [[[0.4, 0.2], [0.7, 0.2], [0.7, 0.5], [0.4, 0.5]],
                                   [[1.3, -0.9], [1.6, -0.9], [1.6, -0.6], [1.3, -0.6]]])
        head.mover = mover
    else:
        from vln_scene import Head, nav

        head = Head(args.model, args.timeout_s,
                    os.path.join(cr.MAP_DIR, "look", time.strftime("%Y%m%d_%H%M%S")))
        robot["head"] = head
        base = nav.Base(*head.neck_home)
        robot["base"] = base
        head.attach(base)
        mover = cr.Mover(cmap, view, base)
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
        boundary = None
    cmap = cr.make_map(boundary, mover) if boundary is not None else growing_map(mover)
    mover.cmap = cmap
    view.set_map(cmap)
    robot["cmap"] = cmap
    print(f"不先扫描，从当前朝向开始 S 形，每 {args.check_m:.2f} m 看一次前方")
    mover.show("S 形直行")
    result = sweep(mover, head, cmap, args.check_m)
    print(f"已清扫约 {cmap.stats()['sweep'] * 100:.0f}% 的可扫地面")
    msg = "S 形走完" if result == "done" else "S 形中途停下"
    mover.show(msg, [])
    print(msg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="电机和相机都不用，假围栏里放两个假纸箱演示")
    p.add_argument("--check-m", type=float, default=cr.LOOK_M, help="每走这么多米用 VLN 看一次前方")
    p.add_argument("--no-imu", action="store_true", help="不用 IMU（转角和直线只靠速度×时间）")
    p.add_argument("--imu-port", default=None, help="IMU 串口，默认自动找")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default="qwen3-vl-plus")
    p.add_argument("--timeout-s", type=float, default=20.0)
    args = p.parse_args()
    if args.check_m < 0.05:
        p.error("--check-m 至少 0.05")
    cr.LOOK_M = args.check_m  # vln_step never drives further than this between two looks

    def _interrupt(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, _interrupt)

    view = MapView(cr.placeholder_map())
    view.write(cr.MAP_DIR)
    cr.ensure_map_server(args.port)
    writer = view.start_writer(cr.MAP_DIR)
    print(f"状态页：http://{cr.local_ip()}:{args.port}")

    robot = {"head": None, "base": None, "imu": None, "cmap": view.cmap}
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
        writer.set()
        view.write(cr.MAP_DIR, running=False)
        print(f"底盘已停。状态页仍开着：http://{cr.local_ip()}:{args.port}")


if __name__ == "__main__":
    main()
