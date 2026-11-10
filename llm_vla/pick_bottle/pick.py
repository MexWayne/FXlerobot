#!/usr/bin/env python
"""Pick a bottle lying on the floor by its cap with the left arm.

    python pick.py                      # park, look, plan, hover, grasp, grip, lift: all in one go
    python pick.py look                 # only check: photos, find bottle and cap, plan; arm stays
    python pick.py place RUN            # put it back where it was, open, park
    python pick.py park                 # the right arm's rest pose, out of the floor in front
    python pick.py park --keep-grip     # same, slowly, gripper left as it is (carrying a bottle)
    python pick.py open                 # open the gripper
    python pick.py relax                # torque off: the arm falls, hold it

One step at a time, for checking (RUN from `look`):
    python pick.py go RUN hover|grasp|lift
    python pick.py grip
    python pick.py nudge RUN dx dy dz   # shift every target (base frame, mm)
    python pick.py demo RUN             # arm hand-guided onto the cap: how far off the plan was

The bottle and then its cap are found in each eye (detect.py) and triangulated; the cap centre and
the label centre give the bottle's axis. Both go into the base frame with the head camera chain and
into the arm frame with arm_mount.yaml. The tool comes down vertically (tilted as little as
reaches, at most TILT_STEPS_DEG[-1], when the cap is too far for that) with the jaws closing across the bottle: HOVER_M above the cap,
down onto it, close, lift (LIFT_HEIGHTS_M, the first that is in reach). Only the left arm moves; the neck
must stay as it was when the photos were taken.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LLM_VLA = HERE.parent
sys.path[:0] = [str(HERE), str(LLM_VLA), str(LLM_VLA.parent / "project_tool"), str(LLM_VLA.parent / "vln")]
import cv2  # noqa: E402
import detect  # noqa: E402
import ik  # noqa: E402
import mount  # noqa: E402
import so101_kin  # noqa: E402
import validate_head_extrinsic as val  # noqa: E402
import yaml  # noqa: E402
from arm import LeftArm, joint_limits  # noqa: E402
from arm_snapshot import read_neck_deg  # noqa: E402
from board_view import HEAD_CAMERAS  # noqa: E402
from cup_compare import triangulate  # noqa: E402
from touch_calib import grab  # noqa: E402
from vln_nav import cam_at  # noqa: E402

RUNS = HERE / "runs"
# The robot's own body in the grasp view (neck yaw -15, pitch 35): below the bumper, plus the left
# arm in PARK. Bottles are looked for outside these polygons (pixels, 640 x 480).
ROBOT_MASK = {"left": [(0, 300), (260, 300), (260, 345), (640, 380), (640, 480), (0, 480)],
              "right": [(0, 240), (260, 240), (260, 300), (560, 440), (640, 440), (640, 480), (0, 480)]}
HOVER_M = 0.06  # stop this far above the grasp point first
LIFT_HEIGHTS_M = (0.20, 0.15, 0.10)  # from 20 cm the bottle (~16 cm) hangs clear; lower if out of reach
CAP_RADIUS_M = 0.015
JAW_CLEARANCE_M = 0.004  # fixed jaw this far off the cap side before closing
# Grasp this far out from the cap's middle, away from the bottle: the neck behind the short (~2 cm)
# cap is much wider, and a jaw that lands on it stops the arm above the cap.
CAP_OUTWARD_M = 0.004
MAX_REPROJ_PX = 2.0  # cap or label seen in the two eyes further apart than this: not the same thing
OPEN_DEG = 0.0  # gripper; LeRobot degrees, closed is near the bottom of its range
MIN_TIP_Z_M = 0.003  # fixed-jaw tip never planned lower than this above the floor
PATH_CLEAR_M = 0.005  # on the way, the tip stays at least this far above the lower end of the move
NECK_TOL_DEG = 0.3
MAX_ROLL_CHANGE_DEG = 30.0  # hover / lift must keep the grasp's wrist orientation
MAX_POS_ERR_M = 0.002  # IK good enough: TCP this close to its goal,
MAX_JAW_ERR_DEG = 3.0  # jaws this close to horizontal and to across the bottle
# When upright is out of reach the tool may lean, in these steps, the least that reaches. Leaning
# more than ~12 deg the jaws hold the cap by a corner and it slips out on the way up (2026-11-03:
# two grasps at 20 deg both lost the bottle).
TILT_STEPS_DEG = (0.0, 4.0, 8.0, 12.0)
HISTORY_SEEDS = 6  # IK starts from this many earlier, different grasps
# After lifting, a bottle whose middle is lower than this is still on the floor: lying it is ~3 cm
# up, standing ~9 cm; hanging from the gripper at least ~12 cm.
LIFTED_MIN_M = 0.10
# The right arm's rest pose (read off it, 2026-11-03), shoulder and wrist kept 4 deg off the stops
# the unpowered right arm leans on: folded upright, 18 cm from the head camera, on the left edge of
# the left eye. The way there passes 13 cm from the camera, so it is taken slower.
PARK = {"shoulder_pan": 3.0, "shoulder_lift": -97.0, "elbow_flex": -16.0, "wrist_flex": 99.0,
        "wrist_roll": -94.0, "gripper": -60.0}
PARK_DEG_S = 30.0
# Carrying: the arm swung in over the body (CARRY, posed by the user by hand, 2026-11-08): the jaws above
# the front middle of the body (base x 0.12, y 0.02, 0.36 m up), what they hold hanging over the deck,
# inside the body's outline and out of both head cameras' view. Held at PARK a bottle filled the lower
# left of the view; leaned forward from PARK (shoulder_lift -60, 2026-11-07) it still hung in front of
# the bumper in every look, and the trips to the can were slow. From PARK the joints' straight way goes
# in over the body, the jaws sinking steadily from 0.51 to 0.36 m. shoulder_pan was posed at 89.4, past
# the arm's limit (89.0, LIMIT_MARGIN_DEG inside its range): the first stow refused to move (2026-11-10).
CARRY = {"shoulder_pan": 88.5, "shoulder_lift": -80.8, "elbow_flex": 34.3, "wrist_flex": 95.9, "wrist_roll": -93.6}
CARRY_DEG_S = 15.0  # with a bottle hanging from the gripper, so it does not swing
FREE_DEG_S = 50.0  # through the air
NEAR_DEG_S = 25.0  # down onto the cap and lifting the bottle
STEPS = ("hover", "grasp", "lift")


def head_cam(neck):
    _, extr, cams = val.load(str(LLM_VLA.parent / "calib" / "head"))
    return cam_at(cams["left"], neck["neck_yaw"] - extr["neck_yaw_pos"], neck["neck_pitch"] - extr["neck_pitch_pos"])


def stereo_base(run, left_px, right_px):
    """Base-frame points of pixel pairs in the run's photos."""
    calib = yaml.safe_load(HEAD_CAMERAS.read_text())
    X_cam, reproj = triangulate(calib, left_px, right_px)
    T = head_cam(run["neck_deg"])["T_base_cam"]
    return (T[:3, :3] @ X_cam.T).T + T[:3, 3], reproj


def load_run(name):
    d = RUNS / name
    return d, json.loads((d / "run.json").read_text())


def save_run(d, run):
    (d / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n")


def photos(d, tag):
    _, frames = grab()
    for eye, img in frames.items():
        cv2.imwrite(str(d / f"{tag}_{eye}.jpg"), img)


def check_neck(run):
    now = read_neck_deg()
    moved = max(abs(now[k] - run["neck_deg"][k]) for k in now)
    if moved > NECK_TOL_DEG:
        raise SystemExit(f"脖子比拍照时动了 {moved:.2f}°：重新 look 和 plan")


def fmt(q):
    return ", ".join(f"{k} {v:.1f}" for k, v in q.items())


def tip_base(q):
    T = mount.load()
    return T[:3, :3] @ so101_kin.tip_position(q) + T[:3, 3]


def move(arm, q, speed, label):
    """One checked move: the fixed-jaw tip must not dip on the way below where the move ends or starts."""
    now = arm.deg()
    full = {**now, **q}
    z = [tip_base({k: now[k] + (full[k] - now[k]) * s for k in now})[2] for s in np.linspace(0, 1, 21)]
    # A tip that starts lower than MIN_TIP_Z_M (the arm sags at a low cap) may still move up out of it.
    if min(z) < min(z[0], z[-1]) - PATH_CLEAR_M or min(z) < min(MIN_TIP_Z_M, z[0] - 0.001):
        raise SystemExit(f"{label}：路上爪尖会低到 {min(z) * 1000:.0f} mm，不动")
    t0 = time.monotonic()
    got = arm.move(q, speed)
    print(f"{label} {time.monotonic() - t0:.1f}s：{fmt(got)}")
    return got


def move_path(arm, path, speed, label):
    """move() through all of `path` (joint dicts) without stopping between, each stretch checked."""
    now = arm.deg()
    for a, b in zip([now] + path, path):
        full_a, full_b = {**now, **a}, {**now, **b}
        z = [tip_base({k: full_a[k] + (full_b[k] - full_a[k]) * s for k in now})[2] for s in np.linspace(0, 1, 11)]
        if min(z) < min(z[0], z[-1]) - PATH_CLEAR_M or min(z) < min(MIN_TIP_Z_M, z[0] - 0.001):
            raise SystemExit(f"{label}：路上爪尖会低到 {min(z) * 1000:.0f} mm，不动")
    t0 = time.monotonic()
    got = arm.move_path(path, speed)
    print(f"{label}（{len(path)} 个路点，不停顿）{time.monotonic() - t0:.1f}s：{fmt(got)}")
    return got


def find_bottle(d, tag):
    """Bottle and cap in both eyes' photos `tag`: pixels of cap centre and label centre per eye."""
    found = {}
    for eye in ("left", "right"):
        img = cv2.imread(str(d / f"{tag}_{eye}.jpg"))
        det = detect.detect(img, exclude=ROBOT_MASK[eye])
        cv2.imwrite(str(d / f"{tag}_{eye}_detect.jpg"), detect.draw(img, det))
        if det is None:
            raise SystemExit(f"{eye} 眼没找到瓶子或瓶盖，看 {d}/{tag}_{eye}_detect.jpg")
        found[eye] = (tuple(det["cap_centre"]), tuple(det["bottle_centre"]))
    return found


def reach(goal, axis_arm, offset, limits, seeds, max_tilt):
    """IK for one TCP goal (arm frame) and whether it is good enough to move to: the TCP within
    MAX_POS_ERR_M, the jaws within MAX_JAW_ERR_DEG of horizontal and of across the bottle, the tool
    within max_tilt of vertical. Returns (q, ok, numbers for the log)."""
    q, err = ik.solve(goal, axis_arm, offset, limits, seeds, max_tilt_deg=max_tilt)
    p, tool, close = ik.pose(q, offset)
    pos = float(np.linalg.norm(p - goal))
    tilt = float(np.degrees(np.arccos(np.clip(-tool[2], -1, 1))))
    jaw = float(max(np.degrees(np.arcsin(min(1.0, abs(close @ axis_arm)))), np.degrees(np.arcsin(min(1.0, abs(close[2]))))))
    ok = pos <= MAX_POS_ERR_M and jaw <= MAX_JAW_ERR_DEG and tilt <= max_tilt + 1.0
    return q, ok, {"ik_error": err, "pos_err_mm": pos * 1000, "tool_tilt_deg": tilt, "jaw_err_deg": jaw}


def solve_steps(run, arm_now):
    """IK for grasp, then hover and lift from it, with the tool upright if it reaches, tilted if not."""
    T = mount.load()
    R, t = T[:3, :3], T[:3, 3]
    axis = np.array(run["bottle_axis_base"])  # from the cap towards the label
    cap = np.array(run["cap_axis_base_m"]) - axis * CAP_OUTWARD_M + np.array(run["nudge_mm"]) / 1000
    axis_arm = R.T @ axis
    offset = ik.tcp_offset(CAP_RADIUS_M + JAW_CLEARANCE_M)
    limits = joint_limits()
    seeds, seen = [], set()
    for r in sorted(RUNS.glob("*/run.json"), reverse=True):  # earlier grasps start near the answer
        old = json.loads(r.read_text())
        if "plan" not in old:
            continue
        q = old["plan"]["grasp"]["joints"]
        key = tuple(round(q[j] / 5) for j in ik.JOINTS)
        if key not in seen:
            seen.add(key)
            seeds.append(q)
        if len(seeds) == HISTORY_SEEDS:
            break
    seeds += [arm_now] + [{**s, "wrist_roll": s["wrist_roll"] + d} for s in seeds[:2] for d in (-90, 90, 180)]
    for max_tilt in TILT_STEPS_DEG:
        q, ok, info = reach(R.T @ (cap - t), axis_arm, offset, limits, seeds, max_tilt)
        if ok:
            break
    else:
        far = np.linalg.norm((cap - t)[:2]) * 100
        raise SystemExit(f"够不着：瓶盖离左臂底座水平 {far:.0f} cm，夹爪倾斜 {TILT_STEPS_DEG[-1]:.0f}° 以内夹不稳"
                         f"（夹取点差 {info['pos_err_mm']:.1f} mm，夹爪方向差 {info['jaw_err_deg']:.1f}°）。"
                         "把瓶子往车这边挪近一点")
    plan = {"grasp": {"joints": q, **info}}
    if max_tilt > 0:
        print(f"工具竖直够不着，倾斜 {info['tool_tilt_deg']:.0f}° 去夹")
    # Hover and lift start from the grasp only: both wrist orientations reach a point exactly, and
    # from the others the arm would turn its wrist half round next to the bottle.
    for step, heights in (("hover", (HOVER_M,)), ("lift", LIFT_HEIGHTS_M)):
        for dz in heights:
            q, ok, info = reach(R.T @ (cap + [0, 0, dz] - t), axis_arm, offset, limits,
                                [plan["grasp"]["joints"]], max_tilt)
            if ok:
                break
        else:
            raise SystemExit(f"{step} 够不着")
        if abs(q["wrist_roll"] - plan["grasp"]["joints"]["wrist_roll"]) > MAX_ROLL_CHANGE_DEG:
            raise SystemExit(f"{step} 和 grasp 不是同一组解（手腕要转半圈），不动")
        plan[step] = {"joints": q, "height_m": dz, **info}
    if plan["lift"]["height_m"] < LIFT_HEIGHTS_M[0]:
        print(f"抬起只能到 {plan['lift']['height_m'] * 100:.0f} cm")
    for step, p in plan.items():
        tip = R @ so101_kin.tip_position(p["joints"]) + t
        p["tip_base_m"] = tip.tolist()
        if tip[2] < MIN_TIP_Z_M:
            raise SystemExit(f"{step} 爪尖会低于地面 {MIN_TIP_Z_M * 1000:.0f} mm")
    return plan


def plan_run(d, run, arm_now):
    found = find_bottle(d, "look")
    pts, reproj = stereo_base(run, list(found["left"]), list(found["right"]))
    if reproj.max() > MAX_REPROJ_PX:
        raise SystemExit(f"两只眼找到的瓶盖/瓶身对不上（重投影 {reproj.max():.1f} px）：看 look_*_detect.jpg")
    cap, label = pts
    axis = label[:2] - cap[:2]
    axis = np.r_[axis / np.linalg.norm(axis), 0.0]
    run.update({"cap_px": found, "cap_axis_base_m": cap.tolist(), "label_base_m": label.tolist(),
                "bottle_axis_base": axis.tolist(), "reproj_px": reproj.tolist()})
    print(f"瓶盖 (mm) {np.round(cap * 1000).astype(int).tolist()}，瓶身方向 {np.degrees(np.arctan2(axis[1], axis[0])):.0f}°，"
          f"重投影 ≤ {reproj.max():.2f} px")
    run["plan"] = solve_steps(run, arm_now)
    save_run(d, run)


def new_run():
    d = RUNS / time.strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True)
    run = {"neck_deg": read_neck_deg(), "nudge_mm": [0.0, 0.0, 0.0]}
    photos(d, "look")
    save_run(d, run)
    return d, run


def bottle_height(d, run, tag):
    """Height (m) of the bottle's middle in the photos `tag`, by stereo; None if either eye misses it."""
    px = [detect.bottle_centre(cv2.imread(str(d / f"{tag}_{eye}.jpg")), exclude=ROBOT_MASK[eye])
          for eye in ("left", "right")]
    if None in px:
        return None
    pts, reproj = stereo_base(run, [px[0]], [px[1]])
    return float(pts[0][2]) if reproj.max() <= MAX_REPROJ_PX else None


def park(arm, keep_grip=False, speed=None):
    """To PARK. keep_grip: leave the gripper where it is (still holding what it holds), move slowly
    (CARRY_DEG_S, a bottle hanging; speed: another, e.g. for a light small thing that does not swing)."""
    target = {k: v for k, v in PARK.items() if not (keep_grip and k == "gripper")}
    now = arm.deg()
    if max(abs(now[k] - v) for k, v in target.items()) < 3:
        return
    if tip_base(now)[2] < HOVER_M:
        # Low (e.g. jaws still at a cap): straight up first so nothing is dragged along the floor.
        offset = ik.tcp_offset(CAP_RADIUS_M + JAW_CLEARANCE_M)
        p, _, close = ik.pose(now, offset)
        axis = np.cross(close, [0, 0, 1.0])
        q, err = ik.solve(p + [0, 0, HOVER_M], axis / np.linalg.norm(axis), offset, joint_limits(), [now])
        if err < 3:
            move(arm, q, NEAR_DEG_S, "竖直抬起")
    move(arm, target, speed or (CARRY_DEG_S if keep_grip else PARK_DEG_S), "收臂")


def stow(arm):
    """What the gripper holds, from PARK to CARRY over the body, out of the head cameras' view. The
    gripper stays as it is."""
    move(arm, CARRY, CARRY_DEG_S, "拿着的东西收到车身上方，让开相机")


def unstow(arm):
    """Back from stow() to PARK, where arm_drop's way out over the can starts. Does no harm on an
    arm already at PARK."""
    move(arm, {k: v for k, v in PARK.items() if k != "gripper"}, CARRY_DEG_S, "收回 PARK")


def cmd_auto(args):
    t0 = time.monotonic()
    arm = LeftArm()
    try:
        park(arm)
        d, run = new_run()
        plan_run(d, run, arm.deg())
        print(f"看+规划完 {time.monotonic() - t0:.1f}s  ({d.name})")
        p = run["plan"]
        move(arm, {**p["hover"]["joints"], "gripper": OPEN_DEG}, FREE_DEG_S, "到瓶盖上方")
        move(arm, p["grasp"]["joints"], NEAR_DEG_S, "下降")
        closed = arm.limits["gripper"][0]
        g = arm.grip(closed)
        held = g > closed + 3
        print(f"夹爪停在 {g:.1f}°（全合 {closed:.1f}°）" + ("，夹住了" if held else "，没夹到东西"))
        run["result"] = {"grip_deg": g, "held": bool(held)}
        save_run(d, run)
        if not held:
            photos(d, "miss")
            move(arm, {**p["hover"]["joints"], "gripper": OPEN_DEG}, NEAR_DEG_S, "退回上方")
            raise SystemExit(f"没夹到：看 {d.name}/look_*_detect.jpg 和 miss_*.jpg")
        move(arm, p["lift"]["joints"], NEAR_DEG_S, "抬起")
        run["result"]["after_lift_deg"] = arm.deg()["gripper"]
        photos(d, "lift")
        z = bottle_height(d, run, "lift")
        run["result"]["bottle_height_m"] = z
        save_run(d, run)
        if z is None:
            print("相机看不清瓶子，无法确认拿起来没有")
        elif z < LIFTED_MIN_M:
            move(arm, {"gripper": OPEN_DEG}, FREE_DEG_S, "松开")
            park(arm)
            raise SystemExit(f"没拿起来：瓶身离地 {z * 100:.0f} cm，还在地上（看 {d.name}/lift_*.jpg）")
        else:
            print(f"拿起来了：瓶身离地 {z * 100:.0f} cm")
    finally:
        arm.close()
    print(f"共 {time.monotonic() - t0:.1f}s")


def cmd_place(args):
    d, run = load_run(args.run)
    check_neck(run)
    p = run["plan"]
    arm = LeftArm()
    try:
        move(arm, p["grasp"]["joints"], NEAR_DEG_S, "放下")
        move(arm, {"gripper": OPEN_DEG}, FREE_DEG_S, "松开")
        move(arm, p["hover"]["joints"], NEAR_DEG_S, "抬起")
        move(arm, PARK, PARK_DEG_S, "收臂")
    finally:
        arm.close()


def cmd_park(args):
    arm = LeftArm()
    try:
        park(arm, keep_grip=args.keep_grip)
    finally:
        arm.close()


def cmd_look(args):
    arm = LeftArm()
    try:
        now = arm.deg()
    finally:
        arm.close()
    d, run = new_run()
    print(d.name)
    try:
        plan_run(d, run, now)
    except SystemExit as err:
        raise SystemExit(f"{err}\n左臂挡在画面里的话，先 python pick.py park")
    for step, s in run["plan"].items():
        print(f"{step:5s}: {fmt(s['joints'])}  爪尖 (mm) {np.round(np.array(s['tip_base_m']) * 1000).astype(int).tolist()}")
    print("能抓：python pick.py 会重新拍照并抓取")


def cmd_nudge(args):
    d, run = load_run(args.run)
    run["nudge_mm"] = (np.array(run["nudge_mm"]) + [args.dx, args.dy, args.dz]).tolist()
    print(f"累计修正 (mm) {run['nudge_mm']}")
    arm = LeftArm()
    try:
        now = arm.deg()
    finally:
        arm.close()
    run["plan"] = solve_steps(run, now)
    save_run(d, run)


def cmd_go(args):
    d, run = load_run(args.run)
    check_neck(run)
    q = dict(run["plan"][args.step]["joints"])
    if args.step == "hover":
        q["gripper"] = OPEN_DEG
    arm = LeftArm()
    try:
        move(arm, q, FREE_DEG_S if args.step == "hover" else NEAR_DEG_S, args.step)
    finally:
        arm.close()
    photos(d, args.step)


def cmd_demo(args):
    """The arm was put on the cap by hand: compare its grasp point with the cap the cameras saw.
    Both go through the same kinematics and arm_mount.yaml, so the difference is the error to nudge."""
    d, run = load_run(args.run)
    arm = LeftArm()
    try:
        q = arm.deg()
    finally:
        arm.close()
    T = mount.load()
    R, t = T[:3, :3], T[:3, 3]
    p, tool, close = ik.pose(q, ik.tcp_offset(CAP_RADIUS_M))
    got = R @ p + t
    cap = np.array(run["cap_axis_base_m"])
    err = (got - cap) * 1000
    across = np.degrees(np.arcsin(min(1.0, abs((R @ close) @ np.array(run["bottle_axis_base"])))))
    print(f"关节 {fmt(q)}")
    print(f"示教的夹取点 (mm) {np.round(got * 1000, 1).tolist()}，相机看到的瓶盖 {np.round(cap * 1000, 1).tolist()}，"
          f"偏差 {np.round(err, 1).tolist()}；工具倾斜 {np.degrees(np.arccos(-(R @ tool)[2])):.0f}°，"
          f"夹爪与垂直瓶身方向差 {across:.0f}°")
    print(f"让计划落到示教位置：python pick.py nudge {args.run} {-err[0]:.1f} {-err[1]:.1f} {-err[2]:.1f}")
    run.setdefault("demos", []).append({"joints": q, "grasp_base_m": got.tolist(), "cap_base_m": cap.tolist(),
                                        "error_mm": err.tolist()})
    save_run(d, run)


def cmd_grip(args):
    arm = LeftArm()
    try:
        closed = arm.limits["gripper"][0]
        g = arm.grip(closed)
    finally:
        arm.close()
    print(f"夹爪停在 {g:.1f}°（全合 {closed:.1f}°）" + ("：没夹到东西" if g < closed + 3 else ""))


def cmd_open(args):
    arm = LeftArm()
    try:
        arm.move({"gripper": OPEN_DEG})
    finally:
        arm.close()
    print("夹爪已张开")


def cmd_relax(args):
    arm = LeftArm()
    try:
        arm.relax()
    finally:
        arm.close()
    print("左臂已卸力")


def main():
    ap = argparse.ArgumentParser(description="不带参数：抓起瓶子并举起来")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("look", "grip", "open", "relax"):
        sub.add_parser(name)
    sub.add_parser("park").add_argument("--keep-grip", action="store_true",
                                        help="leave the gripper as it is (holding a bottle), move slowly")
    for name in ("place", "demo"):
        sub.add_parser(name).add_argument("run")
    p = sub.add_parser("nudge")
    p.add_argument("run")
    for k in ("dx", "dy", "dz"):
        p.add_argument(k, type=float)
    p = sub.add_parser("go")
    p.add_argument("run")
    p.add_argument("step", choices=STEPS)
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd or 'auto'}"](args)


if __name__ == "__main__":
    main()
