#!/usr/bin/env python
"""Where the open jaws really are against the cap. The base never moves; the neck and the left arm do.

    python grasp_check.py look          # neck to the grasp view, photos, bottle and an upright grasp planned
    python grasp_check.py check         # arm (jaws open) to the grasp pose but HOVER_M higher, photos, park
    python grasp_check.py check --up 0.0   # down to the grasp pose itself (jaws still open), photos, park
    python grasp_check.py check --flip     # the wrist turned half round: the jaws swap sides and the
                                           # gripper body no longer hides the cap from the head camera
    python grasp_check.py grab             # the real thing: down, grip, lift, grip again, put it back, open, park
    python grasp_check.py grab --flip      # the same with the wrist half round (the wrist camera sees the cap)
    python grasp_check.py tip 0.26 -0.05   # the closed jaws' tip, tool upright, to (x, y) base frame TIP_Z_M up:
                                           # photos to find the real tip in both eyes, against the kinematics

Into runs/check_<time>/: the photos, *_marked.jpg with what the plan expects drawn on them (red: the
fixed-jaw tip, yellow: the grasp point between the jaws, blue: the cap as the cameras found it, the
line: the jaws' closing line), and check.json. The marks for the arm come from the SO-101 kinematics
and arm_mount.yaml through the head camera model, the cap from detect_any: where the real jaws stand
off the red and yellow marks is how far the arm is from where it was told to be.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "vln"), str(ROOT / "project_tool"), str(ROOT / "llm_vla" / "pick_bottle"),
                str(ROOT / "llm_vla")]
import detect_any  # noqa: E402
from bottle_to_bin import flipped, grasp_nudge_mm, lift_heights  # noqa: E402
import ik  # noqa: E402
import mount  # noqa: E402
import pick  # noqa: E402
import so101_kin  # noqa: E402
import validate_head_extrinsic as val  # noqa: E402
from arm import LeftArm, joint_limits  # noqa: E402
from arm_snapshot import read_neck_deg  # noqa: E402
from neck_look import move_neck  # noqa: E402
from stereo import Stereo  # noqa: E402
from touch_calib import grab  # noqa: E402

import calibrate_head_cameras as cal  # noqa: E402

GRASP_NECK = (-15.0, 35.0)
WRIST_CAM = "/dev/cam_wrist_left"
WRIST_URL = "http://127.0.0.1:8766/frame.jpg"  # wrist_stream.py, when it runs, holds the camera
NECK_TOL_DEG = 1.5  # the neck stops ~1 deg short now and then (backlash); the camera model uses what it reads
HOVER_M = 0.03
TIP_Z_M = 0.03
LAST = HERE / "runs" / "check_last.json"


def photos(d, tag):
    _, frames = grab()
    for eye, img in frames.items():
        cv2.imwrite(str(d / f"{tag}_{eye}.jpg"), img)
    return frames


def wrist_photo(d, tag):
    """One frame of the left wrist camera (the jaws' own view): from wrist_stream.py when it runs."""
    try:
        with urllib.request.urlopen(WRIST_URL, timeout=2.0) as r:
            buf = np.frombuffer(r.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        cv2.imwrite(str(d / f"{tag}_wrist.jpg"), img)
        return img
    except OSError:
        pass
    cap = cv2.VideoCapture(WRIST_CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    img = None
    try:
        for _ in range(15):  # let the exposure settle
            ok, f = cap.read()
            if ok:
                img = f
    finally:
        cap.release()
    if img is None:
        raise SystemExit(f"{WRIST_CAM} 没有画面")
    cv2.imwrite(str(d / f"{tag}_wrist.jpg"), img)
    return img


def to_px(cam, p_base):
    """Left-image pixel of a base-frame point."""
    T = np.linalg.inv(cam["T_base_cam"])
    pc = T[:3, :3] @ p_base + T[:3, 3]
    px, _ = cv2.projectPoints(pc.reshape(1, 3), np.zeros(3), np.zeros(3), cam["K"], cam["dist"])
    return tuple(int(round(v)) for v in px.reshape(2))


def arm_marks(q, offset):
    """Base-frame fixed-jaw tip, grasp point and a point along the closing line for joints q."""
    T = mount.load()
    R, t = T[:3, :3], T[:3, 3]
    p, _, close = ik.pose(q, offset)
    tip = so101_kin.tip_position(q)
    return R @ tip + t, R @ p + t, R @ (p + close * 0.03) + t


def cmd_look(args):
    for _ in range(3):  # the neck has some backlash: try again when it stops short
        neck = move_neck(*GRASP_NECK)
        if max(abs(neck["neck_yaw"] - GRASP_NECK[0]), abs(neck["neck_pitch"] - GRASP_NECK[1])) <= NECK_TOL_DEG:
            break
    else:
        raise SystemExit(f"脖子没到抓取视角：{neck}")
    arm = LeftArm()
    try:
        pick.park(arm)
        q_now = arm.deg()
    finally:
        arm.close()
    d = HERE / "runs" / time.strftime("check_%Y%m%d_%H%M%S")
    d.mkdir(parents=True)
    frames = photos(d, "look")
    neck = read_neck_deg()
    cam = pick.head_cam(neck)
    calib, _, cams = val.load(cal.DEFAULT_DATA)
    stereo = Stereo(calib, cams)
    pts = stereo.points(frames["left"], frames["right"], cam["T_base_cam"])
    got = detect_any.locate(frames["left"], cam, args.model, 20.0, pick.ROBOT_MASK["left"], None, (pts, stereo.uv))
    if got is None:
        raise SystemExit(f"抓取视角里没找到瓶子（{d}/look_left.jpg）：把瓶子放到车头左前方 10-25 cm")
    cv2.imwrite(str(d / "look_left_detect.jpg"), got["vis"])
    got["cap"][2] = max(got["cap"][2], pick.MIN_TIP_Z_M + ik.GRASP_INSET + 0.002)
    run = {"neck_deg": neck, "nudge_mm": grasp_nudge_mm(got["axis"]), "cap_axis_base_m": got["cap"].tolist(),
           "label_base_m": got["mid"].tolist(), "bottle_axis_base": got["axis"].tolist(),
           "cap_px": got["cap_px"], "z_from": got["z_from"]}
    pick.TILT_STEPS_DEG = (0.0,)
    pick.LIFT_HEIGHTS_M = lift_heights(got["length"], got["cut"])
    run["bottle_length_m"], run["bottle_cut"] = got["length"], got["cut"]
    run["plan"] = pick.solve_steps(run, q_now)
    (d / "check.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    LAST.write_text(json.dumps({"dir": str(d)}))
    print(f"瓶盖 (mm) {np.round(got['cap'] * 1000).astype(int).tolist()}（高度按{got['z_from']}），"
          f"离保险杠 {(got['cap'][0] - 0.165) * 100:.0f} cm；竖直夹得到。{d}")


def cmd_check(args):
    d = Path(json.loads(LAST.read_text())["dir"])
    run = json.loads((d / "check.json").read_text())
    if max(abs(read_neck_deg()[k] - run["neck_deg"][k]) for k in run["neck_deg"]) > 0.3:
        raise SystemExit("脖子比 look 时动了：重新 look")
    offset = ik.tcp_offset(pick.CAP_RADIUS_M + pick.JAW_CLEARANCE_M)
    R = mount.load()[:3, :3]
    grasp, hover = run["plan"]["grasp"]["joints"], run["plan"]["hover"]["joints"]
    p, _, close = ik.pose(grasp, offset)
    axis_arm = np.cross(close, [0.0, 0.0, 1.0])
    axis_arm /= np.linalg.norm(axis_arm)
    limits = joint_limits()
    seed_g, seed_h = grasp, hover
    if args.flip:
        roll = grasp["wrist_roll"] + 180.0 if grasp["wrist_roll"] + 180.0 <= limits["wrist_roll"][1] else grasp["wrist_roll"] - 180.0
        seed_g, seed_h = {**grasp, "wrist_roll": roll}, {**hover, "wrist_roll": roll}
    hover_q, err_h = ik.solve(ik.pose(hover, offset)[0], axis_arm, offset, limits, [seed_h])
    q, err = ik.solve(p + R.T @ np.array([0.0, 0.0, args.up]), axis_arm, offset, limits, [seed_g])
    if max(err, err_h) > 3:
        raise SystemExit(f"夹取点上方 {args.up * 100:.0f} cm 解不出 IK（{err:.1f} / {err_h:.1f}）")
    if abs(q["wrist_roll"] - hover_q["wrist_roll"]) > 30:
        raise SystemExit("上方和下面的解手腕差半圈，不动")
    tag = f"up{int(round(args.up * 1000)):03d}" + ("_flip" if args.flip else "")
    arm = LeftArm()
    try:
        pick.move(arm, {**hover_q, "gripper": pick.OPEN_DEG}, pick.FREE_DEG_S, "到瓶盖上方")
        pick.move(arm, q, pick.NEAR_DEG_S, f"到夹取点上方 {args.up * 100:.0f} cm")
        time.sleep(0.5)
        frames = photos(d, tag)
        wrist_photo(d, tag)
        q_real = arm.deg()
        pick.move(arm, hover_q, pick.NEAR_DEG_S, "回到上方")
        pick.park(arm)
    finally:
        arm.close()
    cam = pick.head_cam(run["neck_deg"])
    tip, gp, line = arm_marks(q_real, offset)
    cap = np.array(run["cap_axis_base_m"])
    for eye in ("left",):
        img = frames[eye].copy()
        cv2.drawMarker(img, to_px(cam, cap), (255, 0, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.circle(img, to_px(cam, tip), 6, (0, 0, 255), 2)
        cv2.circle(img, to_px(cam, gp), 6, (0, 220, 255), 2)
        cv2.line(img, to_px(cam, gp), to_px(cam, line), (0, 220, 255), 1)
        cv2.drawMarker(img, to_px(cam, cap + np.array([0, 0, args.up])), (255, 120, 0), cv2.MARKER_TILTED_CROSS, 12, 1)
        cv2.imwrite(str(d / f"{tag}_{eye}_marked.jpg"), img)
    run.setdefault("checks", []).append({"up_m": args.up, "joints": q_real, "tip_base_m": tip.tolist(),
                                         "grasp_point_base_m": gp.tolist(), "cap_base_m": cap.tolist()})
    (d / "check.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    print(f"实际关节 {pick.fmt(q_real)}")
    print(f"按运动学：夹取点 (mm) {np.round(gp * 1000).astype(int).tolist()}，固定爪尖 {np.round(tip * 1000).astype(int).tolist()}；"
          f"瓶盖 {np.round(cap * 1000).astype(int).tolist()}")
    print(d / f"{tag}_left_marked.jpg")


def cmd_grab(args):
    d = Path(json.loads(LAST.read_text())["dir"])
    run = json.loads((d / "check.json").read_text())
    if max(abs(read_neck_deg()[k] - run["neck_deg"][k]) for k in run["neck_deg"]) > 0.3:
        raise SystemExit("脖子比 look 时动了：重新 look")
    if args.dx or args.dy or args.dz:  # aim off the cap by this much more (mm, base frame), plan again
        run["nudge_mm"] = (np.array(run["nudge_mm"]) + [args.dx, args.dy, args.dz]).tolist()
        pick.TILT_STEPS_DEG = (0.0,)
        pick.LIFT_HEIGHTS_M = lift_heights(run.get("bottle_length_m", 0.22), run.get("bottle_cut", True))
        arm = LeftArm()
        try:
            run["plan"] = pick.solve_steps(run, arm.deg())
        finally:
            arm.close()
        print(f"修正量 (mm) {run["nudge_mm"]}")
    p = flipped(run["plan"]) if args.flip else run["plan"]
    p["lift"]["height_m"] = run["plan"]["lift"]["height_m"]
    tag = "grab_flip" if args.flip else "grab"
    arm = LeftArm()
    try:
        pick.move(arm, {**p["hover"]["joints"], "gripper": pick.OPEN_DEG}, pick.FREE_DEG_S, "到瓶盖上方")
        wrist_photo(d, f"{tag}_hover")
        pick.move(arm, p["grasp"]["joints"], pick.NEAR_DEG_S, "下降")
        photos(d, f"{tag}_open")
        wrist_photo(d, f"{tag}_open")
        closed = arm.limits["gripper"][0]
        g = arm.grip(closed)
        photos(d, f"{tag}_closed")
        wrist_photo(d, f"{tag}_closed")
        res = {"grip_deg": g, "held": bool(g > closed + 3)}
        print(f"夹爪停在 {g:.1f}°（全合 {closed:.1f}°）：" + ("夹住了" if res["held"] else "没夹到"))
        if res["held"]:
            pick.move(arm, p["lift"]["joints"], pick.NEAR_DEG_S, f"抬起 {p['lift']['height_m'] * 100:.0f} cm")
            photos(d, f"{tag}_lift")
            wrist_photo(d, f"{tag}_lift")
            time.sleep(5.0)  # longer than the gripper servo's 2 s overload time: does the grip hold?
            photos(d, f"{tag}_held5s")
            g = arm.grip(closed)
            res["after_lift_deg"], res["lifted"] = g, bool(g > closed + 3)
            print(f"举着 5 秒后再合一次夹爪：停在 {g:.1f}°：" + ("瓶子还在" if res["lifted"] else "瓶子掉了"))
            pick.move(arm, p["grasp"]["joints"], pick.NEAR_DEG_S, "放回原处")
        pick.move(arm, {"gripper": pick.OPEN_DEG}, pick.FREE_DEG_S, "松开")
        pick.move(arm, p["hover"]["joints"], pick.NEAR_DEG_S, "退回上方")
        pick.park(arm)
    finally:
        arm.close()
    run.setdefault("grabs", []).append(res)
    (d / "check.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    print(d)


def cmd_tip(args):
    """Closed jaws, tool upright, fixed-jaw tip to (x, y, TIP_Z_M) in the base frame; photos."""
    d = Path(json.loads(LAST.read_text())["dir"])
    T = mount.load()
    R, t = T[:3, :3], T[:3, 3]
    limits = joint_limits()
    goal = np.array([args.x, args.y, TIP_Z_M])
    arm = LeftArm()
    try:
        now = arm.deg()
        seeds = [now] + [json.loads((d / "check.json").read_text())["plan"]["grasp"]["joints"]]
        offset = so101_kin.FIXED_JAW_TIP
        # Which way the jaws close does not matter here: the closing line that solves best.
        best = None
        for deg in range(0, 180, 15):
            axis = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0])
            q, err = ik.solve(R.T @ (goal - t), axis, offset, limits, seeds)
            if best is None or err < best[1]:
                best = (q, err, axis)
        q, err, axis = best
        if err > 3:
            raise SystemExit(f"({args.x}, {args.y}) 解不出竖直的 IK（{err:.1f}）")
        up, err_up = ik.solve(R.T @ (goal + [0, 0, 0.06] - t), axis, offset, limits, [q])
        sol = [up, q]
        closed = arm.limits["gripper"][0] + 2.0
        pick.move(arm, {**sol[0], "gripper": closed}, pick.FREE_DEG_S, "合着夹爪到上方 6 cm")
        pick.move(arm, sol[1], pick.NEAR_DEG_S, f"爪尖到离地 {TIP_Z_M * 100:.0f} cm")
        time.sleep(0.5)
        tag = f"tip_{int(round(args.x * 1000))}_{int(round(args.y * 1000))}"
        photos(d, tag)
        q_real = arm.deg()
        pick.move(arm, sol[0], pick.NEAR_DEG_S, "抬起")
        pick.park(arm)
    finally:
        arm.close()
    tip = R @ so101_kin.tip_position(q_real) + t
    run = json.loads((d / "check.json").read_text())
    run.setdefault("tips", []).append({"tag": tag, "joints": q_real, "tip_base_m": tip.tolist()})
    (d / "check.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    cam = pick.head_cam(run["neck_deg"])
    print(f"按运动学爪尖在 (mm) {np.round(tip * 1000).astype(int).tolist()}，左眼应在像素 {to_px(cam, tip)}；{d}/{tag}_*.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("look").add_argument("--model", default="qwen3.8-max")
    c = sub.add_parser("check")
    c.add_argument("--up", type=float, default=HOVER_M, help="夹取点上方多高 (m)")
    c.add_argument("--flip", action="store_true", help="手腕转半圈：夹爪换边，相机看得见瓶盖")
    gb = sub.add_parser("grab")
    gb.add_argument("--flip", action="store_true", help="手腕转半圈去夹")
    for k in ("--dx", "--dy", "--dz"):
        gb.add_argument(k, type=float, default=0.0, help="再偏这么多 (mm，底盘坐标)")
    tp = sub.add_parser("tip")
    tp.add_argument("x", type=float)
    tp.add_argument("y", type=float)
    args = ap.parse_args()
    {"look": cmd_look, "check": cmd_check, "grab": cmd_grab, "tip": cmd_tip}[args.cmd](args)


if __name__ == "__main__":
    main()
