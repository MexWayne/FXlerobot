#!/usr/bin/env python3
"""Short-range language-guided navigation with the head camera (visual servoing).

Qwen-VL (DashScope) finds the target's bounding box in the left-eye image once; a ViT tracker then
follows the box every frame. The bottom-center of the box is the target's floor contact point, which
head_extrinsic.yaml turns into a floor point in the base frame; turn rate and forward speed are
updated from it every frame, so wheel slip on the carpet is corrected continuously. The base stops
and asks the model again when tracking is lost or every --recheck-s seconds.

The neck takes part: when the target is not in view the head looks left and right (--scan-yaw)
before the base turns, and when the target's floor contact leaves the bottom of the image (the
floor closer than ~0.5 m is out of view with the head level) the head looks down by --near-pitch
and keeps measuring. head_extrinsic.yaml is calibrated at one neck pose; other poses rotate it
about the neck axes (PIVOT_*), which keeps floor points within ~1 cm of the level-head estimate.

  python nlv_nav.py "去红色箱子旁边" --dry-run   # only locate the target, motors stay off
  python nlv_nav.py "去红色箱子旁边"             # navigate; Ctrl-C stops the base

Needs DASHSCOPE_API_KEY. The neck starts at the pose used during cali_extri.
Only targets standing on visible floor within ~3 m; there is no obstacle avoidance,
keep a hand on Ctrl-C.
"""

import argparse
import base64
import json
import math
import os
import re
import signal
import time

import calibrate_head_cameras as cal
import cv2
import numpy as np
import requests
import validate_head_extrinsic as val

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
LOG_ROOT = os.path.expanduser("~/workspaces/FXLerobot/FXlerobot/nav_logs")
TRACKER_MODEL = os.path.expanduser("~/workspaces/FXLerobot/FXlerobot/models/object_tracking_vittrack_2023sep.onnx")
EYE = "left"
MAX_RANGE_M = 3.0
# Neck axes in the base frame (m), fitted on 2026-10-19 by matching floor features across neck poses.
# neck_yaw.pos + turns the head right about a vertical axis; neck_pitch.pos + tilts it down about
# a lateral axis. Both servos turn the camera 1:1 in degrees.
PIVOT_YAW = np.array([-0.017, -0.011, 0.666])
PIVOT_PITCH = np.array([0.023, -0.011, 0.666])

PROMPT = """你是机器人的视觉定位模块，图像来自机器人头部相机。
用户指令：{instruction}
找出指令要去的目标，给出它的边界框，框的下边要落在目标与地面接触的位置。
只输出 JSON，不要其他文字：{{"found": true, "bbox": [x1, y1, x2, y2], "label": "目标名"}}；找不到输出 {{"found": false}}。
{coord_hint}"""
COORD_HINTS = {
    "rel1000": "坐标用 0-1000 归一化，图像左上角为 (0,0)，右下角为 (1000,1000)。",
    "abs": f"坐标用图像像素，图像宽 {cal.WIDTH}、高 {cal.HEIGHT}，左上角为 (0,0)。",
}


def ask_vlm(img, instruction, model, coord, timeout_s):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    text = PROMPT.format(instruction=instruction, coord_hint=COORD_HINTS[coord])
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": text}]}]}
    print(f"  asking {model} ...", flush=True)
    t0 = time.time()
    for attempt in range(3):
        try:
            r = requests.post(DASHSCOPE_URL, json=body, timeout=(5, timeout_s),
                              headers={"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"})
            break
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  request failed after {time.time() - t0:.0f}s ({type(e).__name__}), retry {attempt + 1}/2",
                  flush=True)
    else:
        raise SystemExit("DashScope unreachable, check the robot's network")
    if r.status_code != 200:
        raise SystemExit(f"DashScope HTTP {r.status_code}: {r.text[:300]}")
    reply = r.json()["choices"][0]["message"]["content"]
    print(f"  VLM {time.time() - t0:.1f}s: {reply.strip()}")
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        raise RuntimeError(f"no JSON in VLM reply: {reply!r}")
    ans = json.loads(m.group(0))
    if not ans.get("found"):
        return None
    x1, y1, x2, y2 = (float(v) for v in ans["bbox"][:4])
    if coord == "rel1000":
        x1, x2 = x1 / 1000 * cal.WIDTH, x2 / 1000 * cal.WIDTH
        y1, y2 = y1 / 1000 * cal.HEIGHT, y2 / 1000 * cal.HEIGHT
    x1, x2 = sorted((np.clip(x1, 0, cal.WIDTH - 1), np.clip(x2, 0, cal.WIDTH - 1)))
    y1, y2 = sorted((np.clip(y1, 0, cal.HEIGHT - 1), np.clip(y2, 0, cal.HEIGHT - 1)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def grab(cap):
    # Drop frames buffered while the base was moving.
    for _ in range(5):
        cap.grab()
    ok, img = cap.retrieve()
    if not ok:
        raise SystemExit("camera read failed")
    return img


def cam_at(cam0, d_yaw, d_pitch):
    """Camera model after the neck moved by (d_yaw, d_pitch) degrees from the calibration pose."""
    a = math.radians(d_pitch)
    Rp = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    b = math.radians(-d_yaw)
    Rw = np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
    T = cam0["T_base_cam"].copy()
    T[:3, :3] = Rw @ Rp @ T[:3, :3]
    T[:3, 3] = PIVOT_YAW + Rw @ (PIVOT_PITCH + Rp @ (T[:3, 3] - PIVOT_PITCH) - PIVOT_YAW)
    return {**cam0, "T_base_cam": T}


def target_from_bbox(bbox, cam, edge_px):
    x, y, w, h = bbox
    bottom = y + h
    floor = val.pixel_to_floor(x + w / 2, min(bottom, cal.HEIGHT - 1), cam)
    return floor, bottom >= cal.HEIGHT - edge_px


def save_frame(img, bbox, floor, note, path):
    vis = img.copy()
    if bbox is not None:
        x, y, w, h = (int(v) for v in bbox)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawMarker(vis, (x + w // 2, min(y + h, cal.HEIGHT - 1)), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    # cv2.putText cannot draw Chinese; labels are printed in the terminal.
    text = note if floor is None else f"{note} x={floor[0]:.2f} y={floor[1]:.2f} m"
    cv2.putText(vis, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(path, vis)


def make_tracker(img, bbox):
    params = cv2.TrackerVit_Params()
    params.net = TRACKER_MODEL
    tracker = cv2.TrackerVit.create(params)
    tracker.init(img, tuple(int(v) for v in bbox))
    return tracker


class Base:
    def __init__(self, neck_yaw, neck_pitch):
        from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
        from lerobot.robots.lekiwi.lekiwi import LeKiwiBase, LeKiwiNeck
        from lerobot_kiwi_base_neck import PORT, assert_port_ready

        assert_port_ready(PORT)
        self.base = LeKiwiBase(LeKiwiConfig(port=PORT, id="kiwi_base", cameras={}))
        self.neck = LeKiwiNeck(LeKiwiConfig(port=PORT, id="kiwi_neck", cameras={}))
        self.base.connect(calibrate=False)
        # The neck servos drop a status packet now and then; one miss should not abort the run.
        for attempt in range(3):
            try:
                self.neck.connect(calibrate=False)
                break
            except ConnectionError as e:
                print(f"neck connect failed ({e}), retry {attempt + 1}/2")
                try:
                    self.neck.bus.disconnect(disable_torque=False)
                except Exception:
                    pass
                time.sleep(1.0)
        else:
            raise SystemExit("neck servos on /dev/base_neck do not answer")
        self.home = (neck_yaw, neck_pitch)
        self.look(0.0, 0.0)

    def look(self, d_yaw, d_pitch, settle_s=1.0):
        """Point the head at (d_yaw, d_pitch) degrees from the calibration pose; returns the measured offset."""
        self.neck.send_action({"neck_yaw.pos": self.home[0] + d_yaw, "neck_pitch.pos": self.home[1] + d_pitch})
        time.sleep(settle_s)
        obs = self.neck.get_observation()
        self.pose = (obs["neck_yaw.pos"] - self.home[0], obs["neck_pitch.pos"] - self.home[1])
        return self.pose

    def send(self, x_vel, theta_vel):
        self.base.send_action({"x.vel": x_vel, "y.vel": 0.0, "theta.vel": theta_vel})

    def stop(self):
        self.base.stop_base()

    def turn(self, deg, rot):
        t_end = time.time() + abs(deg) / rot
        while time.time() < t_end:
            self.send(0.0, math.copysign(rot, deg))
            time.sleep(0.05)
        self.stop()

    def close(self):
        for fn in (self.base.stop_base, lambda: self.look(0.0, 0.0, settle_s=0.5),
                   self.base.disconnect, self.neck.disconnect):
            try:
                fn()
            except Exception as e:
                print(f"{fn.__qualname__}: {e}")


def servo(base, cap, cam, tracker, args, log_dir, look):
    """Drive towards the tracked box until arrived, lost or the recheck interval is up."""
    t0 = last_save = time.time()
    frame = 0
    closest = math.inf
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                raise SystemExit("camera read failed")
            ok, bbox = tracker.update(img)
            score = tracker.getTrackingScore()
            floor, at_edge = target_from_bbox(bbox, cam, args.edge_px) if ok else (None, False)
            if not ok or score < args.min_score or floor is None:
                print(f"  tracking lost (score {score:.2f})")
                save_frame(img, bbox if ok else None, None, "lost", os.path.join(log_dir, f"look{look:02d}_lost.jpg"))
                return "lost"
            dist = math.hypot(floor[0], floor[1])
            bearing = math.degrees(math.atan2(floor[1], floor[0]))
            remaining = dist - args.stop_m
            if time.time() - last_save > 1.0 or at_edge or remaining <= args.tol_m:
                save_frame(img, bbox, floor, f"s={score:.2f}",
                           os.path.join(log_dir, f"look{look:02d}_{frame:04d}.jpg"))
                last_save = time.time()
            if at_edge:
                print(f"  target bottom reached the image edge (~{dist:.2f} m)")
                return "edge"
            if remaining <= args.tol_m:
                print(f"  target {dist:.2f} m away, arrived")
                return "arrived"
            # Driving forward can only bring a static target closer; a growing distance means the
            # tracker box slid off the target's floor contact.
            closest = min(closest, dist)
            if dist > closest + args.drift_m:
                print(f"  distance went up from {closest:.2f} to {dist:.2f} m, tracker drifted")
                save_frame(img, bbox, floor, "drift", os.path.join(log_dir, f"look{look:02d}_drift.jpg"))
                return "recheck"
            theta = float(np.clip(args.k_rot * bearing, -args.rot, args.rot))
            fwd = 0.0
            if abs(bearing) < args.turn_deg:
                fwd = float(np.clip(args.k_lin * remaining, args.min_lin, args.lin)) * math.cos(math.radians(bearing))
            base.send(fwd, theta)
            if frame % 15 == 0:
                print(f"  dist {dist:.2f} m bearing {bearing:+.1f} deg score {score:.2f} -> "
                      f"x.vel {fwd:.2f} m/s theta.vel {theta:+.0f} deg/s")
            frame += 1
            if time.time() - t0 > args.recheck_s:
                return "recheck"
    finally:
        base.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("instruction", help="where to go, e.g. 去红色箱子旁边")
    p.add_argument("--data", default=cal.DEFAULT_DATA)
    p.add_argument("--model", default="qwen3-vl-plus")
    p.add_argument("--coord", choices=sorted(COORD_HINTS), default="rel1000",
                   help="box format the model returns: Qwen3-VL uses rel1000, Qwen2.5-VL uses abs")
    p.add_argument("--dry-run", action="store_true", help="locate the target once, do not move")
    p.add_argument("--stop-m", type=float, default=0.6, help="stop when the target is this close")
    p.add_argument("--tol-m", type=float, default=0.05)
    p.add_argument("--edge-px", type=int, default=6,
                   help="box bottom this close to the image bottom counts as floor contact out of view: "
                        "look down, or stop if already looking down")
    p.add_argument("--lin", type=float, default=0.15, help="max forward speed, m/s")
    p.add_argument("--min-lin", type=float, default=0.05, help="min forward speed so the wheels do not stall")
    p.add_argument("--k-lin", type=float, default=0.5, help="forward speed per meter still to go, 1/s")
    p.add_argument("--rot", type=float, default=30.0, help="max turn speed, deg/s")
    p.add_argument("--k-rot", type=float, default=1.5, help="turn speed per degree of bearing, 1/s")
    p.add_argument("--turn-deg", type=float, default=20.0, help="only turn, no forward, when bearing exceeds this")
    p.add_argument("--min-score", type=float, default=0.3, help="tracker score below this counts as lost")
    p.add_argument("--recheck-s", type=float, default=5.0, help="stop and ask the model again after this long")
    p.add_argument("--search-deg", type=float, default=30.0,
                   help="turn the base this much when the target is not seen and the head scan is off")
    p.add_argument("--scan-yaw", type=float, nargs="*", default=[-45.0, 45.0],
                   help="neck yaw offsets (deg, + is right) to look at before turning the base; "
                        "pass no values to turn the base only")
    p.add_argument("--near-pitch", type=float, default=25.0,
                   help="look down this much (deg) once the target's floor contact leaves the image bottom; 0 disables")
    p.add_argument("--drift-m", type=float, default=0.08,
                   help="re-ask the model when the tracked distance grows by this much while driving")
    p.add_argument("--max-looks", type=int, default=20, help="max model queries before giving up")
    p.add_argument("--timeout-s", type=float, default=20.0, help="max wait for model response bytes")
    args = p.parse_args()

    # A dropped SSH session sends SIGHUP; route it through KeyboardInterrupt so the base gets stopped.
    def _interrupt(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, _interrupt)
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set (export it in ~/.bashrc, then open a new terminal)")
    if not os.path.exists(TRACKER_MODEL):
        raise SystemExit(f"{TRACKER_MODEL} missing, download object_tracking_vittrack_2023sep.onnx from opencv_zoo")

    calib, extr, cams = val.load(args.data)
    cam0 = cams[EYE]
    log_dir = os.path.join(LOG_ROOT, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"instruction: {args.instruction}\nlogs: {log_dir}")

    cap = cal.open_camera(cal.CAMS[EYE], calib.get("focus_absolute", 0))
    base = None
    try:
        if not args.dry_run:
            base = Base(extr.get("neck_yaw_pos") or 0.0, extr.get("neck_pitch_pos") or 0.0)
        target = (0.0, 0.0)  # commanded neck offset from the calibration pose
        pose = base.pose if base is not None else (0.0, 0.0)  # measured offset, used for the camera model
        scan = [] if args.dry_run else list(args.scan_yaw)
        # The head scan covers about 2 * max|yaw| plus one image width (~55 deg); turn a bit less than that.
        base_turn = 2 * max(abs(y) for y in scan) + 45.0 if scan else args.search_deg
        pending, scanned_here, searched = [], False, 0.0

        def point_head(d_yaw, d_pitch):
            nonlocal target, pose
            target = (d_yaw, d_pitch)
            pose = base.look(d_yaw, d_pitch)

        def look_down_or_arrive():
            if args.near_pitch > 0 and target[1] < args.near_pitch:
                print(f"  floor contact is below the image, looking down {args.near_pitch:.0f} deg")
                point_head(0.0, args.near_pitch)
                return False
            print("  floor contact is below the image with the head down, arrived")
            return True

        for look in range(args.max_looks):
            head = f" head yaw {target[0]:+.0f} pitch {target[1]:+.0f}" if base is not None else ""
            print(f"[look {look}]{head}")
            img = grab(cap)
            cam = cam_at(cam0, *pose)
            bbox = ask_vlm(img, args.instruction, args.model, args.coord, args.timeout_s)
            floor, at_edge = target_from_bbox(bbox, cam, args.edge_px) if bbox else (None, False)
            path = os.path.join(log_dir, f"look{look:02d}.jpg")
            save_frame(img, bbox, floor, "target" if bbox else "not found", path)
            if bbox is None or floor is None:
                print(f"  target {'not found' if bbox is None else 'above the horizon'} ({path})")
                if args.dry_run:
                    return
                if target[1] != 0.0:
                    print("  looking level again")
                    point_head(target[0], 0.0)
                    continue
                if not pending and not scanned_here and scan:
                    pending, scanned_here = list(scan), True
                if pending:
                    yaw = pending.pop(0)
                    print(f"  turning the head {'right' if yaw > 0 else 'left'} {abs(yaw):.0f} deg to search")
                    point_head(yaw, 0.0)
                    continue
                if searched >= 360:
                    print("target not found after a full turn, stopping")
                    return
                if target != (0.0, 0.0):
                    point_head(0.0, 0.0)
                print(f"  turning the base left {base_turn:.0f} deg to search")
                base.turn(base_turn, args.rot)
                searched += base_turn
                scanned_here = False
                continue
            pending, scanned_here, searched = [], False, 0.0
            dist = math.hypot(floor[0], floor[1])
            bearing = math.degrees(math.atan2(floor[1], floor[0]))
            print(f"  box {bbox} -> floor x={floor[0]:.2f} y={floor[1]:.2f} m, {dist:.2f} m away, "
                  f"bearing {bearing:+.1f} deg{', bottom at image edge' if at_edge else ''} ({path})")
            if dist > MAX_RANGE_M:
                print(f"  beyond {MAX_RANGE_M} m, the floor estimate is rough until it gets closer")
            if args.dry_run:
                return
            if target[0] != 0.0:
                print(f"  found with the head turned, centring the head and turning the base {bearing:+.0f} deg")
                point_head(0.0, target[1])
                base.turn(bearing, args.rot)
                continue
            if at_edge:
                if look_down_or_arrive():
                    return
                continue
            result = servo(base, cap, cam, make_tracker(img, bbox), args, log_dir, look)
            if result == "arrived":
                return
            if result == "edge" and look_down_or_arrive():
                return
        print(f"gave up after {args.max_looks} model queries")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if base is not None:
            base.close()
        cap.release()


if __name__ == "__main__":
    main()
