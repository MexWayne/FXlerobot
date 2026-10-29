"""Head cameras for the coverage run: what is there (VLM) and where it is (stereo, then lidar).

The VLM only names things and boxes them in the left image. Where a boxed thing is comes from
measurements: the stereo points inside the box, moved onto the lidar range where the lidar sees the
same thing (the lidar range is the more accurate one), or the lidar returns in the box's bearings
when the stereo has nothing there (the right third of the left image has no stereo partner). Only a
flat thing lower than the lidar beam outside the stereo overlap, a wire for instance, is put where
the box bottom meets the floor plane; for something lying on the floor that is a measurement too.

Looking up (UP_PITCH) the VLM is asked for the parts of tables, cabinets and shelves over open floor:
the lidar is low and passes under them (see Head._overhead). Glass and mirrors: the lidar is trusted
unless the VLM says the glass reaches the floor; then the stereo ranges it and the map stops lidar
rays there. The robot, the neck-aware camera model and the request style come from vln_nav.py.

Every hit is named for the whole thing (a tabletop, drawer or tray is part of 桌子) and carries how
high it was measured, so the map can model the thing: what it is, how long, which way and how big.
What the name alone would decide is measured instead: a "pick up" thing longer than PICK_MAX_M is an
obstacle, one body is the connected points nearest the robot, a wall is left to the lidar only where
the lidar sees it.
"""

import base64
import json
import math
import os
import re
import sys
import time
from collections import namedtuple

import cv2
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
for path in (os.path.join(PROJECT, "project_tool"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import vln_nav as nav  # noqa: E402
from stereo import Stereo  # noqa: E402

cal, val = nav.cal, nav.val

# Head-down angles (neck_pitch + is down; the camera itself already looks 35° down at 0). The camera
# sees floor 0.87 m out to the horizon at UP_PITCH, 0.30-1.6 m at 10° and 0.04-0.73 m at 30°.
UP_PITCH = -18.0
LANE_PITCHES = (10.0, 30.0)       # far to near
SIDE_PITCH = LANE_PITCHES[-1]     # pitch of the side looks before turns and slides
NEAR_PITCH = 25.0
SELF_NCC = 0.75                    # local correlation above which a textured pixel counts as not moving
SELF_FLAT_VAR = 40.0               # grey variance under which a 21x21 patch counts as flat
SELF_DIFF = 30.0                   # mean grey change under which a flat patch counts as not moving
SELF_TURN_DEG = 30.0               # body turn between the frames that find the body in the picture
SELF_SHARE = 0.6                   # a box this much on the body is the body
SELF_MAX_SHARE = 0.5               # a mask bigger than this means the body did not really turn
SELF_TOP = 0.60  # a box on the image bottom that starts in the bottom 40% is the robot's own deck
UPRIGHT_M = 0.15                   # a box from a bottom corner whose points are this close to the body is its upright
EDGE_PX = 6
KEEP_RUNS = 3      # image folders kept under map/look
STEREO_MIN_Z = 0.10      # the floor itself comes out at about +0.035 m in the base frame
STEREO_MAX_Z = 0.85      # higher than this passes over the 0.80 m body
FLOOR_Z = (-0.03, 0.07)  # stereo points this high are the floor
STEREO_CELL_M = 0.04     # stereo points are counted on a floor grid this fine
STEREO_MIN_PTS = 8       # a cell with fewer points is a mismatch, not a surface
NEAR_STEREO_M = 1.0      # standing points this close to the body center count even outside every box
STEREO_BODY_M = 0.03     # stereo points this close to the chassis outline are the robot itself
STEREO_LABEL = "近处障碍"  # what the stereo measured where the VLM drew no box
NEAR_CONFIRM_M = 0.05    # an unboxed stereo cell is kept once seen again from this far away
NEAR_CONFIRM_DEG = 5.0   # or with the camera this much turned or tilted
NEAR_SEEN_KEEP = 5000
OBJECT_REACH_M = 3.0     # boxed things farther than this are not located
STEREO_TRUST_M = 1.0     # stereo ranges farther than this are too coarse to place a thing
BOX_EDGE_PX = 8.0        # a VLM box edge is about this many pixels off the thing's real edge
BOX_EDGE_M = 0.03        # plus this much between the box bottom and where the lidar beam meets the thing
OBJECT_DEPTH_M = 0.50    # points this far behind a thing's nearest point still belong to it
LIDAR_PAD_DEG = 2.0      # lidar returns this far either side of a box still count
FLOOR_SEEN_M = 2.0       # stereo floor points farther than this do not clear old marks
BOUNDARY_KEYS = ("墙", "围栏", "栏杆", "踢脚", "基座")  # the lidar maps these where it sees them
GLASS_KEYS = ("玻璃", "镜")
# Small things on the floor the arms pick up. Only once measured no longer than PICK_MAX_M: a big
# sheet of cardboard called 包装 or 纸屑 is an obstacle.
PICK_KEYS = ("垃圾", "纸团", "纸屑", "纸巾", "塑料袋", "包装", "瓶", "罐", "果皮", "烟头", "碎屑", "食物", "小物件")
PICK_MAX_M = 0.15        # longest side of a thing the grippers take
# A thing on legs: the legs are apart and the top joins them. Anything else is one connected body.
SPANNING_KEYS = ("桌", "椅", "凳", "支架", "三脚架")
# Part names the VLM may still give, and the thing they belong to.
PART_NAMES = {"桌子": ("桌面", "桌沿", "桌前沿", "桌腿", "桌下", "抽屉", "托盘", "键盘架"), "柜子": ("柜面", "隔板")}
CLUSTER_GAP_M = 0.08     # points of one body are this close: stereo cells are 0.04 m, lidar returns 1° apart
# A tabletop, drawer or tray over open floor. The camera sits about 0.69 m up, below the 0.80 m body
# top, so the ray to the lower edge of a 0.72 m tabletop crosses 0.80 m behind the edge: the picture
# alone gives no range, and stereo finds nothing on a plain horizontal edge. The range comes from what
# holds it up and hangs under it: lidar returns (legs, feet) and stereo cells in the same bearings.
OVERHEAD_TAG = "悬空"      # label prefix older runs gave such hits
OVERHEAD_DEPTH_M = 0.70    # supports this far behind the nearest one still belong to it
OVERHEAD_PAD_DEG = 3.0     # supports this far either side of the box still count
OVERHEAD_RAY_DEG = 4.0     # an edge sample takes the nearest support at most this far round from it

# label, points and sure in the base frame (x forward, y left). kind: floor, person, trash (small, to
# pick up), overhead, glass or near (stereo cells outside every box; sure are the ones seen again from
# another view). height: metres measured to its top (a table: its lower edge), None when not measured;
# top_cut: the box reached the image top, so the thing is at least that high.
Hit = namedtuple("Hit", "label points sure kind how height top_cut", defaults=(None, False))

GLASS_RULE = """如果看到玻璃墙、玻璃门、窗户或镜子：玻璃或镜面一直落到地面、下面没有实心矮墙或踢脚线的，标签写“落地玻璃”或“落地镜子”，框的下边落在它和地面相接处；下面有实心矮墙、踢脚线或柜子挡着的，标签写“玻璃窗”或“镜子”。
透过玻璃看到的东西、镜子里映出来的东西、越过围栏或墙看到的东西都不在这里，一律不要框。别的墙、围栏、纸板挡板不用框。"""

SELF_RULE = """画面最下方左右两角的白色舵机、红白电线、底边中间的蓝色木色板子，以及从画面下边或两边伸进来的黑色竖杆、管子、支架，是机器人自己，不要框。"""

LANE_PROMPT = """你是机器人的视觉模块，场景是办公室，图像来自机器人头部相机，机器人正朝画面中间往前开。机器人本体长 0.30 米、宽 0.50 米、高 0.80 米。
把地面上和立在地面上的东西都框出来，每件东西一个框，离机器人近的先写，一个都不要漏。
标签写整件东西的名字，尽量从这些里选：桌子、柜子、书架、椅子、凳子、箱子、纸箱、纸板、包、垃圾桶、人、电线、数据线、插线板、充电器、细杆、支架、鞋，
地上要捡起来的小东西：纸屑、纸团、纸巾、塑料袋、瓶子、瓶盖、罐子、果皮、烟头、其他小物件。都不合适时写最贴切的名字。
桌子、柜子、椅子框整件，包括桌面、座面和扶手，标签写“桌子”“柜子”“椅子”，不要写桌面、抽屉、隔板这种部件名。人的脚、鞋、腿都算“人”。
颜色相同不等于同一种东西：巴掌大以内的碎纸写“纸屑”或“纸团”；比这大的纸板、纸箱，不管平放还是立着，写“纸板”或“纸箱”。
细小的东西也要框；地面上的阴影、反光、污渍、地毯花纹不算。框要包住整个物体，框的下边落在它和地面接触的地方。
""" + GLASS_RULE + "\n" + SELF_RULE + """
只输出 JSON，不要其他文字：{"objects": [{"bbox": [x1, y1, x2, y2], "label": "名字"}, ...]}；什么都没有输出 {"objects": []}。
坐标用 0-1000 归一化，图像左上角为 (0,0)，右下角为 (1000,1000)。"""

OVERHEAD_PROMPT = """你是机器人的避障模块，场景是办公室，图像来自机器人头部相机，相机离地约 0.7 米，稍微朝下看，机器人正朝画面中间往前开。
机器人本体长 0.30 米、宽 0.50 米、高 0.80 米，比相机高，桌子底下钻不过去。
找家具伸到地面上方、下面是空的、会挡住机器人上半身的部分：桌子的桌面和前沿、桌下的抽屉、托盘、键盘架、线槽，柜子和书架伸出来的柜面、隔板。
它们常常只在画面最上方露出一条边，或者从画面上沿伸下来，也一定要框。框要包住它在画面里的整个宽度，框的下边落在它最下面还能看见的那条边上。
标签写整件家具的名字：“桌子”“柜子”“书架”，不要写桌面、抽屉、隔板这种部件名。
一直落到地面的柜子、窗台下面的矮墙、踢脚线这里不要框；立在地上的东西（人、椅子、凳子、箱子、包、电线、垃圾）这里也不要框，另外检查。
""" + GLASS_RULE + "\n" + SELF_RULE + """
只输出 JSON，不要其他文字：{"objects": [{"bbox": [x1, y1, x2, y2], "label": "名字"}, ...]}；没有这些东西输出 {"objects": []}。
坐标用 0-1000 归一化，图像左上角为 (0,0)，右下角为 (1000,1000)。"""


def ask_json(img, prompt, model, timeout_s):
    """One DashScope request like vln_nav.ask_vlm; None when the answer cannot be trusted."""
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    # Thinking off: with it the newer models took 16-97 s a look, without it qwen3.8-max takes 2-3 s.
    body = {"model": model, "temperature": 0, "enable_thinking": False,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": prompt}]}]}
    t0 = time.time()
    r = None
    for attempt in range(3):
        try:
            r = requests.post(nav.DASHSCOPE_URL, json=body, timeout=(5, timeout_s),
                              headers={"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"})
            break
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  request failed after {time.time() - t0:.0f}s ({type(e).__name__}), retry {attempt + 1}/2",
                  flush=True)
    if r is None:
        print("  DashScope unreachable, check the robot's network")
        return None
    if r.status_code != 200:
        print(f"  DashScope HTTP {r.status_code}: {r.text[:300]}")
        return None
    reply = r.json()["choices"][0]["message"]["content"]
    print(f"  VLM {time.time() - t0:.1f}s: {' '.join(reply.split())}")
    return parse_reply(reply)


_NUM = r"(-?\d+(?:\.\d+)?)"


def _keep_box(pairs):
    """JSON object hook: a "bbox" that is not a list is a slipped "label" (Qwen once wrote
    {"bbox": [248, 370, 417, 522], "bbox": "笔记本电脑"}), so it does not overwrite the box."""
    out = {}
    for key, value in pairs:
        if key == "bbox" and not isinstance(value, list):
            if isinstance(value, str):
                out.setdefault("label", value)
            continue
        out[key] = value
    return out


def _numbers(values, n):
    """The first n entries as floats, or None when they are not all numbers."""
    if not isinstance(values, (list, tuple)) or len(values) < n:
        return None
    try:
        return [float(v) for v in values[:n]]
    except (TypeError, ValueError):
        return None


def parse_reply(reply):
    """JSON from the model, tolerating the slips Qwen makes."""
    m = re.search(r"\{.*\}", reply, re.S)
    if m:
        try:
            return json.loads(m.group(0), object_pairs_hook=_keep_box)
        except json.JSONDecodeError:
            pass
    # Keys seen: "bbox", "bbox_2d", and an unfinished "bbox_ [..] (qwen3.8-max now and then).
    boxes = re.findall(r'"bbox\w*"?\s*:?\s*\[\s*' + r"\s*,\s*".join([_NUM] * 4)
                       + r'\s*\]\s*,\s*"label"\s*:\s*"([^"]*)"', reply)
    return {"objects": [{"bbox": [float(v) for v in b[:4]], "label": b[4]} for b in boxes]} if boxes else None


def _objects(ans):
    """(label, [x1, y1, x2, y2] in pixels) of every box in a reply."""
    objs = (ans or {}).get("objects")
    if objs is None and (ans or {}).get("bbox"):  # older single-box answer
        objs = [ans]
    out = []
    for obj in objs if isinstance(objs, list) else []:
        # Qwen models have used both bbox and bbox_2d for the same pixel box.
        box = (_numbers(obj.get("bbox"), 4) or _numbers(obj.get("bbox_2d"), 4)) if isinstance(obj, dict) else None
        if box is None:
            print(f"  VLM 给的框不是 4 个数，跳过：{obj}", flush=True)
            continue
        x1, x2 = sorted((_px(box[0], cal.WIDTH), _px(box[2], cal.WIDTH)))
        y1, y2 = sorted((_px(box[1], cal.HEIGHT), _px(box[3], cal.HEIGHT)))
        out.append((object_name(str(obj.get("label") or "障碍")), (x1, y1, x2, y2)))
    return out


def object_name(label):
    """The whole thing a boxed part belongs to (桌面, 抽屉 -> 桌子; 隔板 -> 柜子)."""
    label = label.replace(OVERHEAD_TAG, "")
    for name, parts in PART_NAMES.items():
        if any(p in label for p in parts):
            return name
    return label


def spans(label):
    """True for a thing on legs, whose separate leg clusters are one thing."""
    return any(k in label for k in SPANNING_KEYS)


def floor_size(points):
    """(length, width) in metres of the smallest rectangle round floor points."""
    if len(points) < 2:
        return 0.0, 0.0
    _, (w, h), _ = cv2.minAreaRect(np.asarray(points, np.float32))
    return float(max(w, h)), float(min(w, h))


def _connected(points, gap=CLUSTER_GAP_M):
    """The points reached from the first one in steps no longer than `gap`: one body."""
    if not points:
        return []
    xy = np.asarray(points, np.float64)
    keep = np.zeros(len(xy), bool)
    keep[0] = True
    todo = [0]
    while todo:
        i = todo.pop()
        new = (~keep) & (np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1]) <= gap)
        keep |= new
        todo.extend(np.nonzero(new)[0].tolist())
    return [p for p, k in zip(points, keep) if k]


def _prune_runs(root, keep):
    """Delete all but the newest `keep` image folders of past runs; the Jetson disk is small."""
    import shutil

    runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for name in runs[:-keep]:
        shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def _px(v, size):
    return float(np.clip(float(v) / 1000.0 * size, 0, size - 1))


def _cells(xy, min_pts=STEREO_MIN_PTS):
    """Mean point of every STEREO_CELL_M floor cell holding at least min_pts of `xy` (N x 2), nearest first."""
    if len(xy) == 0:
        return []
    xy = np.asarray(xy, np.float64)
    _, cell, count = np.unique(np.floor(xy / STEREO_CELL_M).astype(np.int64), axis=0,
                               return_inverse=True, return_counts=True)
    cell = cell.reshape(-1)
    sx, sy = np.bincount(cell, weights=xy[:, 0]), np.bincount(cell, weights=xy[:, 1])
    good = count >= min_pts
    cells = [(float(a), float(b)) for a, b in zip(sx[good] / count[good], sy[good] / count[good])]
    return sorted(cells, key=lambda p: math.hypot(*p))


def _on_body(px, py, margin=0.02):
    """True when a floor point lies on the robot itself (0.30 m long, 0.50 m wide), within `margin`."""
    return abs(px) <= 0.15 + margin and abs(py) <= 0.25 + margin


def _densify(points, spacing=0.05):
    """Fill the line between neighbouring points; gaps over 0.8 m stay open."""
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = 1 if d > 0.8 else max(1, int(math.ceil(d / spacing)))
        out.extend((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1))
    return out


class Head:
    """Head cameras on the real robot, with the neck pose folded into the camera model."""

    def __init__(self, model, timeout_s, log_dir):
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise SystemExit("DASHSCOPE_API_KEY is not set (export it in ~/.bashrc, then open a new terminal)")
        self.model = model
        self.timeout_s = timeout_s
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        _prune_runs(os.path.dirname(log_dir), KEEP_RUNS)
        calib, extr, cams = val.load(cal.DEFAULT_DATA)
        self.cam0 = cams[nav.EYE]
        self.neck_home = (extr.get("neck_yaw_pos") or 0.0, extr.get("neck_pitch_pos") or 0.0)
        self.cap = cal.open_camera(cal.CAMS[nav.EYE], calib.get("focus_absolute", 0))
        # The stereo grid maps back onto the left image, where the VLM draws its boxes.
        assert nav.EYE == "left", "stereo assumes the VLM sees the left camera"
        self.cap_right = cal.open_camera(cal.CAMS["right"], calib.get("focus_absolute", 0))
        self.stereo = Stereo(calib, cams)
        self.base = None
        self.neck = None
        self.pose = (0.0, 0.0)
        self.shot = 0
        self.self_mask = {}  # (neck yaw, pitch) -> pixels showing the robot's own body
        self.where = None    # map pose (x, y, yaw rad) of the body during the current look
        self.near_seen = {}  # map cell -> camera viewpoints that measured an unboxed stereo cell there
        self.lidar_xy = None  # callable -> N x 2 lidar returns in the base frame, or None
        self.last_hits = []
        self.last_floor = []

    def attach(self, base):
        self.base = base
        self.pose = base.pose
        self.neck = (0.0, 0.0)

    def close(self):
        self.cap.release()
        self.cap_right.release()

    def _grab_pair(self):
        """Left and right frames taken together; what was buffered while moving is dropped."""
        caps = (self.cap, self.cap_right)
        for _ in range(5):
            for cap in caps:
                cap.grab()
        frames = []
        for cap in caps:
            ok, img = cap.retrieve()
            if not ok:
                raise SystemExit("camera read failed")
            frames.append(img)
        return frames

    def _look(self, d_yaw, d_pitch):
        if self.neck != (d_yaw, d_pitch):
            self.pose = self.base.look(d_yaw, d_pitch)
            self.neck = (d_yaw, d_pitch)
        return nav.cam_at(self.cam0, *self.pose)

    def _save(self, img, name, pixels, boxes=(), dots=()):
        vis = img.copy()
        for u, v in dots:
            cv2.circle(vis, (int(u), int(v)), 1, (0, 255, 0), -1)
        for u, v in pixels:
            cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        self.shot += 1
        cv2.imwrite(os.path.join(self.log_dir, f"{self.shot:03d}_{name}.jpg"), vis)

    def learn_self(self, turn, side_yaw):
        """Find the robot's own body in the picture for every head pose the lane check uses.

        turn(deg) turns the body to deg from its heading now (0 turns it back). The same head
        poses are shot at three body headings; whatever does not move is the body.
        """
        poses = [(0.0, p) for p in LANE_PITCHES] + [(side_yaw, SIDE_PITCH), (-side_yaw, SIDE_PITCH)]
        frames = {pose: [] for pose in poses}
        for deg in (0.0, SELF_TURN_DEG, -SELF_TURN_DEG):
            if deg and not turn(deg):
                print("  转身没转到位，不学车身样子了", flush=True)
                break
            for pose in poses:
                self._look(*pose)
                frames[pose].append(cv2.cvtColor(nav.grab(self.cap), cv2.COLOR_BGR2GRAY))
        turn(0.0)
        for pose, grays in frames.items():
            if len(grays) < 3:
                continue
            mask = self_mask_from(grays)
            share = float(mask.mean())
            name = f"self_y{int(pose[0]):+d}_p{int(pose[1]):02d}"
            if share > SELF_MAX_SHARE:
                print(f"  [{name}] 车身占了 {share:.0%} 的画面，不可信，不用", flush=True)
                continue
            self.self_mask[pose] = mask
            vis = cv2.cvtColor(grays[0], cv2.COLOR_GRAY2BGR)
            vis[mask] = (0.5 * vis[mask] + (127, 0, 127)).astype(np.uint8)
            self._save(vis, name, [])
            print(f"  [{name}] 车身占画面 {share:.0%}", flush=True)
        return bool(self.self_mask)

    def lane(self, d_yaw=0.0, near_only=False, where=None):
        """Everything one stop sees ahead, as a list of Hit in the base frame.

        The head looks up first (UP_PITCH: tabletops and shelves), then down (LANE_PITCHES, far to
        near: what lies or stands on the floor). near_only looks down only at the steepest angle,
        for the floor beside the body before a turn or slide. d_yaw turns the head (+ is right).
        `where` is the body's map pose; without it unboxed stereo cells are never confirmed.
        last_floor keeps the floor points the stereo measured, which clear older marks.
        """
        self.where = where
        hits, floor = [], []
        for pitch in (UP_PITCH,) + ((LANE_PITCHES[-1],) if near_only else LANE_PITCHES):
            got, flat = self._lane_at(d_yaw, pitch)
            hits += got
            floor += flat
        self.last_hits, self.last_floor = hits, floor
        return hits

    def _lane_at(self, d_yaw, pitch):
        """(hits, floor points) of one head pose."""
        cam = self._look(d_yaw, pitch)
        img, right = self._grab_pair()
        up = pitch < 0
        ans = ask_json(img, OVERHEAD_PROMPT if up else LANE_PROMPT, self.model, self.timeout_s)
        if ans is None:
            print("  VLM 本帧失败，只用雷达和双目", flush=True)
        pts = self.stereo.points(img, right, cam["T_base_cam"])
        tag = f"p{int(pitch):02d}"
        mask = self.self_mask.get((float(d_yaw), float(pitch)))
        # The robot's own servos and wires sit in the bottom of the frame; the steeper the head,
        # the higher they climb.
        self_top = min(0.9, SELF_TOP - 0.01 * (pitch - NEAR_PITCH))
        hits, boxes, marks = [], [], []
        shown = np.zeros(pts.shape[:2], bool)
        for label, box in _objects(ans):
            x1, y1, x2, y2 = box
            if mask is not None:
                on_self = float(mask[int(y1):int(y2) + 1, int(x1):int(x2) + 1].mean())
                if on_self > SELF_SHARE:
                    print(f"  [{tag}] 框里 {on_self:.0%} 是机器人自己（{label}），不算")
                    continue
            elif y2 >= cal.HEIGHT - EDGE_PX and y1 >= self_top * cal.HEIGHT:
                print(f"  [{tag}] 框到的是机器人自己（{label}），不算")
                continue
            boxes.append(box)
            if any(k in label for k in GLASS_KEYS):
                if "落地" not in label:
                    print(f"  [{tag}] {label}不落地，下面有实心的东西，边界听雷达的", flush=True)
                    continue
                hit, kept = self._glass(label, box, pts, mask, tag)
            elif (any(k in label for k in BOUNDARY_KEYS)
                  and self._lidar_in(self._rays(cam, x1, x2, min(y2, cal.HEIGHT - 1)), LIDAR_PAD_DEG)):
                continue  # the lidar sees this wall or fence and maps it
            elif up:
                hit, kept = self._overhead(label, box, cam, pts, mask, tag)
            else:
                hit, kept = self._locate(label, box, cam, pts, mask, tag)
            shown |= kept
            if hit is None:
                continue
            if (y2 >= cal.HEIGHT - EDGE_PX and (x1 <= EDGE_PX or x2 >= cal.WIDTH - EDGE_PX)
                    and all(_on_body(px, py, UPRIGHT_M) for px, py in hit.points)):
                print(f"  [{tag}] 框到的是机器人两侧的立柱（{label}），不算")
                continue
            hits.append(hit)
            marks.append(((x1 + x2) / 2, min(y2, cal.HEIGHT - 1)))
        near, kept = self._footprint(pts, mask=mask, skip=boxes, reach=NEAR_STEREO_M)
        if near:
            shown |= kept
            sure = self._confirm(near)
            print(f"  [{tag}] 框外双目量到立着的东西 {len(near)} 格（换个角度也量到的 {len(sure)} 格），"
                  f"最近 {math.hypot(*near[0]):.2f} m", flush=True)
            hits.append(Hit(STEREO_LABEL, near, sure, "near", "双目"))
        self._save(img, f"lane_{'hit' if hits else 'clear'}_{tag}", marks, boxes, self.stereo.uv[shown][::3])
        return hits, self._floor(pts, mask)

    # --- where things are ----------------------------------------------------------------------
    def _usable(self, pts, box=None, mask=None):
        """Rectified stereo pixels with a trusted point, not on the robot, optionally inside `box`."""
        u, v = self.stereo.uv[..., 0], self.stereo.uv[..., 1]
        sel = np.isfinite(pts).all(axis=2) & (u >= 0) & (u <= cal.WIDTH - 1) & (v >= 0) & (v <= cal.HEIGHT - 1)
        if box is not None:
            x1, y1, x2, y2 = box
            sel &= (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        if mask is not None:
            sel &= ~mask[np.clip(v, 0, cal.HEIGHT - 1).astype(int), np.clip(u, 0, cal.WIDTH - 1).astype(int)]
        x, y = pts[..., 0], pts[..., 1]
        sel &= ~((np.abs(x) <= 0.15 + STEREO_BODY_M) & (np.abs(y) <= 0.25 + STEREO_BODY_M))
        return sel

    def _footprint(self, pts, box=None, mask=None, skip=(), reach=OBJECT_REACH_M):
        """Floor-plane cells of the stereo points standing between the floor and the top of the body.

        `box` limits them to one pixel box, boxes in `skip` are left out. Returns ([(x, y)] one per
        cell nearest first, rectified pixels behind them).
        """
        sel = self._usable(pts, box, mask)
        z = pts[..., 2]
        with np.errstate(invalid="ignore"):
            sel &= (z >= STEREO_MIN_Z) & (z <= STEREO_MAX_Z) & (np.hypot(pts[..., 0], pts[..., 1]) <= reach)
        u, v = self.stereo.uv[..., 0], self.stereo.uv[..., 1]
        for x1, y1, x2, y2 in skip:
            sel &= ~((u >= x1) & (u <= x2) & (v >= y1) & (v <= y2))
        cells = _cells(pts[sel][:, :2])
        if not cells:
            return [], np.zeros(sel.shape, bool)
        return cells, sel

    def _floor(self, pts, mask):
        """Floor cells the stereo measured (base frame), for clearing marks of things now gone."""
        sel = self._usable(pts, mask=mask)
        z = pts[..., 2]
        with np.errstate(invalid="ignore"):
            sel &= (z >= FLOOR_Z[0]) & (z <= FLOOR_Z[1]) & (np.hypot(pts[..., 0], pts[..., 1]) <= FLOOR_SEEN_M)
        return _cells(pts[sel][:, :2])

    def _rays(self, cam, x1, x2, v, n=7):
        """(bearing, rise per metre along the floor) of rays through n pixels along one image row."""
        rot = cam["T_base_cam"][:3, :3]
        rays = []
        for k in range(n):
            p = cv2.undistortPoints(np.array([[[x1 + (x2 - x1) * k / (n - 1), v]]], float),
                                    cam["K"], cam["dist"]).reshape(2)
            d = rot @ np.array([p[0], p[1], 1.0])
            flat = math.hypot(d[0], d[1])
            if flat > 1e-6:
                rays.append((math.atan2(d[1], d[0]), d[2] / flat))
        return rays

    def _lidar_in(self, rays, pad_deg, reach=OBJECT_REACH_M):
        """Lidar returns (base frame) in the bearings of `rays`, widened by pad_deg each side."""
        xy = self.lidar_xy() if self.lidar_xy is not None else None
        if xy is None or not rays:
            return []
        pad = math.radians(pad_deg)
        lo, hi = min(b for b, _ in rays) - pad, max(b for b, _ in rays) + pad
        out = []
        for px, py in xy:
            px, py = float(px), float(py)
            r, b = math.hypot(px, py), math.atan2(py, px)
            if 0.28 <= r <= reach and lo <= b <= hi and not _on_body(px, py):
                out.append((px, py))
        return out

    def _locate(self, label, box, cam, pts, mask, tag):
        """Where a boxed thing on the floor is.

        Within STEREO_TRUST_M the stereo measures it: its standing parts (STEREO_MIN_Z-STEREO_MAX_Z
        high), or for a flat thing the floor patch it lies on; points more than OBJECT_DEPTH_M behind
        the nearest are background. Farther out the stereo is too coarse and _far places it. Of a
        thing not on legs only the points joined to the nearest one are its body. A thing to pick up
        is one only when it measures no longer than PICK_MAX_M.
        """
        sel = self._usable(pts, box, mask)
        z, rng = pts[..., 2], np.hypot(pts[..., 0], pts[..., 1])
        with np.errstate(invalid="ignore"):
            sel &= (z <= STEREO_MAX_Z) & (rng <= STEREO_TRUST_M)
            standing = sel & (z >= STEREO_MIN_Z)
        cells, how, kept = _cells(pts[standing][:, :2]), "双目", standing
        if not cells:
            cells, how, kept = _cells(pts[sel][:, :2]), "双目（贴地）", sel
        if cells:
            near = math.hypot(*cells[0])
            cells = [c for c in cells if math.hypot(*c) <= near + OBJECT_DEPTH_M]
        else:
            cells, how = self._far(box, cam)
        cells = [c for c in cells if not _on_body(*c)]
        if not cells:
            print(f"  [{tag}] 看到{label}，但双目和雷达都量不到位置，不用", flush=True)
            return None, kept
        cells.sort(key=lambda p: math.hypot(*p))
        if not spans(label):
            cells = _connected(cells)  # what stands behind it in the box is another thing
        near = math.hypot(*cells[0])
        zs = None
        if how == "双目":
            far = max(math.hypot(*c) for c in cells) + STEREO_CELL_M
            zs = pts[standing][np.hypot(pts[standing][:, 0], pts[standing][:, 1]) <= far][:, 2]
        height, top_cut = self._height(box, cam, near, zs)
        print(f"  [{tag}] 看到{label}，{how}量到最近 {near:.2f} m（{len(cells)} 点），"
              f"高{'至少' if top_cut else '约'} {height:.2f} m", flush=True)
        kind = "person" if "人" in label else "floor"
        if kind == "floor" and any(k in label for k in PICK_KEYS):
            length = floor_size(cells)[0] + (STEREO_CELL_M if how.startswith("双目") else 0.0)
            if length <= PICK_MAX_M:
                kind = "trash"
            else:
                print(f"  [{tag}] {label}量到长 {length:.2f} m，比夹爪能夹的 {PICK_MAX_M:.2f} m 大，当障碍",
                      flush=True)
        return Hit(label, cells, cells, kind, how, height, top_cut), kept

    def _height(self, box, cam, r, zs=None):
        """(metres, top_cut) up to the top of a boxed thing whose front is r away: where the ray through
        the box top is at that range, or the highest stereo point on it when that is higher."""
        x1, y1, x2, _ = box
        rays = self._rays(cam, x1, x2, y1)
        height = float(cam["T_base_cam"][2, 3]) + max(rise for _, rise in rays) * r if rays else 0.0
        if zs is not None and len(zs):
            height = max(height, float(np.max(zs)))
        return max(0.0, height), y1 <= EDGE_PX

    def _bottom_error(self, cam, r):
        """Metres a box bottom projected onto the floor at range r may be off (BOX_EDGE_PX of pixel)."""
        h = float(cam["T_base_cam"][2, 3])
        return BOX_EDGE_M + BOX_EDGE_PX / float(cam["K"][0, 0]) * (h * h + r * r) / h

    def _far(self, box, cam):
        """(cells, how) of a boxed thing the stereo did not measure within STEREO_TRUST_M.

        The box bottom, projected onto the floor, is where the thing stands. Lidar returns in the
        box's bearings are the thing only when they agree with that: no nearer than the bottom line
        (less its error) and at most OBJECT_DEPTH_M behind it. A thing lower than the lidar beam
        has no such returns (the lidar sees what stands near or behind it), and the bottom line
        alone places it. A box running off the image bottom reaches nearer than the bottom row:
        only lidar returns within reach of that row place it.
        """
        x1, y1, x2, y2 = box
        cut = y2 >= cal.HEIGHT - EDGE_PX
        v = min(y2, cal.HEIGHT - 1)
        line = []
        for k in range(7):
            f = val.pixel_to_floor(x1 + (x2 - x1) * k / 6.0, v, cam)
            if f is not None and math.hypot(f[0], f[1]) <= OBJECT_REACH_M:
                line.append((float(f[0]), float(f[1])))
        if not line:
            return [], ""
        rf = min(math.hypot(*f) for f in line)
        lo = 0.0 if cut else rf - self._bottom_error(cam, rf)
        own = [p for p in self._lidar_in(self._rays(cam, x1, x2, v), LIDAR_PAD_DEG)
               if lo <= math.hypot(*p) <= rf + OBJECT_DEPTH_M]
        if own:
            rl = min(math.hypot(*p) for p in own)
            own = [p for p in own if math.hypot(*p) <= rl + OBJECT_DEPTH_M]
            return (own, "雷达（框伸出画面下沿）") if cut else (_densify(line, 0.04) + own, "框底投到地面，雷达对上")
        return ([], "") if cut else (_densify(line, 0.04), "框底投到地面")

    def _glass(self, label, box, pts, mask, tag):
        """Floor-standing glass or a mirror, ranged by the stereo along the box: its frame, kickboard
        and marks on it are the nearest points, reflections and what is behind are farther."""
        x1, y1, x2, y2 = box
        sel = self._usable(pts, box, mask)
        rng = np.hypot(pts[..., 0], pts[..., 1])
        with np.errstate(invalid="ignore"):
            sel &= (pts[..., 2] <= 2.0) & (rng <= 4.0)
        u = self.stereo.uv[..., 0]
        line = []
        for k in range(7):
            a, b = x1 + (x2 - x1) * k / 7.0, x1 + (x2 - x1) * (k + 1) / 7.0
            strip = pts[sel & (u >= a) & (u < b)]
            if len(strip) < STEREO_MIN_PTS * 4:
                continue
            r = np.hypot(strip[:, 0], strip[:, 1])
            near = float(np.percentile(r, 20))
            bear = float(np.median(np.arctan2(strip[:, 1], strip[:, 0])))
            line.append((near * math.cos(bear), near * math.sin(bear)))
        if not line:
            print(f"  [{tag}] {label}在双目看不到的地方，量不出距离，这次边界听雷达的", flush=True)
            return None, sel
        line = _densify(sorted(line, key=lambda p: math.atan2(p[1], p[0])), 0.04)
        print(f"  [{tag}] {label}：双目量到离车最近 {min(math.hypot(*p) for p in line):.2f} m，"
              "雷达在它后面的回波不用", flush=True)
        return Hit(label, line, line, "glass", "双目"), sel

    def _overhead(self, label, box, cam, pts, mask, tag):
        """A tabletop, drawer or shelf boxed in an upward look, as a hit along its lower edge.

        The rays through the box bottom give the bearings of the edge and how steeply it is seen,
        not its range. The range is measured by what holds it up or hangs under it: lidar returns
        and stereo cells (STEREO_MIN_Z-STEREO_MAX_Z high) in those bearings below the box top, from
        the nearest one back OVERHEAD_DEPTH_M. Each edge sample takes the nearest of them within
        reach of its bearing; samples with none (open floor between two legs) are interpolated
        along the bearing. The edge height then follows from range and slope: an edge higher than
        STEREO_MAX_Z passes over the body, and one lower than STEREO_MIN_Z is where something
        standing meets the floor, which the lidar and stereo map. Returns (hit or None, stereo
        pixels used).
        """
        x1, y1, x2, y2 = box
        z_cam = float(cam["T_base_cam"][2, 3])
        rays = self._rays(cam, x1, x2, min(y2, cal.HEIGHT - 1))
        cells, kept = self._footprint(pts, (x1, y1, x2, cal.HEIGHT - 1), mask, reach=STEREO_TRUST_M)
        pad = math.radians(OVERHEAD_PAD_DEG)
        lo, hi = min(b for b, _ in rays) - pad, max(b for b, _ in rays) + pad
        support = [(math.atan2(py, px), math.hypot(px, py), px, py, False) for px, py in cells
                   if lo <= math.atan2(py, px) <= hi and math.hypot(px, py) >= 0.28]
        support += [(math.atan2(py, px), math.hypot(px, py), px, py, True)
                    for px, py in self._lidar_in(rays, OVERHEAD_PAD_DEG)]
        if not support:
            print(f"  [{tag}] 看到{label}悬在前方，但这些方向上雷达和双目都没量到东西，量不出距离，不用", flush=True)
            return None, kept
        near = min(s[1] for s in support)
        support = [s for s in support if s[1] <= near + OVERHEAD_DEPTH_M]
        # Wide boxes space the samples further apart than OVERHEAD_RAY_DEG; every support still
        # reaches the sample nearest to it.
        win = max(math.radians(OVERHEAD_RAY_DEG), (hi - lo - 2 * pad) / 12.0 + math.radians(0.5), pad)
        ranges = []
        for b, _ in rays:
            close = [s[1] for s in support if abs(s[0] - b) <= win]
            ranges.append(min(close) if close else None)
        known = [i for i, r in enumerate(ranges) if r is not None]
        for i, r in enumerate(ranges):
            if r is None:
                left = max((j for j in known if j < i), default=None)
                right = min((j for j in known if j > i), default=None)
                if left is None or right is None:
                    ranges[i] = ranges[right if left is None else left]
                else:
                    ranges[i] = ranges[left] + (ranges[right] - ranges[left]) * (i - left) / (right - left)
        heights = [z_cam + rise * r for (_, rise), r in zip(rays, ranges)]
        # Where the box bottom comes out lower than STEREO_MIN_Z it meets the floor: there the box
        # holds the foot of something standing (a low wall under a windowsill, a cabinet side), not an
        # edge over open floor. Mostly so, and the box is no overhead thing at all.
        low = sum(1 for z in heights if z < STEREO_MIN_Z)
        if 2 * low > len(heights):
            print(f"  [{tag}] {label}的下沿量到贴着地面（离地约 {float(np.median(heights)):.2f} m），"
                  "是立在地上的东西，不是悬空的，交给雷达和双目", flush=True)
            return None, kept
        used = [(b, r, z) for (b, _), r, z in zip(rays, ranges, heights) if STEREO_MIN_Z <= z <= STEREO_MAX_Z]
        if not used:
            print(f"  [{tag}] {label}的下沿量到离地 {min(z for z in heights if z >= STEREO_MIN_Z):.2f} m，高过车身，"
                  "能从下面过", flush=True)
            return None, kept
        edge = [(r * math.cos(b), r * math.sin(b)) for b, r, _ in used]
        # Only what holds up the kept part of the edge belongs to it.
        support = [s for s in support if any(abs(s[0] - b) <= win for b, _, _ in used)]
        if low:
            print(f"  [{tag}] {label}的下沿有 {low} 处贴着地面，那几处不算", flush=True)
        points = sorted(_densify(edge, 0.04) + [(s[2], s[3]) for s in support], key=lambda p: math.hypot(*p))
        n_lidar = sum(1 for s in support if s[4])
        height = float(np.median([z for _, _, z in used]))
        print(f"  [{tag}] 看到{label}伸在前方：下面雷达 {n_lidar} 个点、双目 {len(support) - n_lidar} 格，"
              f"前沿最近 {min(r for _, r, _ in used):.2f} m，下沿离地约 {height:.2f} m", flush=True)
        return Hit(label, points, points, "overhead", "雷达+双目", height, False), kept

    def _confirm(self, cells):
        """The cells (base frame) that an earlier look from another viewpoint also measured.

        No VLM box backs these, and a mismatch on repeating texture (window bars behind a thin
        pole) repeats when the same picture is taken again. Another camera position or angle does
        not repeat it.
        """
        if self.where is None:
            return []
        x0, y0, yaw = self.where
        view = (x0, y0, math.degrees(yaw) - self.pose[0], self.pose[1])  # neck yaw + is right
        c, s = math.cos(yaw), math.sin(yaw)
        sure = []
        for px, py in cells:
            mx, my = x0 + c * px - s * py, y0 + s * px + c * py
            key = (int(math.floor(mx / STEREO_CELL_M)), int(math.floor(my / STEREO_CELL_M)))
            views = self.near_seen.pop(key, [])
            if any(math.hypot(view[0] - v[0], view[1] - v[1]) >= NEAR_CONFIRM_M
                   or abs((view[2] - v[2] + 180.0) % 360.0 - 180.0) >= NEAR_CONFIRM_DEG
                   or abs(view[3] - v[3]) >= NEAR_CONFIRM_DEG for v in views):
                sure.append((px, py))
            self.near_seen[key] = (views + [view])[-4:]
        while len(self.near_seen) > NEAR_SEEN_KEEP:
            del self.near_seen[next(iter(self.near_seen))]
        return sure


def self_mask_from(grays):
    """Pixels that stay the same in frames taken from different body headings: the robot's own body.

    Floor and furniture move across the picture when the body turns; the body parts in view do
    not. Grey levels shift with the exposure, which follows the scene, so textured pixels are
    compared by local correlation (blind to brightness and contrast) and flat ones by a loose
    grey difference. Only still regions touching the image bottom count (the body comes in
    from below), slightly grown.
    """
    imgs = [g.astype(np.float32) for g in grays]
    k = (21, 21)
    still = np.ones(imgs[0].shape, bool)
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            a, b = imgs[i], imgs[j]
            ma, mb = cv2.blur(a, k), cv2.blur(b, k)
            va = np.maximum(cv2.blur(a * a, k) - ma * ma, 0.0)
            vb = np.maximum(cv2.blur(b * b, k) - mb * mb, 0.0)
            ncc = (cv2.blur(a * b, k) - ma * mb) / np.sqrt(va * vb + 1e-3)
            flat = (va < SELF_FLAT_VAR) & (vb < SELF_FLAT_VAR) & (np.abs(ma - mb) < SELF_DIFF)
            still &= (ncc > SELF_NCC) | flat
    m = cv2.morphologyEx(still.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    _, lab = cv2.connectedComponents(m)
    keep = np.isin(lab, [i for i in np.unique(lab[-1]) if i != 0])
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((15, 15), np.uint8))
    return keep.astype(bool)
