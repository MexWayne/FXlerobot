"""Head camera questions for the coverage run.

Two questions: where the fence meets the floor (initial scan), and whether
anything stands in the lane ahead (while driving). The robot, the neck-aware
camera model and the request style are taken from vln_nav.py by importing it;
that file is not modified.
"""

import base64
import json
import math
import os
import re
import sys
import time

import cv2
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
for path in (os.path.join(PROJECT, "project_tool"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import vln_nav as nav  # noqa: E402

cal, val = nav.cal, nav.val

NEAR_PITCH = 25.0  # same head-down angle vln_nav uses once the floor contact leaves the image
LANE_PITCHES = (10.0, 20.0, 30.0)  # head-down angles for the obstacle check, far to near
SELF_NCC = 0.75                    # local correlation above which a textured pixel counts as not moving
SELF_FLAT_VAR = 40.0               # grey variance under which a 21x21 patch counts as flat
SELF_DIFF = 30.0                   # mean grey change under which a flat patch counts as not moving
SELF_TURN_DEG = 30.0               # body turn between the frames that find the body in the picture
SELF_SHARE = 0.6                   # a box this much on the body is the body
SELF_MAX_SHARE = 0.5               # a mask bigger than this means the body did not really turn
UPRIGHT_M = 0.15                   # a box from a bottom corner whose contact is this close to the body is its upright
LANE_HALF_M = 0.30                 # half the body width plus a margin: what the body would hit
SNAP_PITCH = LANE_PITCHES[-1]      # floor-texture snapshots use the angle the lane check ends on
EDGE_PX = 6
KEEP_RUNS = 3      # image folders kept under map/look
SELF_TOP = 0.60  # a box on the image bottom that starts in the bottom 40% is the robot's own deck

FENCE_PROMPT = """你是机器人的视觉模块，图像来自机器人头部相机。机器人被一圈围栏围在里面，围栏可能是挡板、墙、箱子，也可能是玻璃墙。
玻璃是透明的，不能当成没有围栏。边界用玻璃下面的实心踢脚线：光滑的灰色或白色基座，和深色条纹地毯相接的那条边。
这条边上方是基座，下方是地毯。沿这条边从左到右给点，大约每隔画面宽度的 1/8 一个。
点要打在两种材质相接的地方，不要打在玻璃上，不要打在基座的上沿，不要打在画面最底下，也不要打在机器人自己的舵机和板子上。看不到这条边的部分不要编。
不是围栏的东西不要算。
只输出 JSON，不要其他文字：{"found": true, "points": [[x, y], [x, y], ...]}；画面里没有围栏输出 {"found": false}。
坐标用 0-1000 归一化，图像左上角为 (0,0)，右下角为 (1000,1000)。"""

LANE_PROMPT = """你是机器人的避障模块，图像来自机器人头部相机，机器人正朝画面中间往前开。
机器人本体长 0.30 米、宽 0.50 米、高 0.80 米。
先看玻璃墙的踢脚线：光滑的灰色或白色基座和深色条纹地毯相接的那条边。看到了就用一个框框住这块基座，标签写“墙”，框的下边落在这条边上，不要落在基座的上沿，也不要落在画面最底下。玻璃能看穿，不要因此当成没有东西。
再把画面里其他立在地面上、机器人可能撞到的东西都框出来，每个东西一个框，最多 6 个：
人（人的脚、鞋、腿都要框，脚是最容易被撞的）、围栏、纸板、墙、箱子、包、椅子腿、凳子腿、桌子腿、椅子底座等。
离机器人越近（越靠画面下方）的越重要，一定不能漏；同一个画面里有好几个东西时全部都要框，不要只框一个。
地面上的阴影、反光、污渍、地毯花纹不算；平放在地上的细电线不算。
画面最下方左右两角的白色舵机、红白电线，以及底边中间的蓝色、木色板子，是机器人自己的身体，不是障碍物，不要框。
从画面下边或左右两边伸进来的黑色竖杆、黑色管子、黑色支架，是机器人自己两侧的立柱，不是障碍物，不要框。
框要包住整个物体的宽度（比如椅子的整个五角底座、整只鞋），框的下边要落在物体与地面接触的位置。
只输出 JSON，不要其他文字：{"objects": [{"bbox": [x1, y1, x2, y2], "label": "名字"}, ...]}；地面上没有东西输出 {"objects": []}。
坐标用 0-1000 归一化，图像左上角为 (0,0)，右下角为 (1000,1000)。"""


def _smooth_edge(raw, width):
    """Drop columns that jump off the edge their neighbours agree on."""
    kept = []
    for u, v in raw:
        nb = [vv for uu, vv in raw if abs(uu - u) <= 64]
        if abs(v - float(np.median(nb))) > 22:
            continue
        kept.append((u, int(v)))
    if len(kept) < 12 or kept[-1][0] - kept[0][0] < width // 3:
        return []
    return kept


def kickboard_pixels(img):
    """The glass-wall boundary: the lower edge of the solid kickboard, in pixels.

    The bright top rail is not the boundary. When that rail is in frame, the boundary is the
    fainter edge below it (the toe of the base). When the camera looks down far enough to see
    the carpet, the boundary is where the smooth base gives way to the striped carpet.
    A carpet stripe has texture above it as well, so it is not a kickboard and is ignored.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(grey, (0, 0), 2)
    h, w = blur.shape
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)
    k = (15, 15)
    mu = cv2.blur(grey, k)
    var = np.maximum(cv2.blur(grey * grey, k) - mu * mu, 0.0)
    limit = int(0.93 * h)
    rails = []
    for u in range(0, w, 8):
        v = int(np.argmax(gy[:int(0.45 * h), u]))
        if gy[v, u] > 500 and v < 0.4 * h:
            rails.append((u, v))
    toe = []
    if len(rails) > w // 20:
        for u in range(6, w - 6, 4):
            near = [v for uu, v in rails if abs(uu - u) <= 24]
            if not near:
                continue
            rail = int(np.median(near))
            col = blur[:, u]
            best_v, best_s = None, 8.0
            for v in range(min(rail + 150, limit - 20), limit - 18):
                score = float(col[v] - col[v + 18])
                if score > best_s:
                    best_v, best_s = v, score
            if best_v is not None:
                toe.append((u, best_v))
        toe = _smooth_edge(toe, w)
    carpet = []
    for u in range(6, w - 6, 4):
        col, tex = blur[:, u], var[:, u]
        best_v, best_d = None, 0.0
        for v in range(40, limit - 16):
            drop = float(col[v - 12:v].mean() - col[v:v + 12].mean())
            if drop < 22:
                continue
            ta = float(tex[max(0, v - 36):max(0, v - 12)].mean()) if v > 36 else 1e9
            tb = float(tex[v + 2:v + 14].mean())
            if tb < 350 or tb < ta * 1.4:
                continue
            if drop > best_d:
                best_v, best_d = v, drop
        if best_v is not None:
            carpet.append((u, best_v))
    carpet = _smooth_edge(carpet, w)
    # The grey base above a real kickboard edge is smooth. Carpet stripes are textured on both
    # sides, and those were being drawn as a boundary all over the floor.
    if _smooth_panel(var, toe):
        return toe
    if _smooth_panel(var, carpet):
        return carpet
    return []


def _smooth_panel(var, pts):
    """True when the band above the edge is a smooth panel, not carpet."""
    if len(pts) < 12:
        return False
    vals = []
    for u, v in pts[::2]:
        a, b = v - 100, v - 28
        if a < 8:
            return False
        vals.append(float(np.median(var[a:b, max(0, u - 2):u + 3])))
    return len(vals) >= 6 and float(np.median(vals)) < 40


def ask_json(img, prompt, model, timeout_s):
    """One DashScope request like vln_nav.ask_vlm; None when the answer cannot be trusted."""
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    body = {"model": model, "temperature": 0,
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
    print(f"  VLM {time.time() - t0:.1f}s: {reply.strip()}")
    return parse_reply(reply)


_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_reply(reply):
    """JSON from the model, tolerating the slips Qwen makes such as a missing outer bracket on points."""
    m = re.search(r"\{.*\}", reply, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    if re.search(r'"found"\s*:\s*false', reply):
        return {"found": False}
    out = {"found": True}
    box = re.search(r'"bbox"\s*:\s*\[\s*' + r"\s*,\s*".join([_NUM] * 4), reply)
    if box:
        out["bbox"] = [float(v) for v in box.groups()]
    pairs = re.findall(r"\[\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\]", reply)
    if pairs and '"points"' in reply:
        out["points"] = [[float(a), float(b)] for a, b in pairs]
    label = re.search(r'"label"\s*:\s*"([^"]*)"', reply)
    if label:
        out["label"] = label.group(1)
    return out if ("bbox" in out or "points" in out) else None


def _prune_runs(root, keep):
    """Delete all but the newest `keep` image folders of past runs; the Jetson disk is small."""
    import shutil

    runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for name in runs[:-keep]:
        shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def _px(v, size):
    return float(np.clip(float(v) / 1000.0 * size, 0, size - 1))


class Head:
    """Left head camera on the real robot, with the neck pose folded into the camera model."""

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
        self.base = None
        self.neck = None
        self.pose = (0.0, 0.0)
        self.shot = 0
        self.self_mask = {}  # (neck yaw, pitch) -> pixels showing the robot's own body

    def attach(self, base):
        self.base = base
        self.pose = base.pose
        self.neck = (0.0, 0.0)

    def close(self):
        self.cap.release()

    def _look(self, d_yaw, d_pitch):
        if self.neck != (d_yaw, d_pitch):
            self.pose = self.base.look(d_yaw, d_pitch)
            self.neck = (d_yaw, d_pitch)
        return nav.cam_at(self.cam0, *self.pose)

    def _save(self, img, name, pixels, box=None):
        vis = img.copy()
        for u, v in pixels:
            cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        boxes = [] if box is None else (box if isinstance(box, list) else [box])
        for x, y, w, h in boxes:
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        self.shot += 1
        cv2.imwrite(os.path.join(self.log_dir, f"{self.shot:03d}_{name}.jpg"), vis)

    def learn_self(self, turn, side_yaw):
        """Find the robot's own body in the picture for every head pose the lane check uses.

        turn(deg) turns the body to deg from its heading now (0 turns it back). The same head
        poses are shot at three body headings; whatever does not move is the body.
        """
        poses = [(0.0, p) for p in LANE_PITCHES] + [(side_yaw, SNAP_PITCH), (-side_yaw, SNAP_PITCH)]
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

    def snapshot(self):
        """Grey frame at the driving pose (head straight, looking down), for floor_motion."""
        cam = self._look(0.0, SNAP_PITCH)
        img = nav.grab(self.cap)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cam

    def floor_motion(self, a, b, tag=None):
        """How far the base really moved between snapshots a and b, from the floor texture.

        Floor points matched in both frames are put on the floor plane with the camera model; a
        point that stayed put on the ground moves by the opposite of the base. Returns
        (dx, dy, dyaw_deg, matches) in the base frame of snapshot a, or None when the floor has
        too little texture. The robot's own body in the image bottom never counts.
        """
        (ga, cam), (gb, _) = a, b
        h, w = ga.shape
        mask = np.zeros_like(ga)
        mask[int(0.30 * h):int(SELF_TOP * h) - 4, :] = 255
        orb = cv2.ORB_create(nfeatures=900, fastThreshold=10)
        ka, da = orb.detectAndCompute(ga, mask)
        kb, db = orb.detectAndCompute(gb, mask)
        if da is None or db is None or len(ka) < 12 or len(kb) < 12:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        src, dst = [], []
        for m in good:
            fa = val.pixel_to_floor(*ka[m.queryIdx].pt, cam)
            fb = val.pixel_to_floor(*kb[m.trainIdx].pt, cam)
            if fa is None or fb is None:
                continue
            if 0.25 <= fa[0] <= 2.0 and 0.25 <= fb[0] <= 2.0:
                src.append((fb[0], fb[1]))
                dst.append((fa[0], fa[1]))
        if len(src) < 10:
            return None
        # floor_a = R floor_b + t, and t is the base motion in frame a.
        mat, inl = cv2.estimateAffinePartial2D(np.array(src), np.array(dst), method=cv2.RANSAC,
                                               ransacReprojThreshold=0.02)
        if mat is None or inl is None or int(inl.sum()) < 10:
            return None
        scale = math.hypot(mat[0, 0], mat[1, 0])
        if not 0.9 <= scale <= 1.1:
            return None
        n = int(inl.sum())
        dyaw = math.degrees(math.atan2(mat[1, 0], mat[0, 0]))
        if tag:
            print(f"  [{tag}] 地面纹理：位移 {mat[0, 2]:+.3f}, {mat[1, 2]:+.3f} m，转 {dyaw:+.1f}°，{n} 个匹配", flush=True)
        return float(mat[0, 2]), float(mat[1, 2]), dyaw, n

    def fence_points(self, tag, d_yaw=0.0):
        """Fence-floor contact points in the base frame (x forward, y left), or None on a failed request.

        Head level first; if the line runs off the image bottom or is not seen, look down like
        vln_nav does when the target's floor contact leaves the image. d_yaw turns the head
        sideways (+ is right, as in vln_nav) without moving the base.
        """
        found = []
        for d_pitch in (0.0, NEAR_PITCH):
            cam = self._look(d_yaw, d_pitch)
            img = nav.grab(self.cap)
            print(f"  [{tag}] 找围栏底边，头 yaw {d_yaw:+.0f}° pitch {d_pitch:+.0f}°", flush=True)
            kb = kickboard_pixels(img)
            if len(kb) >= 12:
                pixels = [(float(u), float(v)) for u, v in kb[::max(1, len(kb) // 8)]]
                print(f"  [{tag}] 用踢脚线下沿当边界", flush=True)
                self._save(img, f"{tag}_p{int(d_pitch):02d}", pixels)
                floor = []
                for u, v in pixels:
                    f = val.pixel_to_floor(u, v, cam)
                    if f is not None and math.hypot(f[0], f[1]) <= nav.MAX_RANGE_M:
                        floor.append((float(f[0]), float(f[1])))
                found.extend(_densify(floor))
                continue
            ans = ask_json(img, FENCE_PROMPT, self.model, self.timeout_s)
            if ans is None:
                return None
            raw = (ans.get("points") if ans.get("found") else None) or []
            if raw and all(isinstance(v, (int, float)) for v in raw):
                # Qwen sometimes flattens the pairs: [x1, y1, x2, y2, ...]
                raw = [raw[i:i + 2] for i in range(0, len(raw) - 1, 2)]
            pixels, low = [], False
            for p in raw or []:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                u, v = _px(p[0], cal.WIDTH), _px(p[1], cal.HEIGHT)
                if v >= cal.HEIGHT - EDGE_PX:
                    low = True
                    continue
                pixels.append((u, v))
            self._save(img, f"{tag}_p{int(d_pitch):02d}", pixels)
            floor = []
            for u, v in pixels:
                f = val.pixel_to_floor(u, v, cam)
                if f is not None and math.hypot(f[0], f[1]) <= nav.MAX_RANGE_M:
                    floor.append((float(f[0]), float(f[1])))
            found.extend(_densify(floor))
            if pixels and not low:
                break
        return found

    def lane(self, d_yaw=0.0, near_only=False):
        """Nearest thing ahead of the camera in the base frame, or None.

        ("clear",) or ("hit", x, y, label, points): (x, y) is the contact under the box center,
        points run along the box bottom so a wide object such as a chair base keeps its width.
        d_yaw turns the head (+ is right, as in vln_nav) to look where the body is about to turn.

        The head looks down in several steps (LANE_PITCHES, far to near). A low object is only
        visible from the steeper angles, and a far wall only from the shallow ones; the hit
        closest to the body wins. The head ends on the last (steepest) angle, where the next
        floor-texture snapshot is taken. near_only looks only at the steepest angle, for checks
        of the floor right beside the body (turns, slides) where far angles add nothing.
        """
        best, best_key, answered = None, None, False
        for pitch in (LANE_PITCHES[-1:] if near_only else LANE_PITCHES):
            hits = self._lane_at(d_yaw, pitch)
            if hits is None:
                continue
            answered = True
            for hit in hits:
                key = _lane_key(hit, d_yaw)
                if best_key is None or key < best_key:
                    best, best_key = hit, key
        if not answered:
            return None
        if best is not None:
            print(f"  最近挡路的是{best[3]}，离车 {math.hypot(best[1], best[2]):.2f} m", flush=True)
        return best if best is not None else ("clear",)

    def _lane_at(self, d_yaw, pitch):
        """Every object the VLM boxes in one look, as ("hit", ...) tuples; [] when none, None on failure."""
        cam = self._look(d_yaw, pitch)
        img = nav.grab(self.cap)
        kb = kickboard_pixels(img)
        ans = ask_json(img, LANE_PROMPT, self.model, self.timeout_s)
        if ans is None and len(kb) < 12:
            return None
        tag = f"p{int(pitch):02d}"
        objs = None if ans is None else ans.get("objects")
        if objs is None and ans.get("bbox"):  # the older single-box answer
            objs = [{"bbox": ans["bbox"], "label": ans.get("label")}]
        hits, boxes, marks = [], [], []
        # The robot's own servos and wires sit in the bottom of the frame; the steeper the head,
        # the higher they climb.
        self_top = min(0.9, SELF_TOP - 0.01 * (pitch - NEAR_PITCH))
        mask = self.self_mask.get((float(d_yaw), float(pitch)))
        for obj in objs or []:
            box = obj.get("bbox") if isinstance(obj, dict) else None
            if not box or len(box) < 4:
                continue
            x1, x2 = sorted((_px(box[0], cal.WIDTH), _px(box[2], cal.WIDTH)))
            y1, y2 = sorted((_px(box[1], cal.HEIGHT), _px(box[3], cal.HEIGHT)))
            label = str(obj.get("label") or "障碍")
            if mask is not None:
                on_self = float(mask[int(y1):int(y2) + 1, int(x1):int(x2) + 1].mean())
                if on_self > SELF_SHARE:
                    print(f"  [{tag}] 框里 {on_self:.0%} 是机器人自己（{label}），不算障碍")
                    continue
            elif y2 >= cal.HEIGHT - EDGE_PX and y1 >= self_top * cal.HEIGHT:
                print(f"  [{tag}] 框到的是机器人自己（{label}），不算障碍")
                continue
            boxes.append((x1, y1, x2 - x1, y2 - y1))
            # A box that runs off the bottom has no visible floor contact. Sample the last row
            # still in frame; never invent a contact at the robot's own center. Where the body
            # covers the box bottom, the contact is the first row above the body.
            low_v = min(y2, cal.HEIGHT - EDGE_PX - 1)

            def contact(u):
                v = low_v if mask is None else _above_self(mask, u, low_v, y1)
                return None if v is None else (v, val.pixel_to_floor(u, v, cam))

            mid = contact((x1 + x2) / 2)
            f = None if mid is None else mid[1]
            marks.append(((x1 + x2) / 2, low_v if mid is None else mid[0]))
            points = []
            for k in range(7):
                c = contact(x1 + (x2 - x1) * k / 6.0)
                g = None if c is None else c[1]
                if g is not None and not _on_body(g[0], g[1]):
                    points.append((float(g[0]), float(g[1])))
            if mask is not None and not points and f is None:
                print(f"  [{tag}] 框的底边全被机器人自己挡住（{label}），不算障碍")
                boxes.pop()
                marks.pop()
                continue
            if f is None or _on_body(f[0], f[1]):
                if not points:
                    print(f"  [{tag}] 框落在机器人身上（{label}），不算障碍")
                    continue
                f = points[len(points) // 2]
            at_side = x1 <= EDGE_PX or x2 >= cal.WIDTH - EDGE_PX
            if y2 >= cal.HEIGHT - EDGE_PX and at_side and all(_on_body(px, py, UPRIGHT_M) for px, py in points):
                # Comes in from a bottom corner and stands right at the body's side: its own upright.
                print(f"  [{tag}] 框到的是机器人两侧的立柱（{label}），不算障碍")
                boxes.pop()
                marks.pop()
                continue
            print(f"  [{tag}] 看到{label}，离车 {math.hypot(f[0], f[1]):.2f} m", flush=True)
            hits.append(("hit", float(f[0]), float(f[1]), label, points))
        if len(kb) >= 12:
            step = max(1, len(kb) // 8)
            floor = []
            for u, v in kb[::step]:
                f = val.pixel_to_floor(u, v, cam)
                if f is not None and not _on_body(f[0], f[1]) and math.hypot(f[0], f[1]) <= nav.MAX_RANGE_M:
                    floor.append((float(f[0]), float(f[1])))
                marks.append((u, v))
            ahead = [p for p in floor if p[0] > 0.05]
            if ahead:
                contact = min(ahead, key=lambda p: abs(p[1]))
                print(f"  [{tag}] 踢脚线下沿，离车 {math.hypot(*contact):.2f} m", flush=True)
                hits.append(("hit", contact[0], contact[1], "墙", ahead))
        self._save(img, f"lane_{'hit' if hits else 'clear'}_{tag}", marks, boxes)
        return hits


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


def _above_self(mask, u, v, top):
    """Lowest row at or above v in column u that is not the body; None if the body fills it up to `top`."""
    col = mask[:, int(min(max(u, 0), mask.shape[1] - 1))]
    v = int(v)
    while v >= top and col[v]:
        v -= 1
    return None if v < top else v


def _lane_key(hit, d_yaw):
    """Sort key: things in the body's path first, nearest first; then the rest by distance.

    The path is the body width along where the head looked (d_yaw + is right).
    """
    a = math.radians(-d_yaw)
    c, s = math.cos(a), math.sin(a)
    _, fx, fy, _, points = hit
    along = [px * c + py * s for px, py in points or [(fx, fy)]
             if abs(-px * s + py * c) <= LANE_HALF_M]
    if along:
        return (0, min(along))
    return (1, math.hypot(fx, fy))


def _on_body(px, py, margin=0.02):
    """True when a floor point lies on the robot itself (0.30 m long, 0.50 m wide), within `margin`."""
    return abs(px) <= 0.15 + margin and abs(py) <= 0.25 + margin


def _densify(points, spacing=0.05):
    """Fill the fence line between neighbouring contact points; a straight chord can only cut a corner inward."""
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        if d > 0.8:
            out.append(b)
            continue
        n = max(1, int(math.ceil(d / spacing)))
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def boundary_from_points(points, center=(0.0, 0.0), bin_deg=5.0, max_gap_deg=60.0):
    """Star-shaped fence polygon around `center` (where the robot turned) from scanned contact points.

    Each bearing keeps its nearest fence point. Unseen bearings take the nearer of their
    neighbours, and a single bearing sticking out past both neighbours is pulled back in,
    so mistakes shrink the area instead of growing it.
    """
    cx, cy = center
    n = int(round(360.0 / bin_deg))
    r = [math.inf] * n
    for px, py in points:
        x, y = px - cx, py - cy
        d = math.hypot(x, y)
        if d < 0.05:
            continue
        i = int((math.degrees(math.atan2(y, x)) % 360.0) // bin_deg) % n
        r[i] = min(r[i], d)
    seen = [i for i in range(n) if math.isfinite(r[i])]
    if len(seen) < n // 3:
        return None, f"只看到 {len(seen) * bin_deg:.0f}° 方向的围栏"
    gap, worst = 0, 0
    for i in range(2 * n):
        gap = 0 if math.isfinite(r[i % n]) else gap + 1
        worst = max(worst, min(gap, n))
    if worst * bin_deg > max_gap_deg:
        return None, f"有 {worst * bin_deg:.0f}° 方向看不到围栏"
    filled = list(r)
    for i in range(n):
        if math.isfinite(filled[i]):
            continue
        left = next(r[(i - k) % n] for k in range(1, n) if math.isfinite(r[(i - k) % n]))
        right = next(r[(i + k) % n] for k in range(1, n) if math.isfinite(r[(i + k) % n]))
        filled[i] = min(left, right)
    clipped = [min(filled[i], max(filled[i - 1], filled[(i + 1) % n])) for i in range(n)]
    poly = []
    for i, d in enumerate(clipped):
        a = math.radians((i + 0.5) * bin_deg)
        poly.append((round(cx + d * math.cos(a), 3), round(cy + d * math.sin(a), 3)))
    return poly, None
