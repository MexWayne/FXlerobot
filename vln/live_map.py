"""Top-down coverage map.

coverage_run.py keeps the state in a MapView and writes it to map/live.npz a few
times a second. map_server.py reads that file and draws it, so the page keeps
showing the last result after coverage_run.py has stopped.
"""

import json
import math
import os
import threading
import time

import cv2
import numpy as np

from coverage_map import ARM_DONE, ARM_ONLY, BLOCKED, BODY_DONE, BODY_REACH, UNKNOWN

LIVE_FILE = "live.npz"

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>清扫状态</title>
<style>
  html, body { margin: 0; height: 100%; background: #161616; color: #f2f2f2;
    font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif; }
  main { display: flex; height: 100%; }
  img { height: 100%; width: auto; max-width: calc(100% - 280px); object-fit: contain;
    background: #111; image-rendering: pixelated; }
  aside { width: 280px; padding: 24px 20px; box-sizing: border-box; }
  h1 { font-size: 28px; margin: 0 0 8px; }
  #status { font-size: 20px; color: #9ad07a; min-height: 28px; }
  #status.stopped { color: #f5b041; }
  #when { font-size: 15px; color: #aaa; margin: 4px 0 12px; }
  #sweep { font-size: 22px; color: #7dcea0; margin: 0 0 8px; min-height: 30px; }
  .row { display: flex; align-items: center; gap: 10px; margin: 10px 0; font-size: 18px; }
  .swatch { width: 22px; height: 22px; border-radius: 3px; }
  ul { list-style: none; padding: 0; margin: 18px 0 0; }
  li { font-size: 18px; margin: 8px 0; }
  .done { color: #7dcea0; }
  .todo { color: #f7dc6f; }
</style>
</head>
<body>
<main>
  <img id="map" alt="已清扫区域">
  <aside>
    <h1>清扫</h1>
    <div id="status"></div>
    <div id="when"></div>
    <div id="sweep"></div>
    <div class="row"><span class="swatch" style="background:#ebebeb"></span>还没扫的地面</div>
    <div class="row"><span class="swatch" style="background:#3cb371"></span>已清扫</div>
    <div class="row"><span class="swatch" style="background:#c0392b"></span>看到的障碍</div>
    <div class="row"><span class="swatch" style="background:#ffc83c"></span>围栏（每次运行重新测）</div>
  </aside>
</main>
<script>
const img = document.getElementById("map");
let version = -1;
function tick() {
  fetch("/stats").then(r => r.json()).then(s => {
    const st = document.getElementById("status");
    st.textContent = s.status || "";
    st.className = s.running ? "" : "stopped";
    document.getElementById("when").textContent = s.when || "";
    document.getElementById("sweep").textContent =
      s.has_map ? "已清扫 " + Math.round((s.sweep || 0) * 100) + "%" : "";
    if (s.version !== version) {
      version = s.version;
      img.src = "/frame.jpg?v=" + version;
    }
  }).catch(() => {});
}
tick();
setInterval(tick, 300);
</script>
</body>
</html>
"""

COLORS = np.array([
    [42, 42, 42],       # UNKNOWN
    [235, 235, 235],    # BODY_REACH
    [180, 220, 246],    # ARM_ONLY  (BGR yellow)
    [90, 170, 70],      # BODY_DONE
    [210, 140, 50],     # ARM_DONE
    [40, 40, 200],      # BLOCKED
], dtype=np.uint8)


class MapView:
    """State of the running program: the map plus robot pose, path and status."""

    def __init__(self, cmap):
        self.lock = threading.Lock()
        self.robot = None
        self.arm = None
        self.remaining = []
        self.scan_points = []
        self.status = "规划中"
        self.set_map(cmap)

    def set_map(self, cmap):
        with self.lock:
            self.cmap = cmap

    def state(self):
        with self.lock:
            cmap = self.cmap
            cmap.refresh()
            stats = cmap.stats()
            return {
                "grid": cmap.grid.copy(),
                "origin": [float(cmap.origin[0]), float(cmap.origin[1])],
                "res": float(cmap.res),
                "boundary": [list(p) for p in cmap.boundary],
                "furniture": [{k: v for k, v in f.items()} for f in cmap.furniture],
                "body_length": float(cmap.body_length),
                "body_width": float(cmap.body_width),
                "robot": None if self.robot is None else [float(v) for v in self.robot],
                "arm": None if self.arm is None else [float(v) for v in self.arm],
                "remaining": [[float(x), float(y), k, fid] for x, y, k, fid in self.remaining],
                "scan_points": [[float(x), float(y)] for x, y in self.scan_points],
                "status": self.status,
                "sweep": stats["sweep"],
                "has_map": bool(cmap.boundary),
            }

    def write(self, map_dir, running=True):
        st = self.state()
        grid = st.pop("grid")
        st["running"] = running
        st["updated"] = time.time()
        os.makedirs(map_dir, exist_ok=True)
        path = os.path.join(map_dir, LIVE_FILE)
        # The writer thread and the final write may overlap; each needs its own temp file.
        tmp = f"{path}.{threading.get_ident()}.tmp.npz"
        np.savez_compressed(tmp, grid=grid, meta=np.array(json.dumps(st, ensure_ascii=False)))
        os.replace(tmp, path)

    def start_writer(self, map_dir, period=0.3):
        stop = threading.Event()

        def loop():
            while not stop.wait(period):
                try:
                    self.write(map_dir)
                except Exception as e:  # a missed frame must not stop the robot
                    print(f"  地图写盘失败：{e}")

        threading.Thread(target=loop, name="map-writer", daemon=True).start()
        return stop


def read_state(map_dir):
    path = os.path.join(map_dir, LIVE_FILE)
    with np.load(path) as data:
        st = json.loads(str(data["meta"]))
        st["grid"] = data["grid"]
    return st


def _crop(st, border=0.30):
    """Rows/cols around the fence, scan points and robot, so the area fills the picture."""
    grid = st["grid"]
    h, w = grid.shape
    res = st["res"]
    ox, oy = st["origin"]
    pts = [tuple(p) for p in st["boundary"]] + [tuple(p) for p in st["scan_points"]]
    if st["robot"] is not None:
        pts.append((st["robot"][0], st["robot"][1]))
    ys, xs = np.where(grid == BODY_DONE)
    if len(xs) and not st["boundary"]:
        pts.append((ox + (float(xs.min()) + 0.5) * res, oy + (float(ys.min()) + 0.5) * res))
        pts.append((ox + (float(xs.max()) + 0.5) * res, oy + (float(ys.max()) + 0.5) * res))
    if not pts:
        return grid, ox, oy
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    c0 = max(0, int(math.floor((min(xs) - border - ox) / res)))
    c1 = min(w, int(math.ceil((max(xs) + border - ox) / res)))
    r0 = max(0, int(math.floor((min(ys) - border - oy) / res)))
    r1 = min(h, int(math.ceil((max(ys) + border - oy) / res)))
    if c1 - c0 < 2 or r1 - r0 < 2:
        return grid, ox, oy
    return grid[r0:r1, c0:c1], ox + c0 * res, oy + r0 * res


def render(st):
    grid, ox, oy = _crop(st)
    h, w = grid.shape
    res = st["res"]
    scale = max(6, min(24, 900 // max(w, h, 1)))
    pad = 12

    def px(x, y):
        return (pad + int(round((x - ox) / res * scale)),
                pad + int(round((h - (y - oy) / res) * scale)))

    small = np.flipud(COLORS[grid])
    big = cv2.resize(small, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((big.shape[0] + 2 * pad, big.shape[1] + 2 * pad, 3), 18, np.uint8)
    canvas[pad:pad + big.shape[0], pad:pad + big.shape[1]] = big

    def poly(points, color, closed, width=2):
        if len(points) < 2:
            return
        cv2.polylines(canvas, [np.array([px(x, y) for x, y in points], np.int32)], closed, color, width, cv2.LINE_AA)

    poly(st["boundary"], (60, 200, 255), True, 3)
    for x, y in st["scan_points"]:
        cv2.circle(canvas, px(x, y), 2, (200, 200, 80), -1, cv2.LINE_AA)
    for item in st["furniture"]:
        color = (120, 200, 120) if item["done"] else (80, 180, 255)
        poly(item["polygon"], color, True)
        cx = sum(p[0] for p in item["polygon"]) / len(item["polygon"])
        cy = sum(p[1] for p in item["polygon"]) / len(item["polygon"])
        org = px(cx, cy)
        cv2.putText(canvas, f"{item['cls'][0].upper()}{item['id']}", (org[0] - 12, org[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    for kind, color in (("body", (80, 200, 80)), ("arm", (220, 160, 40))):
        pts = [px(x, y) for x, y, k, _fid in st["remaining"] if k == kind]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA)
    robot, arm = st["robot"], st["arm"]
    if robot is not None and arm is not None:
        cv2.line(canvas, px(robot[0], robot[1]), px(*arm), (220, 160, 40), 2)
    if arm is not None:
        cv2.circle(canvas, px(*arm), 5, (255, 200, 80), -1)
    if robot is not None:
        x, y, yaw = robot
        hl, hw = st["body_length"] / 2.0, st["body_width"] / 2.0
        c, s = math.cos(yaw), math.sin(yaw)

        def bp(lx, ly):
            return px(x + c * lx - s * ly, y + s * lx + c * ly)

        color = (40, 140, 255) if st.get("running") else (150, 150, 150)
        cv2.polylines(canvas, [np.array([bp(hl, hw), bp(hl, -hw), bp(-hl, -hw), bp(-hl, hw)], np.int32)],
                      True, color, 2, cv2.LINE_AA)
        cv2.fillConvexPoly(canvas, np.array([bp(hl, 0.0), bp(-hl + 0.04, 0.07), bp(-hl + 0.04, -0.07)], np.int32),
                           color)
    return canvas


def blank():
    canvas = np.full((480, 640, 3), 18, np.uint8)
    cv2.putText(canvas, "no map yet - run coverage_run.py", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (200, 200, 200), 2, cv2.LINE_AA)
    return canvas


# Keep the palette aligned with the grid constants. Importing them makes a
# renamed constant fail here instead of painting the wrong colour.
assert COLORS.shape[0] > max(UNKNOWN, BODY_REACH, ARM_ONLY, BODY_DONE, ARM_DONE, BLOCKED)
