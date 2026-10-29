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

from occmap import ARM_DONE, ARM_ONLY, BLOCKED, BODY_DONE, BODY_REACH, TRASH, UNKNOWN

LIVE_FILE = "live.npz"
PX_M = 0.01  # the page image has one pixel per centimetre of floor
CROP_PAD_M = 0.5   # floor drawn around the known part of the grid
CROP_STEP_M = 1.0  # the drawn box grows in steps this big, so the picture keeps its size between most updates
CROP_RAY_M = 0.5   # known floor narrower than this (single lidar rays through gaps) does not widen the box

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>清扫状态</title>
<style>
  html, body { margin: 0; height: 100%; background: #161616; color: #f2f2f2;
    font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif; }
  main { display: flex; height: 100%; width: 100%; }
  .map-wrap { flex: 1 1 auto; min-width: 0; min-height: 0; position: relative;
    overflow: hidden; background: #111; touch-action: none; cursor: grab; }
  .map-wrap.dragging { cursor: grabbing; }
  #map { position: absolute; left: 0; top: 0; max-width: none; image-rendering: pixelated;
    user-select: none; -webkit-user-drag: none; pointer-events: none; }
  .zoom-ctl { position: absolute; left: 16px; bottom: 16px; display: flex; align-items: center;
    gap: 8px; z-index: 2; }
  .zoom-ctl button { height: 40px; min-width: 40px; padding: 0 12px; font-size: 20px;
    border: 1px solid #555; border-radius: 6px; background: #2a2a2a; color: #f2f2f2; cursor: pointer; }
  .zoom-ctl button:hover { background: #3a3a3a; }
  #zoom-label { min-width: 52px; text-align: center; font-size: 16px; color: #ddd; }
  aside { flex: 0 0 340px; width: 340px; padding: 24px 20px; box-sizing: border-box;
    overflow: auto; }
  h2 { font-size: 18px; margin: 18px 0 6px; color: #ddd; }
  ul.things li { font-size: 15px; margin: 5px 0; color: #ffb27a; }
  ul.picks li { color: #fadc28; }
  h1 { font-size: 28px; margin: 0 0 8px; }
  h1.warm { color: #8ec8f8; }
  h1.explore { color: #f5be78; }
  h1.sweep { color: #7dcea0; }
  #stage { font-size: 18px; color: #d0d0d0; min-height: 24px; margin-bottom: 6px; }
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
  <div class="map-wrap" id="viewport" title="滚轮放大，拖动平移，双击继续放大">
    <img id="map" alt="已清扫区域" draggable="false">
    <div class="zoom-ctl">
      <button type="button" id="zout" title="缩小">−</button>
      <span id="zoom-label">100%</span>
      <button type="button" id="zin" title="放大">+</button>
      <button type="button" id="zreset" title="回到整张图">整图</button>
    </div>
  </div>
  <aside>
    <h1 id="title">预热</h1>
    <div id="status"></div>
    <div id="when"></div>
    <div id="sweep"></div>
    <div class="row"><span class="swatch" style="background:#ebebeb"></span>还没扫的地面</div>
    <div class="row"><span class="swatch" style="background:#a8d4f5"></span>预热走过的路</div>
    <div class="row"><span class="swatch" style="background:#f5be78"></span>探索走过的路</div>
    <div class="row"><span class="swatch" style="background:#3cb371"></span>清扫过的区域</div>
    <div class="row"><span class="swatch" style="background:#c0392b"></span>看到的障碍</div>
    <div class="row"><span class="swatch" style="background:#fadc28"></span>要夹起来的小东西（车绕开）</div>
    <div class="row"><span class="swatch" style="background:#ff8c42"></span>认出的物体（编号见下）</div>
    <div class="row"><span class="swatch" style="background:#50c8c8"></span>当前雷达点</div>
    <div class="row"><span class="swatch" style="background:#2a2a2a;border:1px solid #666"></span>未知区域</div>
    <h2 id="things-title"></h2>
    <ul class="things" id="things"></ul>
    <h2 id="picks-title"></h2>
    <ul class="things picks" id="picks"></ul>
  </aside>
</main>
<script>
const img = document.getElementById("map");
const viewport = document.getElementById("viewport");
const zoomLabel = document.getElementById("zoom-label");
let version = -1;
let zoom = 1;
let panX = 0, panY = 0;
const MIN_Z = 1, MAX_Z = 16;

function fitScale() {
  const r = viewport.getBoundingClientRect();
  if (!img.naturalWidth || r.width < 1 || r.height < 1) return 0;
  return Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
}
function apply() {
  const fit = fitScale();
  if (!fit) return;
  const r = viewport.getBoundingClientRect();
  const dw = img.naturalWidth * fit * zoom;
  const dh = img.naturalHeight * fit * zoom;
  const limitX = Math.max(0, (dw - r.width) / 2);
  const limitY = Math.max(0, (dh - r.height) / 2);
  panX = Math.max(-limitX, Math.min(limitX, panX));
  panY = Math.max(-limitY, Math.min(limitY, panY));
  if (zoom <= 1) { zoom = 1; panX = 0; panY = 0; }
  img.style.width = dw + "px";
  img.style.height = dh + "px";
  img.style.left = ((r.width - dw) / 2 + panX) + "px";
  img.style.top = ((r.height - dh) / 2 + panY) + "px";
  zoomLabel.textContent = Math.round(zoom * 100) + "%";
}
function zoomAt(mx, my, next) {
  const fit = fitScale();
  if (!fit) return;
  next = Math.max(MIN_Z, Math.min(MAX_Z, next));
  const r = viewport.getBoundingClientRect();
  const left = (r.width - img.naturalWidth * fit * zoom) / 2 + panX;
  const top = (r.height - img.naturalHeight * fit * zoom) / 2 + panY;
  const ix = (mx - left) / (fit * zoom);
  const iy = (my - top) / (fit * zoom);
  zoom = next;
  const ndw = img.naturalWidth * fit * zoom;
  const ndh = img.naturalHeight * fit * zoom;
  panX = (mx - ix * fit * zoom) - (r.width - ndw) / 2;
  panY = (my - iy * fit * zoom) - (r.height - ndh) / 2;
  apply();
}
viewport.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = viewport.getBoundingClientRect();
  const step = e.deltaMode === 1 ? 0.08 : 0.0016;
  zoomAt(e.clientX - r.left, e.clientY - r.top, zoom * Math.exp(-e.deltaY * step));
}, {passive: false});
const pts = new Map();
let drag = null;
let pinch = null;
viewport.addEventListener("pointerdown", (e) => {
  if (e.target.closest(".zoom-ctl")) return;
  const r = viewport.getBoundingClientRect();
  pts.set(e.pointerId, {x: e.clientX, y: e.clientY, vx: e.clientX - r.left, vy: e.clientY - r.top});
  viewport.setPointerCapture(e.pointerId);
  if (pts.size === 1) {
    drag = {x: e.clientX, y: e.clientY, panX, panY};
    viewport.classList.add("dragging");
  } else if (pts.size === 2) {
    drag = null;
    viewport.classList.remove("dragging");
    const [a, b] = [...pts.values()];
    const mx = (a.vx + b.vx) / 2, my = (a.vy + b.vy) / 2;
    const fit = fitScale();
    if (!fit) return;
    const left = (r.width - img.naturalWidth * fit * zoom) / 2 + panX;
    const top = (r.height - img.naturalHeight * fit * zoom) / 2 + panY;
    pinch = {
      dist: Math.hypot(a.vx - b.vx, a.vy - b.vy) || 1,
      zoom,
      ix: (mx - left) / (fit * zoom),
      iy: (my - top) / (fit * zoom)
    };
  }
});
viewport.addEventListener("pointermove", (e) => {
  if (!pts.has(e.pointerId)) return;
  const r = viewport.getBoundingClientRect();
  const p = pts.get(e.pointerId);
  p.x = e.clientX; p.y = e.clientY; p.vx = e.clientX - r.left; p.vy = e.clientY - r.top;
  if (pts.size >= 2 && pinch) {
    const fit = fitScale();
    if (!fit) return;
    const [a, b] = [...pts.values()];
    const dist = Math.hypot(a.vx - b.vx, a.vy - b.vy) || 1;
    zoom = Math.max(MIN_Z, Math.min(MAX_Z, pinch.zoom * dist / pinch.dist));
    const ndw = img.naturalWidth * fit * zoom;
    const ndh = img.naturalHeight * fit * zoom;
    panX = ((a.vx + b.vx) / 2 - pinch.ix * fit * zoom) - (r.width - ndw) / 2;
    panY = ((a.vy + b.vy) / 2 - pinch.iy * fit * zoom) - (r.height - ndh) / 2;
    apply();
    return;
  }
  if (!drag) return;
  panX = drag.panX + (e.clientX - drag.x);
  panY = drag.panY + (e.clientY - drag.y);
  apply();
});
function endPointer(e) {
  pts.delete(e.pointerId);
  pinch = null;
  drag = null;
  viewport.classList.remove("dragging");
}
viewport.addEventListener("pointerup", endPointer);
viewport.addEventListener("pointercancel", endPointer);
viewport.addEventListener("dblclick", (e) => {
  if (e.target.closest(".zoom-ctl")) return;
  const r = viewport.getBoundingClientRect();
  zoomAt(e.clientX - r.left, e.clientY - r.top, zoom * 2);
});
document.getElementById("zin").onclick = () => {
  const r = viewport.getBoundingClientRect();
  zoomAt(r.width / 2, r.height / 2, zoom * 1.5);
};
document.getElementById("zout").onclick = () => {
  const r = viewport.getBoundingClientRect();
  zoomAt(r.width / 2, r.height / 2, zoom / 1.5);
};
document.getElementById("zreset").onclick = () => { zoom = 1; panX = 0; panY = 0; apply(); };
img.addEventListener("load", apply);
window.addEventListener("resize", apply);

let tickBusy = false;

async function tick() {
  // Do not start another poll while the previous stats/image request is still running.
  // The map image can require a full render and JPEG encode on the Jetson.
  if (tickBusy) return;
  tickBusy = true;
  try {
    const r = await fetch("/stats", {cache: "no-store"});
    const s = await r.json();
    const text = s.status || "";
    let phase = "";
    if (s.stage === "预热建图" || text.indexOf("预热") === 0) phase = "warmup";
    else if (s.stage === "探索建图" || text.indexOf("探索") === 0) phase = "explore";
    else if (s.stage === "清扫" || text.indexOf("清扫") === 0) phase = "sweep";
    const titles = {warmup: "预热", explore: "探索", sweep: "清扫"};
    const counts = {warmup: "预热走过 ", explore: "探索走过 ", sweep: "已清扫 "};
    const title = document.getElementById("title");
    title.textContent = titles[phase] || "建图";
    title.className = phase === "warmup" ? "warm" : phase === "explore" ? "explore" : phase === "sweep" ? "sweep" : "";
    const st = document.getElementById("status");
    st.textContent = text;
    st.className = s.running ? "" : "stopped";
    document.getElementById("when").textContent = s.when || "";
    document.getElementById("sweep").textContent =
      s.has_map && counts[phase] ? counts[phase] + Math.round((s.sweep || 0) * 100) + "%" : "";
    const rows = s.objects || [];
    for (const [id, pick, title] of [["things", false, "认出的物体（长×宽，走向，高）"], ["picks", true, "要夹起来的小东西"]]) {
      const mine = rows.filter((r) => r.pick === pick);
      document.getElementById(id + "-title").textContent = mine.length ? title : "";
      const list = document.getElementById(id);
      list.replaceChildren(...mine.map((r) => { const li = document.createElement("li"); li.textContent = r.text; return li; }));
    }
    if (s.version !== version) {
      version = s.version;
      await new Promise((resolve) => {
        const done = () => { clearTimeout(timeout); resolve(); };
        const timeout = setTimeout(done, 5000);
        img.addEventListener("load", done, {once: true});
        img.addEventListener("error", done, {once: true});
        img.src = "/frame.jpg?v=" + version;
      });
    }
  } catch (_) {
  } finally {
    tickBusy = false;
  }
}
tick();
setInterval(tick, 500);
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
    [40, 220, 250],     # TRASH (yellow): to be cleaned, not an obstacle
], dtype=np.uint8)


def _plain(value):
    """numpy numbers and tuples as plain JSON values."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


class MapView:
    """State of the running program: the map plus robot pose, path and status."""

    def __init__(self, cmap):
        self.lock = threading.Lock()
        self.robot = None
        self.arm = None
        self.remaining = []
        self.scan_points = []
        self.objects = []
        self.lidar_bounds = []
        self.vision_bounds = []
        self.status = "规划中"
        self.stage = ""
        self.set_map(cmap)

    def set_map(self, cmap):
        with self.lock:
            self.cmap = cmap

    def state(self):
        # The planner can grow the grid while the writer reads it. Retry until the
        # grid, masks and paint all belong to the same map size.
        for _ in range(20):
            try:
                state = self._state_once()
                if state["grid"].shape == state["paint"].shape:
                    return state
            except ValueError as err:
                if "broadcast" not in str(err) and "shape" not in str(err):
                    raise
            time.sleep(0.01)
        raise RuntimeError("map arrays kept changing size during snapshot")

    def _state_once(self):
        with self.lock:
            cmap = self.cmap
            cmap.refresh()
            stats = cmap.stats()
            grid = cmap.grid
            paint = (cmap.paint if hasattr(cmap, "paint") else np.zeros(grid.shape, dtype=np.uint8))
            origin = tuple(cmap.origin)
            if grid.shape != paint.shape:
                raise ValueError("map grid and paint shape changed during snapshot")
            grid_copy, paint_copy = grid.copy(), paint.copy()
            if (cmap.grid is not grid or getattr(cmap, "paint", None) is not paint
                    or tuple(cmap.origin) != origin):
                raise ValueError("map shape or origin changed during snapshot")
            return {
                "grid": grid_copy,
                "paint": paint_copy,
                "origin": [float(origin[0]), float(origin[1])],
                "res": float(cmap.res),
                "boundary": [list(p) for p in cmap.boundary],
                "lidar_bounds": [[[float(x), float(y)] for x, y in arc] for arc in self.lidar_bounds],
                "vision_bounds": [[[float(x), float(y)] for x, y in arc] for arc in self.vision_bounds],
                "furniture": [{k: v for k, v in f.items()} for f in cmap.furniture],
                "body_length": float(cmap.body_length),
                "body_width": float(cmap.body_width),
                "robot": None if self.robot is None else [float(v) for v in self.robot],
                "arm": None if self.arm is None else [float(v) for v in self.arm],
                "remaining": [[float(x), float(y), k, fid] for x, y, k, fid in self.remaining],
                "scan_points": [[float(x), float(y)] for x, y in self.scan_points],
                "objects": [_plain(o) for o in self.objects],
                "status": self.status,
                "stage": {"warmup": "预热建图", "explore": "探索建图", "sweep": "清扫"}.get(self.stage, ""),
                "sweep": stats["sweep"],
                "has_map": bool(cmap.boundary) or bool((grid != UNKNOWN).any()),
            }

    def write(self, map_dir, running=True):
        st = self.state()
        grid = st.pop("grid")
        paint = st.pop("paint")
        st["running"] = running
        st["updated"] = time.time()
        os.makedirs(map_dir, exist_ok=True)
        path = os.path.join(map_dir, LIVE_FILE)
        # The writer thread and the final write may overlap; each needs its own temp file.
        tmp = f"{path}.{threading.get_ident()}.tmp.npz"
        np.savez_compressed(tmp, grid=grid, paint=paint, meta=np.array(json.dumps(st, ensure_ascii=False)))
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
        if "paint" in data.files:
            st["paint"] = data["paint"]
        else:
            st["paint"] = np.zeros(st["grid"].shape, dtype=np.uint8)
    return st


def _outlines(obj):
    """A thing's outlines: its rectangle; files from older runs have "outlines" or one hull in "points"."""
    if obj.get("rect"):
        return [obj["rect"]]
    if obj.get("outlines"):
        return obj["outlines"]
    return [obj.get("points") or [(obj["x"], obj["y"])]]


def object_rows(objects):
    """One line of text per modelled thing for the page beside the map: what it is and its measures.
    Things to pick up come last, with where they lie."""
    rows = []
    for obj in sorted(objects, key=lambda o: (bool(o.get("trash")), o.get("id", 0))):
        if "length" not in obj:
            continue
        num = f"{obj['id']} {obj['label']}"
        if obj.get("trash"):
            size = f"长 {obj['length']:.2f} m" if obj["length"] >= 0.05 else "不到 0.05 m"
            rows.append({"pick": True, "text": f"{num}：{size}，在 ({obj['x']:+.2f}, {obj['y']:+.2f})"})
            continue
        size = f"{obj['length']:.2f} × {obj['width']:.2f} m" if obj["width"] >= 0.08 else f"长 {obj['length']:.2f} m（只量到一面）"
        text = f"{num}：{size}，走向 {obj['heading']:.0f}°"
        if obj.get("height") is not None:
            text += f"，高{'≥' if obj.get('top_cut') else ''} {obj['height']:.2f} m"
            if obj["width"] >= 0.08:
                text += f"，约 {obj['length'] * obj['width'] * obj['height']:.2f} m³"
        rows.append({"pick": False, "text": text})
    return rows


def render(st):
    # The known part of the map, always: the page zooms and pans itself.
    grid = st["grid"]
    paint = st.get("paint")
    if paint is None or paint.shape != grid.shape:
        paint = np.zeros(grid.shape, np.uint8)
    ox, oy = st["origin"]
    h, w = grid.shape
    res = st["res"]
    scale = max(1, int(round(res / PX_M)))  # pixels per map cell
    pad = 12

    def px(x, y):
        return (pad + int(round((x - ox) / res * scale)),
                pad + int(round((h - (y - oy) / res) * scale)))

    # Green is only the cleaning pass. Warm-up and explore trails keep their own colours,
    # including after those passes clear the visited cells used by the planner.
    colored = COLORS[grid]
    if paint.shape == grid.shape:
        colored[(grid == BODY_DONE) & (paint != 3)] = COLORS[BODY_REACH]
        colored[paint == 1] = (245, 212, 168)  # light blue, BGR
        colored[paint == 2] = (120, 190, 245)  # light orange, BGR
        colored[paint == 3] = COLORS[BODY_DONE]
    small = np.flipud(colored)
    big = cv2.resize(small, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((big.shape[0] + 2 * pad, big.shape[1] + 2 * pad, 3), 18, np.uint8)
    canvas[pad:pad + big.shape[0], pad:pad + big.shape[1]] = big

    def poly(points, color, closed, width=1):
        if len(points) < 2:
            return
        cv2.polylines(canvas, [np.array([px(x, y) for x, y in points], np.int32)], closed, color, width, cv2.LINE_8)

    for arc in st.get("lidar_bounds", []):
        poly(arc, (255, 200, 40), False, 1)
    for arc in st.get("vision_bounds", []):
        poly(arc, (200, 80, 220), False, 1)
    poly(st["boundary"], (60, 200, 255), True, 1)
    for x, y in st["scan_points"]:
        cv2.circle(canvas, px(x, y), 2, (200, 200, 80), -1, cv2.LINE_8)
    # Only the outline and the number go on the map; the name and measures are listed beside it.
    for obj in st.get("objects") or []:
        color = (40, 220, 250) if obj.get("trash") else (66, 140, 255)
        for footprint in _outlines(obj):
            poly(footprint, color, True, 1 if obj.get("trash") else 2)
        if "id" in obj:
            u, v = px(obj["x"], obj["y"])
            cv2.putText(canvas, str(obj["id"]), (u - 6, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(obj["id"]), (u - 6, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
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
    # The grid spans 24 m; the room is a few metres of it. Fitting the whole grid to the page would
    # leave the room a small patch in the middle, and so would the long thin rays out through gaps.
    k = max(1, int(round(CROP_RAY_M / res)))
    room = cv2.morphologyEx((grid != UNKNOWN).astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0
    ys, xs = np.nonzero(room | (paint > 0))
    xy = [(ox + (c + 0.5) * res, oy + (r + 0.5) * res) for r, c in ((ys.min(), xs.min()), (ys.max(), xs.max()))]         if len(xs) else []
    xy += [tuple(robot[:2])] if robot is not None else []
    xy += [(p[0], p[1]) for obj in st.get("objects") or [] for line in _outlines(obj) for p in line]
    if not xy:
        return canvas
    x0, x1 = min(p[0] for p in xy), max(p[0] for p in xy)
    y0, y1 = min(p[1] for p in xy), max(p[1] for p in xy)
    x0 = math.floor((x0 - CROP_PAD_M) / CROP_STEP_M) * CROP_STEP_M
    y0 = math.floor((y0 - CROP_PAD_M) / CROP_STEP_M) * CROP_STEP_M
    x1 = math.ceil((x1 + CROP_PAD_M) / CROP_STEP_M) * CROP_STEP_M
    y1 = math.ceil((y1 + CROP_PAD_M) / CROP_STEP_M) * CROP_STEP_M
    (left, top), (right, bottom) = px(x0, y1), px(x1, y0)
    left, top = max(0, left), max(0, top)
    right, bottom = min(canvas.shape[1], right), min(canvas.shape[0], bottom)
    return canvas[top:bottom, left:right]


def blank():
    canvas = np.full((480, 640, 3), 18, np.uint8)
    cv2.putText(canvas, "no map yet - run coverage_run.py", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (200, 200, 200), 2, cv2.LINE_AA)
    return canvas


# Keep the palette aligned with the grid constants. Importing them makes a
# renamed constant fail here instead of painting the wrong colour.
assert COLORS.shape[0] > max(UNKNOWN, BODY_REACH, ARM_ONLY, BODY_DONE, ARM_DONE, BLOCKED, TRASH)
