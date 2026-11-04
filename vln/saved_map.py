"""A map kept from one run to the next, and finding the body on it again.

save() writes the whole map at the end of a run: the lidar counts, the camera layers and the
things the cameras named, to map/saved/. load() reads it back. from_live() rebuilds an older map
from the page file (map/live.npz) alone: what was drawn blocked blocks again (lidar and camera
marks are not told apart there), the things come back with their outlines as their points.

relocate() turns the body once in place, slowly, putting the lidar turns together into one round
scan (the lidar sees only +-100 deg at a time), and searches the whole map for where that scan fits
(every free 0.1 m, every 3 deg), then refines the best fits with OccMap.match. While it turns the
body is off the map, so nothing is stamped where it is not.
"""

import json
import math
import os
import time

import numpy as np

import coverage_run as cr
from live_map import _plain
from occmap import BLOCKED, BODY_DONE, HIT_MIN, TRASH, UNKNOWN, OccMap

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(HERE, "map", "saved")
LAYERS = ("hit", "miss", "seen", "glass", "trash", "trash_low", "objects", "stood", "swept", "paint")
OBJECT_KEYS = ("id", "label", "names", "points", "sizes", "parts", "seen", "height", "top_cut", "looks", "heights", "cuts")
OFF_MAP = 1000.0          # where the body is while it is not yet on the map (no stamps land)
RELOC_DPS = 12.0          # deg/s of the locating turn
RELOC_SAMPLE_S = 0.25
RELOC_POINTS = 400        # scan points the search uses
RELOC_STEP_M = 0.10
RELOC_STEP_DEG = 3.0
RELOC_KEEP = 12           # best search fits refined
RELOC_APART_M = 0.30      # fits this far apart (or RELOC_APART_DEG turned) are different places
RELOC_APART_DEG = 15.0
RELOC_MARGIN_M = 0.005    # the best fit beats every other place by at least this much


def save(cmap, mover, folder=SAVED_DIR):
    """Write the map and the modelled things. Camera marks with an end time (people) are left out."""
    os.makedirs(folder, exist_ok=True)
    arrays = {k: getattr(cmap, k) for k in LAYERS}
    arrays["vis_forever"] = np.isinf(cmap.vis_end)
    tmp = os.path.join(folder, "map.tmp.npz")
    np.savez_compressed(tmp, origin=np.array(cmap.origin), res=np.array(cmap.res), **arrays)
    os.replace(tmp, os.path.join(folder, "map.npz"))
    things = {"next_id": mover.next_id, "glass": _plain(mover.glass), "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
              "objects": [_plain({k: o[k] for k in OBJECT_KEYS if k in o}) for o in mover.objects]}
    with open(os.path.join(folder, "things.json"), "w") as f:
        json.dump(things, f, ensure_ascii=False)
    return folder


def _cmap_like(origin, res, shape):
    size = shape[0] * res
    cmap = OccMap(size_m=size, res=res, center=(origin[0] + size / 2, origin[1] + size / 2))
    assert (cmap.h, cmap.w) == shape, "saved map has another size"
    return cmap


def _things(mover, cmap, objects, glass, next_id):
    for o in objects:
        o["points"] = [tuple(p) for p in o["points"]]
        o["parts"] = [[tuple(p) for p in band] for band in o.get("parts", [])]
        if "looks" in o:
            o["looks"] = [[tuple(p) for p in pts] for pts in o["looks"]]
        o.setdefault("sizes", [])
        cr._settle(o, cmap)
    mover.objects = objects
    mover.glass = [[tuple(p) for p in line] for line in glass]
    mover.next_id = max([next_id] + [o["id"] for o in objects])
    cr.block_objects(mover)
    cr.show_objects(mover)


def load(mover, folder=SAVED_DIR):
    """The map saved by save() into mover.cmap (the same grid) and mover's things. Returns when it was saved."""
    with np.load(os.path.join(folder, "map.npz")) as z:
        cmap = mover.cmap
        if tuple(z["origin"]) != tuple(cmap.origin) or float(z["res"]) != cmap.res or z["hit"].shape != (cmap.h, cmap.w):
            raise SystemExit(f"{folder} 的地图大小和现在的不一样")
        for k in LAYERS:
            getattr(cmap, k)[:] = z[k]
        cmap.vis_end[:] = np.where(z["vis_forever"], math.inf, 0.0)
    with open(os.path.join(folder, "things.json")) as f:
        things = json.load(f)
    cmap.rebuild()
    _things(mover, cmap, things["objects"], things["glass"], things["next_id"])
    return things["saved"]


def from_live(mover, path):
    """An older map from a page file (map/live.npz or a copy): blocked cells block (lidar and camera
    alike, kept by both), trash is trash, swept is swept; things have their outlines as points."""
    with np.load(path) as z:
        grid, paint = z["grid"], z["paint"]
        st = json.loads(str(z["meta"]))
    cmap = mover.cmap
    if tuple(st["origin"]) != tuple(cmap.origin) or st["res"] != cmap.res or grid.shape != (cmap.h, cmap.w):
        raise SystemExit(f"{path} 的地图大小和现在的不一样")
    blocked = grid == BLOCKED
    cmap.seen[:] = grid != UNKNOWN
    cmap.hit[blocked] = HIT_MIN        # the lidar map the turn is matched against
    cmap.vis_end[blocked] = math.inf   # and what the cameras had marked: stereo-seen floor clears it
    cmap.swept[:] = grid == BODY_DONE
    cmap.paint[:] = paint
    cmap.stood[:] = paint > 0
    objects = []
    for o in st["objects"]:
        pts = [tuple(p) for line in o.get("outlines") or [] for p in line] or [(o["x"], o["y"])]
        objects.append({"id": o["id"], "label": o["label"], "names": {o["label"]: o.get("seen", 1)},
                        "points": pts, "sizes": [o.get("size", o["length"])], "parts": [],
                        "seen": o.get("seen", 1), "height": o.get("height"), "top_cut": o.get("top_cut", False)})
        if o.get("trash"):
            mask = cmap.mask_of(pts, 0.03) & (grid == TRASH)
            (cmap.trash_low if o.get("passable") else cmap.trash)[mask] = True
    cmap.rebuild()
    _things(mover, cmap, objects, [], 0)
    return st.get("updated")


def _round_scan(mover, status):
    """Turn once in place at RELOC_DPS; the lidar turns put together in the frame the body started
    in. Returns (N x 2 points, degrees turned)."""
    m = mover
    prev = m.imu_yaw()
    turned, pts, next_sample = 0.0, [], 0.0
    t0 = time.time()
    try:
        while True:
            m._need_imu()
            yaw = m.imu_yaw()
            turned += cr.wrap_deg(math.degrees(yaw - prev))
            prev = yaw
            if time.time() >= next_sample:
                xy = m.lidar.latest_xy()
                if xy is not None and len(xy):
                    a = math.radians(turned)
                    c, s = math.cos(a), math.sin(a)
                    pts.append(np.c_[c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])
                next_sample = time.time() + RELOC_SAMPLE_S
                m.show(status)
            err = 360.0 - turned
            if abs(err) <= cr.TURN_TOL_DEG or time.time() - t0 > 360.0 / RELOC_DPS * 1.5 + 10.0:
                break
            cmd = max(-RELOC_DPS, min(RELOC_DPS, cr.TURN_KP * err))
            if abs(cmd) < cr.TURN_MIN_RATE:
                cmd = math.copysign(cr.TURN_MIN_RATE, err)
            m.base.base.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": cmd})
            time.sleep(0.02)
    finally:
        m.base.stop()
    time.sleep(cr.TURN_SETTLE_S)
    yaw = m.imu_yaw()
    turned += cr.wrap_deg(math.degrees(yaw - prev))
    if not pts:
        raise SystemExit("转了一圈，雷达没有数据，定不了位")
    return np.vstack(pts), turned


def _search(cmap, xy):
    """Best poses (cost, x, y, yaw) of the start frame for scan points xy over the whole map."""
    field = cmap.field()
    free = cmap.seen & ~cmap.occupied()
    step = max(1, int(round(RELOC_STEP_M / cmap.res)))
    ys, xs = np.nonzero(free[::step, ::step])
    px = cmap.origin[0] + (xs * step + 0.5) * cmap.res
    py = cmap.origin[1] + (ys * step + 0.5) * cmap.res
    best = []
    for deg in np.arange(0.0, 360.0, RELOC_STEP_DEG):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        rx, ry = c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]
        cost = np.empty(len(px), np.float32)
        for i in range(0, len(px), 2000):
            cx = np.floor((px[i:i + 2000, None] + rx[None] - cmap.origin[0]) / cmap.res).astype(np.int64)
            cy = np.floor((py[i:i + 2000, None] + ry[None] - cmap.origin[1]) / cmap.res).astype(np.int64)
            ok = (cx >= 0) & (cy >= 0) & (cx < cmap.w) & (cy < cmap.h)
            d = np.full(cx.shape, np.float32(field.max()), np.float32)
            d[ok] = field[cy[ok], cx[ok]]
            cost[i:i + 2000] = d.mean(axis=1)
        for k in np.argsort(cost)[:RELOC_KEEP]:
            best.append((float(cost[k]), float(px[k]), float(py[k]), a))
    best.sort()
    return best[:RELOC_KEEP * 4]


def relocate(mover, status="在已有地图上定位：原地转一圈"):
    """Find the body on the map it was given (load or from_live) by one turn in place. Sets its pose."""
    m, cmap = mover, mover.cmap
    m.x = m.y = OFF_MAP  # nothing is stamped while it turns
    xy, turned = _round_scan(m, status)
    cells = np.unique(np.floor(xy / cmap.res).astype(np.int64), axis=0, return_index=True)[1]
    xy = xy[cells]
    if len(xy) > RELOC_POINTS:
        xy = xy[np.random.default_rng(0).choice(len(xy), RELOC_POINTS, replace=False)]
    fits = []
    for cost, x, y, a in _search(cmap, xy):
        got = cmap.match(xy, (x, y, a), 0.15, 6.0)
        if got is not None:
            fits.append((got[1], got[0]))
    if not fits:
        raise SystemExit("在地图上找不到雷达这一圈对得上的地方")
    fits.sort(key=lambda f: f[0])
    fit, (x, y, a) = fits[0]
    others = [f for f, p in fits[1:] if math.hypot(p[0] - x, p[1] - y) > RELOC_APART_M
              or abs(cr.wrap_deg(math.degrees(p[2] - a))) > RELOC_APART_DEG]
    second = min(others) if others else None
    print(f"  定位：转了 {turned:.0f}°，地图上对得最好的是 ({x:+.2f}, {y:+.2f})，朝向 {math.degrees(a):+.0f}°，"
          f"平均差 {fit * 100:.1f} cm" + (f"；别处最好 {second * 100:.1f} cm" if second is not None else ""), flush=True)
    if fit > cr.MATCH_OK_M:
        raise SystemExit(f"在地图上定不了位：最好的地方也差 {fit * 100:.1f} cm（地图可能变了，重新建图）")
    if second is not None and second - fit < RELOC_MARGIN_M:
        raise SystemExit("在地图上定不了位：有两个地方对得一样好，挪一下车再试")
    yaw = a + math.radians(turned)
    m.x, m.y, m.yaw = x, y, yaw
    m.yaw0 = m.imu.yaw_deg() - math.degrees(yaw)
    m.hold = None
    m.locate(cr.MATCH_WIN_M)
    m.show("定位好了")
    return m.pose
