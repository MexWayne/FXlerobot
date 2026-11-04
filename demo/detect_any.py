"""A bottle of any colour lying on the floor, in the grasp view, from the left head eye.

detect.py finds the amber tea bottle by its colour. Any other bottle is found like this:
1. the VLM boxes each bottle lying on the floor and its cap;
2. GrabCut cuts the bottle out of the grey carpet inside its box (the robot's own body, pick.ROBOT_MASK,
   is background). A white cap and a clear neck look like the carpet and are often left out, and the
   bottle's shadow is often taken in, so the cut-out gives only the far end (its pixel farthest from
   the cap) and the width; the cap is the middle of the VLM's cap box, which is tight;
3. pixels are moved onto the plane at the height of what they show through the head camera model:
   on the 11 grasps of 2026-11-02/03 this put the cap within 1 mm of the stereo cap at the same
   height, and 1 cm of height error moves it 3-6 mm. The heights of the cap and the body come from
   the stereo, against the floor right beside the bottle (vln's stereo reads the floor itself ~3 cm
   high). Without enough stereo points the axis is taken one body radius up, the radius from the
   cut-out's width; that is wrong for a flat (crushed) bottle: wide, but its cap 2 cm up, not 4.3
   (2026-11-03: two grasps closed over the cap).
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "vln"), str(ROOT / "project_tool")]
from vln_scene import _numbers, _px, ask_json, cal  # noqa: E402

PROMPT = """你是机器人的视觉模块。图像来自机器人头部相机，朝下看灰色地毯。
找出躺在地上的每一个瓶子，每个瓶子给两个框：整个瓶子（包括瓶盖）的框 "bottle"，和它的瓶盖的框 "cap"。框要紧贴物体。
画面下方和左下方的白色舵机、红白电线、木色板子和黑色机械臂是机器人自己，不要框。
只输出 JSON，不要其他文字：{"bottles": [{"bottle": [x1, y1, x2, y2], "cap": [x1, y1, x2, y2]}, ...]}；没有瓶子输出 {"bottles": []}。
坐标用 0-1000 归一化，图像左上角为 (0,0)，右下角为 (1000,1000)。"""

CROP_PAD_PX = 40       # GrabCut works on the box plus this much carpet round it
BOX_PAD_PX = 6         # the VLM box may cut a few pixels off the bottle
GRABCUT_ITERS = 4
MIN_PX = 300
EDGE_PX = 4            # a box this close to the picture's edge holds a bottle partly out of view
RADIUS_M = (0.012, 0.045)
LENGTH_M = (0.06, 0.40)
CAP_HALF_M = 0.01      # the bottle ends this far beyond the middle of its cap
CAP_ON_AXIS = 0.75     # the cap box lies no further than this many widths off the cut-out's long side
BODY_SLICES = (0.35, 0.85, 12)  # the body's width: median over these slices (share of cap -> end)
FLOOR_RING_PX = 40     # stereo points this far round the bottle's box, off the bottle, are the floor
MIN_STEREO_PTS = 20
HEIGHT_M = (0.008, 0.05)


def to_plane(u, v, cam, z):
    """Base-frame point where the ray through pixel (u, v) is z above the floor, or None."""
    n = cv2.undistortPoints(np.array([[[u, v]]], float), cam["K"], cam["dist"]).reshape(2)
    R, t = cam["T_base_cam"][:3, :3], cam["T_base_cam"][:3, 3]
    d = R @ np.array([n[0], n[1], 1.0])
    if d[2] >= -1e-6:
        return None
    return t + ((z - t[2]) / d[2]) * d


def ask(img, model, timeout_s):
    """[(bottle box, cap box or None)] in pixels, as the VLM sees them."""
    ans = ask_json(img, PROMPT, model, timeout_s) or {}
    out = []
    for b in ans.get("bottles") or []:
        if not isinstance(b, dict):
            continue
        box = _numbers(b.get("bottle"), 4)
        if box is None:
            continue
        cap = _numbers(b.get("cap"), 4)
        to_px = (lambda q: (_px(q[0], cal.WIDTH), _px(q[1], cal.HEIGHT), _px(q[2], cal.WIDTH), _px(q[3], cal.HEIGHT)))
        out.append((to_px(box), to_px(cap) if cap else None))
    return out


def segment(img, box, exclude):
    """Bottle pixels (bool, image size) cut out of the carpet inside `box`, or None."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bx1, by1 = max(0, int(x1) - BOX_PAD_PX), max(0, int(y1) - BOX_PAD_PX)
    bx2, by2 = min(w - 1, int(x2) + BOX_PAD_PX), min(h - 1, int(y2) + BOX_PAD_PX)
    cx1, cy1 = max(0, bx1 - CROP_PAD_PX), max(0, by1 - CROP_PAD_PX)
    cx2, cy2 = min(w - 1, bx2 + CROP_PAD_PX), min(h - 1, by2 + CROP_PAD_PX)
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[by1:by2 + 1, bx1:bx2 + 1] = cv2.GC_PR_FGD
    if exclude is not None:
        cv2.fillPoly(mask, [np.array(exclude, np.int32)], cv2.GC_BGD)
    crop = np.ascontiguousarray(img[cy1:cy2 + 1, cx1:cx2 + 1])
    m = np.ascontiguousarray(mask[cy1:cy2 + 1, cx1:cx2 + 1])
    if not (m == cv2.GC_PR_FGD).any():
        return None
    cv2.grabCut(crop, m, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < MIN_PX:
        return None
    out = np.zeros((h, w), bool)
    out[cy1:cy2 + 1, cx1:cx2 + 1] = lab == k
    return out


def stereo_heights(stereo, mask, box, cap_box):
    """(cap, body) heights above the floor beside the bottle (m) from stereo = (base-frame points on
    the rectified grid, their left-image pixels), or None when the stereo has too few points there."""
    pts, uv = stereo
    h, w = mask.shape
    u, v = uv[..., 0], uv[..., 1]
    ok = np.isfinite(pts).all(axis=-1) & (u >= 0) & (v >= 0) & (u <= w - 1) & (v <= h - 1)
    ui, vi = np.clip(u, 0, w - 1).astype(int), np.clip(v, 0, h - 1).astype(int)
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    x1, y1, x2, y2 = box
    near = (u >= x1 - FLOOR_RING_PX) & (u <= x2 + FLOOR_RING_PX) & (v >= y1 - FLOOR_RING_PX) & (v <= y2 + FLOOR_RING_PX)
    on_body = (np.abs(pts[..., 0]) <= 0.18) & (np.abs(pts[..., 1]) <= 0.28)
    floor = ok & near & ~grown[vi, ui] & ~on_body
    cx1, cy1, cx2, cy2 = cap_box
    cap = ok & (u >= cx1) & (u <= cx2) & (v >= cy1) & (v <= cy2)
    body = ok & mask[vi, ui] & ~cap
    if floor.sum() < MIN_STEREO_PTS or cap.sum() < MIN_STEREO_PTS:
        return None
    z0 = float(np.median(pts[floor][:, 2]))
    zc = float(np.clip(np.median(pts[cap][:, 2]) - z0, *HEIGHT_M))
    zb = float(np.clip(np.median(pts[body][:, 2]) - z0, *HEIGHT_M)) if body.sum() >= MIN_STEREO_PTS else zc
    return zc, zb


def measure(mask, cap_box, cam, heights=None):
    """Cap, middle (base frame, on the axis), axis (cap to bottom), length and radius of a cut-out
    bottle, or None when it does not look like one lying on the floor. heights: (cap, body) above the
    floor when the stereo measured them."""
    pts = np.argwhere(mask)[:, ::-1].astype(np.float64)
    (cx, cy), (w, h), ang = cv2.minAreaRect(cv2.convexHull(pts.astype(np.float32)))
    a = np.radians(ang if w >= h else ang + 90)
    along = np.array([math.cos(a), math.sin(a)])
    width = min(w, h)
    cap_px = np.array([(cap_box[0] + cap_box[2]) / 2, (cap_box[1] + cap_box[3]) / 2])
    off = cap_px - np.array([cx, cy])
    if abs(off @ np.array([-along[1], along[0]])) > CAP_ON_AXIS * width:
        return None  # the cap box is beside the bottle, not at its end
    end_px = pts[int(np.argmax(np.linalg.norm(pts - cap_px, axis=1)))]
    mid_px = (cap_px + end_px) / 2
    # Width across the body, slice by slice: a shadow along one side widens the whole cut-out but
    # few slices, and the neck and cap are left out.
    u = (end_px - cap_px) / np.linalg.norm(end_px - cap_px)
    s, t = (pts - cap_px) @ u / np.linalg.norm(end_px - cap_px), (pts - cap_px) @ np.array([-u[1], u[0]])
    lo, hi, n = BODY_SLICES
    widths = [np.ptp(t[(s >= a) & (s < a + (hi - lo) / n)]) for a in np.linspace(lo, hi, n, endpoint=False)
              if ((s >= a) & (s < a + (hi - lo) / n)).sum() >= 5]
    if widths:
        width = float(np.median(widths))
    # Radius from the width at the middle's range; the height of the axis depends on it, so twice.
    r = 0.03
    for _ in range(2):
        mid = to_plane(*mid_px, cam, r)
        if mid is None:
            return None
        rng = np.linalg.norm(mid - cam["T_base_cam"][:3, 3])
        r = float(np.clip(width * rng / cam["K"][0, 0] / 2, *RADIUS_M))
    z_cap = z_body = r
    if heights is not None:
        z_cap, z_body = heights
        mid = to_plane(*mid_px, cam, z_body)
    cap, bottom = to_plane(*cap_px, cam, z_cap), to_plane(*end_px, cam, z_body)
    if cap is None or bottom is None or mid is None:
        return None
    axis = (bottom - cap)[:2]
    length = float(np.linalg.norm(axis)) + CAP_HALF_M
    if not LENGTH_M[0] <= length <= LENGTH_M[1]:
        return None
    axis = np.r_[axis / np.linalg.norm(axis), 0.0]
    return {"cap": cap, "mid": mid, "axis": axis, "length": length, "radius": r,
            "z_from": "双目" if heights is not None else "瓶身宽度",
            "cap_px": cap_px.tolist(), "end_px": end_px.tolist()}


def locate(img, cam, model, timeout_s, exclude=None, expect=None, stereo=None):
    """The bottle in the picture nearest `expect` (base frame x, y; else nearest the body), as
    measure() gives it plus "cut" (partly out of view) and "vis" (a picture of what was found);
    None when there is none. stereo: (base-frame points, their left pixels) for the heights."""
    vis = img.copy()
    if exclude is not None:
        cv2.polylines(vis, [np.array(exclude, np.int32)], True, (80, 80, 80), 1)
    best = None
    for box, cap_box in ask(img, model, timeout_s):
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
        if cap_box is None:
            continue
        cv2.rectangle(vis, (int(cap_box[0]), int(cap_box[1])), (int(cap_box[2]), int(cap_box[3])), (255, 0, 0), 1)
        mask = segment(img, box, exclude)
        if mask is None:
            continue
        got = measure(mask, cap_box, cam, stereo_heights(stereo, mask, box, cap_box) if stereo is not None else None)
        if got is None:
            continue
        vis[mask] = (0.5 * vis[mask] + [0, 80, 160]).astype(np.uint8)
        cv2.line(vis, tuple(int(v) for v in got["cap_px"]), tuple(int(v) for v in got["end_px"]), (0, 200, 255), 1)
        cv2.drawMarker(vis, tuple(int(v) for v in got["cap_px"]), (255, 0, 0), cv2.MARKER_CROSS, 14, 2)
        h, w = img.shape[:2]
        cut = box[0] <= EDGE_PX or box[1] <= EDGE_PX or box[2] >= w - 1 - EDGE_PX or box[3] >= h - 1 - EDGE_PX
        if exclude is not None and not cut:
            ring = np.zeros((h, w), np.uint8)
            cv2.fillPoly(ring, [np.array(exclude, np.int32)], 1)
            cut = bool((cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) & ring).any())
        got["cut"] = bool(cut)
        ref = np.zeros(2) if expect is None else np.asarray(expect, float)
        d = float(np.linalg.norm(got["mid"][:2] - ref))
        if best is None or d < best[0]:
            best = (d, got)
    if best is None:
        return None
    best[1]["vis"] = vis
    return best[1]
