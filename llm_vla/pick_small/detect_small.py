"""A small flat thing on the floor (a scrap of paper, a wrapper, a cigarette end) in the grasp view.

1. The VLM boxes each small thing lying on the carpet (not the robot, not bottles or furniture).
2. Inside its box GrabCut cuts the thing out (detection first, then the box's own cut: no colour
   thresholds, the user's call, 2026-11-04; a grey chip on the grey carpet and white plastic were
   lost to them). When GrabCut finds nothing that is not most of the box, the box itself, shrunk by
   BOX_SHRINK, stands for the thing (kind "box": its middle is good, its size and direction rough).
   A plastic bag or wrapper is mostly clear film GrabCut does not see: its whole box (kind "bag").
3. Its pixels go onto the floor plane through the head camera model (detect_any.to_plane), at
   THICK_M / 2: the thing is too flat for the stereo. Its rotated rectangle gives the middle, the
   long side's direction, its length and its width in the base frame.
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT / "vln"), str(ROOT / "project_tool"), str(ROOT / "demo")]
from detect_any import to_plane  # noqa: E402
from vln_scene import _numbers, _px, ask_json, cal  # noqa: E402

PROMPT = """你是机器人的视觉模块。图像来自机器人头部相机，朝下看灰色地毯。
找出地上每一个小垃圾：纸屑、纸片、纸团、包装碎片、烟头、小塑料片等能用夹子夹起来的小东西，给出紧贴它的框。
不要框瓶子、家具、电线、地毯本身的花纹和接缝。画面下方和左下方的白色舵机、红白电线、木色板子、蓝色底板和黑色机械臂是机器人自己，不要框。
塑料袋、包装袋的透明部分也是袋子本身：框整个袋子（包括透明的部分），不要只框印了字、有颜色的那一块。
只输出 JSON，不要其他文字：{"objects": [{"bbox": [x1, y1, x2, y2], "label": "名字"}, ...]}；没有就输出 {"objects": []}。
坐标用 0-1000 归一化，图像左上角为 (0,0)，右下角为 (1000,1000)。"""

WRIST_PROMPT = """你是机器人的视觉模块。图像来自机械臂夹爪上的相机，朝下看灰色地毯；画面右侧白色的长条是夹爪自己，不要框。
框出地毯上的每一个小东西（纸屑、纸片、小塑料片、垫圈等），框要紧贴它。
塑料袋、包装袋的透明部分也是袋子本身：框整个袋子（包括透明的部分），不要只框印了字、有颜色的那一块。
只输出 JSON，不要其他文字：{"objects": [{"bbox": [x1, y1, x2, y2], "label": "名字"}, ...]}；没有就输出 {"objects": []}。
坐标用 0-1000 归一化，图像左上角为 (0,0)，右下角为 (1000,1000)。"""

BOX_PAD_PX = 8        # the VLM box may cut a few pixels off the thing
BOX_SHRINK = 0.6      # kind "box": the middle this share of the VLM box is taken as the thing
GRABCUT_ITERS = 4
RING_PX = 25          # GrabCut sees this much carpet round the padded box
MIN_PX = 20
THICK_M = 0.002
SIZE_M = (0.005, 0.12)  # longest side of a thing the jaws can take
# A plastic bag or wrapper is mostly clear film: GrabCut cut out only a snack bag's printed label, and the
# jaws went for the label's edge and closed on nothing, twice (2026-11-07). The user: the clear part is
# the bag, its whole outline is the edge to grasp. For these labels a cut covering less than
# BAG_CUT_SHARE of the VLM's box (or none) gives way to the whole box. Soft, a bag may be longer.
BAG_KEYS = ("袋", "包装", "塑料", "透明")
BAG_CUT_SHARE = 0.5
BAG_SIZE_M = (0.005, 0.25)


def ask(img, model, timeout_s, prompt=PROMPT):
    """[(label, box in pixels)] of the small things the VLM sees."""
    ans = ask_json(img, prompt, model, timeout_s) or {}
    out = []
    for o in ans.get("objects") or []:
        if not isinstance(o, dict):
            continue
        b = next((v for k, v in o.items() if str(k).startswith("bbox")), None)  # bbox, bbox_2d, bbox_
        b = _numbers(b, 4)
        if b is None:
            continue
        out.append((str(o.get("label") or "小东西"),
                    (_px(b[0], cal.WIDTH), _px(b[1], cal.HEIGHT), _px(b[2], cal.WIDTH), _px(b[3], cal.HEIGHT))))
    return out


def segment_grabcut(img, box, exclude):
    """Pixels of a thing GrabCut cuts out of the carpet inside `box` (biggest blob), or None."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(box[0]) - BOX_PAD_PX), max(0, int(box[1]) - BOX_PAD_PX)
    x2, y2 = min(w - 1, int(box[2]) + BOX_PAD_PX), min(h - 1, int(box[3]) + BOX_PAD_PX)
    cx1, cy1 = max(0, x1 - RING_PX), max(0, y1 - RING_PX)
    cx2, cy2 = min(w - 1, x2 + RING_PX), min(h - 1, y2 + RING_PX)
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[y1:y2 + 1, x1:x2 + 1] = cv2.GC_PR_FGD
    if exclude is not None:
        cv2.fillPoly(mask, [np.array(exclude, np.int32)], cv2.GC_BGD)
    crop = np.ascontiguousarray(img[cy1:cy2 + 1, cx1:cx2 + 1])
    m = np.ascontiguousarray(mask[cy1:cy2 + 1, cx1:cx2 + 1])
    if not (m == cv2.GC_PR_FGD).any():
        return None
    try:
        cv2.grabCut(crop, m, None, np.zeros((1, 65)), np.zeros((1, 65)), GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # A cut that keeps most of the padded box is the carpet's stripes, not a thing.
    if stats[k, cv2.CC_STAT_AREA] < MIN_PX or stats[k, cv2.CC_STAT_AREA] > 0.8 * (x2 - x1 + 1) * (y2 - y1 + 1):
        return None
    out = np.zeros((h, w), bool)
    out[cy1:cy2 + 1, cx1:cx2 + 1] = lab_img == k
    return out


def box_mask(shape, box, shrink=BOX_SHRINK):
    """The middle `shrink` of a box as a mask."""
    out = np.zeros(shape[:2], bool)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = max(1.0, (box[2] - box[0]) * shrink / 2), max(1.0, (box[3] - box[1]) * shrink / 2)
    out[int(round(cy - hh)):int(round(cy + hh)) + 1, int(round(cx - hw)):int(round(cx + hw)) + 1] = True
    return out


def measure(mask, cam, size=SIZE_M):
    """Middle (base frame), long-side direction (unit, base frame), length and width (m) of the
    thing's pixels on the floor, or None."""
    vs, us = np.nonzero(mask)
    pts = np.array([to_plane(u, v, cam, THICK_M / 2) for u, v in zip(us, vs)])
    if len(pts) == 0 or any(p is None for p in pts):
        return None
    xy = np.asarray(pts, float)[:, :2]
    (cx, cy), (a, b), ang = cv2.minAreaRect(xy.astype(np.float32))
    t = math.radians(ang if a >= b else ang + 90)
    length, width = max(a, b), min(a, b)
    if not size[0] <= length <= size[1]:
        return None
    return {"mid": np.array([cx, cy, THICK_M / 2]), "axis": np.array([math.cos(t), math.sin(t), 0.0]),
            "length": float(length), "width": float(width),
            "mid_px": [float(us.mean()), float(vs.mean())]}


def locate_all(img, cam, model, timeout_s, exclude=None):
    """(every small thing the VLM boxed and could be measured, plus "label", "box", "px" and "kind"
    (grabcut, bag or box: how it was cut out of its box); the labels that could not, with their boxes; a
    picture of it all)."""
    vis = img.copy()
    robot = np.zeros(img.shape[:2], np.uint8)
    if exclude is not None:
        cv2.polylines(vis, [np.array(exclude, np.int32)], True, (80, 80, 80), 1)
        cv2.fillPoly(robot, [np.array(exclude, np.int32)], 1)
    found, missed = [], []
    for label, box in ask(img, model, timeout_s):
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
        got = None
        bag = any(k in label for k in BAG_KEYS)
        cuts = [("grabcut", lambda: segment_grabcut(img, box, exclude)),
                ("box", lambda: box_mask(img.shape, box) & (robot == 0))]
        if bag:
            cuts.insert(1, ("bag", lambda: box_mask(img.shape, box, 1.0) & (robot == 0)))
        for kind, cut in cuts:
            mask = cut()
            if mask is None or not mask.any():
                continue
            if bag and kind == "grabcut" and mask.sum() < BAG_CUT_SHARE * (box[2] - box[0]) * (box[3] - box[1]):
                continue  # the print on a clear bag, not the bag
            got = measure(mask, cam, (0.003, (BAG_SIZE_M if kind == "bag" else SIZE_M)[1]))
            if got is not None:
                break
        if got is None:
            missed.append((label, [int(v) for v in box]))
            continue
        vis[mask] = (0.4 * vis[mask] + [0, 150, 255]).astype(np.uint8)
        got["label"], got["box"], got["px"], got["kind"] = label, [float(v) for v in box], int(mask.sum()), kind
        found.append(got)
    return found, missed, vis


def locate(img, cam, model, timeout_s, exclude=None):
    """The small thing nearest the body as measure() gives it plus "label" and "vis"; None if none."""
    found, _, vis = locate_all(img, cam, model, timeout_s, exclude)
    if not found:
        return None
    best = min(found, key=lambda g: float(np.linalg.norm(g["mid"][:2])))
    best["vis"] = vis
    return best
