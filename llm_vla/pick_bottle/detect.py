"""Find a bottle lying on the grey carpet, then its cap, in one head-camera image.

The carpet is a bluish grey (hue ~105), the bottle's tea and label are amber to orange (hue 5-35),
the cap is white.
1. bottle: the largest amber blob; its convex hull covers the label's white and blue print too, and
   the long side of the hull's enclosing rectangle runs along the bottle.
2. cap: the white blob furthest out along that axis beyond one end of the bottle.
For a cylinder the middle of its outline is the image of its axis, so the cap blob's centre
triangulated from both eyes is a point on the cap's axis, not on its top.
"""

import cv2
import numpy as np

# Bottle: amber hue and at least this saturated. The carpet's own hue is ~95-120; where its
# saturation is this low its hue is noise, so the floor is SAT_MIN.
HUE_MAX = 35
SAT_MIN = 18
VAL_MIN = 60
CAP_V_MIN = 170  # cap: brighter than this and less saturated than CAP_S_MAX
CAP_S_MAX = 70
MIN_BOTTLE_PX = 800
# The cap sits beyond the end of the amber hull (further when the neck has no tea in it), between
# these many hull half-lengths from its middle,
CAP_SEARCH = (0.7, 2.2)
# and no further than this many half-lengths off its axis.
CAP_ACROSS = 0.45
MIN_CAP_PX = 80


def _blobs(mask):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(labels == i, stats[i, cv2.CC_STAT_AREA], cents[i]) for i in range(1, n)]


def bottle_centre(img, exclude=None):
    """Middle of the largest amber blob's hull (pixels), cap or not, or None."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = np.ones(h.shape, np.uint8)
    if exclude is not None:
        cv2.fillPoly(valid, [np.array(exclude, np.int32)], 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    amber = ((h <= HUE_MAX) | (h >= 170)) & (s >= SAT_MIN) & (v >= VAL_MIN) & valid.astype(bool)
    body = cv2.morphologyEx(cv2.morphologyEx(amber.astype(np.uint8), cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    blobs = [b for b in _blobs(body) if b[1] >= MIN_BOTTLE_PX]
    if not blobs:
        return None
    m = max(blobs, key=lambda b: b[1])[0]
    (cx, cy), _, _ = cv2.minAreaRect(cv2.convexHull(np.argwhere(m)[:, ::-1].astype(np.int32)))
    return [cx, cy]


def detect(img, exclude=None):
    """dict with bottle centre/axis/half length and cap centre (pixels), or None.

    exclude: polygon (pixel corners) to ignore, e.g. the robot's own body."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = np.ones(h.shape, np.uint8)
    if exclude is not None:
        cv2.fillPoly(valid, [np.array(exclude, np.int32)], 0)
    valid = valid.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    amber = ((h <= HUE_MAX) | (h >= 170)) & (s >= SAT_MIN) & (v >= VAL_MIN)
    body = cv2.morphologyEx(cv2.morphologyEx((amber & valid).astype(np.uint8), cv2.MORPH_OPEN, k),
                            cv2.MORPH_CLOSE, k)
    blobs = [b for b in _blobs(body) if b[1] >= MIN_BOTTLE_PX]
    if not blobs:
        return None
    m, area, c = max(blobs, key=lambda b: b[1])
    # Fill the label's white print: the hull is then the bottle's silhouette.
    hull = cv2.convexHull(np.argwhere(m)[:, ::-1].astype(np.int32))
    (cx, cy), (w, h), ang = cv2.minAreaRect(hull)
    a = np.radians(ang if w >= h else ang + 90)
    axis = np.array([np.cos(a), np.sin(a)])
    centre = np.array([cx, cy])
    half = max(w, h) / 2
    m = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    # White print on the label is not the cap: candidates must lie outside the bottle (a few px grown).
    on_label = cv2.dilate(m, np.ones((7, 7), np.uint8)).astype(bool)
    m = m.astype(bool)

    white = ((v > CAP_V_MIN) & (s < CAP_S_MAX) & valid).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)
    best = None
    for wm, warea, wc in _blobs(white):
        if warea < MIN_CAP_PX or on_label[int(round(wc[1])), int(round(wc[0]))]:
            continue
        d = wc - centre
        along, across = d @ axis, abs(d @ np.array([-axis[1], axis[0]]))
        if half * CAP_SEARCH[0] < abs(along) < half * CAP_SEARCH[1] and across < half * CAP_ACROSS:
            # The cap is the far end: highlights on the clear neck and white bands at the label's edge
            # are nearer the label, and often bigger.
            score = abs(along) / (1 + across / 10)
            if best is None or score > best[0]:
                best = (score, wm, wc, along)
    if best is None:
        return None
    _, wm, wc, along = best
    if along < 0:
        axis = -axis  # axis points from the bottle centre towards the cap
    return {"bottle_centre": centre.tolist(), "axis": axis.tolist(), "half_len": half,
            "cap_centre": np.asarray(wc).tolist(), "cap_area": int(wm.sum()), "bottle_area": int(area),
            "_masks": (m, wm)}


def draw(img, det):
    vis = img.copy()
    if det is None:
        return vis
    m, wm = det["_masks"]
    vis[m] = (0.5 * vis[m] + [0, 80, 160]).astype(np.uint8)
    vis[wm] = (0.5 * vis[wm] + [160, 80, 0]).astype(np.uint8)
    c = np.array(det["bottle_centre"])
    a = np.array(det["axis"])
    p0, p1 = (c - a * det["half_len"]).astype(int), (c + a * det["half_len"]).astype(int)
    cv2.line(vis, tuple(p0), tuple(p1), (0, 200, 255), 1)
    cv2.drawMarker(vis, tuple(np.array(det["cap_centre"]).astype(int)), (255, 0, 0), cv2.MARKER_CROSS, 14, 2)
    return vis
