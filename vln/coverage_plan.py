"""Boustrophedon coverage inside an already enclosed area.

The body sweeps every cell it can stand on. Each chair and table then gets a
dock just outside its footprint and an arm sweep across the underside.
"""

import heapq
import math
from collections import deque

import cv2
import numpy as np


def _runs(cols):
    if len(cols) == 0:
        return []
    breaks = np.where(np.diff(cols) > 1)[0]
    starts = np.r_[cols[0], cols[breaks + 1]]
    ends = np.r_[cols[breaks], cols[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _sweep_rows(mask, step):
    """Rows that actually contain passable cells, spaced about `step` apart.

    A fixed stride can fall entirely between the rows of a narrow gap, which
    drops the path even though the body fits.
    """
    rows = [y for y in range(mask.shape[0]) if mask[y].any()]
    if not rows:
        return []
    chosen = [rows[0]]
    for y in rows[1:]:
        if y - chosen[-1] >= max(step, 1):
            chosen.append(y)
    if rows[-1] != chosen[-1]:
        chosen.append(rows[-1])
    return chosen


def _sweep_segments(mask, step):
    segments = []
    forward = True
    for y in _sweep_rows(mask, step):
        row_runs = _runs(np.where(mask[y])[0])
        if not row_runs:
            continue
        ordered = [(b, a) for a, b in reversed(row_runs)] if not forward else row_runs
        segments.extend((((y, a), (y, b)) for a, b in ordered))
        forward = not forward
    return segments


def _astar(mask, start, goal):
    if start == goal:
        return [start]
    if not mask[start] or not mask[goal]:
        return None
    h, w = mask.shape

    def heur(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    openq = [(heur(start, goal), 0, start)]
    came = {}
    cost = {start: 0}
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        if g != cost.get(cur, -1):
            continue
        y, x = cur
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if ny < 0 or nx < 0 or ny >= h or nx >= w or not mask[ny, nx]:
                continue
            nxt = (ny, nx)
            ng = g + 1
            if ng < cost.get(nxt, 1 << 30):
                cost[nxt] = ng
                came[nxt] = cur
                heapq.heappush(openq, (ng + heur(nxt, goal), ng, nxt))
    return None


def _to_points(cmap, mask, segments):
    points = []
    prev = None
    for start, end in segments:
        if prev is not None and prev != start:
            bridge = _astar(mask, prev, start)
            if bridge is None:
                continue
            points.extend(cmap.cell_center(cell) for cell in bridge[1:])
        points.append(cmap.cell_center(start))
        if end != start:
            points.append(cmap.cell_center(end))
        prev = end
    return points, prev


def _nearest(mask, cmap, x, y):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    dx = cmap.origin[0] + (xs + 0.5) * cmap.res - x
    dy = cmap.origin[1] + (ys + 0.5) * cmap.res - y
    k = int(np.argmin(dx * dx + dy * dy))
    return int(ys[k]), int(xs[k])


def plan_coverage(cmap):
    """Return body and arm polylines in map metres.

    Each action is {"kind": "body"|"arm", "furniture_id": int|None, "points": [(x, y), ...]}.
    """
    step = max(1, int(round(cmap.body_brush * 0.8 / cmap.res)))
    points, prev = _to_points(cmap, cmap.plan_mask, _sweep_segments(cmap.plan_mask, step))
    actions = []
    if points:
        actions.append({"kind": "body", "furniture_id": None, "points": points})
    if prev is None:
        prev = _nearest(cmap.plan_mask, cmap, cmap.start[0], cmap.start[1])

    arm_step = max(1, int(round(cmap.arm_brush * 0.75 / cmap.res)))
    pending = list(cmap.furniture)
    while pending and prev is not None:
        px, py = cmap.cell_center(prev)
        pending.sort(key=lambda item: _centroid_dist(item, px, py))
        item = pending.pop(0)
        mask = cmap.owner == item["id"]
        if not mask.any():
            continue
        cen = _centroid(item["polygon"])
        dock = _nearest(cmap.plan_mask, cmap, cen[0], cen[1])
        if dock is None:
            continue
        bridge = _astar(cmap.plan_mask, prev, dock)
        if bridge:
            actions.append({
                "kind": "body", "furniture_id": None,
                "points": [cmap.cell_center(cell) for cell in bridge],
            })
            prev = dock
        arm_points, _ = _to_points(cmap, mask, _sweep_segments(mask, arm_step))
        if arm_points:
            actions.append({"kind": "arm", "furniture_id": item["id"], "points": arm_points})
    return actions


def _centroid(poly):
    arr = np.asarray(poly, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _centroid_dist(item, x, y):
    cx, cy = _centroid(item["polygon"])
    return math.hypot(cx - x, cy - y)


def nearest_cell(cmap, x, y):
    return _nearest(cmap.plan_mask, cmap, x, y)


def sweep_step(cmap):
    return max(1, int(round(cmap.body_brush * 0.8 / cmap.res)))


def todo_rows(cmap, visited):
    """Sweep rows that still need the body, lowest row first.

    Each item is (key, start_cell, end_cell). A row counts as done once most of
    its cells are swept, or once it has been driven (key in visited).
    """
    from coverage_map import BODY_DONE

    out = []
    for a, b in _sweep_segments(cmap.plan_mask, sweep_step(cmap)):
        c0, c1 = sorted((a[1], b[1]))
        key = (a[0], c0, c1)
        if key in visited:
            continue
        if float(np.mean(cmap.grid[a[0], c0:c1 + 1] == BODY_DONE)) >= 0.8:
            continue
        out.append((key, (a[0], c0), (a[0], c1)))
    out.sort(key=lambda item: item[0])
    return out


def line_clear(cmap, mask, a, b):
    n = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / (cmap.res * 0.5))))
    for k in range(n + 1):
        t = k / n
        cell = cmap.world_to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if cell is None or not mask[cell]:
            return False
    return True


def simplify(cmap, mask, pts, corner_mask=None):
    """Drop A* stair steps: keep only the corners needed to stay on mask.

    With corner_mask, intermediate corners (where the body turns) are picked from its cells when
    any reachable one exists.
    """
    if len(pts) <= 2:
        return list(pts)

    def turnable(k):
        if corner_mask is None or k == len(pts) - 1:
            return True
        cell = cmap.world_to_cell(*pts[k])
        return cell is not None and bool(corner_mask[cell])

    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not line_clear(cmap, mask, pts[i], pts[j]):
            j -= 1
        k = j
        while k > i + 1 and not (turnable(k) and line_clear(cmap, mask, pts[i], pts[k])):
            k -= 1
        if turnable(k) and line_clear(cmap, mask, pts[i], pts[k]):
            j = k
        out.append(pts[j])
        i = j
    return out


def _flood(mask, start):
    """Steps from start over 4-connected mask cells; -1 where unreachable."""
    dist = np.full(mask.shape, -1, dtype=np.int32)
    if start is None or not mask[start]:
        return dist
    h, w = mask.shape
    dist[start] = 0
    queue = deque([start])
    while queue:
        y, x = queue.popleft()
        d = dist[y, x] + 1
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = d
                queue.append((ny, nx))
    return dist


def start_cell(cmap, x, y, reach_m=0.8, min_cells=50):
    """Cell the robot drives from.

    When a freshly marked obstacle covers the robot's own cell, the nearest passable cell can be
    a sliver sealed off by that obstacle; prefer a nearby cell with room to go somewhere.
    """
    mask = cmap.drive_mask
    here = cmap.world_to_cell(x, y)
    if here is not None and mask[here]:
        return here
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    dx = cmap.origin[0] + (xs + 0.5) * cmap.res - x
    dy = cmap.origin[1] + (ys + 0.5) * cmap.res - y
    d2 = dx * dx + dy * dy
    order = np.argsort(d2)
    first = (int(ys[order[0]]), int(xs[order[0]]))
    tried = np.zeros(mask.shape, dtype=bool)
    for k in order:
        if d2[k] > reach_m * reach_m:
            break
        cell = (int(ys[k]), int(xs[k]))
        if tried[cell]:
            continue
        part = _flood(mask, cell) >= 0
        if int(part.sum()) >= min_cells:
            return cell
        tried |= part
    return first


def drive_dist(cmap, x, y):
    """Drivable steps from the robot to every cell (breadth-first flood); -1 where unreachable."""
    return _flood(cmap.drive_mask, start_cell(cmap, x, y))


def turn_spot(cmap, x, y):
    """Nearest cell where the body can turn in place that it can drive to from (x, y)."""
    reach = _flood(cmap.drive_mask, start_cell(cmap, x, y)) >= 0
    cell = _nearest(reach & cmap.plan_mask, cmap, x, y)
    return cell if cell is not None else _nearest(cmap.plan_mask, cmap, x, y)


def _approach(cmap, dist, goal, max_m=1.0, tries=80):
    """Turning spot that sees `goal` along a straight drivable line, nearest by driving distance."""
    ys, xs = np.where(cmap.plan_mask & (dist >= 0))
    if len(xs) == 0:
        return None
    gx, gy = cmap.cell_center(goal)
    d2 = (cmap.origin[0] + (xs + 0.5) * cmap.res - gx) ** 2 + (cmap.origin[1] + (ys + 0.5) * cmap.res - gy) ** 2
    order = [k for k in np.argsort(d2)[:tries] if d2[k] <= max_m * max_m]
    best = None
    for k in order:
        cell = (int(ys[k]), int(xs[k]))
        if line_clear(cmap, cmap.drive_mask, cmap.cell_center(cell), (gx, gy)):
            if best is None or dist[cell] < dist[best]:
                best = cell
    return best


def _no_approach(cmap):
    no_way = getattr(cmap, "_no_approach", None)
    if no_way is None or no_way.shape != cmap.drive_mask.shape:
        no_way = cmap._no_approach = np.zeros(cmap.drive_mask.shape, dtype=bool)
    return no_way


def ban_goal(cmap, cell, radius_m):
    """Stop cleanup_goal from picking cells around `cell` again."""
    rad = radius_m / cmap.res
    r = int(math.ceil(rad))
    y0, y1 = max(0, cell[0] - r), min(cmap.h, cell[0] + r + 1)
    x0, x1 = max(0, cell[1] - r), min(cmap.w, cell[1] + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    _no_approach(cmap)[y0:y1, x0:x1] |= (yy - cell[0]) ** 2 + (xx - cell[1]) ** 2 <= rad * rad


def cleanup_goal(cmap, x, y, min_gain=4):
    """Next place to fill in unswept floor, nearest by driving distance.

    Returns (goal, approach, reachable_left, unreachable_left); left counts are floor cells.
    approach is None when the body can turn at goal; otherwise goal is a spot the body may
    only drive into straight from approach (a turning spot) and back out in reverse.
    """
    from coverage_map import BODY_REACH

    left = cmap.sweepable() & (cmap.grid == BODY_REACH)
    rad = max(1, int(math.ceil(cmap.body_brush * 0.5 / cmap.res)))
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    kernel = ((yy * yy + xx * xx) <= rad * rad).astype(np.float32)
    gain = cv2.filter2D(left.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    dist = _flood(cmap.drive_mask, start_cell(cmap, x, y))
    reach = dist >= 0
    reached = cv2.filter2D(reach.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT) > 0
    reachable_left = int((left & reached).sum())
    unreachable_left = int((left & ~reached).sum())
    cand = reach & (gain >= min_gain)
    no_way = _no_approach(cmap)
    cand &= ~no_way
    full = float(kernel.sum())
    score = np.where(cand, dist * cmap.res + 0.6 * (1.0 - gain / full), np.inf)
    checks = 0
    for k in np.argsort(score, axis=None):
        if not np.isfinite(score.flat[k]) or checks >= 200:
            break
        cell = tuple(int(v) for v in np.unravel_index(int(k), score.shape))
        if cmap.plan_mask[cell]:
            return cell, None, reachable_left, unreachable_left
        checks += 1
        via = _approach(cmap, dist, cell)
        if via is not None:
            return cell, via, reachable_left, unreachable_left
        no_way[cell] = True
    return None, None, reachable_left, unreachable_left


def route_to(cmap, x, y, goal_cell):
    """Corner points from (x, y) to goal_cell over passable cells, or None."""
    here = start_cell(cmap, x, y)
    if here is None:
        return None
    path = _astar(cmap.drive_mask, here, goal_cell)
    if path is None:
        return None
    pts = simplify(cmap, cmap.drive_mask, [cmap.cell_center(c) for c in path], cmap.plan_mask)
    return [(x, y)] + pts


def body_path(cmap, x, y):
    """Body sweep from the robot's current position, staying on plan_mask.

    The connection onto the sweep and any return to the skipped prefix are
    searched on the passable cells, so a newly marked obstacle is not crossed.
    """
    points = []
    for act in plan_coverage(cmap):
        if act["kind"] != "body":
            break
        points.extend(act["points"])
    if len(points) < 2:
        return points
    nearest = min(range(len(points)), key=lambda k: math.hypot(points[k][0] - x, points[k][1] - y))
    here = cmap.world_to_cell(x, y)
    if here is None or not cmap.plan_mask[here]:
        here = _nearest(cmap.plan_mask, cmap, x, y)
    goal = cmap.world_to_cell(*points[nearest])
    if goal is None or not cmap.plan_mask[goal]:
        goal = _nearest(cmap.plan_mask, cmap, points[nearest][0], points[nearest][1])
    if here is None or goal is None:
        return []
    bridge = _astar(cmap.plan_mask, here, goal)
    if bridge is None:
        return []
    rest = [cmap.cell_center(cell) for cell in bridge] + points[nearest:]
    if nearest > 0:
        end = cmap.world_to_cell(*rest[-1])
        begin = cmap.world_to_cell(*points[0])
        if end is not None and begin is not None and not cmap.plan_mask[begin]:
            begin = _nearest(cmap.plan_mask, cmap, points[0][0], points[0][1])
        link = _astar(cmap.plan_mask, end, begin) if end is not None and begin is not None else None
        if link is not None:
            rest.extend(cmap.cell_center(cell) for cell in link[1:])
            rest.extend(points[:nearest])
    return rest


def path_length(actions):
    total = 0.0
    for act in actions:
        pts = act["points"]
        for a, b in zip(pts, pts[1:]):
            total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total
