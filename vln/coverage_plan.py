"""Where to go next on the occupancy map, and the route there.

explore_goal picks goals depth first: it keeps going the way the robot faces while there is
something to do close by, stacks the ways not taken, and backtracks to the newest open fork at a
dead end. What there is to do is a mask of target cells: the frontier while exploring, the
unswept floor while cleaning.
"""

import heapq
import math
from collections import deque

import cv2
import numpy as np


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
    """Drivable cell the robot drives from: its own, or a nearby one with room to go somewhere.

    A freshly marked obstacle can cover the robot's own cell; the nearest free cell may then be a
    sliver sealed off by that obstacle.
    """
    mask = cmap.drive_mask
    here = cmap.world_to_cell(x, y)
    if here is not None and mask[here]:
        return here
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    d2 = (cmap.origin[0] + (xs + 0.5) * cmap.res - x) ** 2 + (cmap.origin[1] + (ys + 0.5) * cmap.res - y) ** 2
    order = np.argsort(d2)
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
    return int(ys[order[0]]), int(xs[order[0]])


def drive_dist(cmap, x, y):
    return _flood(cmap.drive_mask, start_cell(cmap, x, y))


def _astar(mask, start, goal):
    if start == goal:
        return [start]
    if not mask[start] or not mask[goal]:
        return None
    h, w = mask.shape
    openq = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0, start)]
    came, cost = {}, {start: 0}
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g != cost.get(cur, -1):
            continue
        y, x = cur
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and g + 1 < cost.get((ny, nx), 1 << 30):
                cost[(ny, nx)] = g + 1
                came[(ny, nx)] = cur
                heapq.heappush(openq, (g + 1 + abs(ny - goal[0]) + abs(nx - goal[1]), g + 1, (ny, nx)))
    return None


def line_clear(cmap, mask, a, b):
    n = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / (cmap.res * 0.5))))
    for k in range(n + 1):
        t = k / n
        cell = cmap.world_to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if cell is None or not mask[cell]:
            return False
    return True


def simplify(cmap, mask, pts):
    """Drop A* stair steps: keep only the corners needed to stay on mask."""
    if len(pts) <= 2:
        return list(pts)
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not line_clear(cmap, mask, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def route_to(cmap, x, y, goal_cell):
    """Corner points from (x, y) to goal_cell over drivable cells, or None."""
    here = start_cell(cmap, x, y)
    if here is None:
        return None
    path = _astar(cmap.drive_mask, here, goal_cell)
    if path is None:
        return None
    return [(x, y)] + simplify(cmap, cmap.drive_mask, [cmap.cell_center(c) for c in path])


def ban_goal(cmap, cell, radius_m):
    """Stop explore_goal from picking cells around `cell` again."""
    rad = radius_m / cmap.res
    r = int(math.ceil(rad))
    y0, y1 = max(0, cell[0] - r), min(cmap.h, cell[0] + r + 1)
    x0, x1 = max(0, cell[1] - r), min(cmap.w, cell[1] + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    cmap._no_approach[y0:y1, x0:x1] |= (yy - cell[0]) ** 2 + (xx - cell[1]) ** 2 <= rad * rad


EXPLORE_LOCAL_M = 1.5       # places this far by road from the robot count as "here" for going on
EXPLORE_TURN_COST_M = 0.6   # turning half round away from the heading costs like driving this far
BRANCH_SECTOR_DEG = 60.0    # places this far apart in direction from the robot are different branches
BRANCH_NEAR_M = 0.40        # a branch is still open while a candidate is this close to it
BRANCH_KEEP = 200


def explore_goal(cmap, x, y, yaw, branches, target, radius_m, min_gain=4):
    """Depth-first: the next drivable cell from which at least `min_gain` target cells lie within
    `radius_m`, keeping the ways not taken in `branches` (a stack of map points).

    Among places within EXPLORE_LOCAL_M by road, the one most nearly straight ahead wins (turning
    away costs EXPLORE_TURN_COST_M per half turn). The best place in each other direction is pushed
    onto `branches`. With nothing nearby, the newest branch that still has a candidate is taken;
    only with no branch left the nearest candidate anywhere.
    Returns (cell, how, reachable, unreachable): how is "ahead", "branch" or "nearest" (None when
    there is no goal); the counts are target cells the body can or cannot get near.
    """
    rad = max(1, int(math.ceil(radius_m / cmap.res)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)).astype(np.float32)
    gain = cv2.filter2D(target.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    dist = drive_dist(cmap, x, y)
    reach = dist >= 0
    near = cv2.dilate(reach.astype(np.uint8), kernel.astype(np.uint8)) > 0
    reachable, unreachable = int((target & near).sum()), int((target & ~near).sum())
    ys, xs = np.nonzero(reach & (gain >= min_gain) & ~cmap._no_approach)
    if len(ys) == 0:
        return None, None, reachable, unreachable
    road = dist[ys, xs] * cmap.res
    cx = cmap.origin[0] + (xs + 0.5) * cmap.res
    cy = cmap.origin[1] + (ys + 0.5) * cmap.res
    bearing = np.arctan2(cy - y, cx - x)
    off = np.abs(np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw)))
    full = float(kernel.sum())
    local = np.nonzero(road <= EXPLORE_LOCAL_M)[0]
    if len(local):
        score = road[local] + EXPLORE_TURN_COST_M * off[local] / math.pi + 0.6 * (1.0 - gain[ys, xs][local] / full)
        k = local[int(np.argmin(score))]
        chosen = bearing[k]
        sector = math.radians(BRANCH_SECTOR_DEG)
        forks = {}
        for j in local:
            d = math.atan2(math.sin(bearing[j] - chosen), math.cos(bearing[j] - chosen))
            if abs(d) < sector / 2.0:
                continue
            key = int(math.floor((d + math.pi) / sector))
            if key not in forks or road[j] < road[forks[key]]:
                forks[key] = j
        # The fork nearest the way taken goes on top: it is the next one to come back to.
        for j in sorted(forks.values(), key=lambda j: -off[j]):
            p = (float(cx[j]), float(cy[j]))
            if all(math.hypot(p[0] - bx, p[1] - by) >= BRANCH_NEAR_M for bx, by in branches):
                branches.append(p)
        del branches[:-BRANCH_KEEP]
        return (int(ys[k]), int(xs[k])), "ahead", reachable, unreachable
    while branches:
        bx, by = branches.pop()
        d2 = (cx - bx) ** 2 + (cy - by) ** 2
        k = int(np.argmin(d2))
        if d2[k] <= BRANCH_NEAR_M ** 2:
            return (int(ys[k]), int(xs[k])), "branch", reachable, unreachable
    k = int(np.argmin(road + 0.6 * (1.0 - gain[ys, xs] / full)))
    return (int(ys[k]), int(xs[k])), "nearest", reachable, unreachable
