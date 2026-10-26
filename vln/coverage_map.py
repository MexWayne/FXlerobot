"""Coverage grid for one enclosed cleaning area.

Floor cells remember whether the body or the cleaning arm has swept them.
A chair or table is done when the swept share of its current footprint is high enough.
"""

import json
import math
import os

import cv2
import numpy as np

UNKNOWN = 0
BODY_REACH = 1
ARM_ONLY = 2
BODY_DONE = 3
ARM_DONE = 4
BLOCKED = 5

# Arm facing direction is the body forward axis.
BODY_LENGTH_M = 0.30  # parallel to the arm normal
BODY_WIDTH_M = 0.50   # perpendicular to the arm normal
BODY_HEIGHT_M = 0.80
# Half-diagonal: the center must stay this far from obstacles so the rectangle can turn.
BODY_CLEARANCE_M = math.hypot(BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0)
# Extra room kept when planning, for dead-reckoning drift and the fence estimate.
PLAN_MARGIN_M = 0.04

NAMES = {"table": "桌子", "chair": "椅子"}
DONE_AT = {"table": 0.85, "chair": 0.90}


def dist_to_boundary(x, y, poly):
    """Distance in metres from a point to the nearest edge of the fence."""
    best = math.inf
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg = dx * dx + dy * dy
        if seg == 0.0:
            best = min(best, math.hypot(x - x1, y - y1))
            continue
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg))
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best


def point_in_poly(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def rasterize(origin, res, shape, poly):
    h, w = shape
    xs = origin[0] + (np.arange(w) + 0.5) * res
    ys = origin[1] + (np.arange(h) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)
    inside = np.zeros(shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        den = yj - yi
        hit = (yi > yy) != (yj > yy)
        cross = hit & (xx < (xj - xi) * (yy - yi) / np.where(hit, den, 1.0) + xi)
        inside ^= cross
        j = i
    return inside


def erode(free, radius_cells):
    """Cells whose center stays at least radius_cells away from blocked cells."""
    if radius_cells <= 0:
        return free.copy()
    r = int(radius_cells)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = ((yy * yy + xx * xx) <= r * r).astype(np.uint8)
    # Cells past the grid edge do not count as blocked.
    out = cv2.erode(free.astype(np.uint8), disk, borderType=cv2.BORDER_CONSTANT, borderValue=1)
    return out.astype(bool)


def clip_polygon(poly, point, normal):
    """The part of `poly` on the side of the line through `point` that `normal` points away from."""
    px, py = point
    nx, ny = normal

    def side(p):
        return (p[0] - px) * nx + (p[1] - py) * ny  # <= 0 is kept

    out = []
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        sa, sb = side(a), side(b)
        if sa <= 0:
            out.append(a)
        if (sa < 0 < sb) or (sb < 0 < sa):
            t = sa / (sa - sb)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


class CleanMap:
    def ensure_room(self, poly, pad=None):
        """Grow the grid when `poly` no longer fits. Old cells stay on the same world positions."""
        if pad is None:
            pad = self.robot_radius + self.body_brush
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        want_x0, want_y0 = min(xs) - pad, min(ys) - pad
        want_x1, want_y1 = max(xs) + pad, max(ys) + pad
        cur_x1 = self.origin[0] + self.w * self.res
        cur_y1 = self.origin[1] + self.h * self.res
        if (want_x0 >= self.origin[0] - 1e-6 and want_y0 >= self.origin[1] - 1e-6
                and want_x1 <= cur_x1 + 1e-6 and want_y1 <= cur_y1 + 1e-6):
            return False
        ox = int(math.floor((self.origin[0] - min(self.origin[0], want_x0)) / self.res + 1e-6))
        oy = int(math.floor((self.origin[1] - min(self.origin[1], want_y0)) / self.res + 1e-6))
        origin = (self.origin[0] - ox * self.res, self.origin[1] - oy * self.res)
        far_x = max(cur_x1, want_x1)
        far_y = max(cur_y1, want_y1)
        w = max(ox + self.w, int(math.ceil((far_x - origin[0]) / self.res - 1e-6)))
        h = max(oy + self.h, int(math.ceil((far_y - origin[1]) / self.res - 1e-6)))
        grid = np.full((h, w), UNKNOWN, dtype=np.uint8)
        owner = np.zeros((h, w), dtype=np.int16)
        grid[oy:oy + self.h, ox:ox + self.w] = self.grid
        owner[oy:oy + self.h, ox:ox + self.w] = self.owner
        paint = np.zeros((h, w), dtype=np.uint8)
        old_paint = getattr(self, "paint", None)
        if old_paint is not None:
            paint[oy:oy + old_paint.shape[0], ox:ox + old_paint.shape[1]] = old_paint
        if hasattr(self, "_no_approach"):
            banned = np.zeros((h, w), dtype=bool)
            old = self._no_approach
            banned[oy:oy + old.shape[0], ox:ox + old.shape[1]] = old
            self._no_approach = banned
        self.origin = origin
        self.h, self.w = h, w
        self.grid = grid
        self.owner = owner
        self.paint = paint
        self._sweep_cache = None
        self._fence_pts = None
        self._fence_field = None
        return True

    def set_boundary(self, poly):
        """Move the fence to `poly`, inwards where a wall was seen, outwards where floor was.

        Floor the new fence leaves out stops being floor; floor it takes in becomes unswept floor.
        Swept and blocked cells stay as they are, and blocked cells keep the body away as before.
        The grid grows when the new fence no longer fits, so exploration can keep moving the outline.
        """
        self.ensure_room(poly)
        self.boundary = [(float(x), float(y)) for x, y in poly]
        inside = rasterize(self.origin, self.res, self.grid.shape, self.boundary)
        self.grid[~inside & (self.grid == BODY_REACH)] = UNKNOWN
        self.grid[inside & (self.grid == UNKNOWN) & (self.owner == 0)] = BODY_REACH
        free = inside & (self.owner == 0) & (self.grid != BLOCKED)
        self.drive_mask = erode(free, max(1, int(math.ceil(self.drive_radius / self.res))))
        self.plan_mask = erode(free, max(1, int(math.ceil(self.turn_radius / self.res))))
        self._sweep_cache = None
        self._fence_pts = None
        self._fence_field = None

    def __init__(self, spec):
        poly = [(float(x), float(y)) for x, y in spec["boundary"]]
        if len(poly) < 3:
            raise ValueError("boundary needs at least 3 points")
        self.boundary = poly
        self.res = float(spec.get("resolution_m", 0.05))
        self.body_length = float(spec.get("body_length_m", BODY_LENGTH_M))
        self.body_width = float(spec.get("body_width_m", BODY_WIDTH_M))
        self.body_height = float(spec.get("body_height_m", BODY_HEIGHT_M))
        self.robot_radius = math.hypot(self.body_length / 2.0, self.body_width / 2.0)
        self.body_brush = float(spec.get("body_brush_m", self.body_width))
        self.arm_brush = float(spec.get("arm_brush_m", 0.16))
        start = spec.get("start", [poly[0][0], poly[0][1], 0.0])
        self.start = (float(start[0]), float(start[1]), math.radians(float(start[2])))

        pts = np.asarray(poly, dtype=float)
        margin = self.robot_radius + self.body_brush
        self.origin = (float(pts[:, 0].min() - margin), float(pts[:, 1].min() - margin))
        self.w = int(math.ceil((pts[:, 0].max() - pts[:, 0].min() + 2 * margin) / self.res))
        self.h = int(math.ceil((pts[:, 1].max() - pts[:, 1].min() + 2 * margin) / self.res))
        self.grid = np.full((self.h, self.w), UNKNOWN, dtype=np.uint8)
        self.owner = np.zeros((self.h, self.w), dtype=np.int16)
        # 1 warm-up trail, 2 explore trail, 3 swept area. Survives reset_visited.
        self.paint = np.zeros((self.h, self.w), dtype=np.uint8)

        inside = rasterize(self.origin, self.res, self.grid.shape, poly)
        self.furniture = []
        for item in spec.get("furniture", []):
            fid = int(item["id"])
            cls = item.get("cls", "table")
            footprint = [(float(x), float(y)) for x, y in item["polygon"]]
            mask = rasterize(self.origin, self.res, self.grid.shape, footprint) & inside
            self.owner[mask] = fid
            self.grid[mask] = ARM_ONLY
            self.furniture.append({
                "id": fid,
                "cls": cls,
                "name": f"{NAMES.get(cls, cls)}-{fid}",
                "polygon": footprint,
                "coverage": 0.0,
                "done": False,
            })

        floor = inside & (self.owner == 0)
        self.grid[floor] = BODY_REACH
        # The body is a 0.30 x 0.50 x 0.80 m cuboid. On the map the footprint is the
        # 0.30 m by 0.50 m base, so the center stays a half-diagonal away at any heading.
        # Turning spots use the same clearance. Corners of a route have to sit on plan_mask.
        self.drive_radius = self.robot_radius + PLAN_MARGIN_M
        self.turn_radius = self.robot_radius + PLAN_MARGIN_M
        self.drive_mask = erode(floor, max(1, int(math.ceil(self.drive_radius / self.res))))
        self.plan_mask = erode(floor, max(1, int(math.ceil(self.turn_radius / self.res))))
        self.refresh()

    def cell_center(self, cell):
        y, x = cell
        return (self.origin[0] + (x + 0.5) * self.res, self.origin[1] + (y + 0.5) * self.res)

    def world_to_cell(self, x, y):
        c = int(math.floor((x - self.origin[0]) / self.res))
        r = int(math.floor((y - self.origin[1]) / self.res))
        if r < 0 or c < 0 or r >= self.h or c >= self.w:
            return None
        return r, c

    def snap_start(self):
        """Move the published start onto a cell the body can actually stand on."""
        x, y, yaw = self.start
        cell = self.world_to_cell(x, y)
        if cell is not None and self.plan_mask[cell]:
            return
        ys, xs = np.where(self.plan_mask)
        if len(xs) == 0:
            raise RuntimeError("围栏里没有本体能进入的地面")
        dx = self.origin[0] + (xs + 0.5) * self.res - x
        dy = self.origin[1] + (ys + 0.5) * self.res - y
        k = int(np.argmin(dx * dx + dy * dy))
        self.start = (float(self.origin[0] + (xs[k] + 0.5) * self.res),
                      float(self.origin[1] + (ys[k] + 0.5) * self.res), yaw)

    def _stamp(self, x, y, radius, accept):
        fx = (x - self.origin[0]) / self.res - 0.5
        fy = (y - self.origin[1]) / self.res - 0.5
        rad = radius / self.res
        r = int(math.ceil(rad))
        c = int(round(fx))
        row = int(round(fy))
        y0, y1 = max(0, row - r), min(self.h, row + r + 1)
        x0, x1 = max(0, c - r), min(self.w, c + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - fy) ** 2 + (xx - fx) ** 2 <= rad * rad
        view = self.grid[y0:y1, x0:x1]
        view[disk & accept(y0, y1, x0, x1)] = accept.write

    def stamp_body(self, x, y):
        def accept(y0, y1, x0, x1):
            state = self.grid[y0:y1, x0:x1]
            # No prior floor map: the cells the body actually drives are the swept area.
            return (state == UNKNOWN) | (state == BODY_REACH) | (state == BLOCKED)
        accept.write = BODY_DONE
        self._stamp(x, y, self.body_brush * 0.5, accept)

    def mark_pass(self, x, y, code, radius):
        """Remember which stage drove here. 1 warm-up, 2 explore, 3 sweep.

        Sweep covers the cleaning footprint and replaces earlier colours.
        Earlier stages leave a narrow trail and do not cover a later one.
        """
        if code not in (1, 2, 3) or not hasattr(self, "paint"):
            return
        fx = (x - self.origin[0]) / self.res - 0.5
        fy = (y - self.origin[1]) / self.res - 0.5
        rad = radius / self.res
        r = int(math.ceil(rad))
        c = int(round(fx))
        row = int(round(fy))
        y0, y1 = max(0, row - r), min(self.h, row + r + 1)
        x0, x1 = max(0, c - r), min(self.w, c + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - fy) ** 2 + (xx - fx) ** 2 <= rad * rad
        layer = self.paint[y0:y1, x0:x1]
        if code == 3:
            layer[disk] = 3
        elif code == 2:
            layer[disk & (layer != 3)] = 2
        else:
            layer[disk & (layer == 0)] = 1

    def reset_visited(self, x, y):
        """Start a new visitation pass while preserving the refined boundary and obstacles."""
        inside = rasterize(self.origin, self.res, self.grid.shape, self.boundary)
        body_done = self.grid == BODY_DONE
        self.grid[body_done & inside & (self.owner == 0)] = BODY_REACH
        self.grid[body_done & ~inside] = UNKNOWN
        self.grid[self.grid == ARM_DONE] = ARM_ONLY
        self._sweep_cache = None
        if hasattr(self, "_no_approach"):
            self._no_approach[:] = False
        # The third pass begins with only the floor physically under the robot counted visited.
        self.stamp_body(x, y)

    def stamp_arm(self, x, y, furniture_id):
        def accept(y0, y1, x0, x1):
            own = self.owner[y0:y1, x0:x1] == furniture_id
            state = self.grid[y0:y1, x0:x1]
            return own & ((state == ARM_ONLY) | (state == ARM_DONE))
        accept.write = ARM_DONE
        self._stamp(x, y, self.arm_brush * 0.5, accept)

    def _disk(self, x, y, radius, protect_xy=None):
        """(rows, cols, mask) of the cells within `radius` of (x, y), minus a small disk at protect_xy."""
        fx = (x - self.origin[0]) / self.res - 0.5
        fy = (y - self.origin[1]) / self.res - 0.5
        rad = radius / self.res
        r = int(math.ceil(rad))
        c = int(round(fx))
        row = int(round(fy))
        y0, y1 = max(0, row - r), min(self.h, row + r + 1)
        x0, x1 = max(0, c - r), min(self.w, c + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - fy) ** 2 + (xx - fx) ** 2 <= rad * rad
        if protect_xy is not None:
            px = (protect_xy[0] - self.origin[0]) / self.res - 0.5
            py = (protect_xy[1] - self.origin[1]) / self.res - 0.5
            disk = disk & ((yy - py) ** 2 + (xx - px) ** 2 > (0.12 / self.res) ** 2)
        return slice(y0, y1), slice(x0, x1), disk

    def block(self, x, y, foot, protect_xy=None):
        """Mark a seen obstacle of radius `foot` at (x, y).

        Only the obstacle itself is drawn as blocked floor. The body center keeps its drive
        clearance from it, and turning spots keep the turn clearance. protect_xy (where the
        robot stands) stays drivable so it can back away, but stops being a turning spot.
        """
        rows, cols, solid = self._disk(x, y, foot)
        view = self.grid[rows, cols]
        view[solid & ((view == BODY_REACH) | (view == BODY_DONE))] = BLOCKED
        for mask, clear, keep in ((self.drive_mask, self.drive_radius, protect_xy),
                                  (self.plan_mask, self.turn_radius, None)):
            rows, cols, disk = self._disk(x, y, foot + clear, keep)
            mask[rows, cols] &= ~disk
        self._sweep_cache = None

    def refresh(self):
        for item in self.furniture:
            cells = self.owner == item["id"]
            total = int(cells.sum())
            done = int(((self.grid == ARM_DONE) & cells).sum())
            item["coverage"] = (done / total) if total else 0.0
            item["done"] = item["coverage"] >= DONE_AT.get(item["cls"], 0.9)

    def sweepable(self):
        """Floor cells the passing body can paint: within half a brush of a drivable center cell."""
        if getattr(self, "_sweep_cache", None) is None:
            self._sweep_cache = self._sweepable()
        return self._sweep_cache

    def _sweepable(self):
        rad = max(1, int(math.ceil(self.body_brush * 0.5 / self.res)))
        out = np.zeros_like(self.drive_mask)
        ys, xs = np.where(self.drive_mask)
        r2 = rad * rad
        h, w = self.drive_mask.shape
        for y, x in zip(ys.tolist(), xs.tolist()):
            y0, y1 = max(0, y - rad), min(h, y + rad + 1)
            x0, x1 = max(0, x - rad), min(w, x + rad + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            out[y0:y1, x0:x1] |= (yy - y) ** 2 + (xx - x) ** 2 <= r2
        floor = (self.grid == BODY_REACH) | (self.grid == BODY_DONE) | (self.grid == BLOCKED)
        return out & floor

    def stats(self):
        body = (self.grid == BODY_REACH) | (self.grid == BODY_DONE)
        total = int(body.sum())
        done = int((self.grid == BODY_DONE).sum())
        sweep = self.sweepable() & (self.grid != BLOCKED)
        sweep_total = int(sweep.sum())
        sweep_done = int(((self.grid == BODY_DONE) & sweep).sum())
        return {
            "body": (done / total) if total else 0.0,
            "sweep": (sweep_done / sweep_total) if sweep_total else 0.0,
            "body_cells": total,
            "furniture": [
                {"id": f["id"], "name": f["name"], "cls": f["cls"],
                 "coverage": f["coverage"], "done": f["done"]}
                for f in self.furniture
            ],
        }

    def save(self, map_dir):
        os.makedirs(map_dir, exist_ok=True)
        self.refresh()
        np.savez_compressed(
            os.path.join(map_dir, "coverage.npz"),
            grid=self.grid, owner=self.owner,
            origin=np.asarray(self.origin), resolution=np.float64(self.res),
        )
        with open(os.path.join(map_dir, "furniture.json"), "w", encoding="utf-8") as f:
            json.dump(self.furniture, f, ensure_ascii=False, indent=2)

    def load(self, map_dir):
        path = os.path.join(map_dir, "coverage.npz")
        if not os.path.exists(path):
            return False
        data = np.load(path)
        grid = data["grid"]
        if grid.shape != self.grid.shape or abs(float(data["resolution"]) - self.res) > 1e-9:
            return False
        origin = data["origin"]
        if abs(float(origin[0]) - self.origin[0]) > 1e-6 or abs(float(origin[1]) - self.origin[1]) > 1e-6:
            return False
        self.grid = grid
        self.owner = data["owner"]
        self.refresh()
        return True
