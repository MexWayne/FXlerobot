"""Occupancy grid built from the lidar, with what only the cameras see on top, and lidar matching.

Every cell counts how often a lidar ray ended in it (hit) and passed through it (miss). A cell is
lidar-occupied once it was hit at least HIT_MIN times and by at least HIT_SHARE of the rays that
reached it, so a person who walks away is passed through and cleared again. The lidar sits low:
tabletops, wires flatter than its beam and floor-standing glass come from the cameras and are kept
in a separate layer with the time until which they block (people expire, furniture does not).
Things the cameras named and measured are modelled as rectangles (coverage_run) in their own layer,
rebuilt whenever a model changes; floor the body stood on is never blocked by them. Small things
on the floor the arms pick up are kept apart (TRASH): the body drives round them, it does not run
them over, and they never hide what the lidar measured.

The grid codes are the ones live_map draws: UNKNOWN (no ray has reached it), BODY_REACH (free floor),
BODY_DONE (free floor the body swept), BLOCKED, TRASH.
"""

import math

import cv2
import numpy as np

UNKNOWN, BODY_REACH, ARM_ONLY, BODY_DONE, ARM_DONE, BLOCKED, TRASH = range(7)

BODY_LENGTH_M = 0.30
BODY_WIDTH_M = 0.50
BODY_HEIGHT_M = 0.80
# Center to a base corner: the in-place turn radius.
BODY_CLEARANCE_M = math.hypot(BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0)
PLAN_MARGIN_M = 0.04  # extra room kept when planning, for pose error

HIT_MIN = 2
HIT_SHARE = 0.25
MATCH_TRIM_M = 0.15   # a scan point farther than this from every occupied cell counts this much


def erode(free, radius_cells):
    """Cells whose center stays at least radius_cells away from cells that are not free."""
    r = int(radius_cells)
    if r <= 0:
        return free.copy()
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(free.astype(np.uint8), disk, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0


def disk(radius_cells):
    r = max(0, int(math.ceil(radius_cells)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)) if r else np.ones((1, 1), np.uint8)


class OccMap:
    def __init__(self, size_m=24.0, res=0.05, center=(0.0, 0.0)):
        self.res = res
        self.w = self.h = int(round(size_m / res))
        self.origin = (center[0] - size_m / 2.0, center[1] - size_m / 2.0)
        shape = (self.h, self.w)
        self.hit = np.zeros(shape, np.int32)
        self.miss = np.zeros(shape, np.int32)
        self.seen = np.zeros(shape, bool)        # a lidar ray or the body has been there
        self.vis_end = np.zeros(shape, np.float64)  # camera obstacle blocks until this time.monotonic()
        self.glass = np.zeros(shape, bool)       # floor-standing glass or mirror: lidar rays stop here
        self.trash = np.zeros(shape, bool)       # small things to pick up: driven round
        self.objects = np.zeros(shape, bool)     # rectangles of the modelled things (coverage_run)
        self.floor_seen = np.zeros(shape, bool)  # the stereo measured bare floor here
        self.stood = np.zeros(shape, bool)       # the body stood here
        self.swept = np.zeros(shape, bool)
        self.paint = np.zeros(shape, np.uint8)   # 1 warm-up, 2 explore, 3 sweep trail (live_map colours)
        self._no_approach = np.zeros(shape, bool)  # goals given up (coverage_plan.ban_goal)
        self.body_length, self.body_width = BODY_LENGTH_M, BODY_WIDTH_M
        self.robot_radius = BODY_CLEARANCE_M
        self.body_brush = BODY_WIDTH_M
        self.drive_radius = self.turn_radius = self.robot_radius + PLAN_MARGIN_M
        self.boundary, self.furniture = [], []   # live_map draws these; the lidar map has none
        self.rebuild()

    # --- cells ---------------------------------------------------------------------------
    def cell_center(self, cell):
        y, x = cell
        return (self.origin[0] + (x + 0.5) * self.res, self.origin[1] + (y + 0.5) * self.res)

    def world_to_cell(self, x, y):
        c = int(math.floor((x - self.origin[0]) / self.res))
        r = int(math.floor((y - self.origin[1]) / self.res))
        if r < 0 or c < 0 or r >= self.h or c >= self.w:
            return None
        return r, c

    def _cells(self, xy):
        """(rows, cols) of map points (N x 2), only those inside the grid."""
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        c = np.floor((xy[:, 0] - self.origin[0]) / self.res).astype(np.int64)
        r = np.floor((xy[:, 1] - self.origin[1]) / self.res).astype(np.int64)
        ok = (r >= 0) & (c >= 0) & (r < self.h) & (c < self.w)
        return r[ok], c[ok]

    def mask_of(self, xy, radius):
        mask = np.zeros((self.h, self.w), np.uint8)
        r, c = self._cells(xy)
        mask[r, c] = 1
        if radius > self.res / 2:
            mask = cv2.dilate(mask, disk(radius / self.res))
        return mask > 0

    # --- what is where ---------------------------------------------------------------------
    def lidar_occ(self):
        return (self.hit >= HIT_MIN) & (self.hit * (1.0 / HIT_SHARE) >= self.hit + self.miss)

    def occupied(self):
        return self._occ

    def rebuild(self, now=None):
        """Recompute obstacles, the drivable cells and the grid codes after the layers changed."""
        import time

        now = time.monotonic() if now is None else now
        self._lidar = self.lidar_occ()
        occ = self._lidar | (self.vis_end > now) | self.trash | (self.objects & ~self.stood) | self.glass
        self._occ = occ
        free = self.seen & ~occ
        # Unknown cells count as blocked: the body only goes where a ray has been.
        self.drive_mask = erode(free, math.ceil(self.drive_radius / self.res))
        self.plan_mask = self.drive_mask
        grid = np.full((self.h, self.w), UNKNOWN, np.uint8)
        grid[free] = BODY_REACH
        grid[free & self.swept] = BODY_DONE
        grid[occ] = BLOCKED
        grid[self.trash] = TRASH  # blocks like the rest, drawn apart: the arms pick these up
        self.grid = grid
        self._sweep_cache = None
        self._field = None

    def insert_scan(self, pose, xy):
        """One lidar turn (N x 2, base frame) taken at map pose (x, y, yaw rad).

        Rays pass through the cells before their return (miss) and end in it (hit). A ray that
        crosses a glass cell stops there: what it returns from is behind the glass or a reflection.
        """
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        if not len(xy):
            return
        x0, y0, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        r = np.hypot(xy[:, 0], xy[:, 1])
        ux = (c * xy[:, 0] - s * xy[:, 1]) / r
        uy = (s * xy[:, 0] + c * xy[:, 1]) / r
        step = self.res * 0.5
        t = np.arange(1, int(r.max() / step) + 1) * step
        along = t[None, :] < (r[:, None] - self.res)       # samples before the return
        sx = x0 + ux[:, None] * t[None, :]
        sy = y0 + uy[:, None] * t[None, :]
        col = np.floor((sx - self.origin[0]) / self.res).astype(np.int64)
        row = np.floor((sy - self.origin[1]) / self.res).astype(np.int64)
        inside = (row >= 0) & (col >= 0) & (row < self.h) & (col < self.w)
        along &= inside
        glass = np.zeros(along.shape, bool)
        glass[inside] = self.glass[row[inside], col[inside]]
        glass &= along
        cut = glass.any(axis=1)
        if cut.any():
            first = np.where(cut, glass.argmax(axis=1), along.shape[1])
            along &= np.arange(along.shape[1])[None, :] < first[:, None]
        flat = np.unique(row[along] * self.w + col[along])
        self.miss.flat[flat] += 1
        self.seen.flat[flat] = True
        end = np.c_[x0 + ux * r, y0 + uy * r][~cut]
        hr, hc = self._cells(end)
        flat = np.unique(hr * self.w + hc)
        self.hit.flat[flat] += 1
        self.seen.flat[flat] = True

    def block(self, xy, radius, until):
        """Camera obstacle at map points `xy`, blocking until time.monotonic() `until`."""
        if len(xy):
            mask = self.mask_of(xy, radius)
            self.vis_end[mask] = np.maximum(self.vis_end[mask], until)

    def fill(self, poly, until):
        """Camera obstacle over a whole polygon (map frame): the floor under a table."""
        mask = np.zeros((self.h, self.w), np.uint8)
        pts = np.array([[(x - self.origin[0]) / self.res, (y - self.origin[1]) / self.res] for x, y in poly])
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        sel = mask > 0
        self.vis_end[sel] = np.maximum(self.vis_end[sel], until)

    def mark_glass(self, xy):
        self.glass |= self.mask_of(xy, self.res)
        self.seen |= self.glass

    def mark_floor(self, xy):
        """The stereo measured bare floor at these map points."""
        self.floor_seen |= self.mask_of(xy, 0.0)

    def mark_trash(self, xy, radius=0.03):
        self.trash |= self.mask_of(xy, radius)

    def is_trash(self, x, y):
        cell = self.world_to_cell(x, y)
        return cell is not None and bool(self.trash[cell])

    def clear_vis(self, mask):
        """The cameras saw bare floor there: older camera obstacles and trash are gone. Returns cells cleared."""
        mask = mask & ((self.vis_end > 0) | self.trash)
        self.vis_end[mask] = 0.0
        self.trash[mask] = False
        return int(mask.sum())

    def unblock(self, mask):
        """The body stood there, so nothing can: lidar and camera marks under it go, and the
        trash it passed over is swept up."""
        mask = mask & (self._occ | (self.vis_end > 0) | self.trash)
        self.hit[mask] = 0
        self.vis_end[mask] = 0.0
        self.trash[mask] = False
        return int(mask.sum())

    def body_mask(self, x, y, yaw, margin=0.0):
        hl, hw = BODY_LENGTH_M / 2.0 + margin, BODY_WIDTH_M / 2.0 + margin
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(x + c * a - s * b, y + s * a + c * b) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        mask = np.zeros((self.h, self.w), np.uint8)
        pts = np.array([[(px - self.origin[0]) / self.res - 0.5, (py - self.origin[1]) / self.res - 0.5]
                        for px, py in corners])
        cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 1)
        return mask > 0

    def stamp_body(self, x, y, yaw, sweep=True):
        """The body stands here: the floor under it is known, and swept when `sweep`."""
        under = self.body_mask(x, y, yaw)
        self.seen |= under
        self.stood |= under
        if sweep:
            self.swept |= self.mask_of([(x, y)], self.body_brush * 0.5)

    def mark_pass(self, x, y, code, radius):
        """Trail colour: 1 warm-up, 2 explore, 3 sweep (sweep covers the brush and replaces the others)."""
        if code not in (1, 2, 3):
            return
        sel = self.mask_of([(x, y)], radius)
        if code == 3:
            self.paint[sel] = 3
        elif code == 2:
            self.paint[sel & (self.paint != 3)] = 2
        else:
            self.paint[sel & (self.paint == 0)] = 1

    # --- coverage --------------------------------------------------------------------------
    def frontier(self):
        """Free cells next to unknown floor: where a look or a spin still adds to the map.

        Small gaps between far rays are closed first; they are not unexplored floor.
        """
        known = cv2.morphologyEx(self.seen.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        free = known & ~self._occ
        return free & (cv2.dilate((~known).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)

    def sweepable(self):
        """Free cells the passing body can paint: within half a brush of a drivable center cell."""
        if self._sweep_cache is None:
            reach = cv2.dilate(self.drive_mask.astype(np.uint8), disk(self.body_brush * 0.5 / self.res)) > 0
            self._sweep_cache = reach & self.seen & ~self._occ
        return self._sweep_cache

    def reset_swept(self):
        self.swept[:] = False
        self._no_approach[:] = False
        self.rebuild()

    def refresh(self):  # live_map calls this; nothing to recount here
        pass

    def stats(self):
        sweep = self.sweepable()
        total = int(sweep.sum())
        return {"sweep": int((sweep & self.swept).sum()) / total if total else 0.0,
                "free_m2": float((self.seen & ~self._occ).sum()) * self.res * self.res,
                "furniture": []}

    # --- localisation ----------------------------------------------------------------------
    def field(self):
        """Metres from every cell to the nearest lidar-occupied cell, capped at MATCH_TRIM_M."""
        if self._field is None:
            dist = cv2.distanceTransform((~self._lidar).astype(np.uint8), cv2.DIST_L2, 5) * self.res
            self._field = np.minimum(dist, MATCH_TRIM_M).astype(np.float32)
        return self._field

    def _cost(self, pts):
        """Mean trimmed distance of map points (..., N, 2) to the lidar map, interpolated between
        cell centers so the fit does not stand still across a whole cell."""
        fx = (pts[..., 0] - self.origin[0]) / self.res - 0.5
        fy = (pts[..., 1] - self.origin[1]) / self.res - 0.5
        c0, r0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        ok = (r0 >= 0) & (c0 >= 0) & (r0 < self.h - 1) & (c0 < self.w - 1)
        d = np.full(fx.shape, MATCH_TRIM_M, np.float32)
        r, c = r0[ok], c0[ok]
        tx, ty = (fx[ok] - c).astype(np.float32), (fy[ok] - r).astype(np.float32)
        f = self.field()
        d[ok] = ((f[r, c] * (1 - tx) + f[r, c + 1] * tx) * (1 - ty)
                 + (f[r + 1, c] * (1 - tx) + f[r + 1, c + 1] * tx) * ty)
        return d.mean(axis=-1)

    def match(self, xy, guess, xy_win, yaw_win_deg, per_m=0.02, per_deg=0.001):
        """Pose near `guess` (x, y, yaw) that lays base-frame scan points `xy` onto the lidar map.

        A coarse then a fine grid search. per_m / per_deg cost metres of fit per metre / degree away
        from the guess, so a corridor that fits equally well along its length keeps the guess.
        Returns (pose, fit): fit is the plain mean trimmed distance at the pose (metres), or None
        when the map has too few occupied cells to match against.
        """
        if int(self._lidar.sum()) < 30 or len(xy) < 10:
            return None
        pts = np.asarray(xy, np.float32).reshape(-1, 2)
        gx, gy, gyaw = guess

        def search(cx, cy, cyaw, win, step, ywin, ystep):
            span = np.arange(-win, win + 1e-9, step, dtype=np.float32)
            tx, ty = np.meshgrid(span, span)
            shifts = np.stack([tx.ravel(), ty.ravel()], axis=1)
            best = (math.inf, guess)
            for d in np.arange(-ywin, ywin + 1e-9, ystep):
                yaw = cyaw + math.radians(float(d))
                c, s = math.cos(yaw), math.sin(yaw)
                base = np.stack([cx + c * pts[:, 0] - s * pts[:, 1], cy + s * pts[:, 0] + c * pts[:, 1]], axis=1)
                cost = self._cost(base[None, :, :] + shifts[:, None, :])
                px, py = cx + shifts[:, 0], cy + shifts[:, 1]
                cost = cost + per_m * np.hypot(px - gx, py - gy) + per_deg * abs(math.degrees(yaw - gyaw))
                k = int(np.argmin(cost))
                if cost[k] < best[0]:
                    best = (float(cost[k]), (float(px[k]), float(py[k]), yaw))
            return best[1]

        step, ystep = max(0.02, xy_win / 10.0), max(0.5, yaw_win_deg / 4.0)
        coarse = search(gx, gy, gyaw, xy_win, step, yaw_win_deg, ystep)
        pose = search(*coarse, step, 0.01, ystep, 0.25)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        placed = np.stack([pose[0] + c * pts[:, 0] - s * pts[:, 1], pose[1] + s * pts[:, 0] + c * pts[:, 1]], axis=1)
        return pose, float(self._cost(placed))
