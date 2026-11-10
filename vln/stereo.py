"""Depth from the two head cameras, as points in the base frame.

The cameras are 14 cm apart and their optical axes 18.5° apart, so only the left two thirds of
the left image (columns 0..410) have a partner in the right image. Rectifying with alpha=1 keeps
all of that overlap. The rectified focal length is then about 350 px: one pixel of disparity is
about 2 cm at 1 m, and 256 disparities reach in to 0.2 m from the camera.
"""

import cv2
import numpy as np

SIZE = (640, 480)
NUM_DISP = 256
BLOCK = 5
TEXTURE = 12.0   # mean horizontal grey gradient under which a patch has nothing to match
LR_PX = 1.5      # the left and the right disparity of a pixel must agree this well
SCALE_PX = 4.0   # the disparity found again on half-size images must agree this well (full-size pixels)
MIN_Z_M, MAX_Z_M = 0.15, 4.0
EDGE_PX = 4 + BLOCK  # matches this close to where a camera's picture ends are not trusted
BACKGROUND_M = 3.0   # the right camera has to see this far behind a pixel for its match to count
# The floor checks the pair. Since 2026-11-07 evening one camera shifts against the other when the body
# moves (the left one was replugged that day): its depth then changes from stop to stop and holds while
# the body stands, the floor 0.4-1 m out came out 3-7 cm high (camera height from the stereo's own floor
# plane 0.57-0.70 m, really ~0.63), and bare striped carpet became "standing" cells, mapped as obstacles
# for good (2026-11-09). The floor in the lower half of the picture (pixels whose disparity lies within
# FLOOR_NEAR_PX of the floor's at that pixel) must come out FLOOR_BASE_Z high within FLOOR_TOL_M, else the
# pair is not used; looks with too little floor in view (looking up) go by the last verdict.
# After the cameras were moved and the left one turned back by hand (2026-11-10) the floor reads lower:
# +0.009/+0.013 at the check, -0.023..+0.004 in the 11 lane looks of 103 that 0.025 +- 0.02 refused.
# Before, sound grasp-view pairs read 0.019 and lane looks ~0.035; the shifted pairs +0.05..+0.10 and
# -0.04..-0.10. Recentre after the next board calibration.
FLOOR_BASE_Z = 0.005
FLOOR_TOL_M = 0.03
FLOOR_NEAR_PX = 15.0  # 8 left too little floor in the worst pairs to judge them
FLOOR_MIN_PIX = 1500


class Stereo:
    def __init__(self, calib, cams):
        kl, dl = cams["left"]["K"], cams["left"]["dist"]
        kr, dr = cams["right"]["K"], cams["right"]["dist"]
        rot = np.array(calib["stereo"]["R_right_left"], float)
        tr = np.array(calib["stereo"]["T_right_left_m"], float)
        full = tuple(calib.get("image_size", (640, 480)))
        self.r1, r2, p1, p2, self.q, _, _ = cv2.stereoRectify(kl, dl, kr, dr, full, rot, tr,
                                                              alpha=1.0, newImageSize=SIZE)
        self.map_l = cv2.initUndistortRectifyMap(kl, dl, self.r1, p1, SIZE, cv2.CV_32FC1)
        self.map_r = cv2.initUndistortRectifyMap(kr, dr, r2, p2, SIZE, cv2.CV_32FC1)
        # Left-image pixel that each rectified pixel was taken from: boxes are drawn on the left image.
        self.uv = np.dstack(self.map_l)
        # Rectified pixels that really come from inside each picture. The black border around them
        # is a strong edge in both images and matches itself at a wrong, steady disparity.
        self.inside_l = self._inside(self.map_l, full)
        self.inside_r = self._inside(self.map_r, full)
        self.sgbm = self._matcher(NUM_DISP, BLOCK)
        self.sgbm_half = self._matcher(NUM_DISP // 2, 3)
        q = self.q
        uu, vv = np.meshgrid(np.arange(SIZE[0], dtype=np.float64), np.arange(SIZE[1], dtype=np.float64))
        # Each rectified pixel's ray (rectified frame, z = 1) in the left camera's frame.
        self.rays = np.dstack([(uu + q[0, 3]) / q[2, 3], (vv + q[1, 3]) / q[2, 3], np.ones_like(uu)]) @ self.r1
        self.trusted = True   # the last pair the floor could check agreed with it
        self.floor_z = None   # the floor's height (base frame) in the last pair that showed enough of it

    @staticmethod
    def _matcher(num_disp, block):
        return cv2.StereoSGBM_create(0, num_disp, block, P1=8 * 3 * block * block, P2=32 * 3 * block * block,
                                     uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    @staticmethod
    def _inside(maps, full):
        w, h = full
        ok = (maps[0] >= EDGE_PX) & (maps[0] <= w - 1 - EDGE_PX) & (maps[1] >= EDGE_PX) & (maps[1] <= h - 1 - EDGE_PX)
        return cv2.erode(ok.astype(np.uint8), np.ones((2 * BLOCK + 1, 2 * BLOCK + 1), np.uint8)) > 0

    @staticmethod
    def _disparity(matcher, a, b):
        # Padding on the left lets the leftmost columns search the full disparity range.
        n = matcher.getNumDisparities()
        pad = cv2.copyMakeBorder(a, 0, 0, n, 0, cv2.BORDER_REPLICATE)
        pbd = cv2.copyMakeBorder(b, 0, 0, n, 0, cv2.BORDER_REPLICATE)
        return matcher.compute(pad, pbd).astype(np.float32)[:, n:] / 16.0

    def points(self, left, right, T_base_cam):
        """HxWx3 base-frame points on the rectified grid, NaN where the match is not trusted.

        A match is kept when the patch has texture across it, both pixels lie inside their camera's
        picture, the disparity found from the right image agrees with the one from the left
        (flipping both images and swapping them runs the same matcher right to left), and the
        half-size images give the same disparity. Plain walls and the carpet's long stripes fail
        one of these and come out as no data, not as an obstacle. The half-size test is for a thin
        pole in front of window bars: the bars repeat, and the pole matched a bar 0.3 m from the
        camera at full size, consistently from both sides.
        Past the right edge of the overlap (left-image columns beyond about 410) only something
        close to the camera is also in the right picture, so any match there comes out close,
        and on bare carpet that put floor 0.25 m in front of the body. A pixel only counts when
        the right camera would also see BACKGROUND_M behind it.
        """
        gl = cv2.cvtColor(cv2.remap(left, *self.map_l, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(cv2.remap(right, *self.map_r, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        d = self._disparity(self.sgbm, gl, gr)
        d_right = cv2.flip(self._disparity(self.sgbm, cv2.flip(gr, 1), cv2.flip(gl, 1)), 1)
        d_half = self._disparity(self.sgbm_half, cv2.pyrDown(gl), cv2.pyrDown(gr))
        h, w = d.shape
        cols = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
        partner = np.clip(np.round(cols - d).astype(int), 0, w - 1)
        agree = np.abs(d - np.take_along_axis(d_right, partner, 1)) <= LR_PX
        texture = cv2.blur(np.abs(cv2.Sobel(gl, cv2.CV_32F, 1, 0, ksize=3)), (7, 7)) > TEXTURE
        p = cv2.reprojectImageTo3D(d, self.q)
        good = (d > 1) & agree & texture & np.isfinite(p).all(-1)
        good &= self.inside_l & np.take_along_axis(self.inside_r, partner, 1)
        good &= np.abs(d - 2.0 * cv2.resize(d_half, (w, h), interpolation=cv2.INTER_NEAREST)) <= SCALE_PX
        good &= (p[..., 2] > MIN_Z_M) & (p[..., 2] < MAX_Z_M)
        far = int(np.ceil((self.q[2, 3] / BACKGROUND_M - self.q[3, 3]) / self.q[3, 2]))
        good &= np.take_along_axis(self.inside_r, np.clip(cols.astype(int) - far, 0, w - 1), 1)
        out = np.full(p.shape, np.nan, np.float32)
        cam = p[good] @ self.r1  # rectified left camera -> left camera
        out[good] = cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
        if not self._floor_agrees(d, good, out, T_base_cam):
            out[:] = np.nan
        return out

    def _floor_agrees(self, d, good, out, T_base_cam):
        """False when the floor in the pair is not where it is (FLOOR_*): one camera has moved against
        the other. Without enough floor in view, the last verdict."""
        R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
        dz = self.rays @ R[2]  # base-frame z of each ray's direction
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(dz < -1e-6, (FLOOR_BASE_Z - t[2]) / dz, np.nan)  # rectified z of the floor
            d_floor = (self.q[2, 3] / depth - self.q[3, 3]) / self.q[3, 2]
            near = good & (np.abs(d - d_floor) < FLOOR_NEAR_PX)
        near[: d.shape[0] // 2] = False
        if near.sum() < FLOOR_MIN_PIX:
            return self.trusted
        self.floor_z = float(np.median(out[near][:, 2]))
        trusted = abs(self.floor_z - FLOOR_BASE_Z) <= FLOOR_TOL_M
        if not trusted:
            print(f"  双目对不上地面：地面量成离地 {self.floor_z:+.3f} m（应为 {FLOOR_BASE_Z:+.3f}），"
                  "两个头部相机可能松动错位了，这次不用双目", flush=True)
        self.trusted = trusted
        return trusted
