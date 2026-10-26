"""RPLIDAR on the robot. One full turn is a set of points in the base frame: x forward, y left.

Glass and mirrors are not in these points. A return from glass is either a reflection or
nothing, so the caller drops every return that lies in a direction the camera called glass.
"""

import glob
import os
import threading
import time

import numpy as np
import serial

BAUD = 460800
# C1: ignore the robot body inside 0.25 m. White targets are valid out to 12 m.
RMIN, RMAX = 0.25, 12.0
EXTENT_BIN_DEG = 5.0
# A return this much closer than the surrounding wall is something standing in the room, not the wall.
INSET_M = 0.40


def find_port():
    for path in sorted(glob.glob("/dev/serial/by-id/*CP2102*")):
        return path
    if os.path.exists("/dev/ttyUSB0"):
        return "/dev/ttyUSB0"
    raise RuntimeError("没有找到雷达的串口（CP2102）")


class Lidar:
    def __init__(self, port=None):
        self.port = port or find_port()
        self._lock = threading.Lock()
        self._xy = None
        self._stamp = 0.0
        self._stop = False
        self._thread = None
        self._error = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="lidar", daemon=True)
        self._thread.start()
        deadline = time.time() + 9.0
        while time.time() < deadline and self.latest_xy() is None:
            if self._error:
                raise RuntimeError(self._error)
            time.sleep(0.1)
        if self.latest_xy() is None:
            self.close()
            raise RuntimeError("雷达没有扫描数据")

    def close(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(3.0)

    def extent(self, shots=4):
        """Rough room outline from a few full turns, plus objects that stand inside it.

        Both are in the base frame (x forward, y left). The outline is the nearest return in
        each direction, except where one direction sticks in: that return is an object and the
        outline keeps the farther surrounding range. Glass is wrong here, either missing or
        mirrored; the camera corrects those directions later.
        """
        scans, seen = [], None
        deadline = time.time() + 3.0
        while len(scans) < shots and time.time() < deadline:
            xy = self.latest_xy()
            if xy is not None and xy is not seen:
                scans.append(xy)
                seen = xy
            time.sleep(0.15)
        if not scans:
            raise RuntimeError("雷达没有一整圈数据")
        return _extent(np.vstack(scans))

    def latest_xy(self):
        """Nx2 array of the newest full turn, or None when it is missing or older than a second."""
        with self._lock:
            if self._xy is None or time.time() - self._stamp > 1.0:
                return None
            return self._xy.copy()

    def _run(self):
        try:
            link = serial.Serial(self.port, BAUD, timeout=1)
        except serial.SerialException as err:
            self._error = str(err)
            return
        try:
            link.dtr = link.rts = False
            link.write(b"\xa5\x25")
            time.sleep(3.0)  # the motor must be spinning before scan points come out
            link.reset_input_buffer()
            link.write(b"\xa5\x20")
            if link.read(7)[:2] != b"\xa5\x5a":
                self._error = "雷达没有返回扫描描述符"
                return
            cur, buf, prev = [], b"", None
            while not self._stop:
                buf += link.read(max(5, min(link.in_waiting, 4096)))
                while len(buf) >= 5:
                    packet = buf[:5]
                    if (packet[0] & 1) == ((packet[0] >> 1) & 1) or not (packet[1] & 1):
                        buf = buf[1:]
                        continue
                    buf = buf[5:]
                    ang = ((packet[1] >> 1) | (packet[2] << 7)) / 64.0
                    if prev is not None and ang < prev - 180 and len(cur) > 60:
                        self._store(np.array(cur, dtype=np.float32))
                        cur = []
                    prev = ang
                    dist = (packet[3] | (packet[4] << 8)) / 4000.0
                    if RMIN < dist < RMAX:
                        cur.append((ang, dist))
        finally:
            try:
                link.write(b"\xa5\x25")
            except serial.SerialException:
                pass
            link.close()

    def _store(self, scan):
        # The angle increases clockwise and 0 is the nose. Flip it so y is left.
        ang = -np.deg2rad(scan[:, 0])
        dist = scan[:, 1]
        xy = np.c_[dist * np.cos(ang), dist * np.sin(ang)]
        with self._lock:
            self._xy = xy
            self._stamp = time.time()


def _extent(xy):
    """Returns (outline, objects), each a list of (x, y)."""
    n = int(round(360.0 / EXTENT_BIN_DEG))
    nearest = [None] * n
    for x, y in xy:
        dist = float(np.hypot(x, y))
        i = int((np.degrees(np.arctan2(y, x)) % 360.0) // EXTENT_BIN_DEG) % n
        if nearest[i] is None or dist < nearest[i][0]:
            nearest[i] = (dist, float(x), float(y))
    known = [i for i, item in enumerate(nearest) if item is not None]
    if len(known) < n // 3:
        raise RuntimeError("雷达只有 %d 个方向有回波" % len(known))
    for i in range(n):
        if nearest[i] is not None:
            continue
        prev_i = max((k for k in known if k < i), default=known[-1])
        next_i = min((k for k in known if k > i), default=known[0])
        nearest[i] = nearest[prev_i] if abs(i - prev_i) <= abs(i - next_i) else nearest[next_i]
    outline, objects = [], []
    for i, (dist, x, y) in enumerate(nearest):
        window = [nearest[(i + k) % n][0] for k in range(-4, 5) if k != 0]
        wall = float(np.median(window))
        ang = np.deg2rad((i + 0.5) * EXTENT_BIN_DEG)
        if dist < wall - INSET_M:
            objects.append((x, y))
            dist = wall
        outline.append((dist * np.cos(ang), dist * np.sin(ang)))
    return outline, objects
