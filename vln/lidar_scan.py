"""RPLIDAR on the robot. One full turn is a set of points in the base frame: x forward, y left,
origin at the base centre (the lidar itself is LIDAR_X_M ahead of it and LIDAR_Y_M to the side).

Floor-standing glass and mirrors are not in these points: a return there is a reflection or what
stands behind. The map stops rays where the stereo measured such glass (occmap.OccMap.insert_scan).
"""

import glob
import os
import threading
import time

import numpy as np
import serial

BAUD = 460800
# Closing: stop commands every STOP_EVERY_S until nothing has arrived for QUIET_S, at most STOP_MAX_S.
STOP_EVERY_S = 0.5
QUIET_S = 0.3
STOP_MAX_S = 4.0
SPIN_UP_S = 3.0      # the motor must be spinning before scan points come out
DESCRIPTOR = b"\xa5\x5a"
# C1: ignore the robot body inside 0.25 m. White targets are valid out to 12 m.
RMIN, RMAX = 0.25, 12.0
# The body blocks the lidar behind: back there it only returns the robot itself (a steady 0.26 m
# echo at -140°) or nothing. Returns farther round from the nose than this are dropped.
VIEW_DEG = 100.0
# The lidar sits this far ahead of the base centre (measured, 2026-11-02). RMIN and VIEW_DEG apply in
# the lidar's own frame; the points handed out are moved into the base frame. Checked against the
# head cameras: the left arm seen by both was 12 cm apart in x before this shift.
LIDAR_X_M = 0.12
# The lidar is also off the centre line, to the right. A tape cannot reach it from the side, so this
# is fitted (2026-11-02): two cups seen by the lidar and by the head stereo were 2.4 cm apart in y
# and agreed in x once the lidar moved this far right. The same cups, located on the ChArUco board
# whose near squares are 47.5 mm from the bumper, put the lidar 45 mm behind the bumper.
LIDAR_Y_M = -0.024


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
        self._raw = None  # the newest full turn as it came (angle deg, range m), for checking VIEW_DEG
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
            self._thread.join(STOP_MAX_S + 4.0)  # the reader drains the stream before it closes the port

    def scans(self, n=2, timeout=2.0):
        """The next n full turns (each N x 2, base frame), for a robot standing still. [] on timeout."""
        out, seen = [], self._stamp
        deadline = time.time() + timeout
        while len(out) < n and time.time() < deadline:
            with self._lock:
                stamp, xy = self._stamp, self._xy
            if stamp != seen and xy is not None:
                out.append(xy.copy())
                seen = stamp
            time.sleep(0.02)
        return out if len(out) == n else []

    def latest_xy(self):
        """Nx2 array of the newest full turn, or None when it is missing or older than a second."""
        with self._lock:
            if self._xy is None or time.time() - self._stamp > 1.0:
                return None
            return self._xy.copy()

    def _run(self):
        try:
            # A write that cannot go out (the CP2102 hung) must not block the reader for good.
            link = serial.Serial(self.port, BAUD, timeout=1, write_timeout=1.0)
        except serial.SerialException as err:
            self._error = str(err)
            return
        try:
            link.dtr = link.rts = False
            link.write(b"\xa5\x25")
            # While the motor spins up, whatever the lidar still sends is read out, not purged: a
            # purge (request 0x12) with data piling up unread times out and hangs the CP2102
            # (2026-11-03, after a 3 s sleep here).
            link.timeout = 0.05
            t_end = time.time() + SPIN_UP_S
            while time.time() < t_end:
                link.read(4096)
            link.timeout = 1
            link.write(b"\xa5\x20")
            # The scan descriptor (7 bytes) starts with A5 5A; anything still in front of it is skipped.
            buf, t_end = b"", time.time() + 2.0
            while DESCRIPTOR not in buf and time.time() < t_end:
                buf += link.read(64)
            if DESCRIPTOR not in buf:
                self._error = "雷达没有返回扫描描述符"
                return
            buf = buf[buf.index(DESCRIPTOR) + 7:]
            cur, prev = [], None
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
            # Stop the scan and read out what is still arriving before closing. Closed while the
            # lidar still streams at 460800 baud, the CP2102's purge and disable requests time out
            # (dmesg: "failed set request 0x12 status: -110") and the next open fails with an I/O
            # error until the USB cable is replugged. Reading for a fixed 2 s and then purging did
            # that too (2026-11-03): the stream had not ended. So the stop is sent again until the
            # line has been quiet for QUIET_S, and there is no purge (request 0x12) at all.
            try:
                link.timeout = 0.05
                t0 = time.time()
                sent = quiet_since = None
                while time.time() - t0 < STOP_MAX_S:
                    now = time.time()
                    if sent is None or now - sent >= STOP_EVERY_S:
                        link.write(b"\xa5\x25")
                        sent = now
                    if link.read(4096):
                        quiet_since = None
                    elif quiet_since is None:
                        quiet_since = now
                    elif now - quiet_since >= QUIET_S:
                        break
                else:
                    print("  雷达停不下来：关串口后可能要拔插雷达 USB", flush=True)
            except serial.SerialException:
                pass
            link.close()

    def latest_raw(self):
        """Nx2 (angle deg clockwise from the nose, range m) of the newest turn, all round, or None."""
        with self._lock:
            return None if self._raw is None else self._raw.copy()

    def _store(self, scan):
        with self._lock:
            self._raw = np.asarray(scan, float)
        # The angle increases clockwise and 0 is the nose. Flip it so y is left.
        ang = -np.deg2rad(scan[:, 0])
        front = np.abs(np.arctan2(np.sin(ang), np.cos(ang))) <= np.deg2rad(VIEW_DEG)
        ang, dist = ang[front], scan[front, 1]
        xy = np.c_[dist * np.cos(ang) + LIDAR_X_M, dist * np.sin(ang) + LIDAR_Y_M]
        with self._lock:
            self._xy = xy
            self._stamp = time.time()
