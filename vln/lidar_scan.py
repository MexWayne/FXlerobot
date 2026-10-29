"""RPLIDAR on the robot. One full turn is a set of points in the base frame: x forward, y left.

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
# C1: ignore the robot body inside 0.25 m. White targets are valid out to 12 m.
RMIN, RMAX = 0.25, 12.0
# The body blocks the lidar behind: back there it only returns the robot itself (a steady 0.26 m
# echo at -140°) or nothing. Returns farther round from the nose than this are dropped.
VIEW_DEG = 100.0


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
            self._thread.join(5.0)  # the reader drains the stream before it closes the port

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
            # Stop the scan and read out what is still arriving before closing. Closed while the
            # lidar still streams at 460800 baud, the CP2102's purge and disable requests time out
            # (dmesg: "failed set request 0x12 status: -110") and the next open fails with an I/O
            # error until the USB cable is replugged.
            try:
                link.write(b"\xa5\x25")
                link.timeout = 0.05
                quiet = time.time() + 2.0
                while time.time() < quiet and link.read(4096):
                    pass
                time.sleep(0.05)
                link.reset_input_buffer()
            except serial.SerialException:
                pass
            link.close()

    def _store(self, scan):
        # The angle increases clockwise and 0 is the nose. Flip it so y is left.
        ang = -np.deg2rad(scan[:, 0])
        front = np.abs(np.arctan2(np.sin(ang), np.cos(ang))) <= np.deg2rad(VIEW_DEG)
        ang, dist = ang[front], scan[front, 1]
        xy = np.c_[dist * np.cos(ang), dist * np.sin(ang)]
        with self._lock:
            self._xy = xy
            self._stamp = time.time()
