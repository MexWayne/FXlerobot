"""Yesense (元生) IMU on a USB serial port: a reader thread that keeps the newest heading.

The module streams 67-byte frames at about 200 Hz, 460800 baud 8N1:
    59 53 | TID (u16) | LEN (u8) | LEN bytes of [id, size, data...] | CK1 CK2
CK1/CK2 are a running sum and a sum of sums over TID..payload. The payload carries
    0x10 acceleration  3 x int32, 1e-6 m/s^2
    0x20 angular rate  3 x int32, 1e-6 deg/s
    0x40 euler angles  pitch, roll, yaw as int32, 1e-6 deg
    0x41 quaternion    w, x, y, z as int32, 1e-6
There is no magnetometer, so yaw is relative to wherever the module powered up and drifts slowly.
Only differences of yaw over seconds to minutes are used.

    python imu.py          # print rate, checksum failures and the newest values for a few seconds
"""

import glob
import math
import os
import struct
import threading
import time

import serial

BAUD = 460800
KNOWN_SERIAL = "5B32028405"   # the CH9102 of this module; the ttyACM number changes with the USB port
SCALE = 1e-6
MAX_STEP_DEG = 15.0  # a bigger yaw change between two consecutive frames is a glitch, not motion


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def _checksum(data):
    ck1 = ck2 = 0
    for b in data:
        ck1 = (ck1 + b) & 0xFF
        ck2 = (ck2 + ck1) & 0xFF
    return ck1, ck2


def parse_frames(buf):
    """Pull complete valid frames out of `buf` (a bytearray, consumed in place).

    Returns (frames, bad): frames is a list of dicts, bad counts checksum failures.
    """
    frames, bad = [], 0
    while True:
        i = buf.find(b"\x59\x53")
        if i < 0:
            del buf[:-1]  # a lone 0x59 may be the start of the next header
            break
        if i:
            del buf[:i]
        if len(buf) < 5:
            break
        n = buf[4]
        total = 5 + n + 2
        if len(buf) < total:
            break
        body = bytes(buf[2:5 + n])
        if _checksum(body) != (buf[5 + n], buf[6 + n]):
            bad += 1
            del buf[:2]  # a header inside the data, not a real one: skip it and look again
            continue
        out = {"tid": buf[2] | (buf[3] << 8)}
        p = 5
        end = 5 + n
        while p + 2 <= end:
            rid, size = buf[p], buf[p + 1]
            data = bytes(buf[p + 2:p + 2 + size])
            p += 2 + size
            if rid in (0x10, 0x20, 0x40) and size == 12:
                out[rid] = tuple(v * SCALE for v in struct.unpack("<3i", data))
            elif rid == 0x41 and size == 16:
                out[rid] = tuple(v * SCALE for v in struct.unpack("<4i", data))
        frames.append(out)
        del buf[:total]
    return frames, bad


def candidate_ports():
    """The module's own port first, then any other 1a86 port that is not the base/neck or an arm."""
    skip = set()
    for link in ["/dev/base_neck"] + glob.glob("/dev/arm_*"):
        if os.path.exists(link):
            skip.add(os.path.realpath(link))
    known = [p for p in glob.glob(f"/dev/serial/by-id/*{KNOWN_SERIAL}*")]
    others = [p for p in sorted(glob.glob("/dev/serial/by-id/*1a86*")) if p not in known]
    return [p for p in known + others if os.path.realpath(p) not in skip]


def open_port(port):
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, BAUD, 0.05
    ser.dtr = False  # on these boards DTR/RTS can reset the module
    ser.rts = False
    ser.open()
    return ser


def find_port(port=None, seconds=1.0):
    """Open `port`, or look for the candidate that streams valid frames. Returns the path."""
    ports = [port] if port else candidate_ports()
    for p in ports:
        try:
            ser = open_port(p)
        except (serial.SerialException, OSError):
            continue
        buf, t0, ok = bytearray(), time.time(), 0
        try:
            while time.time() - t0 < seconds and ok < 5:
                buf.extend(ser.read(512))
                frames, _ = parse_frames(buf)
                ok += len(frames)
        finally:
            ser.close()
        if ok >= 5:
            return p
    return None


class Imu:
    """Newest heading, rate and attitude; yaw is unwrapped so it can run past +-180 deg."""

    STALE_S = 0.3

    def __init__(self, port=None):
        self.port = find_port(port)
        if self.port is None:
            raise RuntimeError("找不到 IMU：没有哪个串口在 460800 波特率上输出 59 53 帧")
        self.sign = 1.0   # +1 when the module's yaw grows in the direction the robot calls left
        self.frames = 0
        self.bad = 0
        self.jumps = 0
        self._lock = threading.Lock()
        self._raw = None
        self._yaw = 0.0
        self._t = 0.0
        self._rate = 0.0
        self._euler = (0.0, 0.0, 0.0)
        self._accel = (0.0, 0.0, 0.0)
        self._quat = (1.0, 0.0, 0.0, 0.0)
        self._stop = threading.Event()
        self._ser = open_port(self.port)
        self._thread = threading.Thread(target=self._loop, name="imu", daemon=True)
        self._thread.start()

    def _loop(self):
        buf = bytearray()
        while not self._stop.is_set():
            try:
                data = self._ser.read(512)
            except (serial.SerialException, OSError):
                break
            if not data:
                continue
            buf.extend(data)
            frames, bad = parse_frames(buf)
            if len(buf) > 4096:
                del buf[:-512]
            now = time.monotonic()
            with self._lock:
                self.bad += bad
                for f in frames:
                    if 0x40 not in f:
                        continue
                    yaw = f[0x40][2]
                    if self._raw is None:
                        self._yaw = yaw
                    else:
                        step = _wrap(yaw - self._raw)
                        if abs(step) > MAX_STEP_DEG:
                            # 200 frames a second: this would be a turn of thousands of deg/s. The
                            # module restarted its heading, so keep the heading and follow from here.
                            self.jumps += 1
                            print(f"  IMU 航向跳变 {step:+.1f}°，已忽略（累计 {self.jumps} 次）", flush=True)
                            step = 0.0
                        self._yaw += step
                    self._raw = yaw
                    self._euler = f[0x40]
                    if 0x20 in f:
                        self._rate = f[0x20][2]
                    if 0x10 in f:
                        self._accel = f[0x10]
                    if 0x41 in f:
                        self._quat = f[0x41]
                    self._t = now
                    self.frames += 1

    def yaw_deg(self):
        """Continuous heading in degrees, positive towards the robot's left."""
        with self._lock:
            return self.sign * self._yaw

    def rate_dps(self):
        with self._lock:
            return self.sign * self._rate

    def age(self):
        """Seconds since the last valid frame."""
        with self._lock:
            t = self._t
        return math.inf if t == 0.0 else time.monotonic() - t

    def fresh(self):
        return self.age() < self.STALE_S

    def snapshot(self):
        with self._lock:
            return {"euler": self._euler, "accel": self._accel, "quat": self._quat,
                    "rate": self._rate, "frames": self.frames, "bad": self.bad}

    def wait_ready(self, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.fresh():
                return True
            time.sleep(0.02)
        return False

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._ser.close()
        except Exception:
            pass


def main():
    imu = Imu()
    print(f"IMU 串口 {imu.port}")
    if not imu.wait_ready():
        raise SystemExit("3 秒内没有收到有效帧")
    f0, t0 = imu.snapshot()["frames"], time.time()
    time.sleep(3.0)
    s = imu.snapshot()
    hz = (s["frames"] - f0) / (time.time() - t0)
    q = s["quat"]
    print(f"{hz:.0f} Hz，累计 {s['frames']} 帧，校验失败 {s['bad']} 次")
    print(f"pitch {s['euler'][0]:+.2f}°  roll {s['euler'][1]:+.2f}°  yaw {s['euler'][2]:+.2f}°")
    print(f"加速度 {s['accel'][0]:+.2f} {s['accel'][1]:+.2f} {s['accel'][2]:+.2f} m/s²  "
          f"四元数模长 {math.sqrt(sum(v * v for v in q)):.4f}  角速度 z {s['rate']:+.2f} °/s")
    imu.close()


if __name__ == "__main__":
    main()
