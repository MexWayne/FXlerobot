#!/usr/bin/env python3
"""原地旋转五圈，叠加激光雷达与双目深度的原始点云。

用法（在 Jetson 终端里）:
    cd ~/workspaces/FXLerobot/FXlerobot
    /home/jetson/anaconda3/envs/lerobot_env/bin/python project_tool/imu_lidar_test.py

脚本自动低速旋转、用 IMU 航向将两个传感器的点变换到起始车体坐标，
然后保存 overlay_map_zoom3m.png 和 raw_point_clouds.npz。
如需提前停车，在终端按 Ctrl-C。
"""
import math
import os
import sys
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import serial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lerobot.robots.lekiwi import LeKiwiBase  # noqa: E402
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig  # noqa: E402

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VLN_DIR = os.path.join(PROJECT, "vln")
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), VLN_DIR]
import calibrate_head_cameras as cal  # noqa: E402
import validate_head_extrinsic as val  # noqa: E402
from stereo import Stereo  # noqa: E402

LIDAR_PORT, LIDAR_BAUD = "/dev/ttyUSB0", 460800
IMU_PORT, IMU_BAUD = "/dev/imu", 460800
BASE_PORT = "/dev/base_neck"
RMIN, RMAX = 0.25, 6.0
G = 9.80665


class ImuReader(threading.Thread):
    """Yesense IMU。帧: 'YS' + tid(2) + len(1) + TLV 负载 + 校验(2), 数值均为 int32 * 1e-6。
    0x10 加速度 m/s², 0x20 角速度 deg/s, 0x40 欧拉角 pitch/roll/yaw deg (IMU 内部融合), 0x41 四元数。
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.acc = np.zeros(3)       # m/s²
        self.gyro = np.zeros(3)      # rad/s
        self.euler = np.zeros(3)     # pitch, roll, yaw (rad)
        self.yaw0 = None
        self.yaw_unwrapped = 0.0     # 相对启动时的航向, 已展开, rad
        self.n = 0
        self.bad = 0
        self.stop_flag = False

    @staticmethod
    def _checksum_ok(f):
        c1 = c2 = 0
        for b in f[2:-2]:
            c1 = (c1 + b) & 0xFF
            c2 = (c2 + c1) & 0xFF
        return f[-2] == c1 and f[-1] == c2

    def _handle(self, f):
        fields, pos, end = {}, 5, 5 + f[4]
        while pos + 2 <= end:
            did, dlen = f[pos], f[pos + 1]
            fields[did] = np.frombuffer(f[pos + 2:pos + 2 + dlen], dtype="<i4") * 1e-6
            pos += 2 + dlen
        if 0x40 not in fields:
            return
        pitch, roll, yaw = np.deg2rad(fields[0x40])
        with self.lock:
            if self.yaw0 is None:
                self.yaw0 = prev = yaw
            else:
                prev = self.yaw0 + self.yaw_unwrapped
            d = (yaw - prev + math.pi) % (2 * math.pi) - math.pi
            self.yaw_unwrapped += d
            self.euler = np.array([pitch, roll, yaw])
            if 0x10 in fields:
                self.acc = fields[0x10]
            if 0x20 in fields:
                self.gyro = np.deg2rad(fields[0x20])
            self.n += 1

    def run(self):
        s = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0.3)
        buf = b""
        while not self.stop_flag:
            buf += s.read(1024)
            if len(buf) > 20000:
                buf = buf[-4000:]
            while True:
                i = buf.find(b"YS")
                if i < 0 or len(buf) < i + 5:
                    break
                flen = 5 + buf[i + 4] + 2
                if len(buf) < i + flen:
                    break
                f = buf[i:i + flen]
                if not self._checksum_ok(f):
                    self.bad += 1
                    buf = buf[i + 2:]
                    continue
                buf = buf[i + flen:]
                self._handle(f)
        s.close()

    @property
    def ready(self):
        return self.n > 0

    def yaw(self):
        with self.lock:
            return self.yaw_unwrapped

    def snapshot(self):
        with self.lock:
            pitch, roll, _ = self.euler
            return roll, pitch, self.yaw_unwrapped, self.acc / G, self.gyro.copy(), self.n


class LidarReader(threading.Thread):
    """RPLIDAR 标准扫描, 每完成一圈把点 (角度°, 距离 m) 放进队列。"""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.scans = []
        self.stop_flag = False

    def run(self):
        s = serial.Serial(LIDAR_PORT, LIDAR_BAUD, timeout=1)
        s.dtr = s.rts = False
        s.write(b"\xa5\x25"); time.sleep(3.0); s.reset_input_buffer()  # 等马达转稳, 否则不输出扫描点
        s.write(b"\xa5\x20")
        if s.read(7)[:2] != b"\xa5\x5a":
            print("雷达没有返回扫描描述符", flush=True)
            return
        cur, buf, prev_ang = [], b"", None
        while not self.stop_flag:
            buf += s.read(max(5, min(s.in_waiting, 4096)))
            while len(buf) >= 5:
                b = buf[:5]
                if (b[0] & 1) == ((b[0] >> 1) & 1) or not (b[1] & 1):
                    buf = buf[1:]
                    continue
                buf = buf[5:]
                ang = ((b[1] >> 1) | (b[2] << 7)) / 64.0
                # 角度回绕才是真正的新一圈。数据里的起始标志每圈出现几十次, 不能用它切分
                if prev_ang is not None and ang < prev_ang - 180 and len(cur) > 60:
                    with self.lock:
                        self.scans.append(np.array(cur))
                    cur = []
                prev_ang = ang
                d = (b[3] | (b[4] << 8)) / 4000.0
                if RMIN < d < RMAX:
                    cur.append((ang, d))
        s.write(b"\xa5\x25"); s.close()

    def take(self):
        with self.lock:
            out, self.scans = self.scans, []
        return out


def to_xy(scan):
    # RPLIDAR 角度顺时针增加, 0° 为车头; 转成 x 向前、y 向左的右手坐标
    a = -np.deg2rad(scan[:, 0]); r = scan[:, 1]
    return np.c_[r * np.cos(a), r * np.sin(a)]


def render_overlay(lidar_xy, stereo_xy, turns_done):
    """原始点云俯视叠加图：蓝色雷达、洋红色双目，x 向前、y 向左。"""
    size, extent = 1400, 3.0
    pad = 90
    img = np.full((size, size, 3), 255, np.uint8)
    scale = (size - 2 * pad) / (2 * extent)

    def pixel(x, y):
        return int(pad + (x + extent) * scale), int(size - pad - (y + extent) * scale)

    for m in range(-3, 4):
        p0, p1 = pixel(m, -extent), pixel(m, extent)
        q0, q1 = pixel(-extent, m), pixel(extent, m)
        cv2.line(img, p0, p1, (225, 225, 225), 1)
        cv2.line(img, q0, q1, (225, 225, 225), 1)
        cv2.putText(img, str(m), (p0[0] - 8, size - pad + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
        cv2.putText(img, str(m), (pad - 38, q0[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    cv2.rectangle(img, pixel(-extent, extent), pixel(extent, -extent), (20, 20, 20), 2)

    def scatter(points, color, radius):
        if not len(points):
            return
        # 只画图窗内的点，保存的 npz 仍保留完整累计数据。
        pts = np.asarray(points)
        keep = np.isfinite(pts[:, :2]).all(1) & (np.abs(pts[:, 0]) <= extent) & (np.abs(pts[:, 1]) <= extent)
        for x, y in pts[keep, :2][::2]:
            cv2.circle(img, pixel(float(x), float(y)), radius, color, -1, cv2.LINE_AA)

    scatter(lidar_xy, (205, 125, 35), 2)      # blue
    scatter(stereo_xy, (205, 55, 205), 2)    # magenta
    ox, oy = pixel(0, 0)
    cv2.drawMarker(img, (ox, oy), (0, 0, 0), cv2.MARKER_TRIANGLE_UP, 28, 3)
    cv2.putText(img, "x forward (m)", (size // 2 - 90, size - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "y left (m)", (12, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, f"5-turn raw point-cloud overlay | turns {turns_done:.2f}/5",
                (pad, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    cv2.circle(img, (size - 310, 42), 5, (205, 125, 35), -1)
    cv2.putText(img, "LiDAR", (size - 292, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.circle(img, (size - 180, 42), 5, (205, 55, 205), -1)
    cv2.putText(img, "Stereo", (size - 162, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return img


def main():
    imu, lidar = ImuReader(), LidarReader()
    imu.start(); lidar.start()
    t0 = time.time()
    while not imu.ready and time.time() - t0 < 5:
        time.sleep(0.1)
    if not imu.ready:
        raise SystemExit("IMU 没有数据, 检查 /dev/imu 是否被其他程序占用")
    print("IMU 已就绪", flush=True)

    left = right = robot = None
    lidar_world, stereo_world, frame_log = [], [], []
    yaw_sign = 1.0
    turns_done = 0.0
    try:
        print("初始化双目相机和标定...", flush=True)
        calib, extr, cams = val.load(cal.DEFAULT_DATA)
        left = cal.open_camera(cal.CAMS["left"], calib.get("focus_absolute", 0))
        right = cal.open_camera(cal.CAMS["right"], calib.get("focus_absolute", 0))
        stereo = Stereo(calib, cams)
        robot = LeKiwiBase(LeKiwiConfig(port=BASE_PORT, id="kiwi_base", cameras={}))
        print("连接底盘...", flush=True)
        robot.connect(calibrate=False)

        # 先低速试转一秒，自动识别 IMU 正方向；随后从静止航向计满 5 圈。
        print("将原地低速旋转 5 圈；请确认车身四周有足够净空。", flush=True)
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 10.0})
        yaw_start = imu.yaw()
        time.sleep(1.0)
        moved = imu.yaw() - yaw_start
        if abs(moved) < math.radians(2):
            raise RuntimeError(f"旋转测试中 IMU 只变化 {math.degrees(moved):.1f}°，停止以免失控")
        yaw_sign = 1.0 if moved > 0 else -1.0
        robot.stop_base()
        time.sleep(0.3)
        yaw_ref = imu.yaw()
        print(f"IMU 旋转方向 {yaw_sign:+.0f}，已开始累计 5 圈。", flush=True)
        out_dir = os.path.join(os.path.dirname(__file__),
                               "sensor_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        next_camera = 0.0
        deadline = time.monotonic() + 300.0
        while turns_done < 5.0 and time.monotonic() < deadline:
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 10.0})
            yaw_now = imu.yaw()
            angle = yaw_sign * (yaw_now - yaw_ref)
            turns_done = max(0.0, angle / (2.0 * math.pi))
            scans = lidar.take()
            if len(scans) > 4:
                scans = scans[-4:]
            for scan in scans:
                if len(scan) < 60:
                    continue
                yaw = yaw_sign * (imu.yaw() - yaw_ref)
                xy = to_xy(scan)
                c, s = math.cos(yaw), math.sin(yaw)
                lidar_world.extend((xy @ np.array([[c, s], [-s, c]])).tolist())

            now = time.monotonic()
            if now >= next_camera:
                for _ in range(2):
                    left.grab(); right.grab()
                ok_l, a = left.retrieve()
                ok_r, b = right.retrieve()
                if ok_l and ok_r:
                    points = stereo.points(a, b, cams["left"]["T_base_cam"])
                    points = points[np.isfinite(points).all(axis=-1)]
                    # 与点云俯视图一致：仅画可作为立体障碍的高度及 3 m 范围。
                    points = points[(points[:, 2] >= 0.07) & (points[:, 2] <= 0.85)
                                    & (np.hypot(points[:, 0], points[:, 1]) <= 3.0)]
                    yaw = yaw_sign * (imu.yaw() - yaw_ref)
                    c, s = math.cos(yaw), math.sin(yaw)
                    xy = points[:, :2] @ np.array([[c, -s], [s, c]]).T
                    stereo_world.extend(np.c_[xy, points[:, 2]].tolist()[::8])
                    frame_log.append((time.time(), turns_done, len(points)))
                next_camera = now + 0.35
            if int(turns_done * 10) != int(max(0.0, (turns_done - 0.006)) * 10):
                print(f"旋转进度：{turns_done:.1f}/5 圈，雷达 {len(lidar_world)} 点，双目 {len(stereo_world)} 点",
                      flush=True)
            time.sleep(0.03)
        robot.stop_base()
        if turns_done < 5.0:
            print(f"达到安全超时，完成 {turns_done:.2f}/5 圈。保存已有数据。", flush=True)
        lidar_arr = np.asarray(lidar_world, dtype=np.float32).reshape(-1, 2)
        stereo_arr = np.asarray(stereo_world, dtype=np.float32).reshape(-1, 3)
        np.savez_compressed(os.path.join(out_dir, "raw_point_clouds.npz"),
                            lidar_xy=lidar_arr, stereo_xyz=stereo_arr,
                            frames=np.asarray(frame_log, dtype=np.float64),
                            turns=np.asarray([turns_done], dtype=np.float32))
        overlay = render_overlay(lidar_arr, stereo_arr, turns_done)
        image_path = os.path.join(out_dir, "overlay_map_zoom3m.png")
        cv2.imwrite(image_path, overlay)
        print(f"叠加图：{image_path}", flush=True)
        print(f"原始点云：{os.path.join(out_dir, 'raw_point_clouds.npz')}", flush=True)
    finally:
        print("停车...", flush=True)
        try:
            if robot is not None:
                robot.stop_base()
                robot.disconnect()
        except Exception as e:
            print(e)
        imu.stop_flag = lidar.stop_flag = True
        if left is not None:
            left.release()
        if right is not None:
            right.release()
        print("IMU 和雷达已停止。", flush=True)


if __name__ == "__main__":
    main()
