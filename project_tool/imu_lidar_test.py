#!/usr/bin/env python3
"""实时显示 IMU 姿态和激光雷达建图, 不控制底盘, 小车保持静止。

用法 (在 Jetson 桌面的终端里):
    cd ~/workspaces/FXLerobot/FXlerobot
    /home/jetson/anaconda3/envs/lerobot_env/bin/python project_tool/imu_lidar_test.py

窗口左侧是栅格地图(黑=障碍, 白=空地, 红线=轨迹, 绿点=起点),
右侧是 IMU 姿态: 航向圆盘、俯仰横滚条和加速度、角速度数值。
画面通过网页提供, 在 Jetson 本机或同一网络的电脑浏览器打开
http://<jetson ip>:8000 查看。
在运行它的终端里按住键控制底盘, 松开即停: w/s 前进后退, a/d 左右平移,
z/x 转向, r/f 加减速度档, q 退出。
"""
import math
import os
import sys
import time
import select
import termios
import tty
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import serial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lerobot.robots.lekiwi import LeKiwiBase  # noqa: E402
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig  # noqa: E402

LIDAR_PORT, LIDAR_BAUD = "/dev/ttyUSB0", 460800
IMU_PORT, IMU_BAUD = "/dev/imu", 460800
BASE_PORT = "/dev/base_neck"
SPEEDS = [{"lin": 0.1, "rot": 30.0}, {"lin": 0.2, "rot": 45.0}, {"lin": 0.3, "rot": 70.0}]

RES, GRID = 0.05, 400          # 5cm 栅格, 20m x 20m
# 实测 0.08~0.15m 的大量回波来自机器人自身/安装结构。柜子从约 0.39m 开始。
RMIN, RMAX = 0.25, 6.0         # 远处零星回波会拉出长射线, 抹掉已建好的障碍
G = 9.80665

# 扫描匹配的接受条件。不满足就丢弃这一圈, 位姿只按 IMU 航向更新。
ICP_MIN_INLIER_RATIO = 0.5
ICP_MAX_MEAN_ERR = 0.08        # m
ICP_MAX_YAW_DEV = math.radians(8.0)  # 与 IMU 转角的最大偏差

# 平移时用 IMU 航向闭环保持直线: 转向速度 = 增益 * 航向误差, 限幅
HEADING_KP = 2.0               # (deg/s) / deg
HEADING_MAX_RATE = 20.0        # deg/s


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


def front_distance(scan, half_width_deg=5.0):
    """车头正前方 ±5° 内回波距离的中位数, 用于和卷尺对比。"""
    a = scan[:, 0]
    m = (a < half_width_deg) | (a > 360 - half_width_deg)
    return float(np.median(scan[m, 1])) if m.any() else None


def transform(pts, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


def nearest(query, ref):
    """分块暴力最近邻, 返回每个 query 点的距离和 ref 下标。"""
    idx = np.empty(len(query), dtype=int)
    dist = np.empty(len(query))
    for i in range(0, len(query), 256):
        d2 = ((query[i:i + 256, None, :] - ref[None, :, :]) ** 2).sum(2)
        idx[i:i + 256] = np.argmin(d2, axis=1)
        dist[i:i + 256] = np.sqrt(d2[np.arange(d2.shape[0]), idx[i:i + 256]])
    return dist, idx


def icp(src, ref, pose, iters=25, maxd=0.5):
    """返回 (位姿, 内点比例, 内点平均误差 m)。"""
    T = pose.copy()
    for it in range(iters):
        # 先用宽门限收敛, 后几轮收紧以排除错误对应
        q = transform(src, T)
        d, idx = nearest(q, ref)
        m = d < (maxd if it < iters // 2 else 0.15)
        if m.sum() < 30:
            break
        A, B = q[m], ref[idx[m]]
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1] *= -1
            R = Vt.T @ U.T
        t = cb - R @ ca
        dth = math.atan2(R[1, 0], R[0, 0])
        c, s = math.cos(dth), math.sin(dth)
        T[:2] = np.array([[c, -s], [s, c]]) @ T[:2] + t
        T[2] += dth
        if math.hypot(*t) < 1e-4 and abs(dth) < 1e-4:
            break
    d, _ = nearest(transform(src, T), ref)
    inl = d < 0.15
    return T, float(inl.mean()), float(d[inl].mean()) if inl.any() else 9.0


def ray_update(grid, pose, pts):
    cx = GRID // 2
    o = np.array([pose[0] / RES + cx, pose[1] / RES + cx])
    for p in pts:
        e = np.array([p[0] / RES + cx, p[1] / RES + cx])
        n = int(np.linalg.norm(e - o)) + 1
        t = np.linspace(0, 1, n, endpoint=False)
        xs = (o[0] + (e[0] - o[0]) * t).astype(int)
        ys = (o[1] + (e[1] - o[1]) * t).astype(int)
        ok = (xs >= 0) & (xs < GRID) & (ys >= 0) & (ys < GRID)
        grid[ys[ok], xs[ok]] -= 0.4
        ex, ey = int(e[0]), int(e[1])
        if 0 <= ex < GRID and 0 <= ey < GRID:
            grid[ey, ex] += 0.9
    np.clip(grid, -5, 5, out=grid)


def render_map(grid, traj, latest_points=None, status=()):
    img = (1 - 1 / (1 + np.exp(-grid))) * 255
    vis = np.where(np.abs(grid) < 0.01, 128, img).astype(np.uint8)
    # grid[y, x] -> 屏幕: 上方 = 起始车头方向 (+x), 左方 = +y。栅格和叠加物必须用同一映射
    vis = np.ascontiguousarray(vis.T[::-1, ::-1])
    bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cx = GRID // 2

    def px(p):
        return int(GRID - 1 - (p[1] / RES + cx)), int(GRID - 1 - (p[0] / RES + cx))

    for a, b in zip(traj, traj[1:]):
        cv2.line(bgr, px(a), px(b), (0, 0, 255), 1)
    cv2.circle(bgr, px(traj[0]), 4, (0, 255, 0), -1)
    # 当前一圈原始回波直接画成洋红色，不经过占用栅格和 ICP。
    # 即使累计地图漂移，眼前的柜子/障碍仍应立即可见。
    if latest_points is not None:
        for p in latest_points:
            x, y = px(p)
            if 0 <= x < GRID and 0 <= y < GRID:
                cv2.circle(bgr, (x, y), 2, (255, 0, 255), -1)
    h = math.radians(0) if len(traj) < 2 else traj[-1][2]
    tip = px(traj[-1][:2] + 0.3 * np.array([math.cos(h), math.sin(h)]))
    cv2.arrowedLine(bgr, px(traj[-1]), tip, (0, 128, 255), 2, tipLength=0.4)
    cv2.putText(
        bgr, "magenta = current raw scan", (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1
    )
    for i, line in enumerate(status):
        cv2.putText(bgr, line, (8, 40 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 160), 1)
    return bgr


def render_imu(roll, pitch, yaw, acc, gyro, n, front=None):
    p = np.full((640, 420, 3), 245, np.uint8)
    cv2.putText(p, "IMU", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    c = (210, 250)
    cv2.circle(p, c, 120, (180, 180, 180), 2)
    for deg in range(0, 360, 30):
        a = math.radians(deg - 90)
        cv2.line(p, (int(c[0] + 112 * math.cos(a)), int(c[1] + 112 * math.sin(a))),
                 (int(c[0] + 120 * math.cos(a)), int(c[1] + 120 * math.sin(a))), (120, 120, 120), 2)
    a = -yaw - math.pi / 2  # 屏幕上方为车头
    cv2.arrowedLine(p, c, (int(c[0] + 100 * math.cos(a)), int(c[1] + 100 * math.sin(a))),
                    (0, 0, 220), 3, tipLength=0.25)
    cv2.putText(p, f"yaw {math.degrees(yaw):7.1f} deg", (120, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y0 = 430
    for name, val, lim in (("roll", math.degrees(roll), 45), ("pitch", math.degrees(pitch), 45)):
        cv2.putText(p, f"{name} {val:6.1f}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.rectangle(p, (170, y0 - 14), (400, y0 + 4), (200, 200, 200), -1)
        x = int(285 + np.clip(val / lim, -1, 1) * 110)
        cv2.rectangle(p, (285, y0 - 14), (x, y0 + 4), (40, 140, 40), -1)
        y0 += 36
    txt = [f"acc  {acc[0]:6.2f} {acc[1]:6.2f} {acc[2]:6.2f} g",
           f"gyro {gyro[0]:6.2f} {gyro[1]:6.2f} {gyro[2]:6.2f} rad/s",
           f"frames {n}"]
    front_txt = "front: no return" if front is None else f"front lidar: {front:.3f} m"
    cv2.putText(p, front_txt, (20, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
    for i, t in enumerate(txt):
        cv2.putText(p, t, (20, 520 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    return p


PAGE = """<!doctype html><meta charset="utf-8">
<title>lidar + imu</title>
<body style="margin:0;background:#222;text-align:center">
<img id="v" style="width:100%;max-width:1100px">
<div><button onclick="fetch('/stop')"
 style="font-size:18px;margin:8px;padding:6px 24px">停止并保存</button></div>
<script>function r(){document.getElementById('v').src='/view.jpg?t='+Date.now()}
setInterval(r,200);r()</script>"""


class ViewServer(threading.Thread):
    frame = b""

    def __init__(self, port=8000):
        super().__init__(daemon=True)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/stop"):
                    outer.request_stop()
                    self.send_response(200); self.end_headers(); self.wfile.write(b"stopped")
                elif self.path.startswith("/view.jpg"):
                    body = outer.frame
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)

    def run(self):
        self.httpd.serve_forever()


class Keys:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.held = {}
        self.old = None
        if sys.stdin.isatty():
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        """返回当前按住的键。终端只在按下时发字符, 所以用重复按键的间隔判断是否仍按住。"""
        if self.old is None:
            return set()
        now = time.time()
        while select.select([self.fd], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch:
                self.held[ch] = now
        return {k for k, t in self.held.items() if now - t < 0.12}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    imu, lidar = ImuReader(), LidarReader()
    imu.start(); lidar.start()
    t0 = time.time()
    while not imu.ready and time.time() - t0 < 5:
        time.sleep(0.1)
    if not imu.ready:
        raise SystemExit("IMU 没有数据, 检查 /dev/imu 是否被其他程序占用")
    print("IMU 已就绪", flush=True)

    robot = LeKiwiBase(LeKiwiConfig(port=BASE_PORT, id="kiwi_base", cameras={}))
    print("连接底盘...", flush=True)
    robot.connect(calibrate=False)

    web = ViewServer()
    web.start()
    print("浏览器打开 http://<jetson ip>:8000 查看画面", flush=True)
    print("终端按住键才走, 松开即停: w/s 前后, a/d 平移, z/x 转向, r/f 速度档, q 退出", flush=True)

    grid = np.zeros((GRID, GRID))
    pose = np.zeros(3)
    traj = [pose.copy()]
    world = []
    latest_world = None
    front = None
    speed_idx = 0
    # IMU 航向的正方向是否与车体逆时针一致, 由第一次转向指令自动判定
    yaw_sign, sign_evidence = None, 0.0
    imu_yaw_prev_loop = imu.yaw()
    imu_yaw_last_scan = imu.yaw()
    accepted = rejected = 0
    hold_yaw = None
    cmd = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    cmd_lock = threading.Lock()

    stop = threading.Event()
    web.request_stop = stop.set

    def drive():
        """独立于建图运行, 保证松键后尽快停车。"""
        while not stop.is_set():
            with cmd_lock:
                current = dict(cmd)
            robot.send_action(current)
            stop.wait(0.02)

    threading.Thread(target=drive, daemon=True).start()

    try:
        with Keys() as keys:
            while not stop.is_set():
                pressed = keys.poll()
                if "q" in pressed:
                    break
                if "r" in pressed:
                    speed_idx = min(speed_idx + 1, len(SPEEDS) - 1)
                if "f" in pressed:
                    speed_idx = max(speed_idx - 1, 0)
                m = SPEEDS[speed_idx]
                new_cmd = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                for k, axis, sign in (("w", "x.vel", 1), ("s", "x.vel", -1),
                                      ("a", "y.vel", 1), ("d", "y.vel", -1)):
                    if k in pressed:
                        new_cmd[axis] += sign * m["lin"]
                if "z" in pressed:
                    new_cmd["theta.vel"] += m["rot"]
                if "x" in pressed:
                    new_cmd["theta.vel"] -= m["rot"]

                yaw_now = imu.yaw()
                translating = new_cmd["x.vel"] != 0 or new_cmd["y.vel"] != 0
                if translating and new_cmd["theta.vel"] == 0 and yaw_sign is not None:
                    if hold_yaw is None:
                        hold_yaw = yaw_now
                    err_deg = math.degrees(yaw_sign * (hold_yaw - yaw_now))
                    new_cmd["theta.vel"] = float(np.clip(HEADING_KP * err_deg,
                                                         -HEADING_MAX_RATE, HEADING_MAX_RATE))
                else:
                    hold_yaw = None
                with cmd_lock:
                    cmd.update(new_cmd)

                if yaw_sign is None and new_cmd["theta.vel"] != 0:
                    sign_evidence += math.copysign(1, new_cmd["theta.vel"]) * (yaw_now - imu_yaw_prev_loop)
                    if abs(sign_evidence) > math.radians(15):
                        yaw_sign = 1 if sign_evidence > 0 else -1
                        print(f"\nIMU 航向方向已判定: {'+' if yaw_sign > 0 else '-'}1", flush=True)
                imu_yaw_prev_loop = yaw_now

                scans = lidar.take()
                if len(scans) > 2:  # 建图落后时只保留最新的, 避免越积越慢
                    scans = scans[-2:]
                for sc in scans:
                    if len(sc) < 60:
                        continue
                    xy = to_xy(sc)
                    front = front_distance(sc)
                    prior = pose.copy()
                    if yaw_sign is not None:
                        prior[2] += yaw_sign * (yaw_now - imu_yaw_last_scan)
                    imu_yaw_last_scan = yaw_now
                    ok = True
                    if world:
                        ref = np.vstack(world[-8:])
                        ref = ref[::max(1, len(ref) // 1500)]
                        est, ratio, err = icp(xy, ref, prior)
                        ok = (ratio >= ICP_MIN_INLIER_RATIO and err <= ICP_MAX_MEAN_ERR
                              and (yaw_sign is None or abs(wrap(est[2] - prior[2])) <= ICP_MAX_YAW_DEV))
                        pose = est if ok else prior
                    latest_world = transform(xy, pose)
                    if ok:
                        accepted += 1
                        world.append(latest_world)
                        ray_update(grid, pose, latest_world)
                        traj.append(pose.copy())
                    else:
                        rejected += 1

                roll, pitch, yaw, acc, gyro, n = imu.snapshot()
                status = [f"scans used {accepted}  rejected {rejected}",
                          "imu yaw sign: " + ("press z/x to detect" if yaw_sign is None
                                              else f"{yaw_sign:+d}")]
                view = np.hstack([cv2.resize(render_map(grid, traj, latest_world, status), (640, 640)),
                                  render_imu(roll, pitch, yaw, acc, gyro, n, front)])
                web.frame = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                time.sleep(0.03)
    finally:
        print("停车...", flush=True)
        try:
            robot.stop_base()
            robot.disconnect()
        except Exception as e:
            print(e)
        imu.stop_flag = lidar.stop_flag = True
        np.savez(os.path.join(os.path.dirname(__file__), "imu_lidar_map.npz"),
                 grid=grid, traj=np.array(traj))
        print("地图已保存到 project_tool/imu_lidar_map.npz", flush=True)


if __name__ == "__main__":
    main()
