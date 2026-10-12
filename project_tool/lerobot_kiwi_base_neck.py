#!/usr/bin/env python3
import os
import sys
import time
import termios
import tty
import select
import subprocess

from lerobot.robots.lekiwi.lekiwi import LeKiwiBase, LeKiwiNeck
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

FPS = 30
PORT = "/dev/base_neck"


def log(msg: str) -> None:
    print(msg, flush=True)


def assert_port_ready(port: str) -> None:
    if not os.path.exists(port):
        raise SystemExit(f"串口不存在: {port}  （先确认底盘/脖子板 USB 已插入）")
    real = os.path.realpath(port)
    r = subprocess.run(["fuser", real], capture_output=True, text=True)
    users = r.stdout.strip()
    if users:
        raise SystemExit(
            f"串口已被占用: {port} -> {real} (PID {users})\n"
            f"先结束旧进程:  pkill -f lerobot_kiwi  或  fuser -k {real}"
        )
    log(f"串口就绪: {port} -> {real}")


class StdinKeyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get_keys(self):
        pressed = set()
        while True:
            r, _, _ = select.select([self.fd], [], [], 0)
            if not r:
                break
            ch = sys.stdin.read(1)
            if ch:
                pressed.add(ch)

        keys = {
            "w": False, "a": False, "s": False, "d": False,
            "z": False, "x": False,
            "r": False, "t": False,
            "v": False, "b": False,
            "q": False,
        }
        for k in pressed:
            if k in keys:
                keys[k] = True
        return keys


def main():
    log("=== LeKiwi base + neck teleop ===")
    assert_port_ready(PORT)

    base_cfg = LeKiwiConfig(port=PORT, id="kiwi_base", cameras={})
    neck_cfg = LeKiwiConfig(port=PORT, id="kiwi_neck", cameras={})
    base = LeKiwiBase(base_cfg)
    neck = LeKiwiNeck(neck_cfg)

    log("正在连接底盘舵机 (ID 7/8/9)...")
    log("如果这里长时间空白，通常是底盘板没供电或舵机总线不通，请检查板子灯/电源。")
    t0 = time.time()
    base.connect(calibrate=False)
    log(f"底盘已连接 ({time.time() - t0:.1f}s)")

    # Same USB board as base. Skip interactive calibration so the script
    # cannot hang on "Press ENTER to use provided calibration file...".
    log("正在连接脖子舵机 (ID 10/11), calibrate=False ...")
    t0 = time.time()
    neck.connect(calibrate=False)
    log(f"脖子已连接 ({time.time() - t0:.1f}s)")

    yaw = 0.0
    pitch = 0.0
    step = 5.0

    log("==== Base + Neck Teleop ====")
    log("Base:")
    log("  w/s : 前进 / 后退")
    log("  a/d : 左移 / 右移")
    log("  z/x : 左转 / 右转")
    log("Neck:")
    log("  r/t : yaw + / - 5°")
    log("  v/b : pitch + / - 5°")
    log("q     : quit")
    log("=============================")

    with StdinKeyboard() as kb:
        try:
            while True:
                t0 = time.time()
                keys = kb.get_keys()

                if keys["q"]:
                    log("Exiting teleop ...")
                    break

                base_action = base._from_keyboard_to_base_action(keys)
                base.send_action(base_action)

                if keys["r"]:
                    yaw += step
                if keys["t"]:
                    yaw -= step
                if keys["v"]:
                    pitch += step
                if keys["b"]:
                    pitch -= step

                neck.send_action({
                    "neck_yaw.pos": yaw,
                    "neck_pitch.pos": pitch,
                })

                print(
                    f"\rBASE: {base_action} | NECK: yaw={yaw:.1f}, pitch={pitch:.1f}   ",
                    end="",
                    flush=True,
                )

                next_t = t0 + 1.0 / FPS
                while time.time() < next_t:
                    pass
        finally:
            print(flush=True)
            log("停车并断开...")
            try:
                base.stop_base()
            except Exception as e:
                log(f"stop_base: {e}")
            try:
                base.disconnect()
            except Exception as e:
                log(f"base.disconnect: {e}")
            try:
                neck.disconnect()
            except Exception as e:
                log(f"neck.disconnect: {e}")
            log("已退出")


if __name__ == "__main__":
    main()
