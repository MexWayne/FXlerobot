#!/usr/bin/env python3
import os
import sys
import time
import termios
import tty
import select
import subprocess

from lerobot.robots.lekiwi import LeKiwiBase
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
            f"先结束旧进程:  pkill -f lerobot_kiwi_base  或  fuser -k {real}"
        )
    log(f"串口就绪: {port} -> {real}")


class StdinKeyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_keys(self) -> dict[str, bool]:
        pressed = set()
        while True:
            rlist, _, _ = select.select([self.fd], [], [], 0)
            if not rlist:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            pressed.add(ch)

        keys = {k: False for k in ["w", "a", "s", "d", "z", "x", "r", "f", "q"]}
        for k in pressed:
            if k in keys:
                keys[k] = True
        return keys


def main():
    log("=== LeKiwi base teleop ===")
    assert_port_ready(PORT)

    config = LeKiwiConfig(port=PORT, id="kiwi_base", cameras={})
    robot = LeKiwiBase(config)

    log("正在连接底盘舵机 (ID 7/8/9)...")
    log("如果这里长时间无白，通常是底盘板没供电或舵机总线不通，请检查板子灯/电源。")
    t0 = time.time()
    robot.connect(calibrate=False)
    log(f"底盘已连接 ({time.time() - t0:.1f}s)")

    log("本地键盘控制：")
    log("  w/s: 前进/后退")
    log("  a/d: 左移/右移")
    log("  z/x: 左转/右转")
    log("  r/f: 提升/降低速度档位")
    log("  q  : 退出")

    with StdinKeyboard() as kb:
        try:
            while True:
                t0 = time.time()
                keys = kb.get_keys()
                if keys.get("q", False):
                    log("退出控制")
                    break

                base_action = robot._from_keyboard_to_base_action(keys)
                print(
                    f"\rkeys={ {k:v for k,v in keys.items() if v} } "
                    f"action={base_action}   ",
                    end="",
                    flush=True,
                )
                robot.send_action(base_action)

                next_t = t0 + 1.0 / FPS
                while time.time() < next_t:
                    pass
        finally:
            print(flush=True)
            log("停车并断开...")
            try:
                robot.stop_base()
            except Exception as e:
                log(f"stop_base: {e}")
            try:
                robot.disconnect()
            except Exception as e:
                log(f"disconnect: {e}")
            log("已退出")


if __name__ == "__main__":
    main()
