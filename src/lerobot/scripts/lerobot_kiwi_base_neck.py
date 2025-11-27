#!/usr/bin/env python3

import sys
import time
import termios
import tty
import select

from lerobot.robots.lekiwi.lekiwi import LeKiwiBase, LeKiwiNeck
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

FPS = 30


# ---------------- 键盘读取 ---------------- #

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


# ---------------- Base + Neck Teleop ---------------- #

def main():
    # Base config
    base_cfg = LeKiwiConfig(
        port="/dev/ttyACM0",
        id="kiwi_base",
        cameras={},    # 不用相机
    )

    # Neck config（用同一个串口）
    neck_cfg = LeKiwiConfig(
        port="/dev/ttyACM0",
        id="kiwi_neck",
        cameras={},    # 不用相机
    )

    base = LeKiwiBase(base_cfg)
    neck = LeKiwiNeck(neck_cfg)

    # 都不需要再次校准
    base.connect(calibrate=False)
    neck.connect(calibrate=True)

    dt = 1.0 / FPS
    yaw = 0.0
    pitch = 0.0
    step = 5.0

    print("==== Base + Neck Teleop ====")
    print("Base:")
    print("  w/s : 前进 / 后退")
    print("  a/d : 左移 / 右移")
    print("  z/x : 左转 / 右转")
    print("Neck:")
    print("  r/t : yaw + / - 5°")
    print("  v/b : pitch + / - 5°")
    print("q     : quit")
    print("=============================")

    with StdinKeyboard() as kb:
        try:
            while True:
                t0 = time.time()
                keys = kb.get_keys()

                if keys["q"]:
                    print("Exiting teleop ...")
                    break

                # ------------ Base action ------------ #
                base_action = base._from_keyboard_to_base_action(keys)
                base.send_action(base_action)

                # ------------ Neck action ------------ #
                if keys["r"]:
                    yaw += step
                if keys["t"]:
                    yaw -= step
                if keys["v"]:
                    pitch += step
                if keys["b"]:
                    pitch -= step

                neck_action = {
                    "neck_yaw.pos": yaw,
                    "neck_pitch.pos": pitch,
                }
                neck.send_action(neck_action)

                # ------------ 打印调试 ------------ #
                print(
                    f"\rBASE: {base_action} | "
                    f"NECK: yaw={yaw:.1f}, pitch={pitch:.1f}",
                    end="",
                    flush=True
                )

                # 控制循环频率
                next_t = t0 + dt
                while time.time() < next_t:
                    pass

        finally:
            base.stop_base()
            base.disconnect()
            neck.disconnect()


if __name__ == "__main__":
    main()
