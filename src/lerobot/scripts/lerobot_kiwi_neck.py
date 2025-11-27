#!/usr/bin/env python

import sys
import time
import termios
import tty

import draccus

from lerobot.robots.lekiwi.lekiwi import LeKiwiNeck
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lerobot.utils.errors import DeviceNotConnectedError


def getch() -> str:
    """
    在终端下无回车读取单个按键。
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


@draccus.wrap()
def main(cfg: LeKiwiConfig):
    """
    键盘控制 LeKiwiNeck:

      y / Y : neck_yaw + / - (默认 5 度)
      p / P : neck_pitch + / - (默认 5 度)
      h     : 回零 (0, 0)
      q     : 退出
    """
    robot = LeKiwiNeck(cfg)

    # 这里先不自动校准，假设你已经写好 calibration 或不在意 0 点
    robot.connect(calibrate=False)

    yaw = 0.0      # 当前目标 yaw 角（度）
    pitch = 0.0    # 当前目标 pitch 角（度）
    step = 5.0     # 每次按键步进角度

    # 给一点简单的安全范围（根据你实际机械结构可再调整）
    YAW_MIN, YAW_MAX = -90.0, 90.0
    PITCH_MIN, PITCH_MAX = -45.0, 45.0

    print("==== LeKiwi Neck Teleop ====")
    print("  y / Y : neck_yaw + / - 5 deg")
    print("  p / P : neck_pitch + / - 5 deg")
    print("  h     : home -> (0, 0)")
    print("  q     : quit")
    print("=============================")

    try:
        while True:
            ch = getch()

            if ch in ("q", "\x03"):  # q 或 Ctrl+C
                print("\nExiting...")
                break

            if ch == "r":
                yaw += step
            elif ch == "t":
                yaw -= step
            elif ch == "v":
                pitch += step
            elif ch == "b":
                pitch -= step
            elif ch == "h":
                yaw = 0.0
                pitch = 0.0
            else:
                # 其他键忽略
                continue

            # 限幅
            yaw = max(min(yaw, YAW_MAX), YAW_MIN)
            pitch = max(min(pitch, PITCH_MAX), PITCH_MIN)

            action = {
                "neck_yaw.pos": yaw,
                "neck_pitch.pos": pitch,
            }

            try:
                robot.send_action(action)
            except DeviceNotConnectedError:
                print("Robot not connected, abort.")
                break

            sys.stdout.write(f"\ryaw = {yaw:7.2f} deg   pitch = {pitch:7.2f} deg")
            sys.stdout.flush()

            time.sleep(0.02)

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
