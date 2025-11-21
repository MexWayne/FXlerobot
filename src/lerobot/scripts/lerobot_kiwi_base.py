#from lerobot.robots.lekiwi.lekiwi import LeKiwiBase
#from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
#
#config = LeKiwiConfig(
#    port="/dev/ttyACM0",   # 换成你实际的 Feetech USB 口
#    cameras={},            # 先不连摄像头也可以
#)
#
#robot = LeKiwiBase(config)
#robot.connect(calibrate=False)  # 如果你已经做过一次 base-only 校准
#
## 让机器人往前走一点
#robot.send_action({"x.vel": 0.2, "y.vel": 0.0, "theta.vel": 0.0})




import sys
import time
import termios
import tty
import select

from lerobot.robots.lekiwi import LeKiwiBase
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

FPS = 30


class StdinKeyboard:
    """
    用 stdin + termios 实现的简单键盘读取器：
    - 非阻塞读取
    - 返回一个 dict[str,bool]，与 _from_keyboard_to_base_action 兼容
    """

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def __enter__(self):
        # 进入 raw 模式
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出时恢复终端设置
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_keys(self) -> dict[str, bool]:
        """
        一次调用中，把这帧时间窗口内按下的所有键读出来。
        只关心单字符键：w/a/s/d/z/x/r/f/q
        """
        pressed = set()

        # non-blocking 检查 stdin 是否有数据
        while True:
            rlist, _, _ = select.select([self.fd], [], [], 0)
            if not rlist:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            pressed.add(ch)

        # 构造与 _from_keyboard_to_base_action 兼容的 dict
        keys = {k: False for k in ["w", "a", "s", "d", "z", "x", "r", "f", "q"]}
        for k in pressed:
            if k in keys:
                keys[k] = True

        return keys


def main():
    config = LeKiwiConfig(
        port="/dev/ttyACM0",  # 换成你的实际串口
        id="kiwi_base",
        cameras={},           # 现在不接相机也没关系
    )

    robot = LeKiwiBase(config)
    robot.connect(calibrate=False)  # 先跳过校准，后面再做也行

    dt = 1.0 / FPS

    print("本地键盘控制模式：")
    print("  w/s: 前进/后退")
    print("  a/d: 左移/右移")
    print("  z/x: 左转/右转")
    print("  r/f: 提升/降低速度档位")
    print("  q  : 退出")
    print()

    with StdinKeyboard() as kb:
        try:
            while True:
                t0 = time.time()

                keys = kb.get_keys()
                if keys.get("q", False):
                    print("退出控制")
                    break

                base_action = robot._from_keyboard_to_base_action(keys)
                print("keys:", keys, " -> base_action:", base_action)

                robot.send_action(base_action)

                next_t = t0 + dt
                # 简单 busy-wait
                while time.time() < next_t:
                    pass

        finally:
            robot.stop_base()
            robot.disconnect()


if __name__ == "__main__":
    main()

