#!/usr/bin/env python

import draccus

from lerobot.robots.lekiwi import LeKiwiNeck
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig


@draccus.wrap()
def main(cfg: LeKiwiConfig):
    """
    独立给 LeKiwiNeck 设置电机 ID 的脚本。

    用法示例：
        python src/lerobot/scripts/setup_lekiwi_neck_motors.py \
            --port=/dev/ttyACM0 \
            --use_degrees=true
    """
    # 不需要指定 type，LeKiwiConfig 默认 type="lekiwi"
    robot = LeKiwiNeck(cfg)
    robot.setup_motors()


if __name__ == "__main__":
    main()
