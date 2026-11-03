"""Move the neck to the extrinsic-calibration pose and leave the servos holding."""
import os
import sys
import time

os.chdir("/home/jetson/workspaces/FXLerobot/FXlerobot/project_tool")
sys.path.insert(0, os.getcwd())

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lerobot.robots.lekiwi.lekiwi import LeKiwiNeck
from lerobot_kiwi_base_neck import PORT, assert_port_ready

assert_port_ready(PORT)
neck = LeKiwiNeck(LeKiwiConfig(port=PORT, id="kiwi_neck", cameras={}))
neck.connect(calibrate=False)
before = neck.get_observation()
print(f"before yaw={before.get('neck_yaw.pos')} pitch={before.get('neck_pitch.pos')}")
neck.send_action({"neck_yaw.pos": 0.0, "neck_pitch.pos": 0.0})
time.sleep(1.5)
after = neck.get_observation()
print(f"after yaw={after.get('neck_yaw.pos')} pitch={after.get('neck_pitch.pos')}")
neck.bus.disconnect(disable_torque=False)
print("neck holding at 0, 0")
