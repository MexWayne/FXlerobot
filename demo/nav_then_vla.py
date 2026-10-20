import subprocess
import sys

ROOT = "/home/jetson/workspaces/FXLerobot/FXlerobot"

subprocess.run(
    [sys.executable, "nlv_nav.py", "去椅子旁边", "--lin", "0.10", "--stop-m", "0.3"],
    cwd=f"{ROOT}/project_tool",
    check=True,
)
subprocess.run(
    [sys.executable, "robot_client.py", "--execute", "--duration", "50"],
    cwd=f"{ROOT}/examples/pizero_cci_to_so101",
    check=True,
)
