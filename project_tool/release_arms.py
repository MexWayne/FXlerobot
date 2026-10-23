#!/usr/bin/env python
"""Release locked SO-arm servos: turn torque off on every STS3215 and report faults.

After lerobot-record / teleoperate is killed (Ctrl-Z, SSH drop, crash) the followers keep
holding their last goal with torque on. A gripper that keeps squeezing past what it can reach
also trips overload protection; it then answers every packet with "Overload error", which
LeRobot's startup check reports as "Missing motor IDs: 6".

This talks to the servos with the raw Feetech SDK (so error replies are still readable), writes
Torque_Enable=0 and Lock=0, reads it back, and prints status / temperature.

Usage:
    python release_arms.py                 # all four arms, asks before releasing
    python release_arms.py --yes           # no prompt
    python release_arms.py --ports /dev/arm_A_follower
    python release_arms.py --include-neck  # release all four arms plus neck IDs 10/11
    python release_arms.py --check         # only read status, don't touch torque
"""

import argparse
import os
import subprocess
import sys
import time

import scservo_sdk as scs

DEFAULT_PORTS = ["/dev/arm_A_follower", "/dev/arm_B_follower", "/dev/arm_A_leader", "/dev/arm_B_leader"]
NECK_PORT = "/dev/base_neck"
ARM_MOTORS = [(i, name) for i, name in enumerate(
    ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    start=1,
)]
NECK_MOTORS = [(10, "neck_yaw"), (11, "neck_pitch")]
BAUDRATE = 1_000_000
# STS3215 control table: name -> (address, bytes).
REG = {
    "Torque_Enable": (40, 1),
    "Goal_Position": (42, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
}
# Bits of the error byte in every reply, and of the Status register.
ERROR_BITS = {0: "电压异常", 1: "编码器异常", 2: "过温", 3: "过流", 5: "过载"}
HOT_C = 50
RETRY = 3


def port_holders(port):
    """PIDs + command lines of processes that have the serial port open."""
    try:
        out = subprocess.run(["fuser", os.path.realpath(port)], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    holders = []
    for pid in out.split():
        pid = pid.strip().rstrip("emc")
        if pid.isdigit() and int(pid) != os.getpid():
            try:
                cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()
            except OSError:
                cmd = "?"
            holders.append((int(pid), cmd))
    return holders


def decode(bits):
    return [name for bit, name in ERROR_BITS.items() if bits & (1 << bit)]


class Servos:
    def __init__(self, port):
        self.port = scs.PortHandler(port)
        self.pk = scs.PacketHandler(0)
        if not self.port.openPort() or not self.port.setBaudRate(BAUDRATE):
            raise OSError("openPort/setBaudRate failed")
        # Avoid hanging forever when a USB serial adapter stops accepting writes.
        if getattr(self.port, "ser", None) is not None:
            self.port.ser.write_timeout = 0.5

    def close(self):
        self.port.closePort()

    def ping(self, sid):
        """(alive, error_bits). A reply carrying an error byte still means the servo is alive."""
        for _ in range(RETRY):
            _, comm, err = self.pk.ping(self.port, sid)
            if comm == scs.COMM_SUCCESS:
                return True, err
        return False, 0

    def read(self, sid, name):
        addr, n = REG[name]
        fn = self.pk.read1ByteTxRx if n == 1 else self.pk.read2ByteTxRx
        for _ in range(RETRY):
            val, comm, _ = fn(self.port, sid, addr)
            if comm == scs.COMM_SUCCESS:
                return val
        return None

    def write(self, sid, name, value):
        addr, n = REG[name]
        fn = self.pk.write1ByteTxRx if n == 1 else self.pk.write2ByteTxRx
        for _ in range(RETRY):
            comm, _ = fn(self.port, sid, addr, value)
            if comm == scs.COMM_SUCCESS:
                return True
        return False


def motors_for_port(port):
    if port == NECK_PORT:
        return NECK_MOTORS
    try:
        if os.path.exists(port) and os.path.exists(NECK_PORT) and os.path.samefile(port, NECK_PORT):
            return NECK_MOTORS
    except OSError:
        pass
    return ARM_MOTORS


def handle_port(port, check_only):
    print(f"\n===== {port} -> {os.path.realpath(port)} =====")
    if not os.path.exists(port):
        print("  端口不存在（线没插或 udev 规则没生效），跳过")
        return False
    holders = port_holders(port)
    if holders:
        print("  端口被其他进程占用，先停掉它（Ctrl-C；挂起的用 kill -9）：")
        for pid, cmd in holders:
            print(f"    PID {pid}: {cmd[:120]}")
        return False
    try:
        bus = Servos(port)
    except OSError as e:
        print(f"  打不开串口: {e}")
        return False

    ok = True
    try:
        for sid, name in motors_for_port(port):
            alive, err = bus.ping(sid)
            if not alive:
                print(f"  [{name:13s}] 完全无响应：检查舵机供电/线缆，或断电 5 秒再上电")
                ok = False
                continue
            torque = bus.read(sid, "Torque_Enable")
            status = bus.read(sid, "Status")
            temp = bus.read(sid, "Present_Temperature")
            volt = bus.read(sid, "Present_Voltage")
            pos = bus.read(sid, "Present_Position")
            goal = bus.read(sid, "Goal_Position")
            faults = decode(err | (status or 0))

            action = ""
            if not check_only:
                done = bus.write(sid, "Torque_Enable", 0) and bus.write(sid, "Lock", 0)
                time.sleep(0.02)
                after = bus.read(sid, "Torque_Enable")
                if done and after == 0:
                    action = " -> 已松开"
                else:
                    action = f" -> 未生效（Torque_Enable={after}）"
                    ok = False

            notes = []
            if faults:
                notes.append("故障: " + "/".join(faults))
                ok = False
            if temp is not None and temp >= HOT_C:
                notes.append("偏热，先让它凉一会")
            volt_s = f"{volt / 10:.1f}V" if volt is not None else "?"
            torque_s = {0: "关", 1: "开"}.get(torque, "?")
            print(
                f"  [{name:13s}] 力矩={torque_s} 位置={pos} 目标={goal} 温度={temp}°C 电压={volt_s} "
                f"{' '.join(notes) if notes else '状态正常'}{action}"
            )
    finally:
        bus.close()
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", nargs="+", default=DEFAULT_PORTS)
    ap.add_argument("--check", action="store_true", help="only read status, don't disable torque")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--include-neck", action="store_true", help="also release neck IDs 10/11")
    args = ap.parse_args()
    ports = list(args.ports)
    if args.include_neck and NECK_PORT not in ports:
        ports.append(NECK_PORT)

    if not args.check and not args.yes:
        print("将关闭以下机械臂所有舵机的力矩，从臂会因重力下落，请先用手托住：")
        for p in ports:
            print(f"  {p}")
        if input("托好后按回车继续（输入 n 取消）: ").strip().lower() == "n":
            sys.exit("已取消")

    results = {p: handle_port(p, args.check) for p in ports}

    print("\n===== 汇总 =====")
    for p, ok in results.items():
        print(f"  {p:22s} {'OK' if ok else '有问题，见上方'}")
    if not all(results.values()):
        print("仍有问题：关掉占用端口的进程后重跑；过载/过温的舵机先松开、凉几分钟，"
              "还不行就断开舵机电源 5 秒再上电。")
        sys.exit(1)


if __name__ == "__main__":
    main()
