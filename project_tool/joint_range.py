#!/usr/bin/env python
"""Record how far each joint of a follower arm really turns, moved by hand, and let the arm use it.

    python joint_range.py                          # record: left arm, 180 s
    python joint_range.py --arm right --seconds 120
    python joint_range.py --apply joint_ranges/left_20261110_151041.json

Record: the arm's torque is switched off, its joints are read at READ_HZ while someone moves them by
hand to their ends, and the lowest and highest reading of each joint are saved as JSON under
joint_ranges/. Nothing is written to the arm or to the LeRobot calibration.

Apply: each joint's range is widened, on each side on its own, to where the recording reached past
the calibrated end (MIN_WIDEN_DEG at least); it is never narrowed. The widened ranges go into save_pose.range_path (next to the
LeRobot calibration, which is left as it is) and into the servos' own position limits (EEPROM 9/11,
else a servo stops at the old end). The calibration stays because the programs in this repo count a
joint's angle from the middle of its calibrated range (save_pose.degrees): every angle written down so
far (PARK, CARRY, the arm's mount on the body) keeps its meaning. lerobot-calibrate would move that
middle; LeRobot's own configure writes the calibrated limits back into the servos.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from save_pose import ARMS, MOTORS, Servos, degrees, load_calibration, load_range, range_path

READ_HZ = 20
PRINT_S = 1.0
OUT_DIR = Path(__file__).resolve().parent / "joint_ranges"
MIN_LIMIT_ADDR, MAX_LIMIT_ADDR, LOCK_ADDR = 9, 11, 55
WRITE_TRIES = 3  # an EEPROM write's reply gets lost now and then: read it back instead
# A side is widened only when the recording reached this far past it: a hand pressing a joint against
# its stop reads a tick or two past the calibrated end.
MIN_WIDEN_DEG = 1.0


def release(bus):
    """Torque off on all joints; names of the ones that would not answer."""
    deaf = []
    for sid, name in enumerate(MOTORS, start=1):
        if not bus.write(sid, "Torque_Enable", 0) or bus.read(sid, "Torque_Enable") != 0:
            deaf.append(name)
    return deaf


def deg(ticks, cal):
    return degrees(ticks, cal) if ticks is not None else None


def record(args):
    cal = load_calibration(args.arm)
    bus = Servos(ARMS[args.arm])
    try:
        deaf = release(bus)
        if deaf:
            print(f"这些舵机没回应，读不到：{', '.join(deaf)}")
        print(f"{args.arm} 臂（{ARMS[args.arm]}）力矩已松开，手臂会往下落，先扶住。")
        print(f"记录 {args.seconds:.0f} 秒：用手把要量的关节慢慢转到两头，转不动或快碰到车身就停。Ctrl-C 提前结束。")
        lo, hi, now = {}, {}, {}
        t0 = last = time.monotonic()
        try:
            while time.monotonic() - t0 < args.seconds:
                for sid, name in enumerate(MOTORS, start=1):
                    t = bus.read(sid, "Present_Position")
                    if t is None:
                        continue
                    now[name] = t
                    lo[name] = min(lo.get(name, t), t)
                    hi[name] = max(hi.get(name, t), t)
                if time.monotonic() - last >= PRINT_S:
                    last = time.monotonic()
                    left = args.seconds - (last - t0)
                    print(f"[{left:4.0f}s] " + "  ".join(
                        f"{n} {deg(now.get(n), cal[n]):+.1f}({deg(lo[n], cal[n]):+.0f}~{deg(hi[n], cal[n]):+.0f})"
                        for n in MOTORS if n in now), flush=True)
                time.sleep(1.0 / READ_HZ)
        except KeyboardInterrupt:
            print("提前结束")
    finally:
        bus.close()

    result = {"arm": args.arm, "port": ARMS[args.arm], "time": datetime.now().isoformat(timespec="seconds"),
              "seconds": args.seconds, "joints": {}}
    print(f"\n{'关节':13s} {'记录到的范围 (°)':>18s} {'标定范围 (°)':>16s}")
    for name in MOTORS:
        if name not in lo:
            continue
        c = cal[name]
        result["joints"][name] = {"recorded_ticks": [lo[name], hi[name]],
                                  "recorded_deg": [round(deg(lo[name], c), 1), round(deg(hi[name], c), 1)],
                                  "calibrated_ticks": [c["range_min"], c["range_max"]]}
        wider = [s for s, ok in (("小的一头", lo[name] < c["range_min"]), ("大的一头", hi[name] > c["range_max"])) if ok]
        print(f"{name:13s} {deg(lo[name], c):+8.1f} ~ {deg(hi[name], c):+7.1f} "
              f"{deg(c['range_min'], c):+8.1f} ~ {deg(c['range_max'], c):+6.1f}  "
              + (f"比标定宽：{'、'.join(wider)}" if wider else ""))
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{args.arm}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\n存到 {path}。标定文件和舵机都没改；要用这个范围：python joint_range.py --apply {path}")


def read2(bus, sid, addr):
    """A 2-byte register, or None. A reply cut short makes the SDK raise IndexError: asked again."""
    for _ in range(WRITE_TRIES):
        try:
            got, comm, _ = bus.pk.read2ByteTxRx(bus.port, sid, addr)
        except IndexError:
            comm = -1
        if comm == 0:
            return got
        time.sleep(0.1)
    return None


def write_eeprom(bus, sid, addr, value):
    """Torque off, unlock, write a 2-byte EEPROM register, check it, lock. True when it holds value."""
    pk, port = bus.pk, bus.port
    bus.write(sid, "Torque_Enable", 0)
    pk.write1ByteTxRx(port, sid, LOCK_ADDR, 0)
    try:
        for _ in range(WRITE_TRIES):
            pk.write2ByteTxRx(port, sid, addr, int(value))
            time.sleep(0.1)
            if read2(bus, sid, addr) == value:
                return True
        return False
    finally:
        pk.write1ByteTxRx(port, sid, LOCK_ADDR, 1)


def apply(args):
    rec = json.loads(Path(args.apply).read_text())
    side = rec["arm"]
    cal = load_calibration(side)
    old = load_range(side)
    new = dict(old)
    for name, j in rec["joints"].items():
        lo, hi = j["recorded_ticks"]
        a, b = old[name]
        new[name] = (lo if degrees(a, cal[name]) - degrees(lo, cal[name]) >= MIN_WIDEN_DEG else a,
                     hi if degrees(hi, cal[name]) - degrees(b, cal[name]) >= MIN_WIDEN_DEG else b)
    changed = [n for n in MOTORS if new[n] != old[n]]
    print(f"{side} 臂（{ARMS[side]}），按 {args.apply}：")
    for n in MOTORS:
        c = cal[n]
        print(f"  {n:13s} {deg(old[n][0], c):+7.1f} ~ {deg(old[n][1], c):+6.1f}°  →  "
              f"{deg(new[n][0], c):+7.1f} ~ {deg(new[n][1], c):+6.1f}°" + ("  放宽" if n in changed else ""))
    widened = {n: [int(new[n][0]), int(new[n][1])] for n in MOTORS
               if new[n] != (cal[n]["range_min"], cal[n]["range_max"])}
    if changed:
        path = range_path(side)
        if path.is_file():
            backup = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}.json")
            backup.write_text(path.read_text())
            print(f"旧的范围文件备份到 {backup}")
        path.write_text(json.dumps(widened, indent=2) + "\n")
        print(f"范围写入 {path}（标定文件没改）")
    else:
        print("范围文件已经是这样，不用改。")
    # The servos' own limits, wherever they differ from the widened range (a write that failed before
    # is done now).
    bus = Servos(ARMS[side])
    try:
        for name, (lo, hi) in widened.items():
            sid = MOTORS.index(name) + 1
            now = (read2(bus, sid, MIN_LIMIT_ADDR), read2(bus, sid, MAX_LIMIT_ADDR))
            if now == (lo, hi):
                print(f"  {name} 舵机限位已是 {lo}~{hi}")
                continue
            ok = all(write_eeprom(bus, sid, addr, v) for addr, v in ((MIN_LIMIT_ADDR, lo), (MAX_LIMIT_ADDR, hi)))
            print(f"  {name} 舵机限位 {now[0]}~{now[1]} {'改成' if ok else '没写成'} {lo}~{hi}")
    finally:
        bus.close()
    print("舵机力矩是松开的。")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="left")
    ap.add_argument("--seconds", type=float, default=180.0, help="how long to record (default 180)")
    ap.add_argument("--apply", metavar="RECORDING", help="widen the arm's range to a saved recording")
    args = ap.parse_args()
    apply(args) if args.apply else record(args)


if __name__ == "__main__":
    sys.exit(main())
