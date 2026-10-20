"""Read the SO101 followers and ask the pizero server on 4090_cci for one action chunk.

Leader arms are never opened. Inference reads and commands only the followers:
left /dev/arm_B_follower, right /dev/arm_A_follower.

The checkpoint was trained on three views (front_left, wrist_left, wrist_right).
front_right is captured with them and saved, and is not part of the model input.
"""

import argparse
import logging
import subprocess
import threading
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import websockets.sync.client
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Leader ports (/dev/arm_A_leader, /dev/arm_B_leader) stay closed.
FOLLOWER_PORTS = {
    "left": "/dev/arm_B_follower",
    "right": "/dev/arm_A_follower",
}
CAMERAS = {
    "front_left": "/dev/cam_front_left",
    "front_right": "/dev/cam_front_right",
    "wrist_left": "/dev/cam_wrist_left",
    "wrist_right": "/dev/cam_wrist_right",
}
# The server skips the training repack, so send the repacked names
# (global_image/left_image/right_image -> cam_high/cam_left_wrist/cam_right_wrist).
# front_right has no slot in this checkpoint.
MODEL_CAMERAS = {
    "cam_high": "front_left",
    "cam_left_wrist": "wrist_left",
    "cam_right_wrist": "wrist_right",
}


def pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def arm_vector(obs, side):
    names = (*JOINTS, "gripper")
    return np.array([float(obs[f"{side}_{name}.pos"]) for name in names], dtype=np.float32)


def so_arm_state(left, right):
    """Inverse of the training layout: degrees and 0..100 -> radians and 0..1, with two pad joints."""
    pad = np.zeros(2, dtype=np.float32)
    joints = np.concatenate(
        [np.deg2rad(left[:5]), pad, np.deg2rad(right[:5]), pad]
    ).astype(np.float32)
    gripper = np.array([left[5], right[5]], dtype=np.float32) / 100.0
    return joints, gripper


def chunk_to_degrees(actions):
    """16-dim absolute actions -> 12-dim degrees / 0..100 used by bi_so_follower."""
    left_joints = np.rad2deg(actions[:, 0:5])
    right_joints = np.rad2deg(actions[:, 8:13])
    left_gripper = actions[:, 7:8] * 100.0
    right_gripper = actions[:, 15:16] * 100.0
    return np.concatenate([left_joints, left_gripper, right_joints, right_gripper], axis=-1)


def as_hwc_uint8(image):
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def save_frames(out_dir, frames):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def make_robot(max_relative_target=None):
    for side, port in FOLLOWER_PORTS.items():
        if "leader" in port:
            raise RuntimeError(f"refusing to open a leader arm for {side}: {port}")
    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=path, fps=30, width=640, height=480, fourcc="MJPG"
        )
        for name, path in CAMERAS.items()
    }
    logging.info(
        "followers only: left %s, right %s; leader ports are not opened",
        FOLLOWER_PORTS["left"],
        FOLLOWER_PORTS["right"],
    )
    return BiSOFollower(
        BiSOFollowerConfig(
            id="bimanual_follower",
            left_arm_config=SOFollowerConfig(port=FOLLOWER_PORTS["left"]),
            right_arm_config=SOFollowerConfig(
                port=FOLLOWER_PORTS["right"], max_relative_target=max_relative_target
            ),
            cameras=cameras,
        )
    )


def _ping_follower(port, result):
    import scservo_sdk as scs

    handler = scs.PortHandler(port)
    packet = scs.PacketHandler(0)
    try:
        opened = handler.openPort()
    except Exception as exc:
        result["error"] = f"open failed: {exc}"
        return
    if not opened:
        result["error"] = "open failed"
        return
    try:
        handler.setBaudRate(1_000_000)
        handler.setPacketTimeoutMillis(50)
        result["found"] = [
            motor_id
            for motor_id in range(1, 7)
            if packet.ping(handler, motor_id)[1] == scs.COMM_SUCCESS
        ]
    finally:
        handler.closePort()


def check_followers(timeout_s=8.0):
    """A wedged USB-serial board blocks pyserial forever, so probe each follower in a thread."""
    for side, port in FOLLOWER_PORTS.items():
        if not Path(port).exists():
            raise RuntimeError(
                f"{side} follower {port} does not exist; its USB board is not enumerated, "
                "check that the cable is seated (see `ls -l /dev/arm_*`)"
            )
        result = {}
        worker = threading.Thread(target=_ping_follower, args=(port, result), daemon=True)
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            raise RuntimeError(
                f"{side} follower {port} ({Path(port).resolve()}) did not respond within {timeout_s:.0f}s; "
                "replug that arm's USB cable or power-cycle its board"
            )
        if "found" not in result:
            raise RuntimeError(f"{side} follower {port}: {result.get('error', 'ping failed')}")
        missing = sorted(set(range(1, 7)) - set(result["found"]))
        if missing:
            raise RuntimeError(f"{side} follower {port}: motor ids {missing} did not answer")
        logging.info("%s follower %s ok, motors 1-6", side, port)


def disable_auto_exposure_fps():
    for path in (CAMERAS["wrist_left"], CAMERAS["wrist_right"]):
        subprocess.run(
            ["v4l2-ctl", "-d", path, "-c", "exposure_dynamic_framerate=0"],
            check=False,
            capture_output=True,
        )


class PolicyClient:
    def __init__(self, host, port):
        uri = f"ws://{host}:{port}"
        logging.info("connecting to %s", uri)
        self._ws = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            open_timeout=20,
            legacy=True,
        )
        self._packer = msgpack.Packer(default=pack_array)
        self.metadata = msgpack.unpackb(self._ws.recv(), object_hook=unpack_array)

    def infer(self, obs):
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return msgpack.unpackb(response, object_hook=unpack_array)

    def close(self):
        self._ws.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3568)
    parser.add_argument("--task", default="vacuum under the chairs.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send the chunk to the follower arms only; leaders are not connected",
    )
    parser.add_argument(
        "--steps", type=int, default=25, help="chunk steps executed before re-inferring"
    )
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument(
        "--duration", type=float, default=20.0, help="seconds of closed-loop execution"
    )
    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=10.0,
        help="cap on how far a right-follower goal may be from its present position",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    out_dir = Path(__file__).resolve().parent / "last_infer"
    check_followers()
    disable_auto_exposure_fps()

    robot = make_robot(args.max_step_deg if args.execute else None)
    client = None
    try:
        robot.connect()
        client = PolicyClient(args.host, args.port)
        logging.info("server metadata keys %s", list(client.metadata))

        # The left follower holds where it started; this checkpoint never learned it.
        hold_left = None
        period = 1.0 / args.hz
        deadline = time.monotonic() + args.duration
        chunk_index = 0
        while True:
            obs = robot.get_observation()
            frames = {name: as_hwc_uint8(obs[name]) for name in CAMERAS}
            left = arm_vector(obs, "left")
            right = arm_vector(obs, "right")
            if hold_left is None:
                hold_left = left
                save_frames(out_dir, frames)
                for name, frame in frames.items():
                    logging.info("camera %s %s", name, frame.shape)
                logging.info("left  deg/gripper %s", np.round(left, 2))
            logging.info("chunk %s right now %s", chunk_index, np.round(right, 2))

            joints, gripper = so_arm_state(left, right)
            started = time.monotonic()
            result = client.infer(
                {
                    # _decode_xe expects CHW, like the training frames.
                    "images": {
                        key: np.ascontiguousarray(frames[name].transpose(2, 0, 1))
                        for key, name in MODEL_CAMERAS.items()
                    },
                    "state.joints": joints,
                    "state.gripper_w": gripper,
                    "prompt": args.task,
                }
            )
            actions = np.asarray(result["actions"], dtype=np.float32)
            elapsed = time.monotonic() - started
            degrees = chunk_to_degrees(actions)[: args.steps]
            span = np.ptp(degrees[:, 6:], axis=0)
            logging.info(
                "chunk %s infer %.2fs step0 delta %s range over %s steps %s",
                chunk_index,
                elapsed,
                np.round(degrees[0, 6:] - right, 2),
                len(degrees),
                np.round(span, 2),
            )
            if chunk_index == 0:
                np.save(out_dir / "actions_16.npy", actions)
                np.save(out_dir / "actions_deg.npy", degrees)
                (out_dir / "summary.txt").write_text(
                    f"shape {actions.shape}\ninfer_s {elapsed:.3f}\n"
                    f"left_now {np.round(left, 3).tolist()}\n"
                    f"right_now {np.round(right, 3).tolist()}\n"
                    f"step0_deg {np.round(degrees[0], 3).tolist()}\n"
                    f"right_delta0 {np.round(degrees[0, 6:] - right, 3).tolist()}\n"
                )

            if not args.execute:
                break

            for goal in degrees:
                action = {}
                for name, value in zip(JOINTS, hold_left[:5]):
                    action[f"left_{name}.pos"] = float(value)
                action["left_gripper.pos"] = float(hold_left[5])
                for name, value in zip(JOINTS, goal[6:11]):
                    action[f"right_{name}.pos"] = float(value)
                action["right_gripper.pos"] = float(goal[11])
                robot.send_action(action)
                time.sleep(period)

            reached = arm_vector(robot.get_observation(), "right")
            logging.info(
                "chunk %s goal %s reached %s moved %s",
                chunk_index,
                np.round(degrees[-1, 6:], 2),
                np.round(reached, 2),
                np.round(reached - right, 2),
            )
            chunk_index += 1
            if time.monotonic() >= deadline:
                break
    finally:
        if client is not None:
            client.close()
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            logging.exception("robot disconnect failed")


if __name__ == "__main__":
    main()
