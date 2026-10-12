import os

if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"
_xauth = "/run/user/%d/gdm/Xauthority" % os.getuid()
if not os.environ.get("XAUTHORITY") and os.path.exists(_xauth):
    os.environ["XAUTHORITY"] = _xauth

import argparse
import subprocess
import time

import cv2
import numpy as np

BY_PATH = "/dev/v4l/by-path/platform-3610000.usb-usb-0:{}:1.0-video-index0"
LEFT_PORT = "2.3.4"   # USB hub 2.3 also carries arm_B (left arm) boards
RIGHT_PORT = "2.1.3"  # USB hub 2.1 also carries arm_A (right arm) boards


def open_camera(port, width, height, fps):
    path = BY_PATH.format(port)
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Without this the wrist cameras drop to 8-15 fps whenever the scene is dark.
    subprocess.run(
        ["v4l2-ctl", "-d", os.path.realpath(path), "-c", "exposure_dynamic_framerate=0"],
        check=False, capture_output=True,
    )
    return cap


def annotate(frame, label, port, fps, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    info = f"USB {port}  {fps:4.1f} fps"
    (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, info, (frame.shape[1] - tw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 3)
    return frame


def no_signal(width, height):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, "NO SIGNAL", (width // 2 - 110, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Live view of left/right wrist cameras")
    parser.add_argument("--left-port", default=LEFT_PORT)
    parser.add_argument("--right-port", default=RIGHT_PORT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    ports = {"LEFT wrist": args.left_port, "RIGHT wrist": args.right_port}
    colors = {"LEFT wrist": (0, 200, 0), "RIGHT wrist": (0, 140, 255)}
    caps = {}
    for label, port in ports.items():
        caps[label] = open_camera(port, args.width, args.height, args.fps)
        print(f"{label:12s} USB {port:6s} opened: {caps[label].isOpened()}")

    print("Window shows on the robot screen. Keys: s = swap left/right, q / ESC = quit")

    swapped = False
    counts = {label: 0 for label in caps}
    fps = {label: 0.0 for label in caps}
    t_last = time.time()
    window = "Wrist cameras (s: swap, q: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = {}
            for label, cap in caps.items():
                ok, frame = cap.read()
                if ok:
                    counts[label] += 1
                    frames[label] = cv2.resize(frame, (args.width, args.height))
                else:
                    frames[label] = no_signal(args.width, args.height)

            now = time.time()
            if now - t_last >= 1.0:
                for label in caps:
                    fps[label] = counts[label] / (now - t_last)
                    counts[label] = 0
                t_last = now

            # When swapped, the stream from the other port gets the label.
            left_src, right_src = ("RIGHT wrist", "LEFT wrist") if swapped else ("LEFT wrist", "RIGHT wrist")
            left_img = annotate(frames[left_src].copy(), "LEFT wrist", ports[left_src],
                                fps[left_src], colors["LEFT wrist"])
            right_img = annotate(frames[right_src].copy(), "RIGHT wrist", ports[right_src],
                                 fps[right_src], colors["RIGHT wrist"])
            cv2.imshow(window, np.hstack([left_img, right_img]))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                swapped = not swapped
                print(current_mapping(ports, swapped))
    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        print("Final " + current_mapping(ports, swapped))


def current_mapping(ports, swapped):
    left = ports["RIGHT wrist"] if swapped else ports["LEFT wrist"]
    right = ports["LEFT wrist"] if swapped else ports["RIGHT wrist"]
    return f"mapping: LEFT wrist = USB {left}, RIGHT wrist = USB {right}"


if __name__ == "__main__":
    main()
