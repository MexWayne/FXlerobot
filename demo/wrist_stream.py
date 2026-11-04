#!/usr/bin/env python
"""The left wrist camera live in a browser, and its newest frame for the grasp scripts.

    python wrist_stream.py            # http://<jetson>:8766  (Ctrl-C or kill to stop)

The camera is opened once here; grasp_check.py and bottle_to_bin.py take /frame.jpg from it instead
of opening the device themselves (a V4L2 camera can be opened by one program only).
"""

import argparse
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

DEVICE = "/dev/cam_wrist_left"
PORT = 8766
PAGE = b"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>left wrist</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif}img{width:100%;max-width:960px;display:block;margin:auto}</style>
</head><body><img src="/stream"></body></html>"""


class Camera:
    def __init__(self, device):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Without this the wrist cameras drop to 8-15 fps whenever the scene is dark (wrist_camera_test.py).
        subprocess.run(["v4l2-ctl", "-d", os.path.realpath(device), "-c", "exposure_dynamic_framerate=0"],
                       check=False, capture_output=True)
        if not self.cap.isOpened():
            raise SystemExit(f"打不开 {device}")
        self.lock = threading.Lock()
        self.jpeg, self.stamp = None, 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            ok, img = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                with self.lock:
                    self.jpeg, self.stamp = buf.tobytes(), time.time()

    def latest(self):
        with self.lock:
            return self.jpeg, self.stamp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    cam = Camera(DEVICE)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(PAGE, "text/html; charset=utf-8")
            elif path == "/frame.jpg":
                jpeg, stamp = cam.latest()
                if jpeg is None or time.time() - stamp > 1.0:
                    self.send_error(503, "no fresh frame")
                    return
                self._send(jpeg, "image/jpeg")
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = 0.0
                try:
                    while True:
                        jpeg, stamp = cam.latest()
                        if jpeg is not None and stamp != last:
                            last = stamp
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *a):
            return

    print(f"左腕相机：http://0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
