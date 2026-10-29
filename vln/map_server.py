#!/usr/bin/env python3
"""常驻的二维图网页：coverage_run.py 停了也能看，下次运行后接着更新。

一般不用手动开：coverage_run.py 发现它没在跑，会自己在后台开一个，关掉终端也不会停。

  python map_server.py         # 手动开，浏览器打开 http://<jetson>:8765
  pkill -f map_server.py       # 关掉；改了这个文件后关掉，下次 coverage_run.py 会重新开
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from live_map import LIVE_FILE, PAGE, blank, object_rows, read_state, render  # noqa: E402

MAP_DIR = os.path.join(HERE, "map")
STALE_S = 5.0  # no update for this long while marked running: the program died without saying so


class Cache:
    """Re-read map/live.npz only when it changes, and draw it once per change."""

    def __init__(self, map_dir):
        self.path = os.path.join(map_dir, LIVE_FILE)
        self.map_dir = map_dir
        self.lock = threading.Lock()
        self.mtime = None
        self.state = None
        self.jpeg = None
        self.version = 0

    def get(self):
        with self.lock:
            try:
                mtime = os.path.getmtime(self.path)
            except OSError:
                mtime = None
            if mtime != self.mtime:
                self.mtime = mtime
                self.version += 1
                self.state = self.jpeg = None
                if mtime is not None:
                    try:
                        self.state = read_state(self.map_dir)
                    except Exception as e:  # half-written or old-format file: keep serving
                        print(f"read {self.path}: {e}", flush=True)
            return self.state, self.version

    def frame(self):
        st, _ = self.get()
        with self.lock:
            if self.jpeg is None:
                img = render(st) if st is not None else blank()
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                self.jpeg = buf.tobytes() if ok else b""
            return self.jpeg

    def stats(self):
        st, version = self.get()
        if st is None:
            return {"status": "还没有地图，运行 coverage_run.py", "running": False, "has_map": False,
                    "version": version, "furniture": []}
        updated = float(st.get("updated", 0.0))
        running = bool(st.get("running")) and time.time() - updated < STALE_S
        when = time.strftime("%m-%d %H:%M:%S", time.localtime(updated))
        status = st["status"] if running else f"程序没在运行 · 上次：{st['status']}"
        return {"status": status, "stage": st.get("stage", ""), "running": running,
                "when": f"更新于 {when}", "has_map": st.get("has_map", False),
                "sweep": st.get("sweep", 0.0), "furniture": st.get("furniture", []), "version": version,
                "objects": object_rows(st.get("objects") or [])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    cache = Cache(MAP_DIR)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body, ctype = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            elif path == "/frame.jpg":
                body, ctype = cache.frame(), "image/jpeg"
            elif path == "/stats":
                body = json.dumps(cache.stats(), ensure_ascii=False).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"map server on :{args.port}, reading {cache.path}", flush=True)
    threading.Thread(target=httpd.serve_forever, name="http", daemon=True).start()
    # The server outlives coverage_run.py, so it would keep drawing with old code after an update.
    # When the drawing code changes it starts itself again (the open page reconnects on its own).
    sources = [os.path.abspath(__file__), os.path.join(HERE, "live_map.py"), os.path.join(HERE, "occmap.py")]
    stamp = [os.path.getmtime(f) for f in sources]
    while True:
        time.sleep(2.0)
        try:
            changed = [os.path.getmtime(f) for f in sources] != stamp
        except OSError:
            continue
        if changed:
            print("drawing code changed, restarting the map server", flush=True)
            httpd.server_close()
            os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    main()
