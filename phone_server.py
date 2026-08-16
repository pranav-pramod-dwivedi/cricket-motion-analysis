"""
Phone-as-camera server.
=======================

Architecture (chosen to eliminate network latency):

    PHONE (browser)                        PC (browser)
    ---------------                        ------------
    getUserMedia @60fps                    dashboard
    -> canvas                                  ^
    -> HSV bat detection                       |
    -> Kalman + skeleton fusion                | tiny JSON
    -> draws overlay ON PHONE  ----------------+  (~200 bytes/frame)
                                            via SSE

No video ever crosses the network. Only stats do. That is why latency is
near-zero compared with MJPEG streaming (IP Webcam / DroidCam), where every
frame is JPEG-encoded, sent over WiFi, and decoded on the PC.

Run:
    python3 phone_server.py

Then open the printed https:// URL on the phone, and the same URL + /pc on
this computer.

HTTPS is mandatory: browsers refuse camera access on plain HTTP for any
non-localhost origin. We self-sign a certificate, so the phone will show a
"Not private" warning once — tap Advanced -> Proceed.
"""

import http.server
import json
import os
import queue
import socket
import ssl
import subprocess
import threading
import webbrowser

PORT = 8443
HERE = os.path.dirname(os.path.abspath(__file__))
CERT = os.path.join(HERE, ".cert.pem")
KEY = os.path.join(HERE, ".key.pem")

# --- fan-out of phone stats to any connected dashboards --------------------
_clients = []
_clients_lock = threading.Lock()
_latest = {}

# most recent low-rate preview frame (JPEG bytes) for the PC mirror
_frame = {"data": None}
# saved swing clips: newest first, capped so memory can't grow without bound
_clips = []
_clips_lock = threading.Lock()
MAX_CLIPS = 12


def _broadcast(obj):
    global _latest
    _latest = obj
    dead = []
    with _clients_lock:
        for q in _clients:
            try:
                q.put_nowait(obj)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _clients.remove(q)


def local_ip():
    """Best-guess LAN IP (the address the phone must reach)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert():
    """Self-signed cert so the phone browser will grant camera permission."""
    if os.path.exists(CERT) and os.path.exists(KEY):
        return
    ip = local_ip()
    print("Generating self-signed certificate…")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", KEY, "-out", CERT, "-days", "825",
        "-subj", "/CN=cricket-tracker",
        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
    ], check=True, capture_output=True)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass  # keep the console clean

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name, ctype):
        try:
            with open(os.path.join(HERE, "web", name), "rb") as f:
                self._send(f.read(), ctype)
        except FileNotFoundError:
            self._send("not found", code=404)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/phone"):
            self._file("phone.html", "text/html; charset=utf-8")
        elif path == "/pc":
            self._file("pc.html", "text/html; charset=utf-8")
        elif path == "/latest":
            self._send(json.dumps(_latest), "application/json")
        elif path == "/events":
            self._sse()
        elif path == "/frame":
            data = _frame["data"]
            if not data:
                self._send("no frame", code=404)
            else:
                self._send(data, "image/jpeg")
        elif path == "/clips":
            with _clips_lock:
                meta = [{k: c[k] for k in ("id", "ts", "peak", "angle", "frames")}
                        for c in _clips]
            self._send(json.dumps(meta), "application/json")
        elif path.startswith("/clip/"):
            self._clip(path[len("/clip/"):])
        elif path.startswith("/vendor/"):
            self._vendor(path[len("/vendor/"):])
        elif path == "/virtualbat.js":
            self._file("virtualbat.js", "text/javascript")
        else:
            self._send("not found", code=404)

    # MediaPipe assets are vendored locally so this works with no internet.
    _MIME = {
        ".mjs": "text/javascript", ".js": "text/javascript",
        ".wasm": "application/wasm", ".task": "application/octet-stream",
    }

    def _vendor(self, name):
        if "/" in name or ".." in name:      # no path traversal
            self._send("bad path", code=400)
            return
        full = os.path.join(HERE, "web", "vendor", name)
        if not os.path.isfile(full):
            self._send("not found", code=404)
            return
        ext = os.path.splitext(name)[1]
        ctype = self._MIME.get(ext, "application/octet-stream")
        size = os.path.getsize(full)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        with open(full, "rb") as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)

    def do_POST(self):
        path = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""

        if path == "/stats":
            try:
                _broadcast(json.loads(raw))
            except json.JSONDecodeError:
                pass
            self._send("{}", "application/json")

        elif path == "/frame":
            # low-rate live preview for the PC mirror
            _frame["data"] = raw
            self._send("{}", "application/json")

        elif path == "/clip":
            # a completed swing: JSON envelope with base64 JPEG frames
            try:
                clip = json.loads(raw)
            except json.JSONDecodeError:
                self._send('{"error":"bad json"}', "application/json", 400)
                return
            clip.setdefault("id", str(int(clip.get("ts", 0))))
            clip["frames"] = len(clip.get("jpegs", []))
            with _clips_lock:
                _clips.insert(0, clip)
                del _clips[MAX_CLIPS:]
            # tell dashboards a new swing landed
            _broadcast({**_latest, "newClip": clip["id"], "clipPeak": clip.get("peak")})
            print(f"  swing saved: {clip['frames']} frames, "
                  f"peak {clip.get('peak', 0):.0f} px/s")
            self._send(json.dumps({"ok": True, "id": clip["id"]}),
                       "application/json")
        else:
            self._send("not found", code=404)

    def _clip(self, cid):
        with _clips_lock:
            for c in _clips:
                if c["id"] == cid:
                    self._send(json.dumps(c), "application/json")
                    return
        self._send("not found", code=404)

    def _sse(self):
        """Server-sent events: push phone stats to the PC dashboard."""
        q = queue.Queue(maxsize=8)
        with _clients_lock:
            _clients.append(q)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    obj = q.get(timeout=15)
                    payload = f"data: {json.dumps(obj)}\n\n"
                except queue.Empty:
                    payload = ": keepalive\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _clients_lock:
                if q in _clients:
                    _clients.remove(q)


class ThreadedServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ensure_cert()
    ip = local_ip()
    srv = ThreadedServer(("0.0.0.0", PORT), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT, KEY)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)

    url = f"https://{ip}:{PORT}"
    print("\n" + "=" * 58)
    print("  CRICKET TRACKER — phone camera server")
    print("=" * 58)
    print(f"  ON YOUR PHONE :  {url}")
    print(f"  ON THIS MAC   :  {url}/pc")
    print("=" * 58)
    print("  Both devices must be on the SAME WiFi.")
    print("  The phone will warn 'Not private' — tap Advanced > Proceed.")
    print("  Ctrl+C to stop.\n")
    try:
        webbrowser.open(f"https://127.0.0.1:{PORT}/pc")
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
