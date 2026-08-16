"""
Cricket Motion-Analysis Engine — Milestone 1
============================================
Live webcam GUI with:
  - Full-body skeleton tracking (MediaPipe)
  - Geometric bat detection (blue elongated object) -> bat vector, angle, speed
  - Swing recognition + shot-quality scores (heuristic)
  - Per-frame JSON Game Output API (shown live, and printable)
  - HSV tuning sliders, mask preview, FPS counter, distance estimate

Run:
    python3 cricket_app.py

This is the modular foundation. Future features (calibration wizard, 3D hitbox,
ghost comparison, ball detection) bolt onto engine/ modules.
"""

import json
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from engine.pose_module import PoseTracker, SKELETON, LM
from engine.bat_module import BatTracker
from engine import game_api


def _clean(obj):
    """Recursively convert numpy scalars/arrays to native Python types so the
    dict is JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def lighting_quality(frame):
    """Classify overall brightness -> warning string."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m = gray.mean()
    if m < 55:
        return "dark", "⚠ Lighting too dark"
    if m > 210:
        return "bright", "⚠ Too bright / overexposed"
    return "good", ""


class CricketApp:
    def __init__(self, root):
        self.root = root
        root.title("Cricket Motion Engine — M1")
        root.configure(bg="#141414")

        self.cap = self._open_camera()
        if self.cap is None:
            raise RuntimeError(
                "Could not open the webcam.\n\n"
                "This is almost always a macOS permission issue:\n"
                "  System Settings > Privacy & Security > Camera > enable Terminal,\n"
                "  then FULLY QUIT Terminal (Cmd+Q) and reopen it.\n\n"
                "Also make sure no other app (Zoom, FaceTime, Photo Booth) is using "
                "the camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.pose = PoseTracker()
        self.bat = BatTracker()

        # Pose is by far the most expensive stage (~29 ms vs ~2 ms for the bat),
        # and the body moves far slower than the bat. So run pose every Nth
        # frame and reuse the last skeleton in between — the bat, which is what
        # actually moves fast, still updates on every single frame.
        self.pose_interval = 2
        self._pose_i = 0
        self._last_pose = None

        self.mirror = tk.BooleanVar(value=True)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.show_masks = tk.BooleanVar(value=False)
        self.show_panel = tk.BooleanVar(value=True)
        self.sliders = {}
        self.disp_w, self.disp_h = 960, 540  # big mirror view

        self.fps = 0.0
        self.sharpness = 0.0   # laplacian variance; <100 means soft/blurry
        self._last_t = time.time()
        self._frames = 0
        self.latest_json = {}

        self._build()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.loop()

    def _open_camera(self):
        """Open a camera source.

        Accepts a phone streaming over Wi-Fi (IP Webcam / DroidCam) via the
        CRICKET_CAM env var or a --cam argument, e.g.

            python3 cricket_app.py --cam http://192.168.1.7:8080/video

        Falls back to scanning local USB/built-in camera indexes.
        """
        src = os.environ.get("CRICKET_CAM")
        for i, a in enumerate(sys.argv):
            if a == "--cam" and i + 1 < len(sys.argv):
                src = sys.argv[i + 1]
            elif a.startswith("--cam="):
                src = a.split("=", 1)[1]

        if src:
            url = self._normalize_cam_url(src)
            print(f"Connecting to phone camera: {url}")
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    # keep latency low: don't let frames pile up in the buffer
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print("Phone camera connected ✓")
                    return cap
            cap.release()
            print("Could not open that stream — falling back to local camera.")

        for idx in (0, 1, 2):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"Camera opened on index {idx}")
                    return cap
            cap.release()
        return None

    @staticmethod
    def _normalize_cam_url(src):
        """Accept a bare IP/host and turn it into an IP Webcam video URL."""
        if src.isdigit():
            return int(src)
        s = src.strip()
        if not s.startswith(("http://", "https://", "rtsp://")):
            s = "http://" + s
        # bare host or host:port -> add IP Webcam's default port/path
        tail = s.split("://", 1)[1]
        if ":" not in tail.split("/")[0]:
            s = s.rstrip("/") + ":8080"
        if not s.rstrip("/").endswith(("/video", "/videofeed", ".mjpg")) \
                and "/" not in s.split("://", 1)[1].split(":", 1)[-1]:
            s = s.rstrip("/") + "/video"
        return s

    # -------------------------------------------------------------- UI
    def _build(self):
        wrap = tk.Frame(self.root, bg="#141414")
        wrap.pack(padx=10, pady=10, fill="both", expand=True)

        # ---- LEFT: big mirror view ----
        left = tk.Frame(wrap, bg="#141414")
        left.grid(row=0, column=0, sticky="n")
        self.video = tk.Label(left, bg="black")
        self.video.pack()

        bar = tk.Frame(left, bg="#141414")
        bar.pack(fill="x", pady=(8, 0))
        self.status = tk.Label(bar, text="Starting…", fg="#eee", bg="#141414",
                               font=("Helvetica", 14, "bold"))
        self.status.pack(side="left")
        self.warn = tk.Label(bar, text="", fg="#ff6b6b", bg="#141414",
                             font=("Helvetica", 12))
        self.warn.pack(side="left", padx=12)
        tk.Checkbutton(bar, text="Show controls", variable=self.show_panel,
                       command=self._toggle_panel, fg="#eee", bg="#141414",
                       selectcolor="#333", activebackground="#141414",
                       activeforeground="#fff").pack(side="right")

        # ---- RIGHT: collapsible control panel ----
        self.panel = tk.Frame(wrap, bg="#141414")
        self.panel.grid(row=0, column=1, sticky="n", padx=(14, 0))

        self._color_panel(self.panel, "BLUE (bat)", "blue", "#7aa2f7")

        opts = tk.Frame(self.panel, bg="#141414")
        opts.pack(fill="x")
        for i, (txt, var) in enumerate([("Mirror", self.mirror),
                                        ("Skeleton", self.show_skeleton),
                                        ("Masks", self.show_masks)]):
            tk.Checkbutton(opts, text=txt, variable=var, fg="#eee", bg="#141414",
                           selectcolor="#333", activebackground="#141414",
                           activeforeground="#fff").grid(row=0, column=i, sticky="w")
        ttk.Button(self.panel, text="Reset colors", command=self.reset).pack(fill="x", pady=(6, 2))
        ttk.Button(self.panel, text="Print JSON to console", command=self.print_json).pack(fill="x")

        tk.Label(self.panel, text="Game Output (live JSON)", fg="#eee", bg="#141414",
                 font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(10, 0))
        self.json_box = tk.Text(self.panel, width=40, height=24, bg="#0c0c0c",
                                fg="#8fe38f", font=("Courier", 9), wrap="none")
        self.json_box.pack()

    def _toggle_panel(self):
        if self.show_panel.get():
            self.panel.grid()
        else:
            self.panel.grid_remove()

    def _color_panel(self, parent, title, color, accent):
        f = tk.LabelFrame(parent, text=title, fg=accent, bg="#141414",
                          font=("Helvetica", 10, "bold"), padx=6, pady=4)
        f.pack(fill="x", pady=(0, 8))
        rng = self.bat.ranges[color]
        rows = [("H", 0, 179), ("S", 1, 255), ("V", 2, 255)]
        for r, (nm, idx, mx) in enumerate(rows):
            tk.Label(f, text=nm, fg="#aaa", bg="#141414", width=2).grid(row=r, column=0)
            for c, which in ((1, "lower"), (2, "upper")):
                v = tk.IntVar(value=rng[which][idx])
                self.sliders[(color, which, idx)] = v
                tk.Scale(f, from_=0, to=mx, orient="horizontal", variable=v,
                         length=90, bg="#141414", fg="#eee", troughcolor="#333",
                         highlightthickness=0,
                         command=lambda _v, cl=color, wh=which: self._sync(cl, wh)
                         ).grid(row=r, column=c, padx=1)

    def _sync(self, color, which):
        vals = [self.sliders[(color, which, i)].get() for i in range(3)]
        self.bat.set_range(color, which, vals)

    # -------------------------------------------------------------- loop
    def loop(self):
        ok, frame = self.cap.read()
        if ok:
            if self.mirror.get():
                frame = cv2.flip(frame, 1)

            self._pose_i += 1
            if self._pose_i % self.pose_interval == 0 or self._last_pose is None:
                self._last_pose = self.pose.process(frame)
            pose = self._last_pose
            bat = self.bat.process(frame, pose)
            light_key, warn_txt = lighting_quality(frame)

            # sharpness probe (every 10th frame, on a downscaled copy — a
            # full-res laplacian every frame would cost us real FPS)
            if self._pose_i % 10 == 0:
                small = cv2.resize(frame, (320, 180))
                g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                self.sharpness = float(cv2.Laplacian(g, cv2.CV_64F).var())

            # fps
            self._frames += 1
            now = time.time()
            if now - self._last_t >= 0.5:
                self.fps = self._frames / (now - self._last_t)
                self._frames = 0
                self._last_t = now

            self.latest_json = _clean(game_api.build_frame_output(pose, bat, light_key, self.fps))
            self._draw(frame, pose, bat)
            self._draw_hud(frame, bat.get("confidence", 0.0),
                           bat.get("source", "-"))
            self._update_panels(pose, bat, warn_txt)
            self._render(frame, bat)

        self.root.after(1, self.loop)

    def _draw(self, frame, pose, bat):
        # skeleton
        if pose and self.show_skeleton.get():
            for a, b in SKELETON:
                pa, pb = pose.get(a), pose.get(b)
                if pa and pb and pose.vis(a) > 0.3 and pose.vis(b) > 0.3:
                    cv2.line(frame, pa, pb, (0, 255, 180), 2)
            for name in LM:
                p = pose.get(name)
                if p and pose.vis(name) > 0.3:
                    cv2.circle(frame, p, 4, (0, 200, 255), -1)

        # bat line
        if bat.get("detected"):
            base, tip = np.array(bat["base"]), np.array(bat["tip"])
            # amber while coasting on prediction/skeleton, red when it's real pixels
            col = (0, 190, 255) if bat.get("predicted") else (0, 0, 255)
            d = tip - base
            n = np.linalg.norm(d)
            if n > 1:
                end = (tip + d / n * 40).astype(int)
                cv2.line(frame, tuple(base), tuple(end), col, 6)
            cv2.circle(frame, bat["base"], 8, (255, 0, 0), -1)
            cv2.circle(frame, bat["tip"], 8, (30, 90, 160), -1)
            # swing arc
            for i in range(1, len(bat.get("path", []))):
                cv2.line(frame, bat["path"][i - 1], bat["path"][i], (0, 140, 255), 2)

    def _draw_hud(self, frame, bat_conf=0.0, bat_src="-"):
        """Draw a translucent stats panel directly onto the mirror view."""
        j = self.latest_json
        b, env = j["bat"], j["environment"]
        q = j["quality"] or {}
        lines = [
            f"FPS      {env['fps']}",
            f"Dist     {env['distance_m_est']} m (est)",
            f"Stance   {j['player'].get('stance')}",
            f"Swing    {b.get('swing') if b.get('detected') else '-'}",
            f"Angle    {b.get('angle') if b.get('detected') else '-'}",
            f"Speed    {b.get('speed') if b.get('detected') else '-'}",
            f"Peak     {b.get('peakSpeed') if b.get('detected') else '-'}",
            f"Conf     {round(bat_conf * 100)}%" if bat_conf else "Conf     -",
            f"Track    {bat_src}",
            f"Sharp    {self.sharpness:.0f}{'  (blurry)' if self.sharpness < 100 else ''}",
        ]
        if q:
            lines.append(f"Overall  {q.get('overall','-')}")

        h = 22 * len(lines) + 16
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 10 + h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (20, 36 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 140), 1,
                        cv2.LINE_AA)

    def _update_panels(self, pose, bat, warn_txt):
        j = self.latest_json
        if bat.get("detected") and pose:
            self.status.config(text="TRACKING ✓  player + bat", fg="#7fe08a")
        elif bat.get("detected"):
            self.status.config(text="Bat ✓  (no full body)", fg="#e0c060")
        elif pose:
            self.status.config(text="Player ✓  (show the bat)", fg="#e0c060")
        else:
            self.status.config(text="Searching…", fg="#e08a8a")
        self.warn.config(text=warn_txt)

        self.json_box.delete("1.0", tk.END)
        self.json_box.insert(tk.END, json.dumps(j, indent=2))

    def _render(self, frame, bat):
        if self.show_masks.get() and "masks" in bat:
            blue_m = bat["masks"][0]
            img = cv2.cvtColor(blue_m, cv2.COLOR_GRAY2BGR)
        else:
            img = frame
        disp = cv2.resize(img, (self.disp_w, self.disp_h))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.configure(image=photo)
        self.video.image = photo

    # -------------------------------------------------------------- buttons
    def reset(self):
        from engine.bat_module import DEFAULT_RANGES
        for (color, which, idx), var in self.sliders.items():
            var.set(DEFAULT_RANGES[color][which][idx])
            self._sync(color, which)

    def print_json(self):
        print(json.dumps(self.latest_json, indent=2))
        print("-" * 50)

    def close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    CricketApp(root)
    root.mainloop()
