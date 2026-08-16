"""
Bat Tracker GUI
===============
Live webcam view + tracking overlay + HSV tuning sliders in one window.

Detects a dull-BLUE object (bat body) and a dull-BROWN tip, draws a line
representing the bat, and lets you tune the color ranges live with sliders.

Run:
    python3 bat_gui.py

Tips:
    - Use the sliders on the right to dial in each color until the little
      "Blue mask" / "Brown mask" previews light up cleanly on your object.
    - "Show masks" swaps the main view to the raw masks so you can see exactly
      what is being detected.
    - "Reset" restores default ranges. "Save" prints current values to console.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk  # Pillow: for showing OpenCV frames in Tkinter

MIN_AREA = 300  # ignore blobs smaller than this (noise)

# Default HSV ranges (OpenCV hue 0-179). Tuned for dull/muted colors.
DEFAULTS = {
    "blue":  {"hl": 95, "hh": 135, "sl": 50, "sh": 255, "vl": 40, "vh": 255},
    "brown": {"hl": 5,  "hh": 28,  "sl": 30, "sh": 200, "vl": 30, "vh": 190},
}


def color_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def largest_centroid(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


class BatGUI:
    def __init__(self, root):
        self.root = root
        root.title("Bat Tracker")
        root.configure(bg="#1e1e1e")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")

        self.show_masks = tk.BooleanVar(value=False)
        self.mirror = tk.BooleanVar(value=True)
        self.sliders = {}  # (color, key) -> tk.IntVar

        self._build_layout()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_frame()

    # ---------------------------------------------------------------- layout
    def _build_layout(self):
        main = tk.Frame(self.root, bg="#1e1e1e")
        main.pack(padx=10, pady=10)

        # Left: video + status
        left = tk.Frame(main, bg="#1e1e1e")
        left.grid(row=0, column=0, sticky="n")

        self.video_label = tk.Label(left, bg="black")
        self.video_label.pack()

        self.status = tk.Label(left, text="Starting…", fg="#e0e0e0",
                               bg="#1e1e1e", font=("Helvetica", 12), anchor="w")
        self.status.pack(fill="x", pady=(8, 0))

        # small mask previews under the video
        prev = tk.Frame(left, bg="#1e1e1e")
        prev.pack(pady=(8, 0))
        self.blue_prev = tk.Label(prev, bg="black")
        self.blue_prev.grid(row=1, column=0, padx=4)
        self.brown_prev = tk.Label(prev, bg="black")
        self.brown_prev.grid(row=1, column=1, padx=4)
        tk.Label(prev, text="Blue mask", fg="#7aa2f7", bg="#1e1e1e").grid(row=0, column=0)
        tk.Label(prev, text="Brown mask", fg="#c99", bg="#1e1e1e").grid(row=0, column=1)

        # Right: controls
        right = tk.Frame(main, bg="#1e1e1e")
        right.grid(row=0, column=1, sticky="n", padx=(16, 0))

        self._build_color_panel(right, "BLUE  (bat body)", "blue", "#7aa2f7")
        self._build_color_panel(right, "BROWN  (tip)", "brown", "#c99")

        btns = tk.Frame(right, bg="#1e1e1e")
        btns.pack(fill="x", pady=(6, 0))
        tk.Checkbutton(btns, text="Show masks", variable=self.show_masks,
                       fg="#e0e0e0", bg="#1e1e1e", selectcolor="#333",
                       activebackground="#1e1e1e", activeforeground="#fff"
                       ).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(btns, text="Mirror", variable=self.mirror,
                       fg="#e0e0e0", bg="#1e1e1e", selectcolor="#333",
                       activebackground="#1e1e1e", activeforeground="#fff"
                       ).grid(row=0, column=1, sticky="w")
        ttk.Button(btns, text="Reset", command=self.reset).grid(row=1, column=0, pady=6, sticky="ew")
        ttk.Button(btns, text="Save (print)", command=self.save).grid(row=1, column=1, pady=6, sticky="ew")

    def _build_color_panel(self, parent, title, color, accent):
        frame = tk.LabelFrame(parent, text=title, fg=accent, bg="#1e1e1e",
                              font=("Helvetica", 11, "bold"), padx=8, pady=6)
        frame.pack(fill="x", pady=(0, 10))
        specs = [("Hue", "hl", "hh", 179), ("Sat", "sl", "sh", 255), ("Val", "vl", "vh", 255)]
        for row, (name, lo, hi, mx) in enumerate(specs):
            tk.Label(frame, text=name, fg="#bbb", bg="#1e1e1e", width=4
                     ).grid(row=row, column=0)
            for col, key in ((1, lo), (2, hi)):
                var = tk.IntVar(value=DEFAULTS[color][key])
                self.sliders[(color, key)] = var
                tk.Scale(frame, from_=0, to=mx, orient="horizontal", variable=var,
                         length=120, bg="#1e1e1e", fg="#e0e0e0", troughcolor="#333",
                         highlightthickness=0, showvalue=True
                         ).grid(row=row, column=col, padx=2)

    # ------------------------------------------------------------- bounds
    def bounds(self, color):
        s = self.sliders
        lower = np.array([s[(color, "hl")].get(), s[(color, "sl")].get(), s[(color, "vl")].get()])
        upper = np.array([s[(color, "hh")].get(), s[(color, "sh")].get(), s[(color, "vh")].get()])
        return lower, upper

    # ------------------------------------------------------------- loop
    def update_frame(self):
        ok, frame = self.cap.read()
        if ok:
            if self.mirror.get():
                frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)

            blue_mask = color_mask(hsv, *self.bounds("blue"))
            brown_mask = color_mask(hsv, *self.bounds("brown"))
            blue = largest_centroid(blue_mask)
            brown = largest_centroid(brown_mask)

            if blue:
                cv2.circle(frame, blue, 9, (255, 0, 0), -1)
            if brown:
                cv2.circle(frame, brown, 9, (19, 69, 139), -1)

            if blue and brown:
                base = np.array(blue, float)
                tip = np.array(brown, float)
                d = tip - base
                n = np.linalg.norm(d)
                if n > 1:
                    end = tip + (d / n) * 40
                    cv2.line(frame, tuple(base.astype(int)), tuple(end.astype(int)),
                             (0, 0, 255), 6)
                self.status.config(text="TRACKING  ✓   bat line drawn", fg="#7fe08a")
            elif blue:
                self.status.config(text="Blue found — show the BROWN tip", fg="#e0c060")
            elif brown:
                self.status.config(text="Brown found — show the BLUE body", fg="#e0c060")
            else:
                self.status.config(text="Searching… show blue body + brown tip", fg="#e08a8a")

            # main view: frame or masks
            if self.show_masks.get():
                combo = cv2.cvtColor(cv2.bitwise_or(blue_mask, brown_mask), cv2.COLOR_GRAY2BGR)
                self._show(self.video_label, combo)
            else:
                self._show(self.video_label, frame)

            self._show(self.blue_prev, blue_mask, size=(160, 120), gray=True)
            self._show(self.brown_prev, brown_mask, size=(160, 120), gray=True)

        self.root.after(15, self.update_frame)

    def _show(self, widget, img, size=None, gray=False):
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if size:
            img = cv2.resize(img, size)
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        widget.configure(image=photo)
        widget.image = photo  # keep a reference so it isn't garbage-collected

    # ------------------------------------------------------------- buttons
    def reset(self):
        for (color, key), var in self.sliders.items():
            var.set(DEFAULTS[color][key])

    def save(self):
        for color in ("blue", "brown"):
            lo, hi = self.bounds(color)
            print(f"{color.upper()}_LOWER = np.array({lo.tolist()})")
            print(f"{color.upper()}_UPPER = np.array({hi.tolist()})")
        print("-" * 40)

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    BatGUI(root)
    root.mainloop()
