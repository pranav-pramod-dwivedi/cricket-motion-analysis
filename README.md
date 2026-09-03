# Cricket Motion-Analysis Engine

A webcam computer-vision engine for cricket batting, using **MediaPipe** pose
tracking plus geometric bat detection. It reads a live camera, tracks your body and
the bat, recognises swings, estimates shot quality and hands a game a per-frame JSON
stream. No secret sauce, no magic depth cameras — heuristics over a single webcam.

Built as a modular foundation (Milestone 1) so later features bolt onto `engine/`.

## Dates

| Milestone | Date |
|---|---|
| Project started (first files) | **2026-07-18** |
| Desktop GUI (webcam skeleton + bat tracking) | 2026-07-18 → 07-19 |
| Discrete bat tracker (`engine/bat_module.py`) | 2026-07-19 |
| Phone-as-camera / PC dashboard server (`phone_server.py`, `web/`) | 2026-07-19 |
| Virtual-bat skeleton model (`web/virtualbat.js`) | 2026-07-19 |
| **Milestone 1 complete** | **2026-07-19** |

Milestone 1 = live skeleton + bat tracking + swing analysis + per-frame
JSON Game Output API in one working GUI, plus the phone-camera setup.

## Two ways to use it

1. **Desktop GUI** — `python3 cricket_app.py`. One window that does everything:
   skeleton overlay, bat vector, swing scores, HSV tuning and the live JSON output.
2. **Phone camera mode** — `python3 phone_server.py`. Your phone becomes the camera
   (it does the detection on-device over HTTPS), and a PC dashboard shows live stats
   and swing clips over Server-Sent Events. No video crosses the WiFi — only stats do,
   so latency is near-zero.

## How it works (single-webcam pipeline)

```
camera
 ├─► engine/pose_module.py   MediaPipe skeleton → named landmarks (px),
 │     handedness, body orientation, standing height in px
 ├─► engine/bat_module.py    the bat as ONE elongated blue object:
 │     HSV mask → contour filtering → best candidate → minAreaRect
 │     → tip/base from wrists → Kalman + velocity-aware gating
 ├─► engine/analysis_module.py  swing classifier, shot-quality sub-scores,
 │     distance estimate (labeled `*_est`)
 └─► engine/game_api.py      assemble the per-frame JSON schema
       └─► live GUI  ·  or  ─► phone_server.py ─► PC dashboard (SSE)
```

Performance note: pose is the dominant cost, so it runs every Nth frame and the last
skeleton is reused in between; the fast-moving bat updates every single frame.

## What you get in the desktop GUI

- **Left**: live webcam with skeleton (green bones, cyan joints), the bat line
  (contact point → tip), swing arc, status, lighting warnings, and a live stats
  readout (FPS, estimated distance, swing, speed, quality scores).
- **Middle**: HSV sliders to tune the blue range for your lighting (toggle *Masks* to
  see exactly what's being detected), plus Mirror / Skeleton / Masks toggles and Reset.
- **Right**: the live **JSON Game Output** — the same data a game engine reads. Click
  *Print JSON to console* to dump a frame.

## Dependencies & install

Python 3, OpenCV (`cv2`), `numpy`, Pillow, `mediapipe`. Tkinter ships with Python on
macOS. The MediaPipe pose model lives in `models/` and is committed so it just runs.

```bash
python3 -m pip install -r requirements.txt
python3 cricket_app.py
```

On first launch macOS asks for **camera permission** — allow it, fully quit your
terminal (Cmd+Q), and rerun.

## API / JSON Game Output schema

Every frame produces this (from `engine/game_api.py`):

| Block | Contents |
|---|---|
| `player` | stance (handedness guess), orientation, joint pixel positions |
| `bat` | detected flag, base/tip, sweet spot, angle, length_px, speed, accel, peakSpeed, swing |
| `quality` | timing, balance, footwork, bat_path, overall (0–100) |
| `environment` | `distance_m_est` (estimate, not true depth), lighting, fps |

## Functions & module reference

### `engine/pose_module.py` — skeleton
- `PoseTracker.process(frame)` → `PoseResult` or `None`
- `PoseResult.handedness()` · `orientation()` · `height_px()` · `vis(name)`
- `LM` landmark-name table · `SKELETON` bone pairs for drawing

### `engine/bat_module.py` — bat
- `BatTracker.process(frame, pose)` → bat dict (detected, base, tip, angle, speed, accel)
- `BatTracker.set_range(color, which, hsv)` · Kalman prediction (`_PointKF`) ·
  velocity-aware teleport gating · `_orient()` disambiguates tip/base from wrists

### `engine/analysis_module.py` — analysis
- `classify_swing(bat, angle_hist)` — idle / defence / straight_drive / cover_drive /
  on_drive / pull / cut / flick (heuristic)
- `shot_quality(bat, pose)` — 0–100 timing / balance / footwork / bat_path
- `estimate_distance_m(pose)` — labeled estimate from body height

### `engine/game_api.py` — output
- `build_frame_output(pose, bat, lighting, fps)` → the JSON dict above

### `cricket_app.py` — desktop GUI
- `class CricketApp` — camera open, per-N-frame pose loop, layout, draw/HUD,
  panel sync, reset, JSON dump. Performs the whole desktop experience.

### `bat_gui.py` / `bat_tracker.py` — standalone tuning helpers
- `color_mask()`, `largest_centroid()` and a small Tk HSV mask-viewer
  (`BatGUI`) for dialling in detection without running the full app.

### `phone_server.py` — phone-camera server
Self-signed **HTTPS** on port `8443` (browsers refuse camera access on plain HTTP).
- Phone browses `https://<ip>:8443/` → `phone.html` (detects on-device)
- PC browses `https://<ip>:8443/pc` → `pc.html` dashboard
- Endpoints: `/stats` (POST), `/events` (SSE fan-out), `/frame` (preview JPEG),
  `/clips` + `/clip/{id}` (saved swings), `/vendor/*` (local MediaPipe assets)

### `web/` — phone/dashboard front-end
- `phone.html` — on-device getUserMedia + HSV detection, posts stats, saves swing clips
- `virtualbat.js` — the **Virtual Bat model**: skeleton → hands → virtual bat, with
  color as a correction layer (inverted pipeline, robust when the bat blurs)
- `pc.html` — dashboard over SSE; shows live stats, preview mirror and swing clips
- `vendor/` — vendored MediaPipe WASM + model so it works with no internet

## Tuning tips

Dull/muted colours flicker. Turn on **Masks**, then drag each colour's **S**
(saturation) and **V** (value) *lower* bounds down until your object lights up
cleanly without the background bleeding in. The defaults assume a plain blue bat.

## Honest limits (single webcam)

- **Distance is an estimate** from apparent body height — labelled `distance_m_est`.
- Swing classification is a **heuristic** starter, not a trained classifier.
- True 3D (floor plane, hitboxes) needs a depth camera or a second camera — later.

## Roadmap

2. Calibration wizard (player height, bat length, saved profile)
3. Environment detection ("move 40 cm back", floor plane)
4. Richer swing recognition (trained per-shot)
5. Motion prediction + Kalman smoothing
6. Depth / stereo camera for true 3D hitbox
7. Ghost comparison vs. pro references, ball detection, live coaching cues

## Security note

`phone_server.py` needs a TLS key pair. It **generates a self-signed cert on first
run** (`ensure_cert()`). The generated `.key.pem` / `.cert.pem` are **git-ignored** —
they are regenerated on any fresh clone, and your private key never leaves your
machine. The phone will show a one-time "Not private" warning: tap Advanced → Proceed.