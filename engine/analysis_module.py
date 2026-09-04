"""
Analysis module — heuristic swing recognition, shot-quality scoring, and a
labeled distance estimate. These are approximations from a single webcam:
distance is inferred from apparent body height and is NOT a true metric depth.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Reference: an adult ~1.7 m tall filling ~60% of a 720p frame at ~2.5 m.
# Purely heuristic — calibrate per setup for better numbers.
_REF_HEIGHT_PX = 430.0
_REF_DISTANCE_M = 2.5

# Below this apparent body height (px) the skeleton is too small / noisy to
# give a usable distance estimate, so we bail out instead of returning noise.
_MIN_HEIGHT_PX = 20


def estimate_distance_m(pose) -> Optional[float]:
    """Rough distance estimate (labeled as estimate). Returns meters or None."""
    if pose is None:
        return None
    hpx = pose.height_px()
    if not hpx or hpx < _MIN_HEIGHT_PX:
        return None
    return round(_REF_DISTANCE_M * (_REF_HEIGHT_PX / hpx), 2)


def classify_swing(bat: dict) -> str:
    """
    Very rough shot classifier from bat angle + motion. This is a heuristic
    starter — it improves as you add labeled data later.
    """
    if not bat.get("detected"):
        return "idle"
    speed = float(bat.get("speed", 0))
    raw_angle = float(bat.get("angle", 0))
    angle = raw_angle if raw_angle >= 0 else (raw_angle + 360.0) % 360.0
    if angle > 180.0:
        angle = 360.0 - angle  # fold to upper hemisphere 0-180
    if speed < 120:
        return "defence"
    # angle: 90=vertical(up), 0=horizontal-right, 180=horizontal-left
    if 60 <= angle <= 120:
        return "straight_drive"
    if 30 <= angle < 60:
        return "cover_drive"
    if 120 < angle <= 150:
        return "on_drive"
    if angle < 30 or angle > 150:
        return "pull" if speed > 400 else "cut"
    # No angle in [0, 180] reaches here: the bands above (0-30, 30-60,
    # 60-120, 120-150, 150-180) cover the entire folded range. Kept as a
    # defensive fallback rather than an unreachable "flick" return.
    return "cut"


def shot_quality(bat: dict, pose) -> dict:
    """Produce 0-100 sub-scores. Heuristic; tune weights over time."""
    # Coerce to float: in phone-camera mode the bat dict arrives from JSON
    # posted over the network, where numeric fields may be strings. This
    # mirrors the defensive coercion already used in classify_swing() and
    # avoids a TypeError on the arithmetic below.
    speed = float(bat.get("speed", 0))
    # Timing proxy: moderate, controlled speed scores best
    timing = int(np.clip(100 - abs(speed - 500) / 8, 40, 99))
    # Bat path proxy: straighter (less angle noise) is better — placeholder
    bat_path = int(np.clip(70 + min(speed, 300) / 12, 40, 99))
    # Balance/footwork from pose symmetry if available
    balance = footwork = 75
    if pose is not None:
        la, ra = pose.get("left_ankle"), pose.get("right_ankle")
        ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
        if la and ra and ls and rs:
            stance_w = abs(la[0] - ra[0]) + 1e-6
            shoulder_w = abs(ls[0] - rs[0]) + 1e-6
            footwork = int(np.clip(60 + 40 * min(stance_w / shoulder_w, 1.5) / 1.5, 40, 99))
            mid_feet = (la[0] + ra[0]) / 2
            mid_sh = (ls[0] + rs[0]) / 2
            balance = int(np.clip(99 - abs(mid_feet - mid_sh) / 3, 40, 99))
    overall = int(round(np.mean([timing, balance, footwork, bat_path])))
    return {"timing": timing, "balance": balance, "footwork": footwork,
            "bat_path": bat_path, "overall": overall}
