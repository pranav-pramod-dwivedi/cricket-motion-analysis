"""
Game Output API — assembles the per-frame JSON that a game engine consumes.
Matches the schema you specified: player / bat / environment blocks.
"""

from . import analysis_module as analysis


def build_frame_output(pose, bat, lighting, fps):
    """Return a plain dict (JSON-serializable) for the current frame."""
    player = {"stance": "unknown", "orientation": "unknown"}
    if pose is not None:
        player = {
            "stance": pose.handedness(),
            "orientation": pose.orientation(),
            "head": list(pose.get("nose")) if pose.get("nose") else None,
            "shoulders": [pose.get("left_shoulder"), pose.get("right_shoulder")],
            "elbows": [pose.get("left_elbow"), pose.get("right_elbow")],
            "wrists": [pose.get("left_wrist"), pose.get("right_wrist")],
            "hips": [pose.get("left_hip"), pose.get("right_hip")],
            "knees": [pose.get("left_knee"), pose.get("right_knee")],
            "ankles": [pose.get("left_ankle"), pose.get("right_ankle")],
            "feet": [pose.get("left_foot"), pose.get("right_foot")],
        }

    bat_out = {"detected": False}
    if bat.get("detected"):
        swing = analysis.classify_swing(bat)
        base = bat.get("base")
        tip = bat.get("tip")
        sweet = None
        if base and tip and len(base) >= 2 and len(tip) >= 2:
            sweet = [int(base[0] + 0.7 * (tip[0] - base[0])),
                     int(base[1] + 0.7 * (tip[1] - base[1]))]
        bat_out = {
            "detected": True,
            "base": list(base) if base else None,
            "tip": list(tip) if tip else None,
            "sweetSpot": sweet,
            "angle": round(float(bat.get("angle", 0.0)), 1),
            "length_px": round(float(bat.get("length_px", 0.0)), 1),
            "speed": round(float(bat.get("speed", 0.0)), 1),
            "accel": round(float(bat.get("accel", 0.0)), 1),
            "peakSpeed": round(float(bat.get("peak_speed", 0.0)), 1),
            "confidence": round(float(bat.get("confidence", 0.0)), 2),
            "source": bat.get("source", "color"),
            "swing": swing,
        }

    quality = analysis.shot_quality(bat, pose) if bat.get("detected") else None

    return {
        "player": player,
        "bat": bat_out,
        "quality": quality,
        "environment": {
            "distance_m_est": analysis.estimate_distance_m(pose),
            "distance_note": "estimate from body height, not true depth",
            "lighting": lighting,
            "fps": round(fps, 1),
        },
    }
