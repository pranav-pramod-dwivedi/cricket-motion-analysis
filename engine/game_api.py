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
        swing = analysis.classify_swing(bat, None)
        # sweet spot ~ 70% from grip toward tip
        base = bat["base"]
        tip = bat["tip"]
        sweet = [int(base[0] + 0.7 * (tip[0] - base[0])),
                 int(base[1] + 0.7 * (tip[1] - base[1]))]
        bat_out = {
            "detected": True,
            "base": list(bat["base"]),
            "tip": list(bat["tip"]),
            "sweetSpot": sweet,
            "angle": round(bat["angle"], 1),
            "length_px": round(bat["length_px"], 1),
            "speed": round(bat["speed"], 1),
            "accel": round(bat["accel"], 1),
            "peakSpeed": round(bat["peak_speed"], 1),
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
