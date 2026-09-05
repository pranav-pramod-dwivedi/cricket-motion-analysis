"""
Tests for engine.game_api.build_frame_output — the per-frame JSON output
contract that a game engine consumes.

Covers:
  - top-level schema (player / bat / quality / environment)
  - the no-detection path (pose None, bat not detected)
  - the full-detection path (pose + bat), including sweet-spot computation
  - JSON-serializability of every frame produced

Run with pytest:
    python3 -m pytest test_game_api.py -v
or standalone (no pytest installed):
    python3 test_game_api.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.game_api import build_frame_output


# ---------------------------------------------------------------------------
# Minimal test double for PoseResult — only the attributes game_api reads
# (get / handedness / orientation / height_px).
# ---------------------------------------------------------------------------
class FakePose:
    def __init__(self, points=None, height=None,
                 handedness="right", orientation="side"):
        self._points = points or {}
        self._height = height
        self._handedness = handedness
        self._orientation = orientation

    def get(self, name):
        return self._points.get(name)

    def handedness(self):
        return self._handedness

    def orientation(self):
        return self._orientation

    def height_px(self):
        return self._height


def _bat(detected=True, base=(0, 0), tip=(100, 100), speed=300, angle=45,
         accel=10, peak_speed=500, length_px=141.4):
    """Build the minimal bat dict build_frame_output reads."""
    return {
        "detected": detected, "base": base, "tip": tip,
        "angle": angle, "length_px": length_px, "speed": speed,
        "accel": accel, "peak_speed": peak_speed,
    }


# --- top-level schema ------------------------------------------------------

def test_output_has_four_top_level_keys():
    out = build_frame_output(None, {"detected": False}, "good", 30.0)
    assert set(out.keys()) == {"player", "bat", "quality", "environment"}


def test_output_is_json_serializable():
    """The whole point of build_frame_output is a JSON-ready dict."""
    pose = FakePose(points={"nose": (50, 20)}, height=430)
    out = build_frame_output(pose, _bat(), "good", 29.97)
    # must not raise
    json.dumps(out)


# --- no-detection path -----------------------------------------------------

def test_no_pose_no_bat_gives_safe_defaults():
    out = build_frame_output(None, {"detected": False}, "dark", 15.0)
    assert out["player"] == {"stance": "unknown", "orientation": "unknown"}
    assert out["bat"] == {"detected": False}
    assert out["quality"] is None
    assert out["environment"]["distance_m_est"] is None
    assert out["environment"]["lighting"] == "dark"
    assert out["environment"]["fps"] == 15.0


def test_bat_not_detected_skips_swing_and_quality():
    pose = FakePose(points={"nose": (10, 10)}, height=430,
                    handedness="left", orientation="front")
    out = build_frame_output(pose, {"detected": False}, "good", 30.0)
    assert out["bat"] == {"detected": False}
    assert out["quality"] is None
    # pose is still surfaced even without a bat
    assert out["player"]["stance"] == "left"


# --- player block ----------------------------------------------------------

def test_player_block_carries_landmarks_from_pose():
    pose = FakePose(points={
        "nose": (100, 50),
        "left_shoulder": (120, 100),
        "right_shoulder": (80, 100),
    }, height=430)
    out = build_frame_output(pose, {"detected": False}, "good", 30.0)
    p = out["player"]
    assert p["stance"] == "right"
    assert p["orientation"] == "side"
    assert p["head"] == [100, 50]
    # shoulders/elbows are pairs; present ones are the raw (x, y) tuples
    assert p["shoulders"] == [(120, 100), (80, 100)]
    # landmarks not in the pose come through as None, not missing keys
    assert p["elbows"] == [None, None]
    assert p["wrists"] == [None, None]


# --- bat block + sweet spot ------------------------------------------------

def test_bat_block_fields_when_detected():
    out = build_frame_output(None, _bat(), "good", 30.0)
    b = out["bat"]
    assert b["detected"] is True
    assert b["base"] == [0, 0]
    assert b["tip"] == [100, 100]
    assert b["angle"] == 45.0
    assert b["swing"] == "cover_drive"   # angle 45, speed 300 -> cover_drive
    assert b["peakSpeed"] == 500.0


def test_sweet_spot_is_70_percent_from_base_to_tip():
    out = build_frame_output(None, _bat(base=(0, 0), tip=(100, 100)), "good", 30.0)
    assert out["bat"]["sweetSpot"] == [70, 70]


def test_sweet_spot_none_when_endpoints_missing():
    bat = _bat(base=None, tip=None)
    out = build_frame_output(None, bat, "good", 30.0)
    assert out["bat"]["sweetSpot"] is None


def test_sweet_spot_none_for_short_endpoints():
    bat = _bat(base=(5,), tip=(9,))  # len < 2
    out = build_frame_output(None, bat, "good", 30.0)
    assert out["bat"]["sweetSpot"] is None


# --- environment block -----------------------------------------------------

def test_environment_carries_lighting_and_fps():
    out = build_frame_output(None, {"detected": False}, "bright", 59.9)
    env = out["environment"]
    assert env["lighting"] == "bright"
    assert env["fps"] == 59.9
    assert "distance_note" in env  # labelled estimate, not true depth


def test_distance_estimate_flows_through_from_pose():
    pose = FakePose(height=860)  # 2x reference -> 1.25 m
    out = build_frame_output(pose, {"detected": False}, "good", 30.0)
    assert out["environment"]["distance_m_est"] == 1.25


# --- standalone runner (no pytest required) -------------------------------

if __name__ == "__main__":
    failures = 0
    g = globals()
    tests = sorted(n for n in g if n.startswith("test_") and callable(g[n]))
    for name in tests:
        try:
            g[name]()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failures += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
