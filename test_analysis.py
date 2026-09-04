"""
Tests for engine.analysis_module — pure functions, no camera/MediaPipe needed.

Covers the three public functions:
  - estimate_distance_m(pose)
  - classify_swing(bat)
  - shot_quality(bat, pose)

Run with pytest:
    python3 -m pytest test_analysis.py -v
or standalone (no pytest installed):
    python3 test_analysis.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.analysis_module import classify_swing, shot_quality, estimate_distance_m


# ---------------------------------------------------------------------------
# Minimal test double for PoseResult — only the attributes the analysis
# module actually reads (get / vis / height_px).
# ---------------------------------------------------------------------------
class FakePose:
    def __init__(self, points=None, height=None):
        self._points = points or {}
        self._height = height

    def get(self, name):
        return self._points.get(name)

    def vis(self, name):
        return 0.9 if name in self._points else 0.0

    def height_px(self):
        return self._height


def _bat(speed=0, angle=0, detected=True):
    """Build the minimal bat dict the analysis functions read."""
    return {"detected": detected, "speed": speed, "angle": angle}


# --- estimate_distance_m ---------------------------------------------------

def test_distance_none_when_no_pose():
    assert estimate_distance_m(None) is None


def test_distance_none_when_pose_too_small():
    assert estimate_distance_m(FakePose(height=15)) is None


def test_distance_scales_inversely_with_height():
    # A player who appears taller in frame (closer to camera) should be
    # estimated as nearer than one who appears smaller (farther away).
    near = estimate_distance_m(FakePose(height=860))   # 2x reference -> closer
    far = estimate_distance_m(FakePose(height=215))    # 0.5x reference -> farther
    assert near is not None and far is not None
    assert near < far
    # 2.5 * (430 / 860) = 1.25,  2.5 * (430 / 215) = 5.0
    assert abs(near - 1.25) < 0.01
    assert abs(far - 5.0) < 0.01


# --- classify_swing --------------------------------------------------------

def test_swing_idle_when_not_detected():
    assert classify_swing(_bat(detected=False)) == "idle"


def test_swing_defence_at_low_speed():
    assert classify_swing(_bat(speed=80, angle=90)) == "defence"


def test_swing_straight_drive_vertical():
    assert classify_swing(_bat(speed=500, angle=90)) == "straight_drive"


def test_swing_cover_drive_diagonal():
    assert classify_swing(_bat(speed=500, angle=45)) == "cover_drive"


def test_swing_on_drive_other_diagonal():
    assert classify_swing(_bat(speed=500, angle=135)) == "on_drive"


def test_swing_pull_fast_horizontal():
    assert classify_swing(_bat(speed=600, angle=10)) == "pull"


def test_swing_cut_slow_horizontal():
    assert classify_swing(_bat(speed=200, angle=170)) == "cut"


def test_swing_negative_angle_normalised():
    # arctan2 returns negative angles; -90 should normalise to 90 (vertical)
    assert classify_swing(_bat(speed=500, angle=-90)) == "straight_drive"


def test_swing_angle_above_180_folded():
    # 270 folds: (270+0) % 360 = 270, then 360-270 = 90
    assert classify_swing(_bat(speed=500, angle=270)) == "straight_drive"


def test_swing_pull_vs_cut_threshold():
    # Same angle, different speed: >400 is pull, <=400 is cut
    assert classify_swing(_bat(speed=401, angle=5)) == "pull"
    assert classify_swing(_bat(speed=400, angle=5)) == "cut"


def test_swing_angle_coverage_fast():
    """At swing speed (>400), every folded angle maps to a named shot and
    horizontal angles resolve to 'pull' (not 'cut', which needs low speed)."""
    shots = set()
    for angle in range(0, 181):
        shots.add(classify_swing(_bat(speed=500, angle=angle)))
    assert shots == {"straight_drive", "cover_drive", "on_drive", "pull"}


def test_swing_angle_coverage_moderate():
    """At moderate speed (120-400), horizontal angles resolve to 'cut'."""
    shots = set()
    for angle in range(0, 181):
        shots.add(classify_swing(_bat(speed=200, angle=angle)))
    assert shots == {"straight_drive", "cover_drive", "on_drive", "cut"}


# --- shot_quality ----------------------------------------------------------

def test_quality_returns_dict_with_expected_keys():
    q = shot_quality(_bat(speed=500), None)
    assert set(q.keys()) == {"timing", "balance", "footwork", "bat_path", "overall"}


def test_quality_scores_in_valid_range():
    q = shot_quality(_bat(speed=500), None)
    for key, val in q.items():
        assert 0 <= val <= 100, f"{key}={val} out of [0, 100]"


def test_quality_timing_best_near_target_speed():
    good = shot_quality(_bat(speed=500), None)["timing"]
    bad = shot_quality(_bat(speed=50), None)["timing"]
    assert good > bad


def test_quality_with_pose_uses_balance_and_footwork():
    pose = FakePose(points={
        "left_ankle": (100, 600),
        "right_ankle": (300, 600),
        "left_shoulder": (200, 200),
        "right_shoulder": (250, 200),
    })
    q = shot_quality(_bat(speed=500), pose)
    assert 40 <= q["balance"] <= 99
    assert 40 <= q["footwork"] <= 99


def test_quality_overall_is_mean_of_subscores():
    q = shot_quality(_bat(speed=400), None)
    expected = round(
        (q["timing"] + q["balance"] + q["footwork"] + q["bat_path"]) / 4
    )
    assert q["overall"] == expected


def test_quality_string_speed_does_not_crash():
    """In phone-camera mode the bat dict arrives from JSON posted over the
    network, where numeric fields may be strings. shot_quality must coerce
    them (as classify_swing already does) instead of raising TypeError."""
    bat = {"detected": True, "speed": "500", "angle": "90"}
    q = shot_quality(bat, None)
    assert 0 <= q["timing"] <= 100
    # must match the result for the numeric equivalent
    assert q == shot_quality({"detected": True, "speed": 500, "angle": 90}, None)


# --- standalone runner (no pytest required) --------------------------------

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
