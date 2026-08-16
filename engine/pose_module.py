"""
Pose module — full-body skeleton tracking via MediaPipe Tasks API.

Wraps PoseLandmarker and exposes named landmarks (head, shoulders, elbows,
wrists, hips, knees, ankles, feet) plus derived attributes (handedness guess,
body orientation). All coordinates are returned in PIXELS for the given frame.
"""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# "lite" is ~25% faster than "full" with equivalent landmark quality at webcam
# distances — it matters because pose is the pipeline's dominant cost.
def _model_path(prefer="lite"):
    for name in (prefer, "full", "lite"):
        p = os.path.join(_MODEL_DIR, f"pose_landmarker_{name}.task")
        if os.path.exists(p):
            return p
    return os.path.join(_MODEL_DIR, f"pose_landmarker_{prefer}.task")


MODEL_PATH = _model_path()

# MediaPipe Pose landmark indices we care about (33-point model)
LM = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot": 31, "right_foot": 32,
}

# Bone connections for drawing the skeleton (pairs of LM keys)
SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot"), ("right_ankle", "right_foot"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
]


class PoseResult:
    """Landmarks in pixel space + convenience accessors."""

    def __init__(self, points, visibility, frame_shape):
        self.points = points          # dict name -> (x, y) pixels
        self.visibility = visibility  # dict name -> 0..1
        self.h, self.w = frame_shape[:2]

    def get(self, name):
        return self.points.get(name)

    def vis(self, name):
        return self.visibility.get(name, 0.0)

    def handedness(self):
        """Rough guess: which wrist is lower (batting hand) — heuristic only."""
        lw, rw = self.get("left_wrist"), self.get("right_wrist")
        if not lw or not rw:
            return "unknown"
        return "right" if rw[1] > lw[1] else "left"

    def orientation(self):
        """facing camera / turned, from shoulder width vs. body height."""
        ls, rs = self.get("left_shoulder"), self.get("right_shoulder")
        lh = self.get("left_hip")
        if not (ls and rs and lh):
            return "unknown"
        shoulder_w = abs(ls[0] - rs[0])
        torso_h = abs(((ls[1] + rs[1]) / 2) - lh[1]) + 1e-6
        ratio = shoulder_w / torso_h
        return "facing" if ratio > 0.6 else "side-on"

    def height_px(self):
        """Approx standing height in pixels (nose to lowest ankle)."""
        nose = self.get("nose")
        ankles = [self.get("left_ankle"), self.get("right_ankle")]
        ankles = [a for a in ankles if a]
        if not nose or not ankles:
            return None
        low = max(a[1] for a in ankles)
        return abs(low - nose[1])


class PoseTracker:
    def __init__(self, min_conf=0.5, model=None):
        path = model or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pose model missing: {path}")
        base = mp_python.BaseOptions(model_asset_path=path)
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._ts = 0

    def process(self, frame_bgr):
        """frame_bgr: OpenCV BGR frame. Returns PoseResult or None."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33  # ~30fps timestamps (must be monotonically increasing)
        result = self.landmarker.detect_for_video(mp_image, self._ts)
        if not result.pose_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        lms = result.pose_landmarks[0]
        points, visibility = {}, {}
        for name, idx in LM.items():
            lm = lms[idx]
            points[name] = (int(lm.x * w), int(lm.y * h))
            visibility[name] = float(lm.visibility)
        return PoseResult(points, visibility, frame_bgr.shape)
