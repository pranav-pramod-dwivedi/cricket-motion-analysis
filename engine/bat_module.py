"""
Bat module — tracks the bat as a single elongated blue object rather than a
pair of color blobs.

Pipeline per frame:
    1. blue HSV mask (morphological cleanup)
    2. contour filtering: min area, aspect ratio, extent
    3. best candidate scored on shape + agreement with the last known bat
    4. endpoints from minAreaRect long axis (not centroids)
    5. tip/base disambiguated by distance from the player's wrists (pose),
       falling back to temporal continuity, then to "lower point is the tip"
    6. temporal gate: reject teleports, coast through brief dropouts
    7. confidence score; callers should ignore frames below MIN_CONFIDENCE

Brown is no longer detected. The tip is inferred geometrically, which removes
the failure mode where skin / wood / furniture all read as "brown".
"""

import time

import cv2
import numpy as np

# HSV range (OpenCV hue 0-179), tuned for a dull/muted blue. Adjustable live.
DEFAULT_RANGES = {
    "blue": {"lower": [95, 50, 40], "upper": [135, 255, 255]},
}

# --- geometric constraints ------------------------------------------------
MIN_AREA = 1500          # px; kills specks and small blue background objects
MIN_ASPECT = 3.0         # long/short of the fitted rect; a bat is elongated
MIN_EXTENT = 0.35        # contour area / rect area; rejects sprawling blobs
MIN_LENGTH_PX = 80       # a bat shorter than this is almost certainly noise

# --- temporal constraints -------------------------------------------------
# Gating is velocity-aware: a fast swing legitimately moves the tip a long way
# between frames, so the budget scales with predicted speed instead of being a
# flat cap (a flat cap is what made fast swings drop out).
BASE_JUMP_PX = 260       # allowance at rest
JUMP_VEL_FACTOR = 1.8    # extra allowance per predicted px of motion
MAX_JUMP_PX = 900        # absolute ceiling

COAST_FRAMES = 8         # frames we predict through before giving up
MIN_CONFIDENCE = 0.55    # below this, `detected` is False

EMA_ALPHA = 0.5          # endpoint smoothing at low speed
EMA_ALPHA_FAST = 0.85    # less smoothing when moving fast (avoid lag)
FAST_PX_PER_FRAME = 25   # above this, treat the bat as "swinging"

SEARCH_PAD = 180         # px padding around the predicted bat for the ROI


def _mask(hsv, lo, hi):
    m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def _rect_endpoints(box):
    """Midpoints of the two short edges of a minAreaRect -> the long axis ends."""
    # order the 4 corners so consecutive points share an edge
    edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    edges.sort(key=lambda e: np.linalg.norm(e[0] - e[1]))
    # two shortest edges are the ends of the elongated rect
    (a0, a1), (b0, b1) = edges[0], edges[1]
    return (a0 + a1) / 2.0, (b0 + b1) / 2.0


def _candidates(mask):
    """Contours that pass the geometric filters, as dicts of measurements."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if w < 1 or h < 1:
            continue
        long_side, short_side = max(w, h), min(w, h)
        aspect = long_side / short_side
        if aspect < MIN_ASPECT:
            continue
        if area / (w * h) < MIN_EXTENT:
            continue
        if long_side < MIN_LENGTH_PX:
            continue
        box = cv2.boxPoints(rect)
        p1, p2 = _rect_endpoints(box)
        out.append({
            "area": float(area),
            "aspect": float(aspect),
            "length": float(long_side),
            "ends": (p1, p2),
            "center": np.array(rect[0], dtype=float),
        })
    return out


def _make_kalman():
    """Constant-velocity Kalman filter on a 2D point: state = [x, y, vx, vy]."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    # trust motion model moderately; measurements are noisy under blur
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.35
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


class _PointKF:
    """Thin wrapper: predict() then correct(), with velocity readout."""

    def __init__(self):
        self.kf = None

    def start(self, p):
        self.kf = _make_kalman()
        s = np.array([[p[0]], [p[1]], [0], [0]], np.float32)
        self.kf.statePost = s
        # correct() reads statePre, which stays zero until predict() runs — seed
        # it so a correct() immediately after start() returns the real point.
        self.kf.statePre = s.copy()

    def predict(self):
        if self.kf is None:
            return None
        s = self.kf.predict().flatten()
        return np.array([float(s[0]), float(s[1])])

    def correct(self, p):
        if self.kf is None:
            self.start(p)
            return np.array(p, dtype=float)
        m = np.array([[np.float32(p[0])], [np.float32(p[1])]])
        s = self.kf.correct(m).flatten()
        return np.array([float(s[0]), float(s[1])])

    @property
    def velocity(self):
        if self.kf is None:
            return np.zeros(2)
        s = self.kf.statePost.flatten()
        return np.array([float(s[2]), float(s[3])])

    def reset(self):
        self.kf = None


def _wrist_anchor(pose):
    """Midpoint of the visible wrists — where the player is holding the bat."""
    if pose is None:
        return None
    pts = []
    for name in ("left_wrist", "right_wrist"):
        p = pose.get(name)
        if p and pose.vis(name) > 0.3:
            pts.append(np.array(p, dtype=float))
    if not pts:
        return None
    return np.mean(pts, axis=0)


class BatTracker:
    def __init__(self):
        self.ranges = {k: dict(v) for k, v in DEFAULT_RANGES.items()}
        self.tip_ema = None
        self.base_ema = None
        self.prev_tip = None
        self.prev_time = None
        self.speed = 0.0            # px/s
        self.accel = 0.0            # px/s^2
        self.peak_speed = 0.0
        self.prev_speed = 0.0
        self.path = []              # recent tip positions for swing arc
        self.confidence = 0.0
        self._misses = 0            # consecutive unconfirmed frames
        self.tip_kf = _PointKF()    # predicts tip through blur/dropouts
        self.base_kf = _PointKF()
        self.predicted = False      # True when this frame's pose is a prediction
        self._grip_offset = None    # tip relative to wrists, for pose recovery
        self._last_len = None

    def set_range(self, color, which, hsv):
        if color in self.ranges:
            self.ranges[color][which] = list(hsv)

    def _masks(self, frame, roi=None):
        """Full-frame mask, or ROI-only (rest zeroed) when we have a prediction.

        Restricting the search both speeds up segmentation and prevents distant
        look-alike blobs from ever being considered.
        """
        if roi is None:
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
            return _mask(hsv, self.ranges["blue"]["lower"], self.ranges["blue"]["upper"])

        x0, y0, x1, y1 = roi
        sub = frame[y0:y1, x0:x1]
        if sub.size == 0:
            return self._masks(frame)
        hsv = cv2.cvtColor(cv2.GaussianBlur(sub, (5, 5), 0), cv2.COLOR_BGR2HSV)
        m = _mask(hsv, self.ranges["blue"]["lower"], self.ranges["blue"]["upper"])
        full = np.zeros(frame.shape[:2], np.uint8)
        full[y0:y1, x0:x1] = m
        return full

    def _roi(self, shape, pred_tip, pred_base):
        """Box around the predicted bat, padded for how fast it's moving."""
        if pred_tip is None or pred_base is None:
            return None
        h, w = shape[:2]
        pad = SEARCH_PAD + float(np.linalg.norm(self.tip_kf.velocity)) * 1.5
        pad = min(pad, max(w, h))  # a huge pad is just a full-frame search
        xs = [pred_tip[0], pred_base[0]]
        ys = [pred_tip[1], pred_base[1]]
        x0 = int(max(0, min(xs) - pad))
        y0 = int(max(0, min(ys) - pad))
        x1 = int(min(w, max(xs) + pad))
        y1 = int(min(h, max(ys) + pad))
        if x1 - x0 < 40 or y1 - y0 < 40:
            return None
        # if the box covers most of the frame there's no point sub-setting
        if (x1 - x0) * (y1 - y0) > 0.75 * w * h:
            return None
        return (x0, y0, x1, y1)

    def _score(self, cand):
        """Shape quality plus agreement with where we last saw the bat."""
        # elongation: saturates around aspect 8
        shape = min(cand["aspect"] / 8.0, 1.0)
        # size: saturates around 12k px
        size = min(cand["area"] / 12000.0, 1.0)
        score = 0.5 * shape + 0.3 * size

        # continuity: did it move plausibly since last frame?
        if self.tip_ema is not None:
            d = min(np.linalg.norm(e - self.tip_ema) for e in cand["ends"])
            score += 0.2 * max(0.0, 1.0 - d / MAX_JUMP_PX)
        else:
            score += 0.1  # no history yet; don't punish the first detection
        return min(score, 1.0)

    def _orient(self, ends, anchor):
        """Return (base, tip). Base is the grip end, tip is the toe."""
        p1, p2 = ends
        if anchor is not None:
            # the end nearer the hands is the grip
            d1 = np.linalg.norm(p1 - anchor)
            d2 = np.linalg.norm(p2 - anchor)
            return (p1, p2) if d1 < d2 else (p2, p1)
        if self.tip_ema is not None:
            # keep whichever end stayed closer to last frame's tip
            d1 = np.linalg.norm(p1 - self.tip_ema)
            d2 = np.linalg.norm(p2 - self.tip_ema)
            return (p2, p1) if d1 < d2 else (p1, p2)
        # cold start with no pose: assume the tip is the lower end
        return (p1, p2) if p1[1] < p2[1] else (p2, p1)

    def process(self, frame, pose=None):
        anchor = _wrist_anchor(pose)

        # --- 1. predict where the bat should be, before looking -------------
        pred_tip = self.tip_kf.predict()
        pred_base = self.base_kf.predict()
        roi = self._roi(frame.shape, pred_tip, pred_base)

        blue_mask = self._masks(frame, roi)
        # `masks` stays a tuple so existing mask-preview callers keep working
        result = {"detected": False, "masks": (blue_mask, blue_mask),
                  "confidence": 0.0, "source": "none"}

        cands = _candidates(blue_mask)
        if not cands and roi is not None:
            # nothing in the predicted box — fall back to a full-frame sweep
            blue_mask = self._masks(frame)
            result["masks"] = (blue_mask, blue_mask)
            cands = _candidates(blue_mask)

        if not cands:
            return self._miss(result, pose, pred_tip, pred_base)

        best = max(cands, key=self._score)
        conf = self._score(best)
        base, tip = self._orient(best["ends"], anchor)

        # --- 2. velocity-aware temporal gate --------------------------------
        # Budget scales with predicted speed so genuine fast swings survive.
        ref = pred_tip if pred_tip is not None else self.tip_ema
        if ref is not None:
            speed_px = float(np.linalg.norm(self.tip_kf.velocity))
            budget = min(BASE_JUMP_PX + speed_px * JUMP_VEL_FACTOR, MAX_JUMP_PX)
            # Each coasted frame widens the gate: a drifting prediction must
            # never permanently reject the real bat when it reappears.
            budget *= (1.0 + 0.8 * self._misses)
            jump = np.linalg.norm(tip - ref)
            if jump > budget:
                return self._miss(result, pose, pred_tip, pred_base)
            conf *= max(0.4, 1.0 - jump / budget)

        if conf < MIN_CONFIDENCE:
            return self._miss(result, pose, pred_tip, pred_base)

        # --- 3. skeleton-assisted correction (used every frame, not just on
        # recovery): snap the grip end toward the hands when they disagree.
        source = "color"
        if anchor is not None:
            drift = np.linalg.norm(base - anchor)
            if drift > 40:
                base = 0.65 * base + 0.35 * anchor
                source = "color+skeleton"

        # Re-locking after a coast: the filter's state is stale, so restart it
        # on the real measurement instead of dragging the old velocity in.
        relocked = self._misses > 0
        if relocked:
            self.tip_kf.start(tip)
            self.base_kf.start(base)
            self.tip_ema, self.base_ema = None, None

        self._misses = 0
        self.confidence = conf
        self.predicted = False

        # --- 4. Kalman correct -> smoothed, lag-free endpoints --------------
        tip = self.tip_kf.correct(tip)
        base = self.base_kf.correct(base)

        # speed-adaptive EMA: heavy smoothing at rest, light during a swing
        moving = float(np.linalg.norm(self.tip_kf.velocity))
        a = EMA_ALPHA_FAST if moving > FAST_PX_PER_FRAME else EMA_ALPHA
        self.base_ema = base if self.base_ema is None else a * base + (1 - a) * self.base_ema
        self.tip_ema = tip if self.tip_ema is None else a * tip + (1 - a) * self.tip_ema

        base_s, tip_s = self.base_ema, self.tip_ema
        self._remember_geometry(base_s, tip_s, anchor)
        result["source"] = source
        vec = tip_s - base_s
        length = np.linalg.norm(vec)
        angle = float(np.degrees(np.arctan2(-vec[1], vec[0])))  # 0=right, 90=up

        # kinematics from tip motion
        now = time.time()
        if self.prev_time is not None and self.prev_tip is not None:
            dt = max(now - self.prev_time, 1e-3)
            v = np.linalg.norm(tip_s - self.prev_tip) / dt
            self.accel = (v - self.prev_speed) / dt
            self.speed = v
            self.prev_speed = v
            self.peak_speed = max(self.peak_speed, v)
        self.prev_tip = tip_s.copy()
        self.prev_time = now

        self.path.append(tuple(tip_s.astype(int)))
        if len(self.path) > 40:
            self.path.pop(0)

        result.update({
            "detected": True,
            "predicted": False,
            "confidence": float(conf),
            "base": tuple(base_s.astype(int)),
            "tip": tuple(tip_s.astype(int)),
            "vector": vec.tolist(),
            "length_px": float(length),
            "angle": angle,
            "speed": float(self.speed),
            "accel": float(self.accel),
            "peak_speed": float(self.peak_speed),
            "path": list(self.path),
        })
        return result

    def _remember_geometry(self, base, tip, anchor):
        """Cache bat length and the grip->tip offset from the hands, so the
        skeleton alone can reconstruct the bat while it's invisible."""
        self._last_len = float(np.linalg.norm(tip - base))
        if anchor is not None:
            self._grip_offset = tip - anchor

    def _miss(self, result, pose=None, pred_tip=None, pred_base=None):
        """No confident color detection — fall back down the priority ladder:
        Kalman prediction, then skeleton reconstruction, then lost."""
        self._misses += 1
        self.prev_time = None  # don't compute speed across a gap

        if self._misses <= COAST_FRAMES and self.tip_ema is not None:
            anchor = _wrist_anchor(pose)
            tip = base = None
            source = None

            if pred_tip is not None and pred_base is not None:
                tip, base, source = pred_tip, pred_base, "predicted"

            # Skeleton reconstruction: if we know where the hands are and how
            # the bat sat relative to them, rebuild it even with zero pixels.
            if anchor is not None and self._grip_offset is not None:
                s_tip = anchor + self._grip_offset
                if tip is None:
                    tip, base, source = s_tip, anchor, "skeleton"
                else:
                    # blend prediction with the skeleton estimate
                    tip = 0.5 * tip + 0.5 * s_tip
                    base = 0.5 * base + 0.5 * anchor
                    source = "skeleton+predicted"

            if tip is not None:
                self.tip_ema, self.base_ema = tip, base
                self.predicted = True
                # confidence decays the longer we go without real pixels
                self.confidence = max(0.25, 0.8 - 0.1 * self._misses)
                vec = tip - base
                result.update({
                    "detected": True,
                    "predicted": True,
                    "source": source,
                    "confidence": float(self.confidence),
                    "base": tuple(np.asarray(base).astype(int)),
                    "tip": tuple(np.asarray(tip).astype(int)),
                    "vector": np.asarray(vec).tolist(),
                    "length_px": float(np.linalg.norm(vec)),
                    "angle": float(np.degrees(np.arctan2(-vec[1], vec[0]))),
                    "speed": float(self.speed),
                    "accel": float(self.accel),
                    "peak_speed": float(self.peak_speed),
                    "path": list(self.path),
                })
                return result

        # fully lost — reset so the next acquisition starts clean
        if self._misses > COAST_FRAMES:
            self.tip_ema = None
            self.base_ema = None
            self.prev_tip = None
            self.confidence = 0.0
            self.predicted = False
            self.tip_kf.reset()
            self.base_kf.reset()
            self._grip_offset = None
            self.path.clear()
        return result
