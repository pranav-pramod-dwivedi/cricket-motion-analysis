// ============================================================================
// Virtual Bat Model
// ----------------------------------------------------------------------------
// Inverts the usual pipeline. Instead of:
//        color -> bat
// we do:
//        skeleton -> hands -> virtual bat -> color corrects it
//
// The skeleton is temporally stable (wrists rarely vanish, even mid-swing),
// while the bat itself blurs badly at speed. So the body drives the estimate
// and color is a correction layer that refines it whenever it's trustworthy.
//
//   Left hand ●────────● Right hand
//                 │  grip = midpoint, orientation from both hands + forearms
//                 ▼
//            virtual bat
//
// Calibration learns how THIS player holds the bat: length in wrist-widths,
// and the bat's angular offset from the forearm.
// ============================================================================

export const CAL_FRAMES = 60;          // ~2s at 30fps of "hold it naturally"

export class VirtualBat {
  constructor() {
    this.cal = null;        // {lenRatio, angOffset, samples}
    this.calBuf = [];
    this.calibrating = false;

    this.tip = null;        // smoothed output, in processing-space px
    this.base = null;
    this.vel = [0, 0];
    this.lastGrip = null;
    this.confidence = 0;
    this.source = 'none';
    this.blend = 0;         // 0 = pure model, 1 = pure color
  }

  // ---- calibration -------------------------------------------------------
  startCalibration() { this.calibrating = true; this.calBuf = []; }

  get calProgress() {
    return this.calibrating ? Math.min(1, this.calBuf.length / CAL_FRAMES) : 1;
  }

  /** Feed a frame during calibration: needs both a skeleton and a color bat. */
  _calibrate(sk, color) {
    if (!sk || !color) return;
    const grip = sk.grip;
    if (!grip) return;
    // bat length relative to shoulder width -> scale-invariant (works at any
    // distance from the camera, unlike raw pixels)
    const ref = sk.shoulderWidth || 1;
    const far = dist(color.tip, grip) > dist(color.base, grip) ? color.tip : color.base;
    const lenRatio = dist(far, grip) / ref;
    // bat direction vs forearm direction
    const angOffset = norm180(ang(sub(far, grip)) - sk.forearmAngle);
    this.calBuf.push({ lenRatio, angOffset });

    if (this.calBuf.length >= CAL_FRAMES) {
      // median is robust to the odd bad frame
      const L = this.calBuf.map(c => c.lenRatio).sort((a, b) => a - b);
      const A = this.calBuf.map(c => c.angOffset).sort((a, b) => a - b);
      this.cal = {
        lenRatio: L[L.length >> 1],
        angOffset: A[A.length >> 1],
        samples: this.calBuf.length,
      };
      this.calibrating = false;
    }
  }

  /** Predict the bat purely from the body. Returns {base, tip} or null. */
  predictFromSkeleton(sk) {
    if (!sk || !sk.grip || !this.cal) return null;
    const len = this.cal.lenRatio * (sk.shoulderWidth || 1);
    const a = (sk.forearmAngle + this.cal.angOffset) * Math.PI / 180;
    const dir = [Math.cos(a), -Math.sin(a)];
    return { base: sk.grip.slice(), tip: [sk.grip[0] + dir[0] * len,
                                          sk.grip[1] + dir[1] * len] };
  }

  // ---- main update -------------------------------------------------------
  /**
   * @param sk     skeleton {grip, forearmAngle, shoulderWidth, ok} or null
   * @param color  color detection {base, tip, conf} or null
   */
  update(sk, color) {
    if (this.calibrating) {
      this._calibrate(sk, color);
      if (color) { this.base = color.base.slice(); this.tip = color.tip.slice(); }
      this.source = 'calibrating';
      this.confidence = this.calProgress;
      return this._out();
    }

    const model = this.predictFromSkeleton(sk);
    const cconf = color ? color.conf : 0;

    // orient color detection so `tip` is the end farther from the hands
    let cbase = null, ctip = null;
    if (color) {
      const g = sk && sk.grip ? sk.grip : (this.base || color.base);
      const d1 = dist(color.p1 || color.base, g), d2 = dist(color.p2 || color.tip, g);
      if (d1 <= d2) { cbase = color.p1 || color.base; ctip = color.p2 || color.tip; }
      else          { cbase = color.p2 || color.tip;  ctip = color.p1 || color.base; }
    }

    // Reject color that wildly disagrees with the body model — that's the
    // "random detection on a brown door" case the model is immune to.
    let colorOK = !!color && cconf >= 0.45;
    if (colorOK && model) {
      const err = dist(ctip, model.tip);
      const tol = 1.15 * dist(model.tip, model.base) + 40;
      if (err > tol) colorOK = false;
    }

    let target;
    if (colorOK && model) {
      // both available: color leads, model stabilizes
      const w = Math.min(0.85, 0.45 + cconf * 0.5);
      target = { base: lerp2(model.base, cbase, w), tip: lerp2(model.tip, ctip, w) };
      this.source = 'color+model';
      this.confidence = Math.min(1, 0.55 + cconf * 0.45);
    } else if (colorOK) {
      target = { base: cbase, tip: ctip };
      this.source = 'color';
      this.confidence = cconf;
    } else if (model) {
      target = model;
      this.source = 'model';               // blur/occlusion — body carries it
      this.confidence = 0.55;
    } else if (this.tip) {
      // no body, no color: coast on last velocity, decaying
      target = { base: add(this.base, this.vel), tip: add(this.tip, this.vel) };
      this.source = 'coast';
      this.confidence = Math.max(0, this.confidence - 0.08);
      if (this.confidence <= 0.05) { this.tip = this.base = null; this.source = 'none'; }
    } else {
      this.source = 'none'; this.confidence = 0;
      return this._out();
    }

    if (!target) return this._out();

    // Smooth toward the target. Snap harder when moving fast so we don't lag
    // the swing; ease when slow so it doesn't jitter.
    const speed = Math.hypot(this.vel[0], this.vel[1]);
    const k = speed > 12 ? 0.8 : 0.45;
    if (!this.tip) { this.tip = target.tip.slice(); this.base = target.base.slice(); }
    else {
      const nt = lerp2(this.tip, target.tip, k);
      this.vel = [0.6 * this.vel[0] + 0.4 * (nt[0] - this.tip[0]),
                  0.6 * this.vel[1] + 0.4 * (nt[1] - this.tip[1])];
      this.tip = nt;
      this.base = lerp2(this.base, target.base, k);
    }
    return this._out();
  }

  _out() {
    return {
      base: this.base, tip: this.tip,
      source: this.source, confidence: this.confidence,
      calibrated: !!this.cal, calProgress: this.calProgress,
      angle: (this.tip && this.base)
        ? +ang(sub(this.tip, this.base)).toFixed(1) : null,
    };
  }

  reset() {
    this.tip = this.base = null; this.vel = [0, 0];
    this.confidence = 0; this.source = 'none';
  }
}

// ---- small vector helpers -------------------------------------------------
const sub  = (a, b) => [a[0] - b[0], a[1] - b[1]];
const add  = (a, b) => [a[0] + b[0], a[1] + b[1]];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const ang  = v => Math.atan2(-v[1], v[0]) * 180 / Math.PI;
function norm180(d) { while (d > 180) d -= 360; while (d < -180) d += 360; return d; }

/**
 * Build the skeleton anchor from MediaPipe pose landmarks.
 * Uses BOTH hands plus forearms: more stable orientation than one wrist, and
 * survives one wrist being briefly occluded.
 */
export function skeletonAnchor(lms, w, h) {
  if (!lms) return null;
  const P = i => lms[i] ? [lms[i].x * w, lms[i].y * h] : null;
  const V = i => (lms[i] && lms[i].visibility != null) ? lms[i].visibility : 1;

  const LW = P(15), RW = P(16), LE = P(13), RE = P(14), LS = P(11), RS = P(12);
  const lwOK = LW && V(15) > 0.3, rwOK = RW && V(16) > 0.3;
  if (!lwOK && !rwOK) return null;

  // grip = midpoint of both hands when both are visible
  let grip;
  if (lwOK && rwOK) grip = lerp2(LW, RW, 0.5);
  else grip = lwOK ? LW : RW;

  // forearm direction: average both elbows->wrists that we can see
  const dirs = [];
  if (lwOK && LE && V(13) > 0.3) dirs.push(sub(LW, LE));
  if (rwOK && RE && V(14) > 0.3) dirs.push(sub(RW, RE));
  let fdir = dirs.length
    ? dirs.reduce((a, b) => add(a, b), [0, 0]).map(v => v / dirs.length)
    : (lwOK && rwOK ? sub(RW, LW) : [0, -1]);
  if (Math.hypot(fdir[0], fdir[1]) < 1e-3) fdir = [0, -1];

  const shoulderWidth = (LS && RS) ? dist(LS, RS) : (h * 0.18);

  return {
    grip, forearmAngle: ang(fdir), shoulderWidth,
    bothHands: lwOK && rwOK, ok: true,
    pts: { LW, RW, LE, RE, LS, RS },
  };
}
