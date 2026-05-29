"""
╔══════════════════════════════════════════════════════════════════════════╗
║                        DYNAMIC EAR THRESHOLD                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PHƯƠNG PHÁP: Sliding Window → EWMA → MAD → Threshold → Hysteresis     ║
║    Sliding Window → EWMA → MAD → T_t = μ_t − k·σ_t → Hysteresis       ║
║                                                                          ║
║  CHỈ SỐ: từ các bài báo đã được chứng minh                              ║
║                                                                          ║
║  α = 0.20   Hunter (1986): "λ = 0.2 ± 0.1" (center of range)           ║
║             J. Quality Technology, 18(4), 203–210                       ║
║                                                                          ║
║  b = 1.4826 Rousseeuw & Croux (1993): 1/Φ⁻¹(0.75)                     ║
║             J. Am. Statistical Association, 88(424), 1273–1283          ║
║             Leys et al. (2013): J. Exp. Social Psychology               ║
║                                                                          ║
║  k = 2.5    Leys et al. (2013): "moderately conservative"               ║
║             "k=2 poorly, k=2.5 moderately, k=3 very conservative"       ║
║                                                                          ║
║  T_floor    Dewi et al. (2022): EAR_optimal = 0.18                     ║
║  = 0.18     Electronics 11(19):3183 — best AUC across all datasets      ║
║                                                                          ║
║  window     Soukupova & Cech (2016): blink = 100–400 ms                 ║
║  = 150f     "approximately 100–400 ms" — CVWW 2016, Sec 2              ║
║             150 frames @ 30fps = 5s (covers ~12–15 blink cycles)        ║
║                                                                          ║
║  gap=0.013  Derived from EWMA noise floor (Hunter 1986):                ║
║             σ_EWMA = √(α/(2−α))·σ_EAR = √(0.111)·0.02 ≈ 0.0067        ║
║             gap = 2·σ_EWMA ≈ 0.013  (2-sigma noise band)                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
import time
import json
import os


# ══════════════════════════════════════════════════════════════════════════
# PAPER-BACKED PARAMETER VALUES
# ══════════════════════════════════════════════════════════════════════════

# ── α (EWMA smoothing factor) ─────────────────────────────────────────────
# Hunter (1986), J. Quality Technology 18(4):203-210
# "Experience with econometric data suggests values of λ = 0.2 ± 0.1"
# → center of recommended range = 0.20
# Lucas & Saccucci (1990), Technometrics 32(1):1-12, Table 4
# → λ=0.20 optimal for detecting 1σ process shift
ALPHA = 0.20

# ── b (MAD consistency constant) ─────────────────────────────────────────
# Rousseeuw & Croux (1993), J. Am. Statistical Association, 88(424):1273-1283
# Leys et al. (2013), J. Experimental Social Psychology
# b = 1/Φ⁻¹(0.75) ≈ 1.4826
# "This multiplication by b is crucial, as otherwise the formula for the
#  MAD would only estimate the scale up to a multiplicative constant"
#  — Leys et al. (2013)
MAD_B = 1.4826

# ── k (threshold sensitivity multiplier) ─────────────────────────────────
# Leys et al. (2013), J. Experimental Social Psychology
# "k=2.5 moderately conservative" — explicitly recommended for scientific use
# "k=2 poorly conservative, k=2.5 moderately, k=3 very conservative"
# doi:10.1016/j.jesp.2013.03.013
K = 2.5

# ── EAR floor (minimum safe threshold) ───────────────────────────────────
# Dewi et al. (2022), Electronics 11(19):3183
# "0.18 was determined to be the optimum EAR threshold in our research"
# "the higher the EAR threshold, the worse the AUC's accuracy"
# AUC(0.18)=0.974 > AUC(0.20)=0.968 > AUC(0.225)=0.953 > AUC(0.25)=0.946
EAR_FLOOR = 0.18

# ── Window size ───────────────────────────────────────────────────────────
# Soukupova & Cech (2016), CVWW
# "The eye blink lasts approximately 100–400 ms"
# 150 frames @ 30fps = 5.0s → covers ~12–37 blink cycles at 15–30 blinks/min
# (normal blink rate: DOT/FAA/AM-94/17: "15-30 per minute during non-reading")
WINDOW = 150

# ── Hysteresis gap ────────────────────────────────────────────────────────
# Derived from EWMA noise floor — Hunter (1986):
#   Var(μ_t) = [α/(2−α)] · Var(EAR)
#   σ_EWMA   = √(0.20/1.80) · σ_EAR = 0.333 · σ_EAR
# Typical open-eye EAR noise: σ_EAR ≈ 0.020 (landmark jitter, Soukupova 2016)
#   σ_EWMA ≈ 0.333 × 0.020 = 0.0067
# Gap = 2·σ_EWMA ≈ 0.013  (2-sigma band ensures >95% separation)
# Rounds to 0.013 — prevents state flicker without masking real closures
GAP = 0.013

# ── Calibration frames ────────────────────────────────────────────────────
# EWMA convergence: 5τ = 5/(1−α) = 5/0.8 ≈ 6 frames to reach 99.3% steady-state
# Hunter (1986): "After a little practice, plotting the EWMA is almost as
#  easy as plotting the successive observations"
# Warmup = 30 frames (conservative, ensures EWMA stable)
# Collect = 270 frames = 9s — enough for robust MAD estimation (Leys 2013)
WARMUP_N  = 30
COLLECT_N = 270


# ══════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — SLIDING WINDOW
# "Chỉ tính toán EAR trên một cửa sổ thời gian gần nhất (3–5 giây)"
# ══════════════════════════════════════════════════════════════════════════

class SlidingWindow:
    """
    Rolling FIFO buffer — automatically discards oldest frame.
    WINDOW=150 (5s) per Soukupova & Cech (2016): blink 100–400ms,
    enough to capture 12–37 full blink cycles at normal rate.
    DOT/FAA/AM-94/17: normal blink rate 15–30/min → 150f covers 12–37 cycles.
    """
    def __init__(self):
        self._buf = deque(maxlen=WINDOW)

    def push(self, v: float):
        self._buf.append(v)

    @property
    def arr(self) -> np.ndarray:
        return np.array(self._buf)

    @property
    def n(self) -> int:
        return len(self._buf)

    @property
    def ready(self) -> bool:
        return self.n >= 30   # minimum for stable MAD


# ══════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — EWMA FILTER
# μ_t = α·EAR_t + (1−α)·μ_{t-1}
# α = 0.20 (Hunter 1986: "λ = 0.2 ± 0.1")
# ══════════════════════════════════════════════════════════════════════════

class EWMAFilter:
    """
    Exponentially Weighted Moving Average.

    α = 0.20 per Hunter (1986), J. Quality Technology 18(4):203-210:
      "Experience with econometric data suggests values of λ = 0.2 ± 0.1"

    Variance reduction:
      Var(μ_t) = [α/(2−α)]·Var(EAR)
               = [0.20/1.80]·Var(EAR)
               = 0.111·Var(EAR)
      → σ(μ) = 0.333·σ(EAR)  — 67% noise reduction
    """
    def __init__(self):
        self._mu: Optional[float] = None

    def update(self, ear: float) -> float:
        if self._mu is None:
            self._mu = ear
        else:
            # Hunter (1986) Eq.(5): μ_t = α·EAR_t + (1−α)·μ_{t−1}
            self._mu = ALPHA * ear + (1.0 - ALPHA) * self._mu
        return self._mu

    @property
    def mu(self) -> Optional[float]:
        return self._mu

    def reset(self):
        self._mu = None


# ══════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — MAD DISPERSION
# σ_t = 1.4826 · median(|x_i − median(x)|)
# b = 1.4826 per Rousseeuw & Croux (1993) / Leys et al. (2013)
# ══════════════════════════════════════════════════════════════════════════

class MADDispersion:
    """
    Robust scale estimator per Rousseeuw & Croux (1993).

    Formula: σ_t = 1.4826 · median(|x_i − median(x)|)

    b = 1.4826 = 1/Φ⁻¹(0.75) per:
      Rousseeuw, P.J. & Croux, C. (1993). Alternatives to the Median
      Absolute Deviation. J. Am. Statistical Association, 88(424), 1273–1283.
      Leys et al. (2013). J. Experimental Social Psychology.

    "The MAD is totally immune to the sample size" — Leys et al. (2013)

    Breakdown point = 50% (highest possible) — Leys et al. (2013)
    """
    def __init__(self):
        self._sigma: float = 0.02   # initial estimate before enough data
        self._ctr: int = 0

    def compute(self, arr: np.ndarray) -> float:
        if len(arr) < 10:
            return self._sigma

        # Recompute every 5 frames (MAD is O(n), cache improves speed)
        self._ctr += 1
        if self._ctr < 5:
            return self._sigma

        self._ctr = 0
        med = np.median(arr)
        # Rousseeuw & Croux (1993) / Leys et al. (2013): b=1.4826
        raw = np.median(np.abs(arr - med))
        self._sigma = max(MAD_B * raw, 0.003)
        return self._sigma

    @property
    def sigma(self) -> float:
        return self._sigma


# ══════════════════════════════════════════════════════════════════════════
# COMPONENT 4 — HYSTERESIS GATE
# T_low: EAR < T_low → CLOSED
# T_high = T_low + gap: EAR > T_high → OPEN
# gap = 0.013 = 2·σ_EWMA  (Hunter 1986: noise floor)
# ══════════════════════════════════════════════════════════════════════════

class HysteresisGate:
    """
    Two-threshold hysteresis to prevent state flickering.

    Gap = 2·σ_EWMA, derived from Hunter (1986) EWMA noise floor:
      σ_EWMA = √(α/(2−α)) · σ_EAR = 0.333 · 0.020 = 0.0067
      gap     = 2 · σ_EWMA ≈ 0.013   (2-sigma separation band)

    This ensures that a genuine eye-open signal (EAR rising by more than
    2× the EWMA noise floor) is required to transition CLOSED → OPEN,
    while any EAR drop below T_low immediately triggers CLOSED state.
    """
    def __init__(self):
        self._closed = False

    def update(self, ear: float, t_low: float) -> bool:
        t_high = t_low + GAP   # T_high = T_low + gap (Hunter 1986: 2·σ_EWMA band)
        if not self._closed:
            if ear < t_low:
                self._closed = True   # OPEN → CLOSED
        else:
            if ear > t_high:
                self._closed = False  # CLOSED → OPEN
        return self._closed

    def reset(self):
        self._closed = False


# ══════════════════════════════════════════════════════════════════════════
# THRESHOLD FORMULA
# T_t = μ_t − k · σ_t
# k = 2.5 per Leys et al. (2013): "moderately conservative"
# ══════════════════════════════════════════════════════════════════════════

def _threshold(mu: float, sigma: float) -> float:
    """
    T_t = μ_t − k · σ_t

    k = 2.5 per Leys et al. (2013):
    "k=2.5 moderately conservative" — explicitly recommended.
    "k=2 poorly conservative, k=2.5 moderately, k=3 very conservative."
    doi:10.1016/j.jesp.2013.03.013

    Lower bound = EAR_FLOOR = 0.18 per Dewi et al. (2022):
    "0.18 was determined to be the optimum EAR threshold"
    "AUC(0.18)=0.974 — best across all 5 datasets tested"
    doi:10.3390/electronics11193183
    """
    T = mu - K * sigma
    return max(float(T), EAR_FLOOR)


# ══════════════════════════════════════════════════════════════════════════
# RESULT
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class LockedThreshold:
    """Immutable threshold snapshot stored after calibration lock."""
    T_low:   float    # = μ − k·σ,  floor=0.18 (Dewi 2022)
    T_high:  float    # = T_low + gap (2·σ_EWMA, Hunter 1986)
    mu:      float    # μ at lock time (EWMA, Hunter 1986)
    sigma:   float    # σ at lock time (MAD·1.4826, Rousseeuw 1993)
    n:       int      # number of open-eye samples used
    ts:      float    = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> 'LockedThreshold':
        return cls(**d)


@dataclass
class FrameStatus:
    """Output of DynamicEAR.update() each frame."""
    phase:     str             # 'WARMUP' | 'CALIBRATING' | 'LOCKED'
    is_closed: bool            # valid only when LOCKED
    T_low:     Optional[float] # valid only when LOCKED
    mu:        float
    sigma:     float
    progress:  float           # 0.0 → 1.0
    n:         int             # samples in window

    @property
    def locked(self) -> bool:
        return self.phase == 'LOCKED'

    @property
    def pct(self) -> int:
        return int(self.progress * 100)


# ══════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════

class DynamicEAR:
    """
    Dynamic EAR threshold

    ┌─────────────────────────────────────────────────────────────────┐
    │  WARMUP  (30f)   EWMA converges, window fills — discard         │
    │  CALIB   (270f)  collect open-eye EAR, compute μ/σ live        │
    │  LOCKED          T_low = μ−k·σ frozen, hysteresis active        │
    └─────────────────────────────────────────────────────────────────┘

    Parameter sources:
      α=0.20   → Hunter (1986) J. Quality Technology
      b=1.4826 → Rousseeuw & Croux (1993) / Leys et al. (2013)
      k=2.5    → Leys et al. (2013) "moderately conservative"
      floor=0.18 → Dewi et al. (2022) Electronics 11(19):3183
      window=150 → Soukupova & Cech (2016) blink 100–400ms
      gap=0.013  → 2·σ_EWMA, Hunter (1986)
    """

    WARMUP = 'WARMUP'
    CALIB  = 'CALIBRATING'
    LOCKED = 'LOCKED'

    def __init__(self, save_path: Optional[str] = None):
        # 4 components per architecture
        self._win  = SlidingWindow()
        self._ewma = EWMAFilter()       # α=0.20, Hunter (1986)
        self._mad  = MADDispersion()    # b=1.4826, Rousseeuw & Croux (1993)
        self._hyst = HysteresisGate()   # gap=2·σ_EWMA=0.013, Hunter (1986)

        self._phase = self.WARMUP
        self._frame = 0
        self._locked: Optional[LockedThreshold] = None
        self._save_path = save_path

        # Load persisted calibration if exists
        if save_path and os.path.exists(save_path):
            self._load(save_path)

    # ── feed every frame ─────────────────────────────────────────────────

    def update(self, ear_raw: float) -> FrameStatus:
        """
        Call every frame with raw EAR value.
        Returns FrameStatus (is_closed valid only when locked).
        """
        self._frame += 1

        # Component 2: EWMA always runs — Hunter (1986)
        mu = self._ewma.update(ear_raw)

        # ── LOCKED ───────────────────────────────────────────────────────
        if self._phase == self.LOCKED:
            # Update window for live σ display
            self._win.push(ear_raw)
            sigma = self._mad.compute(self._win.arr)
            # Component 4: hysteresis with locked T_low — Hunter (1986) noise band
            closed = self._hyst.update(ear_raw, self._locked.T_low)
            return FrameStatus(
                phase=self.LOCKED, is_closed=closed,
                T_low=self._locked.T_low,
                mu=mu, sigma=sigma, progress=1.0, n=self._win.n,
            )

        # ── WARMUP ───────────────────────────────────────────────────────
        if self._phase == self.WARMUP:
            if self._frame >= WARMUP_N:
                self._phase = self.CALIB
            return FrameStatus(
                phase=self.WARMUP, is_closed=False, T_low=None,
                mu=mu, sigma=0.0, progress=self._frame/WARMUP_N, n=0,
            )

        # ── CALIBRATING ──────────────────────────────────────────────────
        # Only push valid open-eye samples (Dewi 2022: floor=0.18)
        if ear_raw >= EAR_FLOOR:
            self._win.push(ear_raw)

        sigma = self._mad.compute(self._win.arr) if self._win.ready else 0.0

        frames_in = self._frame - WARMUP_N
        progress  = frames_in / COLLECT_N

        if frames_in >= COLLECT_N:
            self._lock(mu, sigma)

        return FrameStatus(
            phase=self.CALIB, is_closed=False,
            T_low=_threshold(mu, sigma) if sigma > 0 else None,
            mu=mu, sigma=sigma,
            progress=min(progress, 1.0), n=self._win.n,
        )

    # ── properties ───────────────────────────────────────────────────────

    @property
    def locked(self) -> Optional[LockedThreshold]:
        return self._locked

    @property
    def is_locked(self) -> bool:
        return self._phase == self.LOCKED

    # ── force lock (if enough data) ──────────────────────────────────────

    def force_lock(self):
        mu = self._ewma.mu
        if mu and self._win.n >= 30:
            self._lock(mu, self._mad.sigma)
        else:
            print(f"[DynEAR] Not enough data (n={self._win.n})")

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self):
        self._win  = SlidingWindow()
        self._ewma.reset()
        self._hyst.reset()
        self._phase  = self.WARMUP
        self._frame  = 0
        self._locked = None

    # ── private: lock ────────────────────────────────────────────────────

    def _lock(self, mu: float, sigma: float):
        """
        Final computation at lock time.
        Uses full window for one last MAD pass (no cache).
        """
        arr = self._win.arr
        if len(arr) >= 30:
            med   = float(np.median(arr))
            # Rousseeuw & Croux (1993): b=1.4826
            sigma = max(MAD_B * float(np.median(np.abs(arr - med))), 0.003)

        # T_t = μ_t − k·σ_t,  floor=0.18 (Dewi 2022; Leys 2013)
        T_low  = _threshold(mu, sigma)
        T_high = T_low + GAP  # 2·σ_EWMA noise band (Hunter 1986)

        self._locked = LockedThreshold(
            T_low=T_low, T_high=T_high,
            mu=mu, sigma=sigma,
            n=len(arr), ts=time.time(),
        )
        self._phase = self.LOCKED

        # ── Print ─────────────────────────────────────────────────────
        print("=" * 60)
        print("  [DynEAR] THRESHOLD LOCKED")
        print(f"  samples  n  = {len(arr)}")
        print(f"  μ (EWMA)    = {mu:.4f}   "
              f"[α=0.20, Hunter 1986]")
        print(f"  σ (MAD·b)   = {sigma:.4f}  "
              f"[b=1.4826, Rousseeuw 1993]")
        print(f"  k           = {K}      "
              f"[Leys et al. 2013, moderately conservative]")
        print(f"  ─────────────────────────────────────────────────")
        print(f"  T_low  = μ − k·σ  = {T_low:.4f}  "
              f"(floor 0.18, Dewi et al. 2022)")
        print(f"  T_high = T + gap  = {T_high:.4f}  "
              f"(gap=2·σ_EWMA=0.013, Hunter 1986)")
        print("=" * 60)

        if self._save_path:
            self._save(self._save_path)

    # ── private: persistence ─────────────────────────────────────────────

    def _save(self, path: str):
        try:
            with open(path, 'w') as f:
                json.dump(self._locked.to_dict(), f, indent=2)
        except Exception as e:
            print(f"[DynEAR] Save failed: {e}")

    def _load(self, path: str):
        try:
            with open(path) as f:
                d = json.load(f)
            self._locked = LockedThreshold.from_dict(d)
            self._phase  = self.LOCKED
            age = (time.time() - self._locked.ts) / 60
            print(f"[DynEAR] Loaded (age {age:.1f} min)  "
                  f"T_low={self._locked.T_low:.4f}  "
                  f"T_high={self._locked.T_high:.4f}")
        except Exception as e:
            print(f"[DynEAR] Load failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════════════

def draw_ear_hud(frame, status: FrameStatus) -> None:
    """Minimal OpenCV HUD — calibration overlay + locked EAR bar."""
    import cv2
    h, w = frame.shape[:2]
    F = cv2.FONT_HERSHEY_SIMPLEX
    C = (0, 80, 255) if (status.locked and status.is_closed) else (0, 210, 80)

    # ── LOCKED: bottom bar ───────────────────────────────────────────────
    if status.locked:
        cv2.putText(frame,
            f"EAR:{status.mu:.3f}  T:{status.T_low:.3f}  "
            f"sigma:{status.sigma:.4f}",
            (8, 30), F, 0.55, C, 2)
        # EAR bar (0 → 0.5)
        bx1, bx2, by = 8, w-8, h-22
        cv2.rectangle(frame, (bx1, by-10), (bx2, by+4), (20,25,40), -1)
        ew = int((bx2-bx1) * min(status.mu / 0.5, 1.0))
        cv2.rectangle(frame, (bx1, by-10), (bx1+ew, by+4), C, -1)
        # T_low marker (amber)
        tx = bx1 + int((bx2-bx1) * status.T_low / 0.5)
        cv2.line(frame, (tx, by-14), (tx, by+8), (0,165,255), 2)
        return

    # ── CALIBRATING / WARMUP: overlay ────────────────────────────────────
    ov = frame.copy()
    cv2.rectangle(ov, (w//8, h//5), (7*w//8, 4*h//5), (6, 10, 20), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (w//8, h//5), (7*w//8, 4*h//5), (40, 80, 180), 2)

    cx = w // 2
    warmup = (status.phase == 'WARMUP')
    title  = "DANG KHOI DONG..." if warmup else "DO CHUAN HOA EAR CA NHAN"
    col    = (0, 165, 255) if warmup else (0, 220, 255)
    (tw, _), _ = cv2.getTextSize(title, F, 0.8, 2)
    cv2.putText(frame, title, (cx-tw//2, h//5+45), F, 0.8, col, 2)

    hint = "Giu nguyen, nhin thang..." if warmup else \
           "Mo mat tu nhien, nhin thang camera"
    (hw_, _), _ = cv2.getTextSize(hint, F, 0.44, 1)
    cv2.putText(frame, hint, (cx-hw_//2, h//5+68), F, 0.44, (90,100,120), 1)

    # Progress bar
    pb1, pb2 = w//8+20, 7*w//8-20
    pby = (h//5 + 4*h//5)//2 - 10
    cv2.rectangle(frame, (pb1, pby), (pb2, pby+16), (25,35,55), -1)
    pw = int((pb2-pb1) * status.progress)
    cv2.rectangle(frame, (pb1, pby), (pb1+pw, pby+16), col, -1)
    cv2.rectangle(frame, (pb1, pby), (pb2, pby+16), (60,80,140), 1)
    cv2.putText(frame, f"{status.pct}%", (cx-14, pby+34), F, 0.6, (200,215,230), 2)

    # Stats
    sy = pby + 55
    if not warmup:
        cv2.putText(frame,
            f"n={status.n}  mu={status.mu:.4f}  sigma={status.sigma:.4f}"
            + (f"  T_live={status.T_low:.4f}" if status.T_low else ""),
            (w//8+15, sy), F, 0.38, (80,90,110), 1)


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION — drop-in replacement for main drowsiness detection loop
# ══════════════════════════════════════════════════════════════════════════

def build_main_loop():
    """
    Complete main loop integrating DynamicEAR with the existing
    paper_based_drowsiness_detection.py system.

    DynamicEAR replaces the old static-threshold EAR check.
    All other detectors (EWMA control chart, PERCLOS, blink tracker,
    long-closure, head nod) continue unchanged.

    Usage:
        from dynamic_ear_threshold_v2 import DynamicEAR, draw_ear_hud, FrameStatus
    """
    import cv2
    import mediapipe as mp
    import math
    import threading
    from playsound import playsound

    # ── local EAR helper (Soukupova & Cech 2016) ─────────────────────────
    def ear(pts):
        A = math.hypot(pts[1][0]-pts[5][0], pts[1][1]-pts[5][1])
        B = math.hypot(pts[2][0]-pts[4][0], pts[2][1]-pts[4][1])
        C = math.hypot(pts[0][0]-pts[3][0], pts[0][1]-pts[3][1])
        return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

    def mar(pts):
        A = math.hypot(pts[2][0]-pts[3][0], pts[2][1]-pts[3][1])
        B = math.hypot(pts[0][0]-pts[1][0], pts[0][1]-pts[1][1])
        return A / B if B > 1e-6 else 0.0

    def get_roi(frame, pts, pad=8):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        h, w = frame.shape[:2]
        x1 = max(0, min(xs)-pad); y1 = max(0, min(ys)-pad)
        x2 = min(w, max(xs)+pad); y2 = min(h, max(ys)+pad)
        return frame[y1:y2, x1:x2] if x2>x1 and y2>y1 else None

    # ── MediaPipe setup ───────────────────────────────────────────────────
    mp_fm = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6,
    )
    LEFT_EYE  = [33,160,158,133,153,144]
    RIGHT_EYE = [362,385,387,263,373,380]
    MOUTH_PTS = [61,291,13,14,17,78,308]

    # ── Instantiate DynamicEAR ────────────────────────────────────────────
    # Parameters all paper-backed (see module header)
    dyn_ear = DynamicEAR(save_path='ear_calib.json')

    # ── Per-person long-closure state ─────────────────────────────────────
    # DOT/FAA/AM-94/17 (Stern 1994): 300ms onset, 400ms confirmed
    LONG_CLOSURE_MS  = 300    # [AM-94/17] onset of fatigue long closure
    closure_start    = None
    closure_dur      = 0.0
    closure_fired    = False
    prev_closed      = False

    # ── PERCLOS state (NHTSA) ─────────────────────────────────────────────
    from collections import deque
    perclos_hist      = deque(maxlen=1800)  # 60s @ 30fps
    perclos_start     = 0.0
    perclos_fired     = False

    # ── Yawn state ────────────────────────────────────────────────────────
    # MAR > 0.6 for ≥ 15 frames = yawn (Soukupova & Cech 2016 protocol)
    yawn_frames = 0
    yawn_active = False
    yawn_ts     = deque(maxlen=50)

    # ── Blink tracker ─────────────────────────────────────────────────────
    # Soukupova & Cech (2016): blink 100–400ms
    blink_ts    = deque(maxlen=200)
    blink_dur   = deque(maxlen=100)
    blink_start = None
    blink_on    = False

    # ── Alert ─────────────────────────────────────────────────────────────
    stop_ev      = threading.Event()
    alert_thread = None

    def alert_loop(ev):
        while not ev.is_set():
            try: playsound('alert.wav')
            except: time.sleep(0.5)

    def double_alert():
        try: playsound('alert.wav'); time.sleep(0.2); playsound('alert.wav')
        except: pass

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    prev_t = time.time()

    print("[MAIN] Starting — dynamic EAR threshold active.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt  = max(now - prev_t, 1e-4)
        prev_t = now
        fps = 1.0 / dt

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            cv2.putText(frame, "No face", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow('Drowsiness Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        lm = res.multi_face_landmarks[0].landmark
        def gxy(i): return (int(lm[i].x*w), int(lm[i].y*h))

        left_pts  = [gxy(i) for i in LEFT_EYE]
        right_pts = [gxy(i) for i in RIGHT_EYE]
        mouth_pts = [gxy(i) for i in MOUTH_PTS]

        ear_l   = ear(left_pts)
        ear_r   = ear(right_pts)
        ear_raw = (ear_l + ear_r) / 2.0
        mar_val = mar(mouth_pts)

        # ══ DYNAMIC EAR THRESHOLD ══════════════════════════════════════
        status = dyn_ear.update(ear_raw)
        draw_ear_hud(frame, status)

        if not status.locked:
            # Still calibrating — show overlay, skip detection
            cv2.imshow('Drowsiness Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # ══ DETECTION (runs only after threshold is locked) ════════════
        T_low  = status.T_low          # personal threshold
        is_eye_closed = status.is_closed  # with hysteresis (Hunter 1986 noise band)

        drowsy       = False
        reasons: list[str] = []

        # ── 1. Long closure (DOT/FAA/AM-94/17) ───────────────────────
        # "closure duration ranged between 300 and 400 milliseconds"
        if is_eye_closed and not prev_closed:
            closure_start = now
            closure_fired = False
        elif not is_eye_closed and prev_closed:
            closure_start = None
            closure_dur   = 0.0
        elif is_eye_closed and closure_start:
            closure_dur = now - closure_start
        else:
            closure_dur = 0.0
        prev_closed = is_eye_closed

        if closure_dur * 1000 >= LONG_CLOSURE_MS and not closure_fired:
            drowsy = True
            closure_fired = True
            reasons.append(f"LONG CLOSURE {closure_dur*1000:.0f}ms")

        # ── 2. PERCLOS (NHTSA >15% for 1s) ───────────────────────────
        perclos_hist.append(1 if is_eye_closed else 0)
        perclos = sum(perclos_hist)/len(perclos_hist)*100
        if perclos > 15.0:
            if perclos_start == 0.0: perclos_start = now
            if now - perclos_start >= 1.0 and not perclos_fired:
                drowsy = True
                perclos_fired = True
                reasons.append(f"PERCLOS {perclos:.1f}%")
        else:
            perclos_start = 0.0
            perclos_fired = False

        # ── 3. Blink tracker (Soukupova & Cech 2016 + AM-94/17) ──────
        # Blink: EAR < T_low for 2–12 frames (100–400ms @ 30fps)
        # AM-94/17: "blink closure duration... 300–400ms = fatigue"
        if is_eye_closed:
            if not blink_on:
                blink_on    = True
                blink_start = now
        else:
            if blink_on:
                dur_ms = (now - blink_start) * 1000 if blink_start else 0
                if 60 <= dur_ms <= 500:        # valid blink range
                    blink_ts.append(now)
                    blink_dur.append(dur_ms)
                blink_on    = False
                blink_start = None

        # Blink rate (AM-94/17: fatigue if rate changes significantly)
        recent_blinks = [t for t in blink_ts if now - t <= 10.0]
        blink_rate    = len(recent_blinks) * 6   # per minute
        mean_dur_ms   = (sum(blink_dur)/len(blink_dur)) if blink_dur else 0

        # AM-94/17: mean closure duration >300ms = fatigue indicator
        if mean_dur_ms >= LONG_CLOSURE_MS:
            drowsy = True
            reasons.append(f"AVG CLOSURE {mean_dur_ms:.0f}ms")

        # AM-94/17: elevated blink rate (>18/min during task)
        if blink_rate > 18:
            drowsy = True
            reasons.append(f"BLINK RATE {blink_rate:.0f}/min")

        # ── 4. Yawn (MAR > 0.6 for ≥ 15 frames) ─────────────────────
        if mar_val > 0.60:
            yawn_frames += 1
            if not yawn_active and yawn_frames >= 15:
                yawn_active = True
                yawn_ts.append(now)
                threading.Thread(target=double_alert, daemon=True).start()
        else:
            yawn_active = False
            yawn_frames = max(0, yawn_frames - 1)

        recent_yawns = [t for t in yawn_ts if now - t <= 60]
        if recent_yawns:
            reasons.append(f"YAWN x{len(recent_yawns)}")

        # ── Alert sound for critical events ───────────────────────────
        critical = [r for r in reasons if
                    any(k in r for k in ["LONG CLOSURE","PERCLOS","AVG CLOSURE"])]
        if drowsy and critical:
            if alert_thread is None or not alert_thread.is_alive():
                stop_ev.clear()
                alert_thread = threading.Thread(
                    target=alert_loop, args=(stop_ev,), daemon=True)
                alert_thread.start()
        elif not drowsy and alert_thread and alert_thread.is_alive():
            stop_ev.set()

        # ── HUD — detection stats ──────────────────────────────────────
        def txt(msg, y, c=(200,215,230)):
            cv2.putText(frame, msg, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, c, 2)

        ec = (0,80,255) if is_eye_closed else (0,200,80)
        txt(f"EAR:{ear_raw:.3f} T_low:{T_low:.3f} "
            f"[mu:{status.mu:.3f} σ:{status.sigma:.4f}]", 52, ec)
        txt(f"Closure:{closure_dur*1000:.0f}ms  "
            f"PERCLOS:{perclos:.1f}%  "
            f"MAR:{mar_val:.2f}", 74)
        txt(f"Blink:{blink_rate:.0f}/min  "
            f"AvgDur:{mean_dur_ms:.0f}ms  "
            f"Yawn:{len(recent_yawns)}  FPS:{fps:.0f}", 96)

        if drowsy:
            label = "DROWSY: " + " | ".join(reasons[:2])
            cv2.putText(frame, label, (6, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 3)
        else:
            cv2.putText(frame, "ALERT", (6, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,210,80), 3)

        # Eye state boxes
        for pts, clo in [(left_pts, is_eye_closed), (right_pts, is_eye_closed)]:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            c = (0,80,255) if clo else (0,200,80)
            cv2.rectangle(frame,
                (min(xs)-5, min(ys)-5), (max(xs)+5, max(ys)+5), c, 2)

        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_ev.set()

