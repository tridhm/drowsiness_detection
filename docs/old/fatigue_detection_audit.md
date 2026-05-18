# Vision-Based Driver Fatigue Detection Audit

## Executive Summary
The current system is **functional as a demo** but not yet robust enough for real-world automotive deployment.  
Major failure modes are driven by:
- fixed anthropometric thresholds,
- limited calibration scope,
- unsmoothed landmark-derived signals,
- abrupt boolean alerting,
- and no standards-style closure-time metric (PERCLOS).

---

## ALGORITHMIC FALLACIES (Code-Evidenced)

### 1) Static human-behavior thresholds are treated as universal constants
- `EAR_THRESH`, `MAR_THRESH`, and head-pose thresholds are hardcoded in runtime logic:  
  - [advanced_drowsiness_detection.py:L382-L391](file:///D:/drowsiness_detection-main/advanced_drowsiness_detection.py#L382-L391)  
  - [drowsiness_detection_with_model.py:L127-L132](file:///D:/drowsiness_detection-main/drowsiness_detection_with_model.py#L127-L132)  
  - [v10.py:L10-L13](file:///D:/drowsiness_detection-main/v10.py#L10-L13)
- Why this fails: eye aperture, eyelid geometry, and mouth opening vary strongly across drivers (monolid, deep-set eye, camera distance, glasses). Fixed thresholds create systematic false positives/negatives.

### 2) Calibration exists, but only for head pose and only 50 frames
- Calibration is present only for pitch/yaw/roll baseline with `CALIBRATION_FRAMES = 50` (~2 s):  
  [advanced_drowsiness_detection.py:L423-L535](file:///D:/drowsiness_detection-main/advanced_drowsiness_detection.py#L423-L535)
- Why this fails: EAR/MAR are still uncalibrated; 2 seconds is too short for stable baseline under natural startup motion.

### 3) Alerting remains mostly boolean and abrupt
- `is_drowsy` is set by threshold/counter triggers and immediately drives alarm loop behavior:  
  - [advanced_drowsiness_detection.py:L565-L699](file:///D:/drowsiness_detection-main/advanced_drowsiness_detection.py#L565-L699)  
  - [advanced_drowsiness_detection.py:L772-L800](file:///D:/drowsiness_detection-main/advanced_drowsiness_detection.py#L772-L800)  
  - [drowsiness_detection_with_model.py:L236-L252](file:///D:/drowsiness_detection-main/drowsiness_detection_with_model.py#L236-L252)
- Why this fails: no explicit graded state machine for escalation/de-escalation, hysteresis, or confidence accumulation.

### 4) No PERCLOS-style closure-time metric in runtime
- Project-wide search found no `perclos` implementation.
- Why this fails: absence of a normalized, time-based eyelid-closure metric weakens fatigue validity and temporal robustness.

### 5) Raw geometric features are used without temporal smoothing
- EAR/MAR are directly computed from frame landmarks and thresholded:  
  - [advanced_drowsiness_detection.py:L365-L381](file:///D:/drowsiness_detection-main/advanced_drowsiness_detection.py#L365-L381)  
  - [train_data.py:L64-L79](file:///D:/drowsiness_detection-main/train_data.py#L64-L79)
- Why this fails: MediaPipe/dlib jitter, vibration, and illumination fluctuations produce unstable detections.

---

## Evidence-Based Corrections (Mathematical + Industrial)

### A) 10-second startup dynamic calibration (mandatory)
Use ~300 frames at 30 FPS to estimate per-driver baselines:
- `EAR_open_baseline = median(EAR_calib)`
- `MAR_rest_baseline = median(MAR_calib)`
- `pitch/yaw/roll_baseline = robust mean`

Then derive personalized thresholds, e.g.:
- `EAR_closed_thr = EAR_open_baseline - k1 * sigma_EAR`
- `MAR_yawn_thr = MAR_rest_baseline + k2 * sigma_MAR`

This directly addresses anthropometric diversity and camera-position variance.

### B) EMA smoothing on all framewise continuous signals
For signal `x_t` (`EAR`, `MAR`, pitch, yaw):

`x_hat_t = alpha * x_t + (1 - alpha) * x_hat_{t-1}`

Recommended starting range: `alpha = 0.15 .. 0.30` (tune by latency/noise trade-off).  
Use `x_hat_t` for all threshold and state decisions.

### C) Implement standards-aligned PERCLOS queue
Maintain a circular buffer over a rolling window (e.g., 60 s):
- `closed_t = 1` if normalized eye opening <= 20% of calibrated open state (equivalent to >=80% closure)
- `PERCLOS = sum(closed_t over window) / window_length`

Also keep event metrics:
- microsleep duration,
- closure run-length,
- closure frequency.

### D) Replace boolean alerting with explicit FSM
Suggested states:
1. `ALERT`
2. `SUSPECT`
3. `DROWSY`
4. `CRITICAL`

Transitions should use fused evidence + hysteresis:
- `SUSPECT`: rising PERCLOS or repeated closures
- `DROWSY`: sustained PERCLOS + corroborating MAR/head cues
- `CRITICAL`: microsleep-length closures and/or very high sustained PERCLOS

Exit criteria should require sustained recovery (not single-frame reset).

---

## Old vs New Parameters / Logic (Required Comparison)

| Dimension | Current Implementation | Recommended Replacement |
|---|---|---|
| EAR threshold | Fixed constants `0.22/0.23/0.25` ([adv](file:///D:/drowsiness_detection-main/advanced_drowsiness_detection.py#L382-L383), [model](file:///D:/drowsiness_detection-main/drowsiness_detection_with_model.py#L127-L128), [v10](file:///D:/drowsiness_detection-main/v10.py#L12-L13)) | Per-driver dynamic threshold from 10s baseline (`median - k*sigma`) |
| MAR threshold | Fixed `MAR_THRESH = 0.6` ([adv](file:///D:/drowsiness_detection-main/advanced_drowsiness_detection.py#L384-L385), [model](file:///D:/drowsiness_detection-main/drowsiness_detection_with_model.py#L129-L130)) | Personalized MAR threshold from resting baseline (`median + k*sigma`) |
| Calibration scope | Head pose only, `CALIBRATION_FRAMES = 50` (~2s) ([adv](file:///D:/drowsiness_detection-main/advanced_drowsiness_detection.py#L423-L535)) | Calibrate EAR/MAR/head pose for ~300 frames (~10s) |
| Signal conditioning | No EMA on EAR/MAR/head pose | EMA-filtered signals (`alpha` tuned 0.15–0.30) |
| Drowsiness metric | Counter + threshold heuristics | Add rolling-window PERCLOS + closure run-length metrics |
| Alert policy | `if is_drowsy` then alarm loop ([adv](file:///D:/drowsiness_detection-main/advanced_drowsiness_detection.py#L772-L800), [model](file:///D:/drowsiness_detection-main/drowsiness_detection_with_model.py#L236-L252)) | Multi-level FSM with hysteresis and recovery criteria |
| Model fusion | Frame-level heuristic combinations | Probabilistic or weighted temporal fusion of ocular + yawn + head signals |

---

## Standards Mapping (NHTSA / Euro NCAP)
- **NHTSA** workstreams and drowsy-driving program framing support technology-based fatigue mitigation and operational DMS development.  
- **PERCLOS lineage** in US DOT/NHTSA/FHWA drowsiness research supports eyelid-closure-over-time as a central metric for alertness degradation.  
- **Euro NCAP 2026 Safe Driving protocols** provide explicit Driver Monitoring dossier/test artifacts and a robustness-oriented validation culture for real driving conditions.

---

## Priority Refactor Order
1. Introduce shared signal pipeline (`calibration -> EMA -> feature normalization`).
2. Add rolling PERCLOS queue + microsleep detector.
3. Implement FSM alert manager (replace direct `is_drowsy` alarm trigger).
4. Refit threshold constants as tunable defaults only (never primary logic).
5. Add offline replay evaluation (false alarm rate, miss rate, detection latency).

---

## References
1. Euro NCAP 2026 Protocols:  
   https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/
2. Euro NCAP SD 201 – Driver Monitoring Dossier Guidance:  
   https://www.euroncap.com/media/91718/sd-201-driver-monitoring-dossier-guidance-v11.pdf
3. Euro NCAP SD 202 – Driver Monitoring Test Procedure:  
   https://www.euroncap.com/media/91719/sd-202-driver-monitoring-test-procedure-v11.pdf
4. NHTSA Drowsy Driving Research and Program Plan (DOT HS 812 252):  
   https://www.nhtsa.gov/sites/nhtsa.gov/files/drowsydriving_strategicplan_030316.pdf
5. NHTSA Drowsy Driver Warning System FOT report URL (as indexed in NHTSA references):  
   http://www.nhtsa.gov/DOT/NHTSA/NRD/Multimedia/PDFs/Crash%20Avoidance/2008/810035.pdf
