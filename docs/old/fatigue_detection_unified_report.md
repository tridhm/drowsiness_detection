# Unified Driver Fatigue Detection Audit Report

## Source Documents Aggregated
This report consolidates findings from:
- [fatigue_detection_audit.md](file:///d:/drowsiness_detection-main/fatigue_detection_audit.md)
- [fatigue_detection_audit_report.md](file:///d:/drowsiness_detection-main/fatigue_detection_audit_report.md)
- [analysis_results.md](file:///d:/drowsiness_detection-main/analysis_results.md)

---

## Executive Summary
The current system is best characterized as a **prototype heuristic vigilance detector**, not a clinically validated fatigue-detection stack. Across runtime and training code, the same core issues recur:

1. **Clinical ground truth weakness**: labels are assigned per video (binary) rather than time-aligned to validated sleepiness scales (e.g., KSS).
2. **Heuristic threshold dominance**: fixed EAR/MAR/head-pose constants and frame counters are treated as universal physiology.
3. **Temporal fragility**: logic relies on frame counts that are not robustly normalized to measured FPS.
4. **Insufficient signal conditioning**: little/no smoothing (EMA) and no standards-style closure-time metric (PERCLOS) in the decision core.
5. **Rule-stack fusion bias**: boolean OR/AND logic creates implicit hard weights and can amplify false positives.
6. **Validation gaps**: no complete subject-disjoint, system-level ablation + significance-testing protocol proving each module is necessary.
7. **Train/runtime inconsistency**: at least one trained RF artifact is not consistently used in online inference.

---

## Scope
Audited artifacts and referenced logic include:
- [advanced_drowsiness_detection.py](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py)
- [drowsiness_detection_with_model.py](file:///d:/drowsiness_detection-main/drowsiness_detection_with_model.py)
- [v10.py](file:///d:/drowsiness_detection-main/v10.py)
- [train_data.py](file:///d:/drowsiness_detection-main/train_data.py)
- [extract_video_frames.py](file:///d:/drowsiness_detection-main/extract_video_frames.py)
- [train_model.py](file:///d:/drowsiness_detection-main/train_model.py)
- [train_advanced_model.py](file:///d:/drowsiness_detection-main/train_advanced_model.py)

---

## Consolidated Code-Flaw Discovery

### 1) Ground-truth construction is not clinically anchored
- Video-level labels (`LABEL_TO_WRITE`) are manually assigned and can ignore within-video state changes.
- This can train models to learn clip-specific artifacts (lighting/camera identity/subject traits) rather than true fatigue states.

**Impact**: Biological validity ceiling is low regardless of downstream model complexity.

### 2) Static thresholds are treated as universal physiology
- Hardcoded thresholds (`EAR_THRESH`, `MAR_THRESH`, `HEAD_PITCH_THRESH`, etc.) dominate behavior.
- Anthropometry, camera geometry, eyelid morphology, eyewear, and face alignment vary significantly across users.

**Impact**: Systematic false positives/negatives across populations.

### 3) Temporal criteria are partly arbitrary and FPS-dependent
- Frame-count logic (e.g., 15, 20, 30 frames) changes physiological meaning when FPS changes.
- A 15-frame closure could mean 250 ms (60 FPS), 500 ms (30 FPS), or 1.0 s (15 FPS).

**Impact**: Decision semantics drift with camera performance.

### 4) Yawn logic is too short and weakly specific
- Sub-second to ~0.67 s MAR episodes can be counted as yawns in current settings.
- Genuine yawning is typically multi-second and morphologically distinct from speech/chewing/cough.

**Impact**: Frequent mouth-motion false alarms.

### 5) Head-nod detection is mostly static-angle based
- Fixed pitch threshold + counter cannot separate drowsy nodding from normal down-looking behavior.
- Sleep-onset nodding is dynamic (drop velocity/recovery), not just static head pose.

**Impact**: Context-insensitive head-pose alarms.

### 6) No strong PERCLOS-style runtime metric
- No robust closure-over-time queue is implemented as a first-class core feature.

**Impact**: Limited standards alignment for ocular fatigue interpretation.

### 7) Fusion logic imposes implicit hard weights
- Boolean decision stacking (`A or B`) lets one noisy trigger dominate others.
- Fixed confidence cutoffs are not equivalent to calibrated probabilities.

**Impact**: Non-transparent behavior and unstable risk calibration.

### 8) Training/validation methodology is incomplete for safety claims
- Partial training metrics exist, but system-level proof is incomplete.
- Missing consistent confusion-matrix-centered evaluation, confidence intervals, and ablation significance tests.
- Potential temporal leakage risk from frame-level splits.

**Impact**: Performance claims may be optimistic and not deployment-safe.

---

## Unified Evidence-Based Redesign

### A) Ground Truth Layer (Clinical Anchor)
1. Collect **Karolinska Sleepiness Scale (KSS)** at fixed time intervals.
2. Time-align KSS to frame windows.
3. Use windowed labels (not whole-video binary tags).

Suggested KSS bands:
- `KSS 1–4`: alert
- `KSS 5–6`: caution/intermediate
- `KSS 7–9`: sleepy/high risk

### B) Signal Pipeline (Robustness)
1. **Startup dynamic calibration (~10 s)** for EAR, MAR, pitch/yaw/roll.
2. **EMA smoothing** on all continuous signals:
   - `x_hat_t = alpha * x_t + (1 - alpha) * x_hat_{t-1}`
   - initial `alpha` range: `0.15–0.30`
3. **FPS normalization** for all time-based event thresholds.

### C) Clinically Motivated Event Definitions
1. **Ocular lapse (EAR)**:
   - event duration criterion: `> 500 ms` (PVT-aligned convention)
   - `eye_lapse_frames = ceil(fps * 0.5)`
2. **Yawn (MAR + duration/morphology)**:
   - candidate window `4–7 s` at measured FPS
   - reject short mouth events as non-specific unless supported by morphology/classifier
3. **Head-drop event (pitch dynamics)**:
   - detect rapid downward pitch excursion + partial recovery
   - use velocity/acceleration features, not static angle alone

### D) Decision Layer (From Rules to Evidence Fusion)
1. Replace direct boolean trigger stack with **tiered state model**:
   - `ALERT -> SUSPECT -> DROWSY -> CRITICAL`
2. Require hysteresis and sustained recovery criteria.
3. Fuse multi-signal evidence over time (ocular + yawn + head dynamics + optional model posterior).
4. Calibrate probabilities and thresholds on clinically aligned validation data.

### E) ML/Statistics Layer (Proof of Irreducibility)
1. Train temporal multimodal model(s) on subject-disjoint splits.
2. Compute:
   - Precision / Recall / F1
   - Confusion matrix
   - PR-AUC
   - 95% bootstrap CI for F1
3. Mandatory ablations:
   - remove yawn block
   - remove head-pose block
   - remove eye-temporal block
   - remove fusion (single-modality baseline)
4. Use paired significance testing (e.g., McNemar for error differences).
5. Keep train/deploy parity in one auditable inference stack.

---

## Consolidated Old vs New Variables / Logic

| Current variable / logic | Current issue | Consolidated replacement |
|---|---|---|
| `LABEL_TO_WRITE` (video-wide binary) | Not clinically time-aligned ground truth | `kss_score`, `kss_band`, timestamp-aligned window labels |
| Fixed `EAR_THRESH` | Person/camera dependent | Subject-calibrated `ear_closed_threshold_subject_specific` |
| `EAR_FRAMES = 15` | FPS-dependent semantics | `eye_lapse_frames = ceil(fps * 0.5)` |
| Fixed `MAR_THRESH` only | Poor specificity (speech vs yawn) | Subject-relative MAR + duration/morphology gates |
| `MAR_FRAMES` / `YAWN_FRAMES` short | Too brief for physiologic yawn | `yawn_min_duration_sec = 4.0`, `yawn_max_duration_sec = 7.0` |
| Static head pitch threshold + counter | Confuses posture with nod event | `pitch_drop_amplitude`, `pitch_velocity_deg_per_sec`, recovery timing |
| `BLINK_FREQ_THRESH` global | Population/context sensitivity | Baseline-relative blink change + supporting signal only |
| Pixel-based gaze-fixation rules | Resolution/scale dependent | Normalized gaze dispersion + context-aware interpretation |
| `CONFIDENCE_THRESH` fixed | Confidence != calibrated risk | Calibrated posterior + validated fusion rule |
| Boolean OR trigger stack | Implicit hard weights | Probabilistic/learned temporal fusion + FSM hysteresis |
| No runtime PERCLOS core | Weak temporal closure evidence | Rolling-window PERCLOS + run-length/microlapse stats |
| Training model/runtime mismatch | Drift and inconsistency | Single train=deploy pipeline with tracked model lineage |

---

## Priority Implementation Roadmap
1. **Ground truth first**: migrate to KSS-aligned window labels.
2. Build shared preprocessing chain: calibration -> EMA -> normalization.
3. Add PERCLOS queue + ocular lapse run-length metrics.
4. Redefine yawn/head events with duration/dynamics.
5. Replace direct `is_drowsy` trigger with FSM + hysteresis.
6. Introduce temporal multimodal learner and calibrated fusion.
7. Run subject-disjoint evaluation + ablation + significance tests.
8. Freeze deployment through a single auditable inference stack.

---

## References
1. Euro NCAP 2026 Protocols:  
   [https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/](https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/)
2. Euro NCAP SD 201 – Driver Monitoring Dossier Guidance:  
   [https://www.euroncap.com/media/91718/sd-201-driver-monitoring-dossier-guidance-v11.pdf](https://www.euroncap.com/media/91718/sd-201-driver-monitoring-dossier-guidance-v11.pdf)
3. Euro NCAP SD 202 – Driver Monitoring Test Procedure:  
   [https://www.euroncap.com/media/91719/sd-202-driver-monitoring-test-procedure-v11.pdf](https://www.euroncap.com/media/91719/sd-202-driver-monitoring-test-procedure-v11.pdf)
4. NHTSA Drowsy Driving Program Plan (DOT HS 812 252):  
   [https://www.nhtsa.gov/sites/nhtsa.gov/files/drowsydriving_strategicplan_030316.pdf](https://www.nhtsa.gov/sites/nhtsa.gov/files/drowsydriving_strategicplan_030316.pdf)
5. NHTSA Drowsy Driver Warning System FOT reference URL:  
   [http://www.nhtsa.gov/DOT/NHTSA/NRD/Multimedia/PDFs/Crash%20Avoidance/2008/810035.pdf](http://www.nhtsa.gov/DOT/NHTSA/NRD/Multimedia/PDFs/Crash%20Avoidance/2008/810035.pdf)
6. Kaida K, et al. (2006). Validation of KSS against performance and EEG variables. *Clinical Neurophysiology*.
7. Dinges DF / Basner M. PVT literature (lapse convention >500 ms).
8. Gallup AC, Eldakar OT, et al. Yawning physiology literature (typical human yawn duration ~4–7 s).
9. Breiman L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. DOI: 10.1023/A:1010933404324.
10. Baltrušaitis T, Ahuja C, Morency L-P. (2019). Multimodal ML survey. *IEEE TPAMI*, 41(2), 423–443. DOI: 10.1109/TPAMI.2018.2798607.
11. Sokolova M, Lapalme G. (2009). Classification performance measures. *Information Processing & Management*, 45(4), 427–437.
12. Scikit-learn: forest feature-importance example and caveats.  
    [https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
