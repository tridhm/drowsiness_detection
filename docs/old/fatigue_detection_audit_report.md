# Vision-Based Driver Fatigue Detection Audit

## Scope
This audit reviewed the runtime and data-labeling code in:
- [advanced_drowsiness_detection.py](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py)
- [drowsiness_detection_with_model.py](file:///d:/drowsiness_detection-main/drowsiness_detection_with_model.py)
- [v10.py](file:///d:/drowsiness_detection-main/v10.py)
- [train_data.py](file:///d:/drowsiness_detection-main/train_data.py)
- [extract_video_frames.py](file:///d:/drowsiness_detection-main/extract_video_frames.py)

## Executive Summary
The codebase is dominated by **heuristic thresholds and frame counters** that are not tied to validated sleep-medicine constructs. In several places, the software equates a short-lived facial configuration with **clinical drowsiness**, even though fatigue is a **state** that must be anchored to validated ground truth and temporally meaningful physiology.

The largest scientific problems are:
1. **Ground truth is not clinically anchored.** The training pipeline writes a fixed label for an entire video (`LABEL_TO_WRITE = 1`) rather than aligning labels to a validated sleepiness scale such as **KSS**.
2. **Temporal logic is inconsistent with sleep physiology.** Some thresholds are extremely short for true fatigue phenomena (e.g. yawning after 15–20 frames), while others are arbitrary with no FPS normalization.
3. **Static geometry is treated as biology.** EAR, MAR, and head pose are highly person-, camera-, and pose-dependent. A single absolute cutoff cannot represent all drivers.
4. **The code conflates warning signs with diagnosis.** A yawn, brief eye closure, fixed gaze, or head movement can be normal behavior, task strategy, speech, or camera artifact. None alone is a clinical diagnosis of drowsiness.

---

## CODE FLAW DISCOVERY

### 1) Video-level labels are hardcoded, not clinically measured
- In [train_data.py:90-98](file:///d:/drowsiness_detection-main/train_data.py#L90-L98), the dataset label is manually assigned with `LABEL_TO_WRITE = 1` for an entire source video.
- In [extract_video_frames.py](file:///d:/drowsiness_detection-main/extract_video_frames.py), comments explicitly admit that frames are saved first and may require later manual separation.

**Why this is biologically flawed**
- Sleepiness is not a binary property of an entire recording. Within one clip, a subject may shift between alertness and drowsiness.
- Clinical fatigue studies typically anchor subjective state to **Karolinska Sleepiness Scale (KSS)** ratings or other validated ground truth, then relate vision signals to those labels.
- Training on video-level labels teaches the model to memorize clip identity, lighting, or pose instead of physiologic sleepiness.

### 2) `v10.py` declares drowsiness from a single EAR cutoff
- [v10.py:10-12](file:///d:/drowsiness_detection-main/v10.py#L10-L12): `EAR_THRESH = 0.25`
- [v10.py:139-141](file:///d:/drowsiness_detection-main/v10.py#L139-L141): if `avg_ear < EAR_THRESH`, the status becomes `"Closed / Drowsy"`.

**Why this is biologically flawed**
- This logic has **no temporal requirement**. A normal blink, squint, downward glance, eyeglasses occlusion, or face-mesh jitter can all transiently drop EAR.
- Clinically relevant ocular lapses are not defined by one low frame; they require **duration**.
- EAR is a geometric proxy, not a direct measure of sleep onset. Without subject calibration and time normalization, it is not a safe drowsiness label.

### 3) Eye-closure logic uses arbitrary frame counts without robust FPS mapping
- [drowsiness_detection_with_model.py:127-131](file:///d:/drowsiness_detection-main/drowsiness_detection_with_model.py#L127-L131): `EAR_FRAMES = 15  # ~0.5 seconds at 30 fps`
- [advanced_drowsiness_detection.py:382-391](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L382-L391): `EAR_FRAMES = 15`
- [advanced_drowsiness_detection.py:695-700](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L695-L700): prolonged eye closure is flagged after counter accumulation.

**Why this is biologically flawed**
- The code assumes **30 FPS** but never normalizes thresholds to the actual measured frame rate.
- A 15-frame closure is 500 ms at 30 FPS, but **1.0 s at 15 FPS** and **250 ms at 60 FPS**.
- If the user camera drops frames, the physiologic meaning changes immediately.

### 4) Yawn detection thresholds are far too short for genuine yawns
- [drowsiness_detection_with_model.py:129-130](file:///d:/drowsiness_detection-main/drowsiness_detection_with_model.py#L129-L130): `MAR_FRAMES = 15`
- [advanced_drowsiness_detection.py:384-387](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L384-L387): `YAWN_FRAMES = 20`, `YAWN_WINDOW = 60`, `MAX_YAWNS_IN_WINDOW = 3`
- [advanced_drowsiness_detection.py:716-733](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L716-L733): a yawn event is counted after 20 consecutive frames and frequent yawning is flagged within a 60 s window.

**Why this is biologically flawed**
- At 30 FPS, 15–20 frames is only **0.5–0.67 s**, much shorter than a typical physiologic yawn.
- Genuine yawns are multi-second motor sequences with gradual gape, inspiratory phase, climax, and release. Very short mouth openings are more likely **speech, singing, chewing, cough, jaw stretch, or landmark noise**.
- A MAR threshold alone cannot distinguish speech from yawning unless the event duration and morphology are modeled.

### 5) Head nodding is reduced to a fixed pitch angle plus counter
- [advanced_drowsiness_detection.py:388-390](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L388-L390): `HEAD_PITCH_THRESH = 20`, `HEAD_YAW_THRESH = 30`, `HEAD_NOD_FRAMES = 30`
- [advanced_drowsiness_detection.py:554-568](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L554-L568): if relative pitch exceeds threshold for enough frames, it becomes `HEAD NODDING`.

**Why this is biologically flawed**
- Sleep-onset head drops are dynamic motor events caused by **loss of postural neck-muscle tone** and brief control failure; they are better described by **pitch velocity/acceleration followed by recovery**, not just static angle.
- A driver can maintain 20° downward pitch for non-drowsy reasons: checking mirrors, dashboard, road grade, or camera placement.
- The current logic partially compensates with calibration, but still treats a static orientation as a neurophysiologic event.

### 6) Blink frequency threshold is unsupported and likely population-dependent
- [advanced_drowsiness_detection.py:395-396](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L395-L396): `BLINK_FREQ_THRESH = 10`, `BLINK_WINDOW = 10.0`
- [advanced_drowsiness_detection.py:600-607](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L600-L607): `blink_freq > 10` in 10 seconds triggers `RAPID BLINKING`.

**Why this is biologically flawed**
- Blink rate is strongly altered by conversation, dry eye, contact lenses, airflow, concentration, screen use, and lighting.
- A global threshold of `>1 blink/s` has no demonstrated clinical specificity for fatigue here.
- If used at all, blink metrics should be modeled relative to an individual baseline and interpreted as a **supporting signal**, not a direct drowsiness diagnosis.

### 7) “Blank stare” is defined by tiny gaze jitter and short fixation time
- [advanced_drowsiness_detection.py:393-394](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L393-L394): `GAZE_FIXATION_TIME = 6.0`, `GAZE_MOVE_THRESH = 1.5`
- [advanced_drowsiness_detection.py:654-668](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L654-L668): low gaze movement over time can trigger `BLANK STARE` or `FIXED GAZE`.

**Why this is biologically flawed**
- A driver can deliberately stabilize gaze on the road center during attentive driving.
- A `1.5`-pixel stability rule is extremely sensitive to camera resolution, face size, crop scale, and landmark jitter.
- “Blank stare” is not a validated clinical surrogate unless tied to richer oculomotor evidence and controlled context.

### 8) The “advanced” model mixes heuristic and classifier logic without validated fusion
- [advanced_drowsiness_detection.py:690-699](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L690-L699): model predictions and EAR are OR-combined with a fixed confidence threshold.
- [advanced_drowsiness_detection.py:391](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L391): `CONFIDENCE_THRESH = 0.75`

**Why this is biologically flawed**
- Model confidence is not the same thing as calibrated probability of physiologic eye closure.
- OR-fusing independent noisy detectors increases false positives unless the fusion is validated on a clinically labeled cohort.
- The system currently behaves more like a rule stack than a clinically interpretable detector.

---

## Evidence-Based Corrections for the Known Metrics

### 1) Ground Truth: use KSS-aligned labels, not arbitrary binary tags
**Current flaw**
- The training set writes a single binary label per source video.

**Medical correction**
- Anchor the target variable to **Karolinska Sleepiness Scale (KSS)** measurements collected at synchronized time intervals.
- Recommended mapping:
  - **KSS 1–4**: alert / low sleepiness
  - **KSS 5–6**: intermediate / caution zone
  - **KSS 7–9**: sleepy / clinically relevant drowsiness risk
- If framewise labels are needed, inherit the most recent time-synchronized KSS block label rather than assigning one label to a whole video blindly.

**Why this is better**
- KSS is a validated subjective sleepiness measure and correlates with EEG and behavioral performance, including lapse counts.

### 2) EAR ocular lapses: use PVT-standard lapse duration and FPS normalization
**Current flaw**
- `EAR_FRAMES = 15` is only meaningful if the stream is truly 30 FPS.

**Medical correction**
- Keep EAR as a geometric marker of eyelid closure, but define an ocular lapse by **duration** using the PVT convention: **lapse > 500 ms**.
- Replace raw frame constants with:
  - `lapse_duration_sec = 0.5`
  - `eye_lapse_frames = ceil(fps * lapse_duration_sec)`
- Example:
  - 30 FPS -> 15 frames
  - 25 FPS -> 13 frames
  - 15 FPS -> 8 frames
- Use actual measured FPS from the capture device, not a comment assumption.

**Clinical note**
- This does **not** prove a microsleep by itself, but it is much closer to a validated vigilance-decrement convention than arbitrary frame counts.

### 3) MAR yawns: require multi-second duration consistent with airway/thermoregulatory physiology
**Current flaw**
- `MAR_FRAMES = 15` and `YAWN_FRAMES = 20` are too short.

**Medical correction**
- Treat a putative yawn as a **mouth-opening episode lasting 4–7 seconds**.
- Replace raw counts with:
  - `yawn_min_duration_sec = 4.0`
  - `yawn_max_duration_sec = 7.0`
  - `yawn_min_frames = ceil(fps * 4.0)`
  - `yawn_max_frames = floor(fps * 7.0)`
- Example at 30 FPS:
  - minimum genuine yawn duration: **120 frames**
  - upper physiologic window: **210 frames**
- Any mouth-opening event shorter than that should be treated as **non-specific mouth activity** unless another validated classifier proves otherwise.

**Clinical note**
- This is conservative, but it prevents the system from misclassifying normal speech or transient mouth motion as fatigue.

### 4) Head pitch: justify nod detection through loss of postural tone, and track velocity
**Current flaw**
- Current code uses a fixed angle plus frame counter.

**Medical correction**
- Keep calibrated baseline pitch, but redefine the signal as a **dynamic pitch-drop event**:
  1. Compute `pitch_velocity_deg_per_sec` and optionally acceleration.
  2. Detect a **rapid downward excursion** from baseline.
  3. Require a short recovery or rebound pattern rather than a sustained static downward posture.
- Conceptually, the target is not “head below 20°,” but **brief failure of neck extensor tone** consistent with sleep onset.

**Clinical note**
- Strictly speaking, classic full muscle atonia is a REM hallmark, while early drowsiness and sleep onset involve reduced postural control and nodding rather than full REM atonia. So the correct framing is: **track pitch velocity as a surrogate for transient loss of neck postural tone during sleep onset**, not as proof of REM atonia.

---

## Old Variables vs Medically Justified Replacements

| Existing variable / logic | File(s) | Current value | Problem | Proposed medically justified variable | Evidence-based rationale | Clinical citations |
|---|---|---:|---|---|---|---|
| `LABEL_TO_WRITE` | `train_data.py` | `0` or `1` per whole video | Entire-video binary labels ignore within-video state changes and lack clinical ground truth | `kss_score`, `kss_band`, `label_timestamp` | KSS is a validated subjective sleepiness scale and should be time-synchronized to video segments | Kaida et al., 2006, *Clin Neurophysiol*; KSS validation against EEG/performance (PubMed) |
| `EAR_THRESH` only -> drowsy | `v10.py` | `0.25` | Single-frame geometry cutoff confuses blink/squint/noise with drowsiness | `ear_closed_threshold_subject_specific` + `eye_lapse_duration_sec` | EAR should be individualized and interpreted with duration, not instantaneous status | PVT lapse convention: RT >500 ms as vigilance lapse standard |
| `EAR_FRAMES` | `advanced_drowsiness_detection.py`, `drowsiness_detection_with_model.py` | `15` | Not FPS normalized; physiologic meaning changes with camera rate | `eye_lapse_frames = ceil(fps * 0.5)` | 500 ms aligns with the standard PVT lapse window | Dinges/Basner PVT literature; standard lapse >500 ms |
| `MAR_FRAMES` / `YAWN_FRAMES` | `drowsiness_detection_with_model.py`, `advanced_drowsiness_detection.py` | `15`, `20` | 0.5–0.67 s at 30 FPS is too short for a physiologic yawn | `yawn_min_duration_sec = 4.0`, `yawn_max_duration_sec = 7.0`, frame-mapped from FPS | Genuine yawns are multi-second motor/airway/thermoregulatory events, not sub-second mouth openings | Gallup/Eldakar yawning literature; human yawns typically ~4–7 s, mean ~6 s |
| `MAR_THRESH` alone defines yawn | both runtime scripts | `0.6` | MAR amplitude alone cannot separate yawn from speech, laugh, cough, jaw stretch | `mar_open_threshold_subject_specific` + `yawn_duration_sec` + morphology rules | Duration + morphology are required for biological plausibility | Yawn physiology literature on prolonged inspiratory gape and release |
| `HEAD_PITCH_THRESH` + `HEAD_NOD_FRAMES` | `advanced_drowsiness_detection.py` | `20°`, `30 frames` | Static angle mislabels dashboard glances and camera pose as nodding | `pitch_drop_amplitude_deg`, `pitch_velocity_deg_per_sec`, `pitch_recovery_sec` | Sleep-onset nodding is dynamic postural failure, better captured by velocity/recovery than angle alone | Sleep-onset motor-control literature; reduced neck postural tone / nodding phenomena |
| `BLINK_FREQ_THRESH` | `advanced_drowsiness_detection.py` | `10 blinks / 10 s` | Unsupported population threshold; blink rate is highly context-dependent | `blink_rate_change_from_baseline`, `blink_duration_change`, `PERCLOS_supporting` | Blink metrics should be relative to subject baseline and used as supporting evidence only | Fatigue literature commonly favors PERCLOS over raw blink count alone |
| `GAZE_MOVE_THRESH` / `GAZE_FIXATION_TIME` | `advanced_drowsiness_detection.py` | `1.5 px`, `6 s` | Resolution-dependent and not clinically validated | `gaze_dispersion_norm`, `off-road gaze metrics`, `contextual attention model` | Pixel thresholds do not transfer across face size/camera setups | Human factors literature, not direct sleep physiology |
| `CONFIDENCE_THRESH` | `advanced_drowsiness_detection.py` | `0.75` | Model confidence is not clinical probability | `calibrated_closed_eye_probability` + validated fusion rule | Confidence should be calibrated on clinically labeled validation data | Model-calibration best practice; not a physiologic citation |

---

## Recommended Baseline Logic Redesign

### A. Ground-truth layer
1. Collect **KSS** at fixed intervals during recording.
2. Time-align each KSS score to frame windows.
3. Train on **windowed labels**, not whole-video labels.

### B. Ocular lapse layer
1. Estimate actual camera FPS.
2. Maintain subject-specific open-eye baseline EAR.
3. Detect sustained closure when EAR remains below subject threshold for **>500 ms**.
4. Track **PERCLOS** over rolling windows as a stronger fatigue feature than raw blink count.

### C. Yawn layer
1. Use subject-relative MAR opening threshold.
2. Segment mouth-opening episodes.
3. Classify as probable yawn only when duration falls within **4–7 s** and morphology matches slow opening + release.

### D. Head-drop layer
1. Use calibrated baseline pitch.
2. Compute pitch derivative in deg/s.
3. Detect **rapid downward drop + brief rebound**, not static down-looking posture.
4. Fuse with ocular evidence before escalating to “drowsy.”

### E. Decision layer
- Replace current OR-stacked rules with a **tiered evidence model**:
  - **Warning**: isolated yawn, isolated blink burst, isolated prolonged closure
  - **High risk**: sustained ocular lapse + head-drop event, or repeated ocular lapses + elevated PERCLOS
  - **Drowsy state**: multi-signal agreement over time and/or KSS-aligned model output

---

## Practical Audit Conclusions
- The current system is best described as a **prototype vigilance heuristic engine**, not a clinically grounded fatigue detector.
- The most serious flaw is **ground truth**: if labels are not KSS-aligned, later threshold tuning cannot fix the biological validity problem.
- Of the user-specified corrections, the strongest immediate upgrades are:
  1. **Replace binary whole-video labels with KSS-aligned window labels**.
  2. **Normalize eye-closure logic to 500 ms using actual FPS**.
  3. **Raise yawn duration to a 4–7 s physiologic window**.
  4. **Reframe head nodding as pitch-velocity/postural-tone loss**, not just pitch angle.

## Key Citations
1. **Kaida K, et al.** Validation of the Karolinska Sleepiness Scale against performance and EEG variables. *Clinical Neurophysiology*. 2006. PubMed-indexed.
2. **Dinges DF / Basner M PVT literature.** Standard PVT lapse definition uses response time **>500 ms** as a vigilance lapse.
3. **Gallup AC, Eldakar OT** and related yawning physiology literature. Human yawns are typically **4–7 seconds**, with average duration around **6 seconds**.
4. Sleep-onset motor-control literature supports monitoring **head-drop dynamics / reduced postural neck tone** rather than static pitch alone; full REM atonia should not be claimed from webcam pitch data.
