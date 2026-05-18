# DA1 Research Roadmap: Vision-Based Driver Drowsiness Monitoring

## 1. Purpose

This roadmap replaces the older AI-generated thesis roadmap for the current codebase. It is written for a HUST **Đồ Án 1 / pre-graduation thesis** goal.

Recommended project framing:

> A vision-based driver drowsiness monitoring prototype that extracts temporal ocular, yawning, gaze, and head-pose features, then evaluates whether those features classify KSS-aligned sleepiness levels.

This is stronger than simply saying “the system detects fatigue.” Drowsiness is a hidden physiological state, so a research-quality project must connect visual evidence to a defensible ground truth such as KSS, PVT, or another sleepiness/performance proxy.

## 2. Current Project Status

The codebase already has a useful engineering foundation:

- Modular runtime entrypoint: `advanced_drowsiness_detection.py`
- Runtime orchestration: `runtime/app.py`
- Perception extraction: `runtime/perception.py`
- Temporal feature pipeline: `runtime/features.py`
- PERCLOS utility: `perclos.py`
- FSM decision logic: `fsm.py` and `runtime/engines/fsm_engine.py`
- Legacy eye-model runtime: `drowsiness_detection_with_model.py`
- Training scripts: `train_eye_model_torch.py`, `train_advanced_model.py`, `train_model.py`
- Lightweight tests for config and engine contracts in `tests/`

The main weakness is not lack of code. The main weakness is lack of a research-grade data and evaluation layer.

Current limitations:

- Some old data collection logic still supports one binary label for a whole video or extraction run.
- Current evaluation is stronger for yawn/MAR than for full drowsiness state.
- Current model training does not yet prove subject-disjoint sleepiness classification.
- `advanced_drowsiness_model.pth` is a training artifact, while the main modular runtime is mostly feature/FSM based.
- FSM thresholds and feature weights are plausible, but not yet tuned from KSS-aligned validation data.

## 3. Main Research Question

Recommended main question:

> Can temporal visual features such as PERCLOS, eye-closure duration, yawn episodes, blink behavior, gaze stability, and head-pose dynamics classify driver sleepiness levels when evaluated against KSS-aligned time-window labels?

Recommended sub-questions:

1. Does PERCLOS alone provide a useful baseline for drowsiness classification?
2. Does a multi-signal FSM improve over single-signal baselines?
3. Does Random Forest fusion on window-level features improve over rule-based baselines?
4. Which feature groups contribute most: ocular, yawn, head pose, or gaze?
5. How many false alerts per minute does the system produce under alert conditions?

## 4. Key Concept

Separate the project into three layers:

| Layer | Meaning | Examples |
|---|---|---|
| Camera events | Direct visual observations | eye closed, yawn-like mouth opening, head drop |
| Sleepiness state | Ground truth or proxy state | KSS alert/caution/sleepy |
| System decision | Runtime warning output | ALERT, CAUTION, DROWSY, CRITICAL |

The research goal is not only to detect camera events. The research goal is to evaluate whether camera events and temporal features match sleepiness state.

Recommended KSS mapping:

| KSS Score | Band | Meaning |
|---|---|---|
| 1-4 | `alert` | Mostly awake / alert |
| 5-6 | `caution` | Transitional sleepiness |
| 7-9 | `sleepy` | High sleepiness risk |

For smaller binary experiments:

```text
KSS 1-6 -> non_sleepy
KSS 7-9 -> sleepy
```

## 5. Roadmap Summary

| Phase | Name | Main Output |
|---|---|---|
| 0 | Scope reset | Honest project claims and DA1 framing |
| 1 | Research protocol | Label schema and evaluation rules |
| 2 | Offline feature exporter | Per-frame and per-window feature CSVs |
| 3 | Dataset strategy | KSS/PVT-aware dataset plan |
| 4 | Baseline experiments | PERCLOS/FSM/single-signal metrics |
| 5 | Random Forest fusion | Interpretable ML baseline and feature importance |
| 6 | Runtime improvement | Evidence-tuned FSM and config |
| 7 | Thesis packaging | Figures, tables, report, and demo |

## 6. Phase 0: Scope Reset

### Goal

Make the project claims honest and defensible.

### Use This Claim

> This project is a vision-based driver drowsiness monitoring prototype with temporal feature extraction and research evaluation against KSS-aligned sleepiness labels.

### Avoid This Claim For Now

> This project is a clinically validated fatigue detector.

### Tasks

- Treat old reports in `docs/old/` as historical audit material, not the active roadmap.
- Keep the current modular runtime as the engineering base.
- Stop prioritizing bigger model training until the data/evaluation layer is improved.
- Separate engineering demo claims from research validation claims.

### Deliverables

- `docs/DA1_RESEARCH_ROADMAP.md`
- Optional later: `docs/research_protocol.md`
- Optional later: `docs/label_schema.md`

## 7. Phase 1: Research Protocol and Label Schema

### Goal

Define what is being measured, how labels are created, and how evaluation will be done.

### Recommended Label Unit

Use time windows, not random frames.

Recommended default:

- Window length: 60 seconds
- Window stride: 10 seconds
- Label source: nearest KSS score, or KSS score collected inside that interval

Alternative for small experiments:

- Window length: 30 seconds
- Window stride: 5 seconds

### Recommended Label Schema

```csv
subject_id,session_id,video_id,start_time_sec,end_time_sec,kss_score,kss_band,notes
S01,session_01,video_01,0,60,3,alert,baseline awake
S01,session_01,video_01,60,120,5,caution,early signs
S01,session_01,video_01,120,180,8,sleepy,clear drowsiness
```

### Subject Split Rule

Use subject-disjoint evaluation.

Bad split:

```text
Randomly split frames from all subjects into train/test.
```

Good split:

```text
Train subjects: S01-S07
Validation subjects: S08
Test subjects: S09-S10
```

This prevents the model from learning subject identity, camera angle, lighting, or background instead of drowsiness.

### Deliverables

- `docs/research_protocol.md`
- `docs/label_schema.md`
- `metadata/kss_annotations.csv`
- `metadata/subject_split.csv`

## 8. Phase 2: Offline Feature Exporter

### Goal

Turn videos into analyzable research data.

The live OpenCV UI is useful for a demo, but research needs exported features and predictions.

### New Script

Recommended script:

```text
tools/research/export_video_features.py
```

### Suggested Input Arguments

```text
--video-path <path>
--subject-id <id>
--session-id <id>
--output-csv <path>
--window-seconds 60
--stride-seconds 10
```

### Per-Frame Output Columns

```text
subject_id
session_id
video_id
frame_index
timestamp_sec
ear
mar
pitch
yaw
roll
eye_closed
mouth_open
head_nod_candidate
gaze_x
gaze_y
fsm_state
fsm_evidence
```

### Per-Window Output Columns

```text
subject_id
session_id
video_id
window_start_sec
window_end_sec
mean_ear
min_ear
mean_mar
max_mar
perclos_30s
perclos_60s
max_eye_closed_duration_sec
ocular_lapse_count
blink_rate_per_min
yawn_count
mean_yawn_duration_sec
head_drop_count
max_pitch_velocity_deg_s
gaze_dispersion
fsm_state_mode
fsm_max_evidence
```

### Implementation Notes

- Reuse `runtime/perception.py` instead of duplicating EAR/MAR/head-pose logic.
- Reuse or adapt `runtime/features.py` for temporal features.
- Export raw signals and derived features separately when possible.
- Store frame timestamps from video FPS or actual capture timing.

### Deliverables

- `tools/research/export_video_features.py`
- `reports/features/frame_features.csv`
- `reports/features/window_features.csv`

## 9. Phase 3: Dataset Strategy

### Goal

Use labels that support research-quality claims.

### Recommended Strategy

Use two dataset tracks:

1. **Public sleepiness dataset** for scientific validation.
2. **Small custom webcam dataset** for local demo and domain adaptation.

### Track A: Public Dataset

DROZY is useful because it includes KSS, PVT reaction times, PSG signals, and video-related data. It is more research-grounded than manually labeling a whole video as “drowsy.”

Use it to answer:

```text
Can the feature pipeline classify KSS-aligned sleepiness windows on a recognized dataset?
```

### Track B: Custom Pilot Dataset

For a HUST DA1 demo, collect a small controlled dataset.

Suggested protocol:

```text
Subjects: 5-10
Sessions per subject: 2-3
Session duration: 10-20 minutes
KSS prompt interval: every 1-2 minutes
Camera: normal webcam
Conditions: normal light, low light, glasses/no glasses if possible
```

Safety rule:

Do not require unsafe sleep deprivation. A pilot dataset can include naturally tired sessions, simulated yawns for event testing, and normal alert sessions for false-alert measurement.

### Dataset Manifest

```csv
video_id,subject_id,session_id,path,fps,duration_sec,condition,has_kss,has_pvt,notes
video_01,S01,session_01,data/videos/S01_session_01.mp4,30,900,normal_light,true,false,custom pilot
```

### Deliverables

- `metadata/data_manifest.csv`
- `metadata/kss_annotations.csv`
- `metadata/subject_split.csv`
- Optional: `metadata/pvt_results.csv`

## 10. Phase 4: Baseline Experiments

### Goal

Establish simple, defensible baselines before using ML fusion.

### Required Baselines

| Baseline | Description |
|---|---|
| Eye-closure rule | Uses long eye closure only |
| PERCLOS-only | Uses PERCLOS threshold only |
| Yawn-only | Uses yawn episode count/duration only |
| Head-pose-only | Uses head-drop or pitch dynamics only |
| Current FSM | Uses the current multi-signal decision engine |

### Metrics

Report:

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
False alerts per minute
Detection latency
```

For thesis quality, prioritize:

- Recall for sleepy windows
- False alerts per minute during alert windows
- F1-score for balanced comparison
- Confusion matrix for interpretability

### Deliverables

- `tools/research/evaluate_baselines.py`
- `reports/baseline_results.csv`
- `reports/confusion_matrices/`
- `reports/figures/perclos_vs_kss.png`
- `reports/figures/fsm_state_timeline.png`

## 11. Phase 5: Random Forest Fusion and Feature Importance

### Goal

Use Random Forest as an interpretable window-level fusion baseline.

Random Forest should not be treated as a magic detector. It should test whether learned feature fusion improves over simple rules.

### Correct Training Setup

Use this:

```text
window-level features -> KSS band -> subject-disjoint train/test split
```

Do not use this:

```text
random frames -> whole-video binary labels -> random train/test split
```

### Recommended Features

```text
perclos_60s
perclos_30s
max_eye_closed_duration_sec
ocular_lapse_count
blink_rate_per_min
mean_ear
min_ear
yawn_count
mean_yawn_duration_sec
max_mar
head_drop_count
max_pitch_velocity_deg_s
gaze_dispersion
```

### Models To Compare

Minimum:

- PERCLOS-only rule
- Current FSM
- Random Forest

Optional:

- Logistic Regression
- Gradient Boosting
- XGBoost or LightGBM if allowed by project constraints

### Feature Importance

Use three methods:

1. Built-in Random Forest importance as a quick overview.
2. Permutation importance as a more reliable explanation.
3. Ablation study as the strongest thesis evidence.

Example ablation plan:

| Variant | Purpose |
|---|---|
| Full features | Best complete model |
| Eye-only | Ocular contribution |
| PERCLOS-only | Strong simple baseline |
| No PERCLOS | Importance of closure-time accumulation |
| No yawn features | Importance of mouth/yawn cues |
| No head-pose features | Importance of head dynamics |
| No gaze features | Importance of gaze stability |

### Deliverables

- `tools/research/train_window_fusion_model.py`
- `reports/random_forest_results.csv`
- `reports/feature_importance.csv`
- `reports/permutation_importance.csv`
- `reports/ablation_results.csv`

## 12. Phase 6: Runtime Improvement From Evidence

### Goal

Update the real-time demo after evaluation, not before.

### Improvements To Prioritize

1. Make PERCLOS timestamp-based instead of only config-FPS based.
2. Convert yawn detection from frame-count threshold to seconds-based episode detection.
3. Convert head nod detection from mostly static pitch threshold to pitch drop plus rebound dynamics.
4. Tune FSM thresholds using validation results.
5. Add a research/evaluation config separate from the default demo config.

### Suggested Runtime Config Split

```text
config.json              # default demo config
config_research.json     # thresholds chosen from validation experiments
```

### Deliverables

- Updated `perclos.py`
- Updated `runtime/features.py`
- Updated `fsm.py` or a new research engine
- `config_research.json`
- `reports/runtime_comparison.csv`

## 13. Phase 7: DA1 Thesis Structure

### Chapter 1: Introduction

- Driver drowsiness problem
- Why camera-based monitoring is useful
- Project motivation
- Scope and limitations
- Research question

### Chapter 2: Background

- Drowsiness as a hidden state
- KSS and sleepiness labeling
- PERCLOS
- EAR and MAR
- Head pose and gaze features
- FSM and temporal decision logic
- Random Forest and interpretable feature fusion

### Chapter 3: System Design

- Runtime architecture
- Perception pipeline
- Temporal feature pipeline
- FSM decision engine
- Offline evaluation pipeline
- Data flow diagram

### Chapter 4: Dataset and Labeling

- Dataset source
- KSS label mapping
- Window construction
- Subject-disjoint split
- Data quality notes

### Chapter 5: Experiments and Results

- Baseline methods
- Random Forest fusion
- Feature importance
- Ablation study
- Error analysis
- Runtime demo results

### Chapter 6: Conclusion

- What the system can do
- What remains limited
- Future work for graduation thesis / Đồ Án tốt nghiệp

## 14. Recommended Timeline

For an 8-10 week DA1 plan:

| Week | Focus |
|---|---|
| 1 | Scope reset, research protocol, label schema |
| 2 | Offline feature exporter |
| 3 | Dataset adapter and KSS annotation format |
| 4 | Window feature generation and KSS alignment |
| 5 | Baseline evaluation |
| 6 | Random Forest fusion and feature importance |
| 7 | Ablation study and error analysis |
| 8 | Runtime FSM improvements |
| 9 | Thesis writing, figures, and tables |
| 10 | Final demo, cleanup, and presentation slides |

## 15. What To Do First

The next practical steps are:

1. Create `docs/research_protocol.md`.
2. Create `docs/label_schema.md`.
3. Create `tools/research/export_video_features.py`.
4. Create a small example `metadata/kss_annotations.csv`.
5. Run one video through the exporter and inspect the generated features.

Do not start by training a larger model. Start by making the project measurable.

## 16. What To Avoid

Avoid these mistakes:

- Do not use random frame-level train/test split for drowsiness-state claims.
- Do not use whole-video binary labels as research ground truth.
- Do not claim clinical fatigue detection.
- Do not rely only on accuracy.
- Do not use Gini feature importance alone as proof.
- Do not tune FSM thresholds only by visual feeling.
- Do not let the old AI-generated thesis define the current project plan.

## 17. Recommended Final DA1 Contribution

The strongest DA1 contribution is:

> A measurable, interpretable, vision-based drowsiness monitoring pipeline that compares rule-based temporal features and Random Forest feature fusion against KSS-aligned sleepiness labels using subject-disjoint evaluation.

This is defensible because it combines:

- Computer vision
- Temporal signal processing
- Ground-truth labeling
- Interpretable machine learning
- Proper evaluation metrics
- Real-time demo capability

## 18. References For Research Direction

- Kaida et al., “Validation of the Karolinska Sleepiness Scale against performance and EEG variables,” Clinical Neurophysiology, 2006: https://pubmed.ncbi.nlm.nih.gov/16679057/
- Abe, “PERCLOS-based technologies for detecting drowsiness: current evidence and future directions,” Sleep Advances, 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10108649/
- DROZY: The ULg Multimodality Drowsiness Database: https://www.drozy.ulg.ac.be/
- Euro NCAP Safe Driving Driver Engagement Protocol, Version 1.1, October 2025: https://cdn.euroncap.com/cars/assets/euro_ncap_protocol_safe_driving_driver_engagement_v11_a30e874152.pdf
