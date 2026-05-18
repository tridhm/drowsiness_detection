# Codebase Visual Map

This page gives a high-level visual of the current modular repository structure and the main data/model flows.

## 1) System Map (Modular)

```mermaid
%%{init: {'theme':'base','themeVariables': {'fontFamily':'Segoe UI','primaryTextColor':'#0f172a','lineColor':'#475569','tertiaryColor':'#ffffff'}}}%%
flowchart TB
    subgraph Inputs["Inputs"]
        direction TB
        I1["Datasets<br/>CEW / Video Database / others"]
        I2["Evaluation metadata<br/>tools/evaluation/metadata/*"]
        I3["Runtime config<br/>config.json + CLI"]
        I4["Research metadata<br/>metadata/*.sample.csv<br/>KSS annotations"]
    end

    subgraph Build["Build + Train"]
        direction TB
        D1["train_data.py"]
        D2["extract_video_frames.py"]
        T1["train_eye_model_torch.py"]
        T2["train_advanced_model.py"]
        T3["train_model.py"]
    end

    subgraph Artifacts["Generated Artifacts"]
        direction TB
        A1["eye_model.pth"]
        A2["advanced_drowsiness_model.pth"]
        A3["drowsiness_model.pkl + scaler.pkl"]
        A4["drowsiness_data.csv"]
    end

    subgraph Runtime["Runtime (Modular)"]
        direction TB
        R0["advanced_drowsiness_detection.py<br/>thin entry"]
        R1["runtime/app.py<br/>orchestrator"]
        R2["runtime/transports.py<br/>webcam|file"]
        R3["runtime/perception.py"]
        R4["runtime/features.py<br/>EMA + PERCLOS + counters"]
        R5["runtime/engines/registry.py"]
        R6["runtime/engines/fsm_engine.py"]
        R7["runtime/engines/legacy_engine.py"]
        R8["runtime/alerts.py"]
        L1["logs/*.log"]
    end

    subgraph Eval["Offline Evaluation"]
        direction TB
        E1["label_yawn_frames.py"]
        E2["evaluate_mar_videos.py"]
        E3["analyze_mar_thresholds.py"]
        E4["report CSVs"]
    end

    subgraph Research["Research Pipeline MVP"]
        direction TB
        Q1["export_video_features.py"]
        Q2["frame_features.csv<br/>window_features.csv"]
        Q3["align_window_labels.py"]
        Q4["labeled_windows.csv"]
        Q5["evaluate_baselines.py"]
        Q6["train_window_fusion_model.py"]
        Q7["metrics + importance<br/>ablation CSVs"]
    end

    I1 --> D1 --> A4 --> T3 --> A3
    I1 --> D2 --> T1 --> A1
    I1 --> D2 --> T2 --> A2

    I3 --> R0 --> R1 --> R2 --> R3 --> R4 --> R5
    R5 --> R6
    R5 --> R7
    R6 --> R8
    R7 --> R8
    R1 --> L1

    A1 --> R1
    A2 --> R1

    I2 --> E2 --> E4
    E1 --> I2
    E3 --> E4

    I1 --> Q1
    R3 --> Q1
    R4 --> Q1
    Q1 --> Q2
    I4 --> Q3
    Q2 --> Q3 --> Q4
    Q4 --> Q5 --> Q7
    Q4 --> Q6 --> Q7

    classDef input fill:#fef3c7,stroke:#d97706,color:#7c2d12,stroke-width:1.2px;
    classDef build fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e,stroke-width:1.2px;
    classDef artifact fill:#ede9fe,stroke:#6d28d9,color:#4c1d95,stroke-width:1.2px;
    classDef runtime fill:#dcfce7,stroke:#15803d,color:#14532d,stroke-width:1.2px;
    classDef eval fill:#ffe4e6,stroke:#be123c,color:#881337,stroke-width:1.2px;
    classDef research fill:#ccfbf1,stroke:#0f766e,color:#134e4a,stroke-width:1.2px;

    class I1,I2,I3,I4 input;
    class D1,D2,T1,T2,T3 build;
    class A1,A2,A3,A4 artifact;
    class R0,R1,R2,R3,R4,R5,R6,R7,R8,L1 runtime;
    class E1,E2,E3,E4 eval;
    class Q1,Q2,Q3,Q4,Q5,Q6,Q7 research;
```

## 2) Advanced Runtime Flow (`runtime/app.py`)

```mermaid
%%{init: {'theme':'base','themeVariables': {'fontFamily':'Segoe UI','lineColor':'#475569'}}}%%
flowchart TB
    A["Load defaults"] --> B["Load config.json"]
    B --> C["Apply CLI overrides"]
    C --> D["Open source transport"]
    D --> E["Capture frame"]
    E --> F["PerceptionExtractor"]
    F --> G["SignalFeaturePipeline"]
    G --> H{"Warm-up + calibration done?"}
    H -- No --> H0["State = ALERT (calibrating)"]
    H -- Yes --> I["ENGINE_REGISTRY resolve"]
    I --> J{"decision-engine"}
    J --> K1["FSMDecisionEngine.update"]
    J --> K2["LegacyRuleEngine.update"]
    K1 --> L["DecisionResult"]
    K2 --> L
    L --> M["AudioAlertController"]
    L --> N["Overlay + debug"]
    N --> Q{"Quit?"}
    M --> Q
    Q -- No --> E
    Q -- Yes --> X["Cleanup resources"]

    classDef stage fill:#e2e8f0,stroke:#475569,color:#0f172a,stroke-width:1.2px;
    classDef decision fill:#fef9c3,stroke:#ca8a04,color:#713f12,stroke-width:1.2px;
    classDef action fill:#dcfce7,stroke:#15803d,color:#14532d,stroke-width:1.2px;
    class A,B,C,D,E,F,G,I,K1,K2,L stage;
    class H,J,Q decision;
    class H0,M,N,X action;
```

## 3) Training + Evaluation Pipeline

```mermaid
%%{init: {'theme':'base','themeVariables': {'fontFamily':'Segoe UI','lineColor':'#475569'}}}%%
flowchart TB
    R["Raw videos + image datasets"] --> D["Dataset prep<br/>train_data.py / extract_video_frames.py"]
    D --> C["drowsiness_data.csv"]
    C --> RF["train_model.py<br/>RandomForest"]
    RF --> PKL["drowsiness_model.pkl + scaler.pkl"]

    R --> EYE["train_eye_model_torch.py"]
    EYE --> EYEW["eye_model.pth"]

    R --> ADV["train_advanced_model.py"]
    ADV --> ADVW["advanced_drowsiness_model.pth"]

    R --> LBL["label_yawn_frames.py"]
    LBL --> LAB["*_labels.csv"]
    LAB --> MAR["evaluate_mar_videos.py"]
    R --> MAR
    MAR --> REP["evaluation reports (*.csv)"]

    MR["mar_result.csv"] --> ANA["analyze_mar_thresholds.py"] --> MREP["mar_threshold_analysis.csv"]

    classDef source fill:#fef3c7,stroke:#d97706,color:#7c2d12,stroke-width:1.2px;
    classDef process fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e,stroke-width:1.2px;
    classDef output fill:#ede9fe,stroke:#6d28d9,color:#4c1d95,stroke-width:1.2px;
    class R,MR source;
    class D,RF,EYE,ADV,LBL,MAR,ANA process;
    class C,PKL,EYEW,ADVW,LAB,REP,MREP output;
```

## 4) Research Pipeline MVP

```mermaid
%%{init: {'theme':'base','themeVariables': {'fontFamily':'Segoe UI','lineColor':'#475569'}}}%%
flowchart LR
    V["Video file"] --> EX["export_video_features.py"]
    EX --> FR["frame_features.csv<br/>one row per frame"]
    EX --> WR["window_features.csv<br/>one row per time window"]
    ANN["KSS annotation CSV<br/>existing labels"] --> AL["align_window_labels.py"]
    WR --> AL
    AL --> LW["labeled_windows.csv"]
    LW --> BASE["evaluate_baselines.py"]
    LW --> RF["train_window_fusion_model.py"]
    BASE --> BREP["baseline_results.csv<br/>confusion matrix"]
    RF --> RREP["RF metrics<br/>feature/permutation importance<br/>ablation results"]

    classDef input fill:#fef3c7,stroke:#d97706,color:#7c2d12,stroke-width:1.2px;
    classDef process fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e,stroke-width:1.2px;
    classDef output fill:#ede9fe,stroke:#6d28d9,color:#4c1d95,stroke-width:1.2px;
    class V,ANN input;
    class EX,AL,BASE,RF process;
    class FR,WR,LW,BREP,RREP output;
```

KSS labels are not generated by the exporter. They must come from an annotation file or a public/validated dataset before metrics can be reported as real research evidence.

## 5) Quick Orientation

- Real-time modular runtime: start with `advanced_drowsiness_detection.py` and `runtime/app.py`.
- Engine selection: use `--decision-engine fsm|legacy`.
- Config path + overrides: use `--config` plus CLI overrides.
- Baseline model runtime: `drowsiness_detection_with_model.py`.
- Offline evaluation workflows: tools under `tools/evaluation/`.
- Research pipeline workflows: tools under `tools/research/`, sample fixtures under `metadata/`, generated reports under `reports/research/`.
- DA1/bundle provenance: docs under `docs/merge/`.
