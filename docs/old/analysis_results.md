# Driver Fatigue Decision-Logic Audit

## Scope and Evidence Base
Audited runtime and training logic in:
- [advanced_drowsiness_detection.py](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py)
- [drowsiness_detection_with_model.py](file:///d:/drowsiness_detection-main/drowsiness_detection_with_model.py)
- [v10.py](file:///d:/drowsiness_detection-main/v10.py)
- [train_model.py](file:///d:/drowsiness_detection-main/train_model.py)
- [train_data.py](file:///d:/drowsiness_detection-main/train_data.py)
- [train_advanced_model.py](file:///d:/drowsiness_detection-main/train_advanced_model.py)

---

## ALGORITHMIC FALLACIES

1. **Threshold absolutism disguised as physiology**
   - Runtime decisions are dominated by fixed constants (`EAR_THRESH`, `MAR_THRESH`, `HEAD_PITCH_THRESH`, counters/windows), e.g. [advanced_drowsiness_detection.py:L382-L397](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L382-L397), [drowsiness_detection_with_model.py:L127-L132](file:///d:/drowsiness_detection-main/drowsiness_detection_with_model.py#L127-L132), [v10.py:L10-L13](file:///d:/drowsiness_detection-main/v10.py#L10-L13).
   - This is a **hard-coded prior**, not an inferred posterior from data.

2. **No explicit linear weights found, but implicit hard weights are everywhere**
   - You do not currently use formulas like `0.6*EAR + 0.4*MAR`.
   - However, boolean fusion (`if A or B`, counter increments/decrements) creates **implicit infinite/zero weighting**, where one trigger can dominate all others (e.g. [advanced_drowsiness_detection.py:L690-L701](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L690-L701)).

3. **Manually imposed task weights in training objective**
   - Advanced model uses `loss = loss_eye + loss_yawn` with fixed 1:1 weighting: [train_advanced_model.py:L270-L273](file:///d:/drowsiness_detection-main/train_advanced_model.py#L270-L273).
   - This is a heuristic assumption that the two tasks contribute equally to safety risk.

4. **Model-training/runtime mismatch**
   - A Random Forest is trained and serialized (`drowsiness_model.pkl`, `scaler.pkl`), but runtime inference does not load or use it:
     - Save: [train_model.py:L54-L55](file:///d:/drowsiness_detection-main/train_model.py#L54-L55)
     - Runtime loads only Torch eye models: [drowsiness_detection_with_model.py:L38](file:///d:/drowsiness_detection-main/drowsiness_detection_with_model.py#L38), [advanced_drowsiness_detection.py:L114-L145](file:///d:/drowsiness_detection-main/advanced_drowsiness_detection.py#L114-L145)

5. **Validation incompleteness and missing system-proof**
   - `train_model.py` prints a classification report (hence precision/recall exist only there), but there is no **system-level** confusion matrix/F1 protocol for the online detector, and no ablation framework.
   - `train_advanced_model.py` reports mostly loss/accuracy, not precision/recall/F1/confusion matrix per module.
   - No structural ablation study appears in codebase search.

6. **Labeling and split methodology risk**
   - Frame labels are assigned per-video constant in collection (`LABEL_TO_WRITE`): [train_data.py:L92-L99](file:///d:/drowsiness_detection-main/train_data.py#L92-L99), [train_data.py:L143](file:///d:/drowsiness_detection-main/train_data.py#L143).
   - Random frame-level split in `train_model.py` can leak near-duplicate temporal neighbors across train/test, inflating apparent generalization.

---

## Evidence-Based Corrections (ML + Mathematics)

### 1) Empirical Weight Determination via Ensemble Attribution

**Mandate:** remove manual threshold fusion as decision core; learn contribution from data.

Use a tabular temporal feature matrix \(X\) with windows over EAR, MAR, Pitch (plus optional Yaw/Roll, blink rate, yawn frequency, confidence stats).

Train:
- `RandomForestClassifier(class_weight='balanced', n_estimators=..., max_depth=...)`
- Evaluate with subject-disjoint folds.

For feature \(j\), estimate importance via mean decrease impurity (MDI):

\[
I_j = \frac{1}{T}\sum_{t=1}^{T}\sum_{n \in \mathcal{N}_{j,t}} \frac{N_n}{N}\,\Delta i_n,
\quad
\Delta i_n = i(n) - p_L i(n_L) - p_R i(n_R)
\]

where \(i(\cdot)\) is Gini impurity and \(N_n/N\) weights split impact by sample mass.

Then normalize \(\sum_j I_j = 1\). This replaces guessed importance with empirically induced variance explanation.

> [!IMPORTANT]
> Also compute **permutation importance** to cross-check MDI bias toward high-cardinality/continuous features.

### 2) Why Temporal Multimodal Fusion (EAR+MAR+Pitch vectors) Is Superior

Single-threshold detectors fail when one sensor is noisy. If each modality misses a drowsy event with probability \(p_i\), then under weak dependence:

\[
P(\text{joint miss}) \approx \prod_{i=1}^{M} p_i
\]

This multiplicative decay beats any single-channel miss rate \(p_k\). Add temporal aggregation over \(T\) frames (e.g., majority/sequence statistics), and error concentration decreases approximately exponentially with \(T\) (Hoeffding/Chernoff-style behavior for bounded independent or weakly dependent votes).

Thus, combining **modal diversity + temporal redundancy** materially increases robustness versus static univariate thresholds.

### 3) Mandatory Ablation Protocol (Only Acceptable Irreducibility Proof)

Define a full model \(\mathcal{M}_{full}\) using temporal EAR/MAR/Pitch fusion. Then run controlled removals:

- \(\mathcal{M}_{-Yawn}\): remove yawn-related features
- \(\mathcal{M}_{-HeadPose}\): remove pitch/yaw/roll block
- \(\mathcal{M}_{-EyeTemporal}\): remove temporal eye dynamics
- \(\mathcal{M}_{-Fusion}\): single-modality baseline

For each model, report on a subject-disjoint test set:
- Precision, Recall, F1
- Confusion matrix
- PR-AUC (recommended for imbalance)
- 95% bootstrap CI for F1
- McNemar test vs \(\mathcal{M}_{full}\) on paired errors

Acceptance rule:
- A module is **irreducible** only if removing it causes statistically significant degradation in F1 and confusion-matrix error structure.

---

## Contrast: Current vs Proposed

| Dimension | Current codebase logic | Proposed ML-driven validated logic | Mathematical/statistical justification |
|---|---|---|---|
| Decision core | Fixed thresholds + counters + boolean rules | Learned classifier on temporal multimodal features | Data-estimated decision boundary minimizes empirical risk instead of hand-picked constants [1][2] |
| Feature contribution | Implicit hard weights via rule precedence; no measured contribution | `feature_importances_` (MDI) + permutation importance | Split-wise impurity decrease quantifies feature contribution [1][2] |
| Fusion strategy | Rule-based OR/AND on instantaneous triggers | Temporal vector fusion of EAR, MAR, Pitch (+ context features) | Joint error drops multiplicatively across modalities and temporally concentrates downward [3] |
| Validation | Partial (`classification_report` in one training script), no system ablation | Subject-disjoint evaluation + confusion matrix + F1 + ablation + significance tests | Reliable comparison requires confusion-matrix-aware metrics and paired tests [4] |
| Scientific claim strength | Heuristic plausibility | Statistical falsifiability | Ablation + CIs + hypothesis tests convert claims into evidence |
| Deployment consistency | RF trained but unused online | Single auditable inference stack (train = deploy parity) | Prevents objective-function/runtime mismatch and undocumented drift |

---

## Citations

1. Scikit-learn documentation, **Feature importances with a forest of trees** (MDI and caveats). [scikit-learn.org](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
2. Breiman, L. (2001). **Random Forests**. *Machine Learning*, 45(1), 5–32. DOI: 10.1023/A:1010933404324.
3. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). **Multimodal Machine Learning: A Survey and Taxonomy**. *IEEE TPAMI*, 41(2), 423–443. DOI: 10.1109/TPAMI.2018.2798607.
4. Sokolova, M., & Lapalme, G. (2009). **A Systematic Analysis of Performance Measures for Classification Tasks**. *Information Processing & Management*, 45(4), 427–437.
