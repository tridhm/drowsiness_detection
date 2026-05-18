# Model Artifact Provenance (DA1 Merge)

## SHA256 Checksums

- Root model: `advanced_drowsiness_model.pth`
  - `AF7C2EEA1DEFC6E8D4A167559C1EB4C081F461FFCC7C038FC13EEFE709C3B642`
- DA1 advanced model (archived): `D:/drowsiness_detection-import-archive/merge_20260418_015536/DA1/drowsiness_detection-main/advanced_drowsiness_model.pth`
  - `A99ECE3D8143F519B15F5C5FEAC0507A2477D87B757C916F8A29046D47C659DC`
- Root eye model: `eye_model.pth`
  - `BDA9FB3F40779C1CDCE0558E6EA4ED82D94274C4E17CAA46388CDC1EABCFCB3E`
- DA1 eye model (archived): `D:/drowsiness_detection-import-archive/merge_20260418_015536/DA1/drowsiness_detection-main/eye_model.pth`
  - `BDA9FB3F40779C1CDCE0558E6EA4ED82D94274C4E17CAA46388CDC1EABCFCB3E`

## Compatibility Notes

### Advanced model pair (`advanced_drowsiness_model.pth`)
- Files expose identical state-dict key sets (`64` tensors each).
- Tensor values differ materially (`8/64` tensors byte-identical).
- Policy: keep both artifacts distinct and never overwrite silently.

### Eye model pair (`eye_model.pth`)
- DA1 and root checksums are identical after sync.
- Baseline runtime can load `eye_model.pth` without archive dependency.

## Operational Default

- Canonical runtime defaults to modular FSM engine (`--decision-engine fsm`).
- Baseline model runtime (`drowsiness_detection_with_model.py`) uses root `eye_model.pth` unless overridden.
