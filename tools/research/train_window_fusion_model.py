from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from tools.research.common import (
        EYE_FEATURES,
        FSM_FEATURES,
        FULL_FEATURES,
        HEAD_POSE_FEATURES,
        PERCLOS_FEATURES,
        YAWN_FEATURES,
        available_features,
        binary_metrics,
        kss_is_sleepy,
        numeric_feature_matrix,
        read_csv_rows,
        write_csv_rows,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.research.common import (
        EYE_FEATURES,
        FSM_FEATURES,
        FULL_FEATURES,
        HEAD_POSE_FEATURES,
        PERCLOS_FEATURES,
        YAWN_FEATURES,
        available_features,
        binary_metrics,
        kss_is_sleepy,
        numeric_feature_matrix,
        read_csv_rows,
        write_csv_rows,
    )


RESULT_COLUMNS = ["variant", "accuracy", "precision", "recall", "f1", "support", "tp", "tn", "fp", "fn"]
IMPORTANCE_COLUMNS = ["feature", "importance"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a window-level Random Forest fusion baseline.")
    parser.add_argument("--input-csv", required=True, help="Input labeled_windows.csv path.")
    parser.add_argument("--output-dir", required=True, help="Directory for RF reports.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=100, help="RandomForest tree count.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="RandomForest parallel jobs.")
    parser.add_argument("--test-subject-ratio", type=float, default=0.3, help="Subject-disjoint test ratio.")
    parser.add_argument("--permutation-repeats", type=int, default=5, help="Permutation importance repeats.")
    parser.add_argument(
        "--allow-random-split",
        action="store_true",
        help="Fallback to stratified random split when subject-disjoint split is impossible.",
    )
    return parser


def train_window_fusion_model(
    input_csv: Path,
    output_dir: Path,
    random_state: int = 42,
    n_estimators: int = 100,
    n_jobs: int = -1,
    allow_random_split: bool = False,
    test_subject_ratio: float = 0.3,
    permutation_repeats: int = 5,
) -> dict[str, Path]:
    rows = read_csv_rows(input_csv)
    if not rows:
        raise ValueError("No labeled window rows found")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be > 0")
    if not 0.0 < test_subject_ratio < 1.0:
        raise ValueError("test_subject_ratio must be between 0 and 1")

    train_rows, test_rows = split_rows(rows, test_subject_ratio, random_state, allow_random_split)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_features = available_features(rows, FULL_FEATURES)
    if not full_features:
        raise ValueError("No usable Random Forest feature columns found")

    metrics, feature_importance, permutation_importance = _train_and_score(
        train_rows,
        test_rows,
        full_features,
        random_state,
        n_estimators,
        n_jobs,
        permutation_repeats,
    )

    ablation_rows = []
    for variant, features in ablation_feature_sets(rows).items():
        if not features:
            continue
        variant_metrics, _feature_importance, _permutation = _train_and_score(
            train_rows,
            test_rows,
            features,
            random_state,
            n_estimators,
            n_jobs,
            0,
        )
        ablation_rows.append({"variant": variant, **variant_metrics})

    result_path = output_dir / "random_forest_results.csv"
    feature_path = output_dir / "feature_importance.csv"
    permutation_path = output_dir / "permutation_importance.csv"
    ablation_path = output_dir / "ablation_results.csv"

    write_csv_rows(result_path, RESULT_COLUMNS, [{"variant": "full", **metrics}])
    write_csv_rows(feature_path, IMPORTANCE_COLUMNS, feature_importance)
    write_csv_rows(permutation_path, IMPORTANCE_COLUMNS, permutation_importance)
    write_csv_rows(ablation_path, RESULT_COLUMNS, ablation_rows)

    return {
        "results": result_path,
        "feature_importance": feature_path,
        "permutation_importance": permutation_path,
        "ablation_results": ablation_path,
    }


def split_rows(
    rows: list[dict[str, Any]],
    test_subject_ratio: float,
    random_state: int,
    allow_random_split: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subjects = sorted({row.get("subject_id", "") for row in rows if row.get("subject_id", "")})
    if len(subjects) >= 2:
        import random

        rng = random.Random(random_state)
        shuffled = subjects[:]
        rng.shuffle(shuffled)
        test_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * test_subject_ratio)))
        test_subjects = set(shuffled[:test_count])
        train_rows = [row for row in rows if row.get("subject_id") not in test_subjects]
        test_rows = [row for row in rows if row.get("subject_id") in test_subjects]
        if _has_two_classes(train_rows) and _has_two_classes(test_rows):
            return train_rows, test_rows

    if not allow_random_split:
        raise ValueError("Cannot create a valid subject-disjoint split; pass --allow-random-split for fixture-only fallback.")

    from sklearn.model_selection import train_test_split

    labels = [1 if kss_is_sleepy(row) else 0 for row in rows]
    train_rows, test_rows = train_test_split(rows, test_size=test_subject_ratio, random_state=random_state, stratify=labels)
    return list(train_rows), list(test_rows)


def ablation_feature_sets(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "full": available_features(rows, FULL_FEATURES),
        "eye_only": available_features(rows, EYE_FEATURES),
        "perclos_only": available_features(rows, PERCLOS_FEATURES),
        "no_perclos": available_features(rows, [feature for feature in FULL_FEATURES if feature not in PERCLOS_FEATURES]),
        "no_yawn": available_features(rows, [feature for feature in FULL_FEATURES if feature not in YAWN_FEATURES]),
        "no_head_pose": available_features(rows, [feature for feature in FULL_FEATURES if feature not in HEAD_POSE_FEATURES]),
        "no_fsm": available_features(rows, [feature for feature in FULL_FEATURES if feature not in FSM_FEATURES]),
    }


def _train_and_score(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    features: list[str],
    random_state: int,
    n_estimators: int,
    n_jobs: int,
    permutation_repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    x_train = numeric_feature_matrix(train_rows, features)
    y_train = [1 if kss_is_sleepy(row) else 0 for row in train_rows]
    x_test = numeric_feature_matrix(test_rows, features)
    y_test = [1 if kss_is_sleepy(row) else 0 for row in test_rows]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    y_pred = list(model.predict(x_test))
    metrics = binary_metrics(y_test, [int(value) for value in y_pred])

    feature_importance = sorted(
        [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(features, model.feature_importances_)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )

    permutation_rows: list[dict[str, Any]] = []
    if permutation_repeats > 0:
        permutation = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=permutation_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        permutation_rows = sorted(
            [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(features, permutation.importances_mean)
            ],
            key=lambda row: row["importance"],
            reverse=True,
        )
    return metrics, feature_importance, permutation_rows


def _has_two_classes(rows: list[dict[str, Any]]) -> bool:
    labels = {1 if kss_is_sleepy(row) else 0 for row in rows}
    return len(labels) >= 2


def run(args: argparse.Namespace) -> int:
    outputs = train_window_fusion_model(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        allow_random_split=args.allow_random_split,
        test_subject_ratio=args.test_subject_ratio,
        permutation_repeats=args.permutation_repeats,
    )
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
