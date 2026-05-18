import argparse
from pathlib import Path

from cli_json_config import parse_args_with_json_config

DEFAULT_COLUMNS = ["ear", "mar", "pitch", "yaw", "roll", "label"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train tabular RandomForest drowsiness model from CSV features.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to shared JSON config file (section: train_model).",
    )
    parser.add_argument(
        "--input-csv",
        default="drowsiness_data.csv",
        help="Input CSV path. Expected columns: ear, mar, pitch, yaw, roll, label.",
    )
    parser.add_argument(
        "--has-header",
        dest="has_header",
        action="store_true",
        help="Set this if input CSV already has a header row.",
    )
    parser.add_argument(
        "--no-header",
        dest="has_header",
        action="store_false",
        help="Treat input CSV as headerless.",
    )
    parser.set_defaults(has_header=False)
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (0-1).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split/model.")
    parser.add_argument("--n-estimators", type=int, default=100, help="RandomForest tree count.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="RandomForest parallel jobs.")
    parser.add_argument("--output-model", default="drowsiness_model.pkl", help="Output model path.")
    parser.add_argument("--output-scaler", default="scaler.pkl", help="Output scaler path.")
    return parser


def _load_dataset(input_csv: Path, has_header: bool):
    import pandas as pd

    if has_header:
        data = pd.read_csv(input_csv)
    else:
        data = pd.read_csv(input_csv, header=None, names=DEFAULT_COLUMNS)
    data = data.dropna()
    missing_cols = [col for col in DEFAULT_COLUMNS if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return data


def train(args: argparse.Namespace) -> int:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1 (exclusive).")
    if args.n_estimators <= 0:
        raise ValueError("--n-estimators must be > 0.")

    data = _load_dataset(input_csv, has_header=args.has_header)

    features = ["ear", "mar", "pitch", "yaw", "roll"]
    target = "label"

    x = data[features]
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("Bắt đầu huấn luyện Random Forest...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train_scaled, y_train)
    print("Huấn luyện hoàn tất!")

    y_pred = model.predict(x_test_scaled)
    print("\n--- Kết quả đánh giá trên tập Test ---")
    print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_test, y_pred, target_names=["Tỉnh táo (0)", "Buồn ngủ (1)"]))

    output_model = Path(args.output_model)
    output_scaler = Path(args.output_scaler)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_scaler.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_model)
    joblib.dump(scaler, output_scaler)

    print(f"\nĐã lưu model: {output_model}")
    print(f"Đã lưu scaler: {output_scaler}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_json_config(parser, argv, section="train_model")
    try:
        return train(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
