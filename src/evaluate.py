import argparse
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config
from .utils import (
    ensure_dir,
    file_size_kb,
    import_tensorflow,
    load_json,
    load_processed_arrays,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained UAV state classifiers.")
    parser.add_argument("--model", choices=["tinytcn", "dscnn"], default=None)
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUTS_DIR)
    parser.add_argument("--compare", action="store_true", help="Write outputs/comparison.csv")
    return parser.parse_args()


def evaluate_model(
    model_name: str,
    processed_dir: Path = config.PROCESSED_DIR,
    output_dir: Path = config.OUTPUTS_DIR,
    model_path: Optional[Path] = None,
) -> Dict[str, object]:
    tf = import_tensorflow()
    processed_dir = Path(processed_dir)
    model_dir = ensure_dir(Path(output_dir) / model_name)
    model_path = Path(model_path) if model_path else model_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    arrays = load_processed_arrays(processed_dir)
    X_test = arrays["X_test"]
    y_test = arrays["y_test"]
    label_encoder = joblib.load(processed_dir / "label_encoder.pkl")
    class_labels = label_encoder.classes_.tolist()
    target_names = [config.CLASS_DISPLAY_NAMES.get(label, label) for label in class_labels]

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_labels)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_labels)),
        target_names=target_names,
        zero_division=0,
    )
    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(report_text)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_labels)))
    fig, ax = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title(f"{model_name} confusion matrix")
    fig.tight_layout()
    fig.savefig(model_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    metrics = {
        "model_name": model_name,
        "num_parameters": int(model.count_params()),
        "keras_model_size_kb": file_size_kb(model_path),
        "tflite_float32_size_kb": file_size_kb(model_dir / "model_float32.tflite"),
        "tflite_int8_size_kb": file_size_kb(model_dir / "model_int8.tflite"),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "class_labels": class_labels,
        "target_names": target_names,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }
    save_json(metrics, model_dir / "metrics.json")
    return metrics


def write_comparison(output_dir: Path = config.OUTPUTS_DIR) -> Path:
    output_dir = Path(output_dir)
    rows: List[Dict[str, object]] = []
    for model_name in ["tinytcn", "dscnn"]:
        model_dir = output_dir / model_name
        metrics_path = model_dir / "metrics.json"
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        benchmark = {}
        for name in ["benchmark_int8.json", "benchmark_float32.json", "benchmark.json"]:
            path = model_dir / name
            if path.exists():
                benchmark = load_json(path)
                break
        rows.append(
            {
                "model_name": model_name,
                "num_parameters": metrics.get("num_parameters", 0),
                "keras_model_size_kb": metrics.get(
                    "keras_model_size_kb", file_size_kb(model_dir / "model.keras")
                ),
                "tflite_float32_size_kb": file_size_kb(model_dir / "model_float32.tflite"),
                "tflite_int8_size_kb": file_size_kb(model_dir / "model_int8.tflite"),
                "test_accuracy": metrics.get("test_accuracy", None),
                "macro_f1": metrics.get("macro_f1", None),
                "avg_inference_time_ms": benchmark.get("avg_inference_time_ms", None),
            }
        )
    ensure_dir(output_dir)
    comparison_path = output_dir / "comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    return comparison_path


def main() -> None:
    args = parse_args()
    if args.compare:
        path = write_comparison(args.output_dir)
        print(f"Saved comparison table to {path}")
        return
    if not args.model:
        raise SystemExit("--model is required unless --compare is used")
    metrics = evaluate_model(args.model, args.processed_dir, args.output_dir)
    print(
        f"{args.model}: accuracy={metrics['test_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
