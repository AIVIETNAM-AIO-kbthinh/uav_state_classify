import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .utils import ensure_dir, file_size_kb


MODEL_NAMES = ["tinytcn", "dscnn"]
MODEL_LABELS = {"tinytcn": "TinyTCN", "dscnn": "DS-CNN"}
COLORS = {"tinytcn": "#2563eb", "dscnn": "#dc2626"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize UAV state classification results with matplotlib.")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def percent(values) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def annotate_bars(ax, bars, fmt="{:.2f}", suffix=""):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{fmt.format(height)}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def load_all_results(output_dir: Path, models: List[str]) -> Dict[str, Dict]:
    results = {}
    for model_name in models:
        model_dir = output_dir / model_name
        results[model_name] = {
            "dir": model_dir,
            "history": load_json(model_dir / "history.json"),
            "metrics": load_json(model_dir / "metrics.json"),
            "benchmark": load_json(model_dir / "benchmark_int8.json")
            if (model_dir / "benchmark_int8.json").exists()
            else {},
            "export": load_json(model_dir / "export_summary.json")
            if (model_dir / "export_summary.json").exists()
            else {},
        }
    return results


def plot_training_history(results: Dict[str, Dict], figures_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    for row, model_name in enumerate(results):
        history = results[model_name]["history"]
        epochs = np.arange(1, len(history["accuracy"]) + 1)

        ax_acc = axes[row, 0]
        ax_loss = axes[row, 1]
        ax_acc.plot(epochs, percent(history["accuracy"]), label="Train", color="#0f766e", linewidth=2)
        ax_acc.plot(epochs, percent(history["val_accuracy"]), label="Validation", color="#f97316", linewidth=2)
        ax_acc.set_title(f"{MODEL_LABELS[model_name]} Accuracy")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.grid(True, alpha=0.25)
        ax_acc.legend()

        ax_loss.plot(epochs, history["loss"], label="Train", color="#0f766e", linewidth=2)
        ax_loss.plot(epochs, history["val_loss"], label="Validation", color="#f97316", linewidth=2)
        best_epoch = int(np.argmin(history["val_loss"]) + 1)
        best_loss = float(np.min(history["val_loss"]))
        ax_loss.scatter([best_epoch], [best_loss], color="#111827", s=35, zorder=3)
        ax_loss.annotate(
            f"best val_loss\nE{best_epoch}",
            xy=(best_epoch, best_loss),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )
        ax_loss.set_title(f"{MODEL_LABELS[model_name]} Loss")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.25)
        ax_loss.legend()

    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")
    fig.suptitle("Training and Validation Curves", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_curves.png", dpi=dpi)
    plt.close(fig)


def plot_metric_comparison(results: Dict[str, Dict], figures_dir: Path, dpi: int) -> None:
    labels = [MODEL_LABELS[name] for name in results]
    accuracy = [results[name]["metrics"]["test_accuracy"] * 100 for name in results]
    macro_f1 = [results[name]["metrics"]["macro_f1"] * 100 for name in results]
    weighted_f1 = [results[name]["metrics"]["weighted_f1"] * 100 for name in results]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars1 = ax.bar(x - width, accuracy, width, label="Accuracy", color="#2563eb")
    bars2 = ax.bar(x, macro_f1, width, label="Macro F1", color="#16a34a")
    bars3 = ax.bar(x + width, weighted_f1, width, label="Weighted F1", color="#f97316")
    for bars in (bars1, bars2, bars3):
        annotate_bars(ax, bars, fmt="{:.3f}", suffix="%")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(min(99.0, min(accuracy + macro_f1 + weighted_f1) - 0.4), 100.15)
    ax.set_title("Test Classification Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "metric_comparison.png", dpi=dpi)
    plt.close(fig)


def plot_per_class_f1(results: Dict[str, Dict], figures_dir: Path, dpi: int) -> None:
    class_names = next(iter(results.values()))["metrics"]["target_names"]
    x = np.arange(len(class_names))
    width = 0.36 if len(results) == 2 else 0.8 / max(len(results), 1)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    offsets = np.linspace(-width * (len(results) - 1) / 2, width * (len(results) - 1) / 2, len(results))
    for offset, model_name in zip(offsets, results):
        report = results[model_name]["metrics"]["classification_report"]
        scores = [report[class_name]["f1-score"] * 100 for class_name in class_names]
        bars = ax.bar(
            x + offset,
            scores,
            width,
            label=MODEL_LABELS[model_name],
            color=COLORS[model_name],
            alpha=0.88,
        )
        annotate_bars(ax, bars, fmt="{:.2f}", suffix="%")

    ax.set_xticks(x, class_names, rotation=18, ha="right")
    ax.set_ylabel("F1-score (%)")
    ax.set_ylim(97.5, 100.25)
    ax.set_title("Per-Class F1-score on Test Set")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "per_class_f1.png", dpi=dpi)
    plt.close(fig)


def plot_confusion_matrices(results: Dict[str, Dict], figures_dir: Path, dpi: int) -> None:
    class_names = next(iter(results.values()))["metrics"]["target_names"]
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5.8), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, results):
        cm = np.asarray(results[model_name]["metrics"]["confusion_matrix"], dtype=int)
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{MODEL_LABELS[model_name]} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names)), class_names, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(class_names)), class_names)

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                value = normalized[row, col]
                text_color = "white" if value > 0.55 else "#111827"
                ax.text(
                    col,
                    row,
                    f"{cm[row, col]}\n{value * 100:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row-normalized rate")
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=dpi)
    plt.close(fig)


def plot_tinyml_footprint(results: Dict[str, Dict], figures_dir: Path, dpi: int) -> None:
    labels = [MODEL_LABELS[name] for name in results]
    params = [results[name]["metrics"]["num_parameters"] for name in results]
    int8_sizes = [
        results[name]["export"].get(
            "tflite_int8_size_kb",
            file_size_kb(results[name]["dir"] / "model_int8.tflite"),
        )
        for name in results
    ]
    latencies = [results[name]["benchmark"].get("avg_inference_time_ms", np.nan) for name in results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    bar_sets = [
        (params, "Parameters", "Count", "{:.0f}", ""),
        (int8_sizes, "Int8 TFLite Size", "KB", "{:.2f}", " KB"),
        (latencies, "Avg Inference Time", "ms/window", "{:.4f}", " ms"),
    ]
    for ax, (values, title, ylabel, fmt, suffix) in zip(axes, bar_sets):
        bars = ax.bar(labels, values, color=[COLORS[name] for name in results], alpha=0.9)
        annotate_bars(ax, bars, fmt=fmt, suffix=suffix)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if np.nanmax(values) > 0:
            ax.set_ylim(0, np.nanmax(values) * 1.18)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("TinyML Deployment Footprint", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "tinyml_footprint.png", dpi=dpi)
    plt.close(fig)


def plot_summary_dashboard(results: Dict[str, Dict], figures_dir: Path, dpi: int) -> None:
    labels = [MODEL_LABELS[name] for name in results]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    acc = [results[name]["metrics"]["test_accuracy"] * 100 for name in results]
    macro_f1 = [results[name]["metrics"]["macro_f1"] * 100 for name in results]
    int8_sizes = [
        results[name]["export"].get("tflite_int8_size_kb", file_size_kb(results[name]["dir"] / "model_int8.tflite"))
        for name in results
    ]
    latency = [results[name]["benchmark"].get("avg_inference_time_ms", np.nan) for name in results]

    panels = [
        (axes[0, 0], acc, "Accuracy (%)", "Test Accuracy", "{:.3f}", "%"),
        (axes[0, 1], macro_f1, "Macro F1 (%)", "Macro F1", "{:.3f}", "%"),
        (axes[1, 0], int8_sizes, "KB", "Int8 Model Size", "{:.2f}", " KB"),
        (axes[1, 1], latency, "ms/window", "Inference Time", "{:.4f}", " ms"),
    ]
    for ax, values, ylabel, title, fmt, suffix in panels:
        bars = ax.bar(labels, values, color=[COLORS[name] for name in results], alpha=0.9)
        annotate_bars(ax, bars, fmt=fmt, suffix=suffix)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if np.nanmax(values) > 0:
            ax.set_ylim(0, np.nanmax(values) * 1.18)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylim(min(99.0, min(acc) - 0.4), 100.15)
    axes[0, 1].set_ylim(min(99.0, min(macro_f1) - 0.4), 100.15)
    fig.suptitle("TinyTCN vs DS-CNN Summary", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "summary_dashboard.png", dpi=dpi)
    plt.close(fig)


def save_readme(results: Dict[str, Dict], figures_dir: Path) -> None:
    lines = [
        "# Generated Figures",
        "",
        "This folder contains matplotlib visualizations generated from `outputs/<model>/` JSON files.",
        "",
        "## Files",
        "",
        "- `summary_dashboard.png`: compact comparison of accuracy, macro F1, int8 size, and latency.",
        "- `training_curves.png`: train/validation accuracy and loss per epoch.",
        "- `metric_comparison.png`: grouped test metric bars.",
        "- `per_class_f1.png`: per-class F1 comparison.",
        "- `confusion_matrices.png`: row-normalized confusion matrices with raw counts.",
        "- `tinyml_footprint.png`: parameter count, int8 size, and inference latency.",
        "",
        "## Source Results",
        "",
    ]
    for model_name, result in results.items():
        metrics = result["metrics"]
        lines.append(
            f"- {MODEL_LABELS[model_name]}: "
            f"accuracy={metrics['test_accuracy']:.6f}, "
            f"macro_f1={metrics['macro_f1']:.6f}, "
            f"params={metrics['num_parameters']}"
        )
    (figures_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = ensure_dir(args.figures_dir or output_dir / "figures")
    results = load_all_results(output_dir, args.models)

    plot_summary_dashboard(results, figures_dir, args.dpi)
    plot_training_history(results, figures_dir, args.dpi)
    plot_metric_comparison(results, figures_dir, args.dpi)
    plot_per_class_f1(results, figures_dir, args.dpi)
    plot_confusion_matrices(results, figures_dir, args.dpi)
    plot_tinyml_footprint(results, figures_dir, args.dpi)
    save_readme(results, figures_dir)
    print(f"Saved matplotlib figures to {figures_dir}")


if __name__ == "__main__":
    main()
