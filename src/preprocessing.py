import argparse
import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config
from .data_loader import (
    add_sequence_metadata,
    audit_labels,
    derive_labels,
    load_raw_csv,
    numeric_forward_fill_features,
    parse_tuple_columns,
    select_feature_columns,
    sort_by_sequence_time,
)
from .utils import ensure_dir, save_json, set_random_seed
from .windowing import create_sliding_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess UAV EKF fusion CSV into windowed arrays.")
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--feature-set", choices=["default", "core"], default="default")
    parser.add_argument(
        "--ambiguous-label-policy",
        choices=["drop", "priority"],
        default=config.AMBIGUOUS_LABEL_POLICY,
    )
    parser.add_argument("--split-group-column", default=config.SPLIT_GROUP_COLUMN)
    parser.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=config.STRIDE)
    parser.add_argument("--min-label-purity", type=float, default=config.MIN_LABEL_PURITY)
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args()


def run_preprocessing(
    data_path: Path = config.DEFAULT_DATA_PATH,
    feature_set: str = "default",
    ambiguous_label_policy: str = config.AMBIGUOUS_LABEL_POLICY,
    split_group_column: str = config.SPLIT_GROUP_COLUMN,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
    min_label_purity: float = config.MIN_LABEL_PURITY,
    processed_dir: Path = config.PROCESSED_DIR,
    reports_dir: Path = config.REPORTS_DIR,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, object]:
    set_random_seed(seed)
    processed_dir = ensure_dir(processed_dir)
    reports_dir = ensure_dir(reports_dir)

    df = load_raw_csv(data_path)
    audit_labels(df, reports_dir)
    df = derive_labels(df, ambiguous_label_policy=ambiguous_label_policy)
    df = parse_tuple_columns(df)
    df, sequence_metadata = add_sequence_metadata(df)
    df = sort_by_sequence_time(df)
    if split_group_column not in df.columns:
        raise ValueError(f"Split group column not found after metadata creation: {split_group_column}")

    feature_columns, missing_feature_columns = select_feature_columns(df, feature_set)
    if missing_feature_columns:
        print(f"Missing requested feature columns skipped: {missing_feature_columns}")
    print(f"Using {len(feature_columns)} feature columns")

    df = numeric_forward_fill_features(df, feature_columns)
    valid_statuses = ["single"] if ambiguous_label_policy == "drop" else ["single", "ambiguous_priority"]

    X, y_labels, window_df, window_stats = create_sliding_windows(
        df=df,
        feature_columns=feature_columns,
        window_size=window_size,
        stride=stride,
        min_label_purity=min_label_purity,
        group_column="flight" if "flight" in df.columns else split_group_column,
        split_group_column=split_group_column,
        valid_label_statuses=valid_statuses,
    )
    if len(X) == 0:
        raise RuntimeError("No valid windows were produced. Check label policy, purity, window size, and stride.")

    label_encoder = LabelEncoder()
    label_encoder.fit(config.LABEL_COLUMNS)
    y = label_encoder.transform(y_labels)

    split_assignments = split_windows_by_group(
        y=y,
        groups=window_df[split_group_column].astype(str).to_numpy(),
        train_split=config.TRAIN_SPLIT,
        val_split=config.VALIDATION_SPLIT,
        test_split=config.TEST_SPLIT,
        seed=seed,
    )
    window_df["split"] = split_assignments

    split_indices = {
        split: np.flatnonzero(window_df["split"].to_numpy() == split)
        for split in ["train", "val", "test"]
    }
    for split, indices in split_indices.items():
        if len(indices) == 0:
            raise RuntimeError(f"The {split} split is empty. Adjust group split settings.")

    X_train, y_train = X[split_indices["train"]], y[split_indices["train"]]
    X_val, y_val = X[split_indices["val"]], y[split_indices["val"]]
    X_test, y_test = X[split_indices["test"]], y[split_indices["test"]]

    X_train, X_val, X_test, train_medians = impute_with_train_median(X_train, X_val, X_test)
    X_train, X_val, X_test, scaler = scale_with_train_only(X_train, X_val, X_test)

    np.save(processed_dir / "X_train.npy", X_train.astype(np.float32))
    np.save(processed_dir / "y_train.npy", y_train.astype(np.int64))
    np.save(processed_dir / "X_val.npy", X_val.astype(np.float32))
    np.save(processed_dir / "y_val.npy", y_val.astype(np.int64))
    np.save(processed_dir / "X_test.npy", X_test.astype(np.float32))
    np.save(processed_dir / "y_test.npy", y_test.astype(np.int64))
    joblib.dump(scaler, processed_dir / "scaler.pkl")
    joblib.dump(label_encoder, processed_dir / "label_encoder.pkl")

    window_df.to_csv(processed_dir / "window_metadata.csv", index=False)
    write_class_distribution(y, window_df, label_encoder, reports_dir)
    write_window_distribution(window_df, window_stats, reports_dir)

    metadata = build_metadata(
        data_path=data_path,
        feature_set=feature_set,
        feature_columns=feature_columns,
        missing_feature_columns=missing_feature_columns,
        ambiguous_label_policy=ambiguous_label_policy,
        split_group_column=split_group_column,
        window_size=window_size,
        stride=stride,
        min_label_purity=min_label_purity,
        sequence_metadata=sequence_metadata,
        window_stats=window_stats,
        train_medians=train_medians,
        label_encoder=label_encoder,
        window_df=window_df,
        df=df,
        seed=seed,
        array_shapes={
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
    )
    save_json(metadata, processed_dir / "metadata.json")
    print(f"Saved processed arrays to {processed_dir}")
    return metadata


def split_windows_by_group(
    y: np.ndarray,
    groups: np.ndarray,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> np.ndarray:
    proportions = {"train": train_split, "val": val_split, "test": test_split}
    if not np.isclose(sum(proportions.values()), 1.0):
        raise ValueError("train/val/test split proportions must sum to 1.0")

    rng = np.random.default_rng(seed)
    group_values = groups.astype(str)
    unique_groups = np.asarray(sorted(pd.unique(group_values)))
    rng.shuffle(unique_groups)

    total_hist = np.bincount(y, minlength=config.NUM_CLASSES).astype(float)
    total_count = float(len(y))
    group_hists: Dict[str, np.ndarray] = {}
    group_sizes: Dict[str, int] = {}
    for group in unique_groups:
        mask = group_values == group
        group_hists[group] = np.bincount(y[mask], minlength=config.NUM_CLASSES).astype(float)
        group_sizes[group] = int(mask.sum())

    remaining = set(unique_groups.tolist())
    test_groups = _choose_group_subset(
        candidates=remaining,
        group_hists=group_hists,
        group_sizes=group_sizes,
        target_count=total_count * test_split,
        target_hist=total_hist * test_split,
        total_count=total_count,
        seed=seed + 1,
    )
    remaining -= set(test_groups)
    val_groups = _choose_group_subset(
        candidates=remaining,
        group_hists=group_hists,
        group_sizes=group_sizes,
        target_count=total_count * val_split,
        target_hist=total_hist * val_split,
        total_count=total_count,
        seed=seed + 2,
    )
    remaining -= set(val_groups)

    assigned_groups: Dict[str, List[str]] = {
        "train": sorted(remaining),
        "val": sorted(val_groups),
        "test": sorted(test_groups),
    }
    if not assigned_groups["train"] or not assigned_groups["val"] or not assigned_groups["test"]:
        raise RuntimeError("Unable to create non-empty train/val/test group split.")

    assignments = np.empty(len(y), dtype=object)
    group_to_split = {
        group: split for split, split_groups in assigned_groups.items() for group in split_groups
    }
    for idx, group in enumerate(group_values):
        assignments[idx] = group_to_split[group]
    return assignments


def _choose_group_subset(
    candidates,
    group_hists: Dict[str, np.ndarray],
    group_sizes: Dict[str, int],
    target_count: float,
    target_hist: np.ndarray,
    total_count: float,
    seed: int,
) -> List[str]:
    candidates = sorted(candidates)
    if not candidates:
        return []

    if len(candidates) <= 24:
        best_subset = None
        best_score = None
        for subset_size in range(1, len(candidates)):
            for subset in itertools.combinations(candidates, subset_size):
                score = _score_group_subset(
                    subset,
                    group_hists=group_hists,
                    group_sizes=group_sizes,
                    target_count=target_count,
                    target_hist=target_hist,
                    total_count=total_count,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_subset = subset
        return list(best_subset or [candidates[0]])

    rng = np.random.default_rng(seed)
    ordered = candidates.copy()
    rng.shuffle(ordered)
    ordered = sorted(ordered, key=lambda group: group_sizes[group], reverse=True)
    selected: List[str] = []
    best_score = _score_group_subset(
        selected,
        group_hists=group_hists,
        group_sizes=group_sizes,
        target_count=target_count,
        target_hist=target_hist,
        total_count=total_count,
    )
    for group in ordered:
        trial = selected + [group]
        score = _score_group_subset(
            trial,
            group_hists=group_hists,
            group_sizes=group_sizes,
            target_count=target_count,
            target_hist=target_hist,
            total_count=total_count,
        )
        if score <= best_score or sum(group_sizes[g] for g in selected) < target_count * 0.85:
            selected = trial
            best_score = score
    return selected or [ordered[0]]


def _score_group_subset(
    subset,
    group_hists: Dict[str, np.ndarray],
    group_sizes: Dict[str, int],
    target_count: float,
    target_hist: np.ndarray,
    total_count: float,
) -> float:
    if not subset:
        subset_count = 0.0
        subset_hist = np.zeros_like(target_hist, dtype=float)
    else:
        subset_count = float(sum(group_sizes[group] for group in subset))
        subset_hist = np.sum([group_hists[group] for group in subset], axis=0)

    count_score = abs(subset_count - target_count) / max(total_count, 1.0)
    if subset_count > 0 and target_count > 0:
        class_score = np.abs((subset_hist / subset_count) - (target_hist / target_count)).mean()
    else:
        class_score = 1.0
    empty_class_penalty = float((subset_hist == 0).sum()) / max(len(subset_hist), 1)
    return (2.0 * count_score) + (0.5 * class_score) + (0.1 * empty_class_penalty)


def impute_with_train_median(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_features = X_train.shape[-1]
    medians = np.nanmedian(X_train.reshape(-1, num_features), axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians).astype(np.float32)

    def impute(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        nan_rows, nan_steps, nan_features = np.where(np.isnan(X))
        if len(nan_rows):
            X[nan_rows, nan_steps, nan_features] = medians[nan_features]
        return X

    return impute(X_train), impute(X_val), impute(X_test), medians


def scale_with_train_only(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    timesteps = X_train.shape[1]
    num_features = X_train.shape[2]
    scaler.fit(X_train.reshape(-1, num_features))

    def transform(X: np.ndarray) -> np.ndarray:
        transformed = scaler.transform(X.reshape(-1, num_features))
        return transformed.reshape(-1, timesteps, num_features).astype(np.float32)

    return transform(X_train), transform(X_val), transform(X_test), scaler


def write_class_distribution(
    y: np.ndarray, window_df: pd.DataFrame, label_encoder: LabelEncoder, reports_dir: Path
) -> None:
    rows = []
    for split in ["all", "train", "val", "test"]:
        mask = np.ones(len(y), dtype=bool) if split == "all" else window_df["split"].to_numpy() == split
        split_y = y[mask]
        total = max(int(len(split_y)), 1)
        for class_idx, label in enumerate(label_encoder.classes_):
            count = int((split_y == class_idx).sum())
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "display_name": config.CLASS_DISPLAY_NAMES.get(label, label),
                    "count": count,
                    "percent": count / total,
                }
            )
    pd.DataFrame(rows).to_csv(Path(reports_dir) / "class_distribution.csv", index=False)


def write_window_distribution(window_df: pd.DataFrame, window_stats: Dict[str, int], reports_dir: Path) -> None:
    rows = [{"metric": key, "value": int(value), "split": "", "label": ""} for key, value in window_stats.items()]
    for (split, label), count in window_df.groupby(["split", "label"]).size().items():
        rows.append(
            {
                "metric": "kept_windows_by_split_label",
                "value": int(count),
                "split": split,
                "label": label,
            }
        )
    pd.DataFrame(rows).to_csv(Path(reports_dir) / "window_distribution.csv", index=False)


def build_metadata(
    data_path: Path,
    feature_set: str,
    feature_columns: List[str],
    missing_feature_columns: List[str],
    ambiguous_label_policy: str,
    split_group_column: str,
    window_size: int,
    stride: int,
    min_label_purity: float,
    sequence_metadata: Dict[str, object],
    window_stats: Dict[str, int],
    train_medians: np.ndarray,
    label_encoder: LabelEncoder,
    window_df: pd.DataFrame,
    df: pd.DataFrame,
    seed: int,
    array_shapes: Dict[str, List[int]],
) -> Dict[str, object]:
    split_details = {}
    for split in ["train", "val", "test"]:
        split_windows = window_df[window_df["split"] == split]
        split_groups = sorted(split_windows[split_group_column].astype(str).unique().tolist())
        source_rows = df[df[split_group_column].astype(str).isin(split_groups)]
        split_details[split] = {
            "num_windows": int(len(split_windows)),
            "base_flight_id": split_groups,
            "flight": sorted(source_rows["flight"].astype(str).unique().tolist())
            if "flight" in source_rows.columns
            else [],
            "uid": sorted(source_rows["uid"].astype(str).unique().tolist())
            if "uid" in source_rows.columns
            else [],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(Path(data_path)),
        "random_seed": seed,
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "missing_feature_columns": missing_feature_columns,
        "label_columns": config.LABEL_COLUMNS,
        "class_display_names": config.CLASS_DISPLAY_NAMES,
        "label_encoder_classes": label_encoder.classes_.tolist(),
        "ambiguous_label_policy": ambiguous_label_policy,
        "priority_label_order": config.PRIORITY_LABEL_ORDER,
        "window_size": window_size,
        "stride": stride,
        "min_label_purity": min_label_purity,
        "split_group_column": split_group_column,
        "splits": split_details,
        "sequence_metadata": sequence_metadata,
        "window_stats": window_stats,
        "train_feature_medians": train_medians.tolist(),
        "array_shapes": array_shapes,
    }


def main() -> None:
    args = parse_args()
    run_preprocessing(
        data_path=args.data,
        feature_set=args.feature_set,
        ambiguous_label_policy=args.ambiguous_label_policy,
        split_group_column=args.split_group_column,
        window_size=args.window_size,
        stride=args.stride,
        min_label_purity=args.min_label_purity,
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
