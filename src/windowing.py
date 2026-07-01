from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def create_sliding_windows(
    df: pd.DataFrame,
    feature_columns: List[str],
    window_size: int,
    stride: int,
    min_label_purity: float,
    group_column: str = "flight",
    split_group_column: str = "base_flight_id",
    valid_label_statuses: Iterable[str] = ("single",),
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, int]]:
    valid_label_statuses = set(valid_label_statuses)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not 0 < min_label_purity <= 1:
        raise ValueError("min_label_purity must be in (0, 1]")

    if group_column not in df.columns:
        group_column = split_group_column if split_group_column in df.columns else None

    windows = []
    labels = []
    metadata_rows = []
    stats = {
        "candidate_windows": 0,
        "kept_windows": 0,
        "skipped_too_short": 0,
        "skipped_label_status": 0,
        "skipped_missing_label": 0,
        "skipped_low_purity": 0,
    }

    grouped = [(None, df)] if group_column is None else df.groupby(group_column, sort=False)
    window_id = 0

    for group_value, group in grouped:
        if len(group) < window_size:
            stats["skipped_too_short"] += 1
            continue

        feature_values = group[feature_columns].to_numpy(dtype=np.float32)
        label_values = group["label"].to_numpy(dtype=object)
        status_values = group["label_status"].to_numpy(dtype=object)
        split_group_values = (
            group[split_group_column].to_numpy(dtype=object)
            if split_group_column in group.columns
            else np.full(len(group), group_value, dtype=object)
        )

        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            stats["candidate_windows"] += 1
            window_status = status_values[start:end]
            if not set(window_status).issubset(valid_label_statuses):
                stats["skipped_label_status"] += 1
                continue

            window_labels = label_values[start:end]
            if any(label is None or (isinstance(label, float) and np.isnan(label)) for label in window_labels):
                stats["skipped_missing_label"] += 1
                continue

            counts = Counter(window_labels)
            majority_label, majority_count = counts.most_common(1)[0]
            purity = majority_count / float(window_size)
            if purity < min_label_purity:
                stats["skipped_low_purity"] += 1
                continue

            windows.append(feature_values[start:end])
            labels.append(majority_label)
            metadata_rows.append(
                {
                    "window_id": window_id,
                    "flight": group_value,
                    "base_flight_id": split_group_values[start],
                    "start_row": int(group.index[start]),
                    "end_row": int(group.index[end - 1]),
                    "label": majority_label,
                    "purity": purity,
                }
            )
            window_id += 1
            stats["kept_windows"] += 1

    if windows:
        X = np.stack(windows).astype(np.float32)
        y = np.asarray(labels, dtype=object)
    else:
        X = np.empty((0, window_size, len(feature_columns)), dtype=np.float32)
        y = np.empty((0,), dtype=object)
    return X, y, pd.DataFrame(metadata_rows), stats
