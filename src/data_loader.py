import ast
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from . import config
from .utils import ensure_dir


def load_raw_csv(data_path: Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()
    validate_required_columns(df, config.LABEL_COLUMNS)

    print(f"Loaded dataframe: shape={df.shape}")
    if "flight_logic_state" in df.columns:
        states = df["flight_logic_state"].dropna().astype(str).unique().tolist()
        print(f"flight_logic_state unique values ({len(states)}): {states[:20]}")

    label_numeric = labels_as_numeric(df)
    print("Raw label column totals:")
    for label in config.LABEL_COLUMNS:
        print(f"  {label}: {int(label_numeric[label].sum())}")

    label_sum = label_numeric.sum(axis=1)
    print(f"Single-label rows: {int((label_sum == 1).sum())}")
    print(f"Missing-label rows: {int((label_sum == 0).sum())}")
    print(f"Ambiguous/multi-hot rows: {int((label_sum > 1).sum())}")
    return df


def validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def labels_as_numeric(df: pd.DataFrame) -> pd.DataFrame:
    label_df = df[config.LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return label_df.fillna(0).astype(int)


def audit_labels(df: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    ensure_dir(reports_dir)
    label_numeric = labels_as_numeric(df)
    label_sum = label_numeric.sum(axis=1)

    rows = [{"metric": "total_rows", "value": int(len(df)), "detail": ""}]
    if "flight_logic_state" in df.columns:
        value_counts = df["flight_logic_state"].fillna("<NA>").astype(str).value_counts()
        for value, count in value_counts.items():
            rows.append(
                {
                    "metric": "flight_logic_state",
                    "value": int(count),
                    "detail": value,
                }
            )

    for label in config.LABEL_COLUMNS:
        rows.append(
            {
                "metric": "label_column_total",
                "value": int(label_numeric[label].sum()),
                "detail": label,
            }
        )

    rows.extend(
        [
            {"metric": "single_label_rows", "value": int((label_sum == 1).sum()), "detail": ""},
            {"metric": "missing_label_rows", "value": int((label_sum == 0).sum()), "detail": ""},
            {"metric": "ambiguous_label_rows", "value": int((label_sum > 1).sum()), "detail": ""},
        ]
    )
    audit = pd.DataFrame(rows)
    audit.to_csv(Path(reports_dir) / "raw_label_audit.csv", index=False)
    return audit


def derive_labels(df: pd.DataFrame, ambiguous_label_policy: str = "drop") -> pd.DataFrame:
    if ambiguous_label_policy not in {"drop", "priority"}:
        raise ValueError("ambiguous_label_policy must be either 'drop' or 'priority'")

    label_numeric = labels_as_numeric(df)
    label_sum = label_numeric.sum(axis=1)
    labels = np.full(len(df), None, dtype=object)
    status = np.full(len(df), "missing", dtype=object)

    single_mask = (label_sum == 1).to_numpy()
    if single_mask.any():
        labels[single_mask] = label_numeric.loc[single_mask, config.LABEL_COLUMNS].idxmax(axis=1).to_numpy()
        status[single_mask] = "single"

    ambiguous_mask = (label_sum > 1).to_numpy()
    if ambiguous_mask.any():
        if ambiguous_label_policy == "priority":
            priority_labels = label_numeric.loc[
                ambiguous_mask, config.PRIORITY_LABEL_ORDER
            ].idxmax(axis=1)
            labels[ambiguous_mask] = priority_labels.to_numpy()
            status[ambiguous_mask] = "ambiguous_priority"
        else:
            status[ambiguous_mask] = "ambiguous"

    df = df.copy()
    df["label_sum"] = label_sum.astype(int)
    df["label_status"] = status
    df["label"] = labels
    return df


def parse_tuple_columns(df: pd.DataFrame, columns: Tuple[str, ...] = ("escSpeed", "escVoltage")) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            continue
        parsed = df[column].apply(_parse_four_tuple)
        expanded = pd.DataFrame(parsed.tolist(), index=df.index)
        for idx in range(4):
            df[f"{column}_{idx + 1}"] = pd.to_numeric(expanded[idx], errors="coerce")
    return df


def _parse_four_tuple(value) -> List[float]:
    if pd.isna(value):
        return [np.nan] * 4
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        try:
            parsed = ast.literal_eval(str(value))
            items = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
        except Exception:
            return [np.nan] * 4

    if len(items) != 4:
        return [np.nan] * 4
    result = []
    for item in items:
        try:
            result.append(float(item))
        except Exception:
            result.append(np.nan)
    return result


def add_sequence_metadata(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = df.copy()
    metadata: Dict[str, object] = {"base_flight_strategy": "unknown"}

    if "uid" in df.columns:
        df["is_synthetic"] = df["uid"].astype(str).str.contains("_synthetic", case=False, na=False)
    else:
        df["is_synthetic"] = False

    if "flight" in df.columns:
        flight_numeric = pd.to_numeric(df["flight"], errors="coerce")
        df["flight"] = flight_numeric.where(flight_numeric.notna(), df["flight"])
        df["base_flight_id"] = df["flight"].astype(str)
        unique_flights = sorted(df["flight"].dropna().unique().tolist())

        flight_synth = (
            df.groupby("flight", dropna=False)["is_synthetic"].max().reset_index().sort_values("flight")
        )
        synthetic_flights = flight_synth.loc[flight_synth["is_synthetic"], "flight"].tolist()
        original_flights = flight_synth.loc[~flight_synth["is_synthetic"], "flight"].tolist()

        if synthetic_flights and len(synthetic_flights) == len(original_flights):
            mapping = {
                str(synthetic): str(original)
                for synthetic, original in zip(sorted(synthetic_flights), sorted(original_flights))
            }
            df["base_flight_id"] = df["flight"].astype(str).replace(mapping)
            metadata["base_flight_strategy"] = "uid_synthetic_pairing"
            metadata["synthetic_to_base_flight"] = mapping
        elif synthetic_flights:
            metadata["base_flight_strategy"] = "flight_with_unpaired_synthetic_uid"
        elif len(unique_flights) % 2 == 0 and len(unique_flights) > 0:
            half = len(unique_flights) // 2
            first_half = unique_flights[:half]
            second_half = unique_flights[half:]
            mapping = {str(synthetic): str(original) for synthetic, original in zip(second_half, first_half)}
            df["base_flight_id"] = df["flight"].astype(str).replace(mapping)
            metadata["base_flight_strategy"] = "numeric_half_pairing_no_uid_marker"
            metadata["synthetic_to_base_flight"] = mapping
        else:
            metadata["base_flight_strategy"] = "flight_identity"

        metadata["num_flights"] = int(df["flight"].nunique(dropna=True))
        metadata["num_base_flight_ids"] = int(df["base_flight_id"].nunique(dropna=True))
    elif "uid" in df.columns:
        df["base_flight_id"] = (
            df["uid"]
            .astype(str)
            .str.replace("_synthetic", "", regex=False)
            .str.replace("-synthetic", "", regex=False)
        )
        metadata["base_flight_strategy"] = "uid_without_synthetic_suffix"
    else:
        df["flight"] = 0
        df["base_flight_id"] = "0"
        metadata["base_flight_strategy"] = "single_sequence_fallback"
        metadata["warning"] = "Neither flight nor uid exists; treating all rows as one sequence."

    return df, metadata


def sort_by_sequence_time(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [col for col in ["flight", "timestamp", "seq"] if col in df.columns]
    if not sort_columns:
        sort_columns = [col for col in ["uid"] if col in df.columns]
    if not sort_columns:
        return df.reset_index(drop=True)
    return df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def select_feature_columns(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str]]:
    if feature_set == "default":
        requested = config.FEATURE_COLUMNS
    elif feature_set == "core":
        requested = config.CORE_FEATURE_COLUMNS
    else:
        raise ValueError("feature_set must be either 'default' or 'core'")

    selected = [col for col in requested if col in df.columns]
    missing = [col for col in requested if col not in df.columns]
    leakage = [col for col in selected if col in config.NON_FEATURE_COLUMNS]
    if leakage:
        raise ValueError(f"Leakage columns selected as features: {leakage}")
    return selected, missing


def numeric_forward_fill_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    group_col = "flight" if "flight" in df.columns else "base_flight_id"
    if group_col in df.columns:
        df[feature_columns] = df.groupby(group_col, sort=False)[feature_columns].ffill()
    else:
        df[feature_columns] = df[feature_columns].ffill()
    return df
