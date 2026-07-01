import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def import_tensorflow():
    try:
        import tensorflow as tf

        return tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for training/export/benchmark. "
            "Install dependencies with `pip install -r requirements.txt`. "
            "If TensorFlow is unavailable on this machine, use Python 3.10 or 3.11."
        ) from exc


def numpy_to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): numpy_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_to_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(numpy_to_python(data), handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def file_size_kb(path: Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / 1024.0


def write_model_summary(model, path: Path) -> None:
    ensure_dir(Path(path).parent)
    lines = []
    model.summary(print_fn=lines.append)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def load_processed_arrays(processed_dir: Path):
    processed_dir = Path(processed_dir)
    arrays = {
        "X_train": np.load(processed_dir / "X_train.npy"),
        "y_train": np.load(processed_dir / "y_train.npy"),
        "X_val": np.load(processed_dir / "X_val.npy"),
        "y_val": np.load(processed_dir / "y_val.npy"),
        "X_test": np.load(processed_dir / "X_test.npy"),
        "y_test": np.load(processed_dir / "y_test.npy"),
    }
    return arrays
