import argparse
from pathlib import Path

import numpy as np

from . import config
from .utils import ensure_dir, file_size_kb, import_tensorflow, load_processed_arrays, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained Keras models to TensorFlow Lite.")
    parser.add_argument("--model", choices=["tinytcn", "dscnn"], required=True)
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUTS_DIR)
    parser.add_argument("--representative-samples", type=int, default=500)
    return parser.parse_args()


def export_tflite(
    model_name: str,
    processed_dir: Path = config.PROCESSED_DIR,
    output_dir: Path = config.OUTPUTS_DIR,
    representative_samples: int = 500,
):
    tf = import_tensorflow()
    model_dir = ensure_dir(Path(output_dir) / model_name)
    model_path = model_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    export_log = []

    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_model = float_converter.convert()
    float_path = model_dir / "model_float32.tflite"
    with open(float_path, "wb") as handle:
        handle.write(float_model)
    export_log.append(f"float32 export ok: {float_path} ({file_size_kb(float_path):.2f} KB)")

    int8_path = model_dir / "model_int8.tflite"
    try:
        arrays = load_processed_arrays(processed_dir)
        X_train = arrays["X_train"].astype(np.float32)
        sample_count = min(representative_samples, len(X_train))

        def representative_dataset():
            for sample in X_train[:sample_count]:
                yield [sample[np.newaxis, ...].astype(np.float32)]

        int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
        int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        int8_converter.representative_dataset = representative_dataset
        int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        int8_converter.inference_input_type = tf.int8
        int8_converter.inference_output_type = tf.int8
        int8_model = int8_converter.convert()
        with open(int8_path, "wb") as handle:
            handle.write(int8_model)
        export_log.append(f"int8 export ok: {int8_path} ({file_size_kb(int8_path):.2f} KB)")
    except Exception as exc:
        export_log.append(f"int8 export failed: {type(exc).__name__}: {exc}")

    with open(model_dir / "export_log.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(export_log) + "\n")

    result = {
        "model": model_name,
        "keras_model_size_kb": file_size_kb(model_path),
        "tflite_float32_size_kb": file_size_kb(float_path),
        "tflite_int8_size_kb": file_size_kb(int8_path),
        "log": export_log,
    }
    save_json(result, model_dir / "export_summary.json")
    return result


def main() -> None:
    args = parse_args()
    result = export_tflite(
        model_name=args.model,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        representative_samples=args.representative_samples,
    )
    for line in result["log"]:
        print(line)


if __name__ == "__main__":
    main()
