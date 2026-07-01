import argparse
import time
from pathlib import Path

import numpy as np

from . import config
from .utils import ensure_dir, file_size_kb, import_tensorflow, load_processed_arrays, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TFLite inference latency.")
    parser.add_argument("--model", choices=["tinytcn", "dscnn"], required=True)
    parser.add_argument("--tflite", choices=["float32", "int8"], default="int8")
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUTS_DIR)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    return parser.parse_args()


def prepare_input(sample: np.ndarray, input_detail) -> np.ndarray:
    input_tensor = sample[np.newaxis, ...].astype(np.float32)
    dtype = input_detail["dtype"]
    if dtype in (np.int8, np.uint8):
        scale, zero_point = input_detail["quantization"]
        if scale == 0:
            scale = 1.0
        input_tensor = np.round(input_tensor / scale + zero_point)
        info = np.iinfo(dtype)
        input_tensor = np.clip(input_tensor, info.min, info.max).astype(dtype)
    else:
        input_tensor = input_tensor.astype(dtype)
    return input_tensor


def benchmark_tflite(
    model_name: str,
    tflite_type: str = "int8",
    processed_dir: Path = config.PROCESSED_DIR,
    output_dir: Path = config.OUTPUTS_DIR,
    num_samples: int = 500,
    warmup: int = 20,
):
    tf = import_tensorflow()
    arrays = load_processed_arrays(processed_dir)
    X_test = arrays["X_test"]
    model_dir = ensure_dir(Path(output_dir) / model_name)
    tflite_path = model_dir / f"model_{tflite_type}.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    sample_count = min(num_samples, len(X_test))
    if sample_count == 0:
        raise RuntimeError("X_test is empty; cannot benchmark.")

    samples = X_test[:sample_count]
    for sample in samples[: min(warmup, sample_count)]:
        interpreter.set_tensor(input_detail["index"], prepare_input(sample, input_detail))
        interpreter.invoke()
        _ = interpreter.get_tensor(output_detail["index"])

    elapsed = []
    for sample in samples:
        interpreter.set_tensor(input_detail["index"], prepare_input(sample, input_detail))
        start = time.perf_counter()
        interpreter.invoke()
        _ = interpreter.get_tensor(output_detail["index"])
        elapsed.append((time.perf_counter() - start) * 1000.0)

    result = {
        "model": model_name,
        "tflite_type": tflite_type,
        "num_samples": int(sample_count),
        "warmup": int(min(warmup, sample_count)),
        "avg_inference_time_ms": float(np.mean(elapsed)),
        "p50_inference_time_ms": float(np.percentile(elapsed, 50)),
        "p95_inference_time_ms": float(np.percentile(elapsed, 95)),
        "model_size_kb": file_size_kb(tflite_path),
    }
    save_json(result, model_dir / f"benchmark_{tflite_type}.json")
    save_json(result, model_dir / "benchmark.json")
    return result


def main() -> None:
    args = parse_args()
    result = benchmark_tflite(
        model_name=args.model,
        tflite_type=args.tflite,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        warmup=args.warmup,
    )
    print(
        f"{args.model} {args.tflite}: "
        f"{result['avg_inference_time_ms']:.4f} ms/window over {result['num_samples']} samples"
    )


if __name__ == "__main__":
    main()
