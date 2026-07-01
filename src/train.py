import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .evaluate import evaluate_model
from .preprocessing import run_preprocessing
from .utils import (
    ensure_dir,
    import_tensorflow,
    load_processed_arrays,
    save_json,
    set_random_seed,
    write_model_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyML UAV state classifiers.")
    parser.add_argument("--model", choices=["tinytcn", "dscnn"], required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=config.STRIDE)
    parser.add_argument("--min-label-purity", type=float, default=config.MIN_LABEL_PURITY)
    parser.add_argument("--feature-set", choices=["default", "core"], default="default")
    parser.add_argument(
        "--ambiguous-label-policy",
        choices=["drop", "priority"],
        default=config.AMBIGUOUS_LABEL_POLICY,
    )
    parser.add_argument("--split-group-column", default=config.SPLIT_GROUP_COLUMN)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUTS_DIR)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args()


def processed_arrays_exist(processed_dir: Path) -> bool:
    required = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
        "scaler.pkl",
        "label_encoder.pkl",
        "metadata.json",
    ]
    return all((Path(processed_dir) / name).exists() for name in required)


def build_model(model_name: str, input_shape, num_classes: int):
    if model_name == "tinytcn":
        from .models.tinytcn import build_tinytcn

        return build_tinytcn(input_shape=input_shape, num_classes=num_classes)
    if model_name == "dscnn":
        from .models.dscnn import build_dscnn

        return build_dscnn(input_shape=input_shape, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def train_model(args: argparse.Namespace):
    tf = import_tensorflow()
    set_random_seed(args.seed)

    if args.force_preprocess or args.data is not None or not processed_arrays_exist(args.processed_dir):
        run_preprocessing(
            data_path=args.data or config.DEFAULT_DATA_PATH,
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

    arrays = load_processed_arrays(args.processed_dir)
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    label_encoder = joblib.load(Path(args.processed_dir) / "label_encoder.pkl")
    num_classes = len(label_encoder.classes_)

    model = build_model(args.model, input_shape=X_train.shape[1:], num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_dir = ensure_dir(Path(args.output_dir) / args.model)
    model_path = model_dir / "model.keras"
    write_model_summary(model, model_dir / "model_summary.txt")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    classes_present = np.unique(y_train)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=y_train,
    )
    class_weight = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes_present, class_weight_values)
    }

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(model_path)
    save_json(history.history, model_dir / "history.json")
    save_json({"class_weight": class_weight}, model_dir / "training_config.json")

    metrics = evaluate_model(
        model_name=args.model,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        model_path=model_path,
    )
    return metrics


def main() -> None:
    args = parse_args()
    metrics = train_model(args)
    print(
        f"Training complete for {args.model}: "
        f"accuracy={metrics['test_accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
