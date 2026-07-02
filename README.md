# UAV State Classification with TinyML Models

This project builds an end-to-end Python/TensorFlow pipeline for multiclass UAV
flight-state classification from the Drone onboard extended EKF fusion dataset.
The raw CSV is expected at:

```text
data/raw/drone_extended_ekf_fusion.csv
```

The main comparison is between two TinyML-friendly models:

- TinyTCN
- DS-CNN

## Dataset and Labels

The CSV contains onboard sensor signals, EKF fusion features, tuple ESC fields,
and five state columns:

```text
IDLE_HOVER, ASCEND, TURN, HMSL, DESCEND
```

`flight_logic_state` is audited for sanity checks only. It is not used as the
training label because this dataset can contain only `Idle-Hover` in that
column.

Output classes are derived from the five state columns:

| Dataset label | Report name |
| --- | --- |
| `IDLE_HOVER` | Hover |
| `ASCEND` | Ascending |
| `TURN` | Turning |
| `HMSL` | Forward flight |
| `DESCEND` | Descending |

Rows with exactly one active label are treated as clean single-label rows. By
default, missing-label and multi-hot rows are excluded from the main multiclass
experiment through:

```text
--ambiguous-label-policy drop
```

An optional `priority` policy is available for side experiments and uses:

```text
TURN > ASCEND > DESCEND > HMSL > IDLE_HOVER
```

## Feature Columns

The default feature set uses sensor and EKF columns only, including parsed ESC
tuple values:

```text
wind_speed, wind_angle,
battery_voltage, battery_current,
altitude,
orientation_x, orientation_y, orientation_z, orientation_w,
velocity_x, velocity_y, velocity_z,
angular_x, angular_y, angular_z,
linear_acceleration_x, linear_acceleration_y, linear_acceleration_z,
payload,
d_roll, d_pitch, d_yaw,
heading, yaw,
kalmanZ,
escSpeed_1..4,
escVoltage_1..4
```

The raw tuple columns `escSpeed` and `escVoltage` are parsed into numeric
columns. Identifier and label columns such as `seq`, `uid`, `timestamp`,
`flight`, `flight_logic_state`, and the five label columns are never used as
model input features.

A smaller `core` feature set is also available:

```text
linear_acceleration_x/y/z,
angular_x/y/z,
velocity_x/y/z,
altitude,
kalmanZ
```

## Preprocessing Pipeline

The preprocessing script:

1. Loads the CSV with pandas and strips column whitespace.
2. Audits label columns and writes `data/reports/raw_label_audit.csv`.
3. Derives labels from the five state columns.
4. Parses `escSpeed` and `escVoltage`.
5. Sorts by `flight`, `timestamp`, and `seq`.
6. Creates `base_flight_id` so original and synthetic paired flights remain in
   the same split.
7. Forward-fills missing feature values within each flight.
8. Builds sliding windows per flight, never across flight boundaries.
9. Drops windows with missing/ambiguous labels under the default policy.
10. Splits windows by `base_flight_id` into train/validation/test.
11. Imputes remaining NaNs using train-set medians only.
12. Fits `StandardScaler` on train only and transforms all splits.

Default windowing:

```text
WINDOW_SIZE = 10
STRIDE = 5
MIN_LABEL_PURITY = 0.8
```

Processed arrays and metadata are written to `data/processed/`.

## Models

TinyTCN uses causal 1D convolutions with residual dilated TCN blocks:

```text
Conv1D -> BN -> ReLU -> Dropout
TCN blocks with dilation 1, 2, 4
GlobalAveragePooling1D -> Dense -> Dropout -> Softmax
```

DS-CNN uses separable 1D convolutions:

```text
Conv1D -> BN -> ReLU
SeparableConv1D blocks
GlobalAveragePooling1D -> Dense -> Dropout -> Softmax
```

Both models use sparse categorical cross-entropy and Adam.

## Setup

TensorFlow is easiest to install on Python 3.10 or 3.11.

```bash
pip install -r requirements.txt
```

## Commands

Preprocess:

```bash
python -m src.preprocessing --data data/raw/drone_extended_ekf_fusion.csv --feature-set default --ambiguous-label-policy drop --split-group-column base_flight_id
```

Train TinyTCN:

```bash
python -m src.train --model tinytcn --feature-set default
```

Train DS-CNN:

```bash
python -m src.train --model dscnn --feature-set default
```

You can force preprocessing during training:

```bash
python -m src.train --model tinytcn --data data/raw/drone_extended_ekf_fusion.csv --force-preprocess
```

Evaluate:

```bash
python -m src.evaluate --model tinytcn
python -m src.evaluate --model dscnn
```

Export TFLite:

```bash
python -m src.export_tflite --model tinytcn
python -m src.export_tflite --model dscnn
```

Benchmark:

```bash
python -m src.benchmark --model tinytcn --tflite int8
python -m src.benchmark --model dscnn --tflite int8
```

If int8 export fails, benchmark float32 instead:

```bash
python -m src.benchmark --model tinytcn --tflite float32
python -m src.benchmark --model dscnn --tflite float32
```

Create the comparison table:

```bash
python -m src.evaluate --compare
```

Generate matplotlib figures:

```bash
python -m src.visualize_results
```

Figures are saved to:

```text
outputs/figures/
```

## Outputs

Each trained model writes to `outputs/<model_name>/`:

```text
model.keras
model_summary.txt
history.json
training_config.json
metrics.json
classification_report.txt
confusion_matrix.png
model_float32.tflite
model_int8.tflite
export_log.txt
benchmark.json
```

The comparison table is:

```text
outputs/comparison.csv
```

Expected columns:

```text
model_name,
num_parameters,
keras_model_size_kb,
tflite_float32_size_kb,
tflite_int8_size_kb,
test_accuracy,
macro_f1,
avg_inference_time_ms
```

## Notes and Limitations

- The project is a 5-class classifier. It does not create a synthetic `Landing`
  class because the dataset does not provide that label.
- The main split is group-based by `base_flight_id` to reduce leakage between
  original and synthetic paired flights.
- `timestamp` is not treated as a reliable sampling-rate signal because the CSV
  can contain repeated integer-second timestamps.
- Accuracy alone is not enough for reporting; macro F1 and the confusion matrix
  are generated for every evaluated model.
