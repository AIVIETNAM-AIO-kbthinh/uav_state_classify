# plan.md — TinyML UAV Flight State Classification using Extended EKF Fusion Sensor Dataset

## 1. Mục tiêu dự án

Xây dựng pipeline huấn luyện và đánh giá mô hình TinyML cho bài toán **phân loại trạng thái bay của UAV** dựa trên dữ liệu cảm biến onboard đa phương thức và đặc trưng EKF fusion.

Dataset đã chốt sử dụng là **Drone onboard extended EKF fusion 2026** link data: https://zenodo.org/records/20392506 . Sau khi giải nén, dữ liệu chính nằm trong một file CSV. Nhãn không nằm trong file annotation riêng mà đã được nhúng trực tiếp trong CSV thông qua cột `flight_logic_state` và các cột one-hot label.

Mục tiêu cuối cùng là so sánh hai phương pháp chính của thành viên này:

- **TinyTCN**
- **DS-CNN**

Các mô hình cần được đánh giá theo cả chất lượng phân loại và khả năng triển khai TinyML:

- Accuracy
- Macro Precision / Macro Recall / Macro F1-score
- Confusion matrix
- Số tham số mô hình
- Dung lượng model sau export, xem như Flash usage proxy
- Thời gian inference trung bình
- TFLite float32 và TFLite int8 nếu có thể

---

## 2. Mô tả bài toán

### 2.1. Loại bài toán

Đây là bài toán **multiclass time-series classification**.

Mỗi mẫu đầu vào là một cửa sổ thời gian liên tiếp của dữ liệu cảm biến UAV. Mỗi cửa sổ được gán một nhãn trạng thái bay. Mô hình nhận đầu vào dạng chuỗi thời gian và dự đoán UAV đang thuộc một trong các trạng thái bay đã định nghĩa.

### 2.2. Input

Input là dữ liệu cảm biến theo thời gian, lấy từ file CSV của dataset. Dữ liệu gốc có nhiều cột như:

```text
seq, uid, timestamp, flight_logic_state,
wind_speed, wind_angle,
battery_voltage, battery_current,
position_x, position_y, position_z, altitude,
orientation_x, orientation_y, orientation_z, orientation_w,
velocity_x, velocity_y, velocity_z,
angular_x, angular_y, angular_z,
linear_acceleration_x, linear_acceleration_y, linear_acceleration_z,
payload,
escSpeed, escVoltage,
d_roll, d_pitch, d_yaw, heading, yaw,
flight,
IDLE_HOVER, ASCEND, TURN, HMSL, DESCEND,
kalmanLat, kalmanLong, kalmanZ
```

Các cột input chính nên sử dụng:

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
kalmanZ
```

Các cột `escSpeed` và `escVoltage` có dạng chuỗi tuple, ví dụ:

```text
"(279, 279, 279, 279)"
"(4927, 4909, 4896, 4918)"
```

Cần parse thành các cột số riêng:

```text
escSpeed_1, escSpeed_2, escSpeed_3, escSpeed_4
escVoltage_1, escVoltage_2, escVoltage_3, escVoltage_4
```

Sau khi parse, có thể đưa các cột này vào input.

Không dùng trực tiếp các cột định danh hoặc nhãn làm feature:

```text
seq
uid
timestamp
flight_logic_state
flight
IDLE_HOVER
ASCEND
TURN
HMSL
DESCEND
```

Lý do: các cột này có thể gây data leakage hoặc không phải tín hiệu cảm biến thực sự.

### 2.3. Output

Output thống nhất gồm 5 trạng thái bay:

| Label trong dataset | Tên class dùng trong báo cáo | Ý nghĩa |
|---|---|---|
| `Idle-Hover` hoặc `IDLE_HOVER` | `Hover` | UAV bay treo hoặc gần như đứng yên |
| `Ascend` hoặc `ASCEND` | `Ascending` | UAV đang tăng độ cao |
| `Descend` hoặc `DESCEND` | `Descending` | UAV đang giảm độ cao |
| `Turn` hoặc `TURN` | `Turning` | UAV đang đổi hướng/quay |
| `HMSL` | `Forward flight` | Horizontal Movement Straight Line, UAV bay ngang theo đường tương đối thẳng |

Lưu ý: đề bài ban đầu có thể nhắc đến 6 trạng thái gồm cả `Landing`, nhưng dataset đã chốt không cung cấp nhãn `Landing` riêng. Vì vậy, output chính thức của dự án này là 5 class theo nhãn gốc của dataset.

---

## 3. Quy ước thư mục dự án

Coding agent cần tạo project theo cấu trúc sau:

```text
uav_state_classify/
│
├── data/
│   ├── raw/
│   │   └── drone_extended_ekf_fusion.csv
│   ├── processed/
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── X_val.npy
│   │   ├── y_val.npy
│   │   ├── X_test.npy
│   │   ├── y_test.npy
│   │   ├── scaler.pkl
│   │   ├── label_encoder.pkl
│   │   └── metadata.json
│   └── reports/
│       ├── class_distribution.csv
│       └── window_distribution.csv
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── windowing.py
│   ├── models/
│   │   ├── tinytcn.py
│   │   └── dscnn.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_tflite.py
│   ├── benchmark.py
│   └── utils.py
│
├── outputs/
│   ├── tinytcn/
│   │   ├── model.keras
│   │   ├── model_float32.tflite
│   │   ├── model_int8.tflite
│   │   ├── metrics.json
│   │   ├── confusion_matrix.png
│   │   └── model_summary.txt
│   └── dscnn/
│       ├── model.keras
│       ├── model_float32.tflite
│       ├── model_int8.tflite
│       ├── metrics.json
│       ├── confusion_matrix.png
│       └── model_summary.txt
│
├── requirements.txt
├── README.md
└── plan.md
```

---

## 4. Yêu cầu môi trường

Tạo `requirements.txt` tối thiểu gồm:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow
joblib
```

Ghi chú:

- Nếu không muốn dùng `seaborn`, có thể dùng hoàn toàn `matplotlib` để vẽ confusion matrix.
- Ưu tiên TensorFlow/Keras vì thuận tiện export sang TensorFlow Lite cho TinyML.
- Nếu môi trường không hỗ trợ TensorFlow, coding agent cần báo lỗi rõ ràng và gợi ý dùng Python 3.10 hoặc 3.11.

---

## 5. Tiền xử lý dữ liệu

### 5.1. Load CSV

File input mặc định:

```text
data/raw/drone_extended_ekf_fusion.csv
```

Yêu cầu:

- Đọc bằng `pandas.read_csv`.
- Tự động strip khoảng trắng ở tên cột.
- Kiểm tra các cột bắt buộc có tồn tại.
- In ra shape ban đầu của dataframe.
- In ra danh sách nhãn có trong `flight_logic_state`.
- In ra số mẫu theo từng nhãn.

### 5.2. Chuẩn hóa nhãn

Tạo mapping nhãn:

```python
LABEL_MAPPING = {
    "Idle-Hover": "IDLE_HOVER",
    "IDLE_HOVER": "IDLE_HOVER",
    "Ascend": "ASCEND",
    "ASCEND": "ASCEND",
    "Turn": "TURN",
    "TURN": "TURN",
    "HMSL": "HMSL",
    "Descend": "DESCEND",
    "DESCEND": "DESCEND",
}
```

Cột label chính:

```text
flight_logic_state
```

Sau khi chuẩn hóa, tạo cột mới:

```text
label
```

Chỉ giữ các dòng có label thuộc 5 class:

```text
IDLE_HOVER, ASCEND, TURN, HMSL, DESCEND
```

### 5.3. Parse cột tuple

Hai cột cần parse:

```text
escSpeed
escVoltage
```

Yêu cầu:

- Nếu cột tồn tại, parse chuỗi tuple sang list số.
- Mỗi tuple thường có 4 phần tử.
- Tạo cột mới:

```text
escSpeed_1, escSpeed_2, escSpeed_3, escSpeed_4
escVoltage_1, escVoltage_2, escVoltage_3, escVoltage_4
```

- Nếu parse lỗi, thay bằng NaN rồi xử lý missing value sau.
- Sau khi parse, không dùng cột gốc `escSpeed`, `escVoltage` làm feature.

### 5.4. Xử lý missing value

Yêu cầu:

- Chuyển toàn bộ feature sang numeric.
- Với missing value ít, dùng forward-fill theo từng `flight`, sau đó backward-fill theo từng `flight`.
- Nếu vẫn còn NaN, dùng median của tập train sau khi split.
- Không được dùng thống kê toàn bộ dataset để fit scaler hoặc imputing trước khi chia train/test.

### 5.5. Sắp xếp dữ liệu

Cần sort theo thứ tự thời gian trong từng chuyến bay:

```text
flight, uid, timestamp, seq
```

Nếu thiếu `flight`, dùng `uid` làm group id. Nếu thiếu cả hai, dùng toàn bộ dataset như một sequence, nhưng phải cảnh báo trong log.

---

## 6. Tạo cửa sổ thời gian

### 6.1. Sampling rate

Dataset này được xem như dữ liệu time-series log khoảng 10 Hz. Vì vậy:

```text
window_size = 10 timestep
```

Tương ứng một cửa sổ khoảng 1 giây.

Nếu sau khi kiểm tra timestamp thấy sampling rate khác, coding agent cần cho phép cấu hình:

```python
WINDOW_SIZE = 10
STRIDE = 5
```

### 6.2. Windowing

Tạo sliding window theo từng `flight`.

Không tạo window băng qua ranh giới giữa hai flight khác nhau.

Input shape sau windowing:

```text
X.shape = (num_windows, window_size, num_features)
y.shape = (num_windows,)
```

Ví dụ nếu dùng `window_size = 10` và có 32 feature:

```text
X.shape = (N, 10, 32)
```

### 6.3. Gán nhãn cho window

Mỗi window có nhiều timestep. Gán nhãn bằng majority voting:

```text
window_label = label xuất hiện nhiều nhất trong window
```

Để tránh nhiễu ở đoạn chuyển trạng thái, thêm điều kiện purity:

```text
purity = số timestep thuộc majority label / window_size
```

Mặc định:

```python
MIN_LABEL_PURITY = 0.8
```

Nếu `purity < 0.8`, bỏ window đó.

### 6.4. Train/validation/test split

Không split ngẫu nhiên theo từng dòng hoặc từng window nếu có thể tránh, vì dễ gây leakage do các window gần nhau rất giống nhau.

Ưu tiên split theo `flight`:

```text
Train: 70%
Validation: 15%
Test: 15%
```

Yêu cầu:

- Cùng một `flight` không được xuất hiện ở cả train và test.
- Nếu số flight ít, dùng `GroupShuffleSplit` hoặc tự chia theo danh sách flight.
- Cố gắng giữ phân phối class tương đối cân bằng giữa các tập.
- Lưu danh sách flight thuộc train/val/test vào `metadata.json`.

### 6.5. Chuẩn hóa feature

Dùng `StandardScaler`:

- Fit scaler chỉ trên tập train.
- Transform train/val/test bằng scaler đã fit.
- Vì input là 3D `(samples, timesteps, features)`, reshape tạm về 2D để fit/transform:

```python
X_train_2d = X_train.reshape(-1, num_features)
```

Sau khi transform, reshape lại 3D.

Lưu scaler vào:

```text
data/processed/scaler.pkl
```

Lưu label encoder vào:

```text
data/processed/label_encoder.pkl
```

---

## 7. Feature set đề xuất

### 7.1. Feature set mặc định

Feature set mặc định nên dùng:

```python
FEATURE_COLUMNS = [
    "wind_speed", "wind_angle",
    "battery_voltage", "battery_current",
    "altitude",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "velocity_x", "velocity_y", "velocity_z",
    "angular_x", "angular_y", "angular_z",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    "payload",
    "d_roll", "d_pitch", "d_yaw",
    "heading", "yaw",
    "kalmanZ",
    "escSpeed_1", "escSpeed_2", "escSpeed_3", "escSpeed_4",
    "escVoltage_1", "escVoltage_2", "escVoltage_3", "escVoltage_4",
]
```

Nếu một số cột không tồn tại, coding agent được phép bỏ qua nhưng phải ghi rõ trong log.

### 7.2. Feature set bám sát đề bài gốc

Để có thêm thí nghiệm phụ, có thể tạo feature set tối giản:

```python
CORE_FEATURE_COLUMNS = [
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    "angular_x", "angular_y", "angular_z",
    "velocity_x", "velocity_y", "velocity_z",
    "altitude",
    "kalmanZ",
]
```

Feature set này gần với yêu cầu ban đầu: Accel, Gyro, GPS speed/velocity và altitude.

---

## 8. Model 1: TinyTCN

### 8.1. Mục tiêu

TinyTCN dùng convolution 1D theo trục thời gian để học pattern động học của UAV trong cửa sổ ngắn. Mô hình cần nhỏ, dễ export sang TFLite và phù hợp hướng TinyML.

### 8.2. Input/Output

Input:

```text
(batch_size, window_size, num_features)
```

Output:

```text
(batch_size, 5)
```

Activation cuối:

```text
softmax
```

Loss:

```text
sparse_categorical_crossentropy
```

### 8.3. Kiến trúc đề xuất

Triển khai trong `src/models/tinytcn.py`.

Kiến trúc gợi ý:

```text
Input
→ Conv1D(filters=32, kernel_size=3, padding="causal", dilation_rate=1)
→ BatchNorm
→ ReLU
→ Dropout(0.1)
→ TCN residual block dilation=1
→ TCN residual block dilation=2
→ TCN residual block dilation=4
→ GlobalAveragePooling1D
→ Dense(32, activation="relu")
→ Dropout(0.1)
→ Dense(5, activation="softmax")
```

Mỗi TCN residual block:

```text
x_in
→ Conv1D(filters=32, kernel_size=3, padding="causal", dilation_rate=d)
→ BatchNorm
→ ReLU
→ Dropout(0.1)
→ Conv1D(filters=32, kernel_size=3, padding="causal", dilation_rate=d)
→ BatchNorm
→ Add residual
→ ReLU
```

Nếu số channel của residual không khớp, dùng `Conv1D(1x1)` để match dimension.

### 8.4. Ràng buộc TinyML

- Tổng số tham số nên giữ ở mức nhỏ, ưu tiên dưới khoảng 100k parameters.
- Không dùng layer khó convert sang TFLite.
- Ưu tiên Conv1D, BatchNorm, ReLU, GlobalAveragePooling, Dense.
- Có thể thử giảm filters từ 32 xuống 16 nếu model quá lớn.

---

## 9. Model 2: DS-CNN

### 9.1. Mục tiêu

DS-CNN sử dụng depthwise separable convolution để giảm số tham số và chi phí tính toán so với CNN thông thường. Đây là kiến trúc rất phù hợp cho TinyML.

### 9.2. Input/Output

Input:

```text
(batch_size, window_size, num_features)
```

Có hai cách triển khai:

Cách A — dùng `SeparableConv1D` trực tiếp trên time-series:

```text
(batch_size, timesteps, features)
```

Cách B — reshape thành dạng 2D:

```text
(batch_size, timesteps, features, 1)
```

Rồi dùng `DepthwiseConv2D` và `Conv2D 1x1`.

Để đơn giản và dễ export, ưu tiên Cách A với `SeparableConv1D`.

### 9.3. Kiến trúc đề xuất

Triển khai trong `src/models/dscnn.py`.

```text
Input
→ Conv1D(filters=32, kernel_size=3, padding="same")
→ BatchNorm
→ ReLU
→ SeparableConv1D(filters=32, kernel_size=3, padding="same")
→ BatchNorm
→ ReLU
→ SeparableConv1D(filters=64, kernel_size=3, padding="same")
→ BatchNorm
→ ReLU
→ SeparableConv1D(filters=64, kernel_size=3, padding="same")
→ BatchNorm
→ ReLU
→ GlobalAveragePooling1D
→ Dense(32, activation="relu")
→ Dropout(0.1)
→ Dense(5, activation="softmax")
```

### 9.4. Ràng buộc TinyML

- Ưu tiên `SeparableConv1D` để giảm tham số.
- Nếu TFLite int8 gặp lỗi với `SeparableConv1D`, có thể chuyển sang `DepthwiseConv2D + Conv2D 1x1`.
- Giữ số filters nhỏ: 16, 32 hoặc 64.
- Không dùng attention hoặc recurrent layer trong phiên bản chính.

---

## 10. Huấn luyện

### 10.1. Cấu hình chung

Tạo file `src/config.py`:

```python
RANDOM_SEED = 42
WINDOW_SIZE = 10
STRIDE = 5
MIN_LABEL_PURITY = 0.8
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_CLASSES = 5
```

### 10.2. Optimizer và callback

Dùng:

```text
Adam learning_rate=1e-3
```

Callbacks:

```text
EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
ModelCheckpoint(save_best_only=True)
```

### 10.3. Class imbalance

Cần kiểm tra class distribution. Nếu lệch lớp lớn, dùng class weight:

```python
sklearn.utils.class_weight.compute_class_weight
```

Lưu class distribution vào:

```text
data/reports/class_distribution.csv
```

### 10.4. CLI train

`src/train.py` cần hỗ trợ:

```bash
python -m src.train --model tinytcn --data data/raw/drone_extended_ekf_fusion.csv
python -m src.train --model dscnn --data data/raw/drone_extended_ekf_fusion.csv
```

Options:

```text
--window-size
--stride
--feature-set default/core
--epochs
--batch-size
--output-dir
```

---

## 11. Đánh giá

### 11.1. Metrics bắt buộc

Trên test set, tính:

```text
Accuracy
Macro Precision
Macro Recall
Macro F1-score
Weighted F1-score
Per-class Precision/Recall/F1
Confusion matrix
```

Lưu vào:

```text
outputs/<model_name>/metrics.json
outputs/<model_name>/classification_report.txt
outputs/<model_name>/confusion_matrix.png
```

### 11.2. So sánh TinyML

Cần báo cáo:

```text
model_name
num_parameters
keras_model_size_kb
tflite_float32_size_kb
tflite_int8_size_kb
test_accuracy
macro_f1
avg_inference_time_ms
```

Lưu thành:

```text
outputs/comparison.csv
```

---

## 12. Export TensorFlow Lite

Tạo script:

```text
src/export_tflite.py
```

Yêu cầu export hai bản:

### 12.1. Float32 TFLite

```text
outputs/<model_name>/model_float32.tflite
```

### 12.2. Int8 TFLite

Dùng representative dataset lấy từ một phần X_train.

```text
outputs/<model_name>/model_int8.tflite
```

Yêu cầu:

- Nếu int8 export thành công, benchmark cả float32 và int8.
- Nếu int8 export lỗi, vẫn giữ float32 và ghi lỗi vào `export_log.txt`.

---

## 13. Benchmark inference

Tạo script:

```text
src/benchmark.py
```

Yêu cầu:

- Load `.tflite` model.
- Chạy inference trên ít nhất 500 sample test hoặc toàn bộ test nếu ít hơn.
- Bỏ qua một số lần warm-up đầu.
- Tính thời gian inference trung bình trên một window.

Kết quả lưu vào:

```text
outputs/<model_name>/benchmark.json
```

Nội dung:

```json
{
  "model": "tinytcn",
  "tflite_type": "int8",
  "num_samples": 500,
  "avg_inference_time_ms": 0.0,
  "model_size_kb": 0.0
}
```

---

## 14. Acceptance criteria

Coding agent chỉ được xem là hoàn thành khi đáp ứng đủ các tiêu chí sau:

1. Đọc được file CSV dataset.
2. Nhận diện được 5 nhãn:
   - `IDLE_HOVER`
   - `ASCEND`
   - `TURN`
   - `HMSL`
   - `DESCEND`
3. Parse được `escSpeed` và `escVoltage` thành các cột số.
4. Tạo được sliding windows không vượt qua ranh giới giữa các flight.
5. Split dữ liệu theo `flight` để giảm leakage.
6. Fit scaler chỉ trên train set.
7. Train được TinyTCN.
8. Train được DS-CNN.
9. Xuất được metrics cho từng model.
10. Xuất được confusion matrix.
11. Export được ít nhất TFLite float32 cho mỗi model.
12. Nếu có thể, export thêm TFLite int8.
13. Tạo được `outputs/comparison.csv` so sánh TinyTCN và DS-CNN.
14. README có hướng dẫn chạy từ đầu đến cuối.

---

## 15. Các lỗi cần tránh

- Không đưa `flight_logic_state`, `IDLE_HOVER`, `ASCEND`, `TURN`, `HMSL`, `DESCEND` vào input feature.
- Không split random theo từng dòng vì sẽ gây leakage rất nặng.
- Không fit scaler trên toàn bộ dataset trước khi chia train/test.
- Không tạo window băng qua hai flight khác nhau.
- Không dùng `seq`, `timestamp`, `uid`, `flight` làm feature trong model chính.
- Không báo cáo mỗi Accuracy nếu dataset mất cân bằng; bắt buộc có Macro F1.
- Không tự thêm class `Landing` vì dataset không có nhãn Landing chính thức.
- Không dùng model quá lớn nếu mục tiêu là TinyML.

---

## 16. Gợi ý nội dung README

README cần có các phần:

```text
1. Project overview
2. Dataset description
3. Output classes
4. Feature columns
5. Preprocessing pipeline
6. Windowing strategy
7. Train/validation/test split
8. TinyTCN architecture
9. DS-CNN architecture
10. Training commands
11. Evaluation commands
12. TFLite export commands
13. Result table format
14. Notes and limitations
```

---

## 17. Lệnh chạy kỳ vọng

Sau khi coding agent hoàn thành, người dùng có thể chạy:

```bash
pip install -r requirements.txt
```

Chuẩn bị dữ liệu:

```bash
python -m src.preprocessing --data data/raw/drone_extended_ekf_fusion.csv --feature-set default
```

Train TinyTCN:

```bash
python -m src.train --model tinytcn --feature-set default
```

Train DS-CNN:

```bash
python -m src.train --model dscnn --feature-set default
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

Tạo bảng so sánh:

```bash
python -m src.evaluate --compare
```

---

## 18. Kết quả mong đợi

Kết quả cuối cùng cần có bảng so sánh như sau:

| Model | Accuracy | Macro F1 | Params | Keras size KB | TFLite float32 KB | TFLite int8 KB | Inference ms/window |
|---|---:|---:|---:|---:|---:|---:|---:|
| TinyTCN | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DS-CNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Ngoài ra cần có confusion matrix cho từng model để xem class nào dễ nhầm nhất, đặc biệt là các cặp có khả năng gần nhau như:

```text
IDLE_HOVER vs HMSL
ASCEND vs DESCEND
TURN vs HMSL
```

---

## 19. Hướng mở rộng nếu còn thời gian

Nếu hoàn thành pipeline chính, có thể bổ sung:

1. So sánh feature set `default` và `core`.
2. Thử window size 10, 20, 30 timestep.
3. Thử stride 1, 5, 10.
4. Thử giảm số filter để tối ưu Flash/RAM.
5. So sánh float32 và int8 accuracy drop.
6. Xuất header `.h` từ TFLite int8 để chuẩn bị deploy ESP32/STM32F4.
7. Viết script inference demo trên một file CSV nhỏ.

---

## 20. Tóm tắt cho coding agent

Hãy xây dựng một project Python/TensorFlow hoàn chỉnh để phân loại trạng thái bay UAV từ CSV của dataset Drone onboard extended EKF fusion 2026. Dataset có nhãn trực tiếp trong cột `flight_logic_state` và các cột one-hot `IDLE_HOVER`, `ASCEND`, `TURN`, `HMSL`, `DESCEND`. Không cần tìm file annotation riêng. Nhiệm vụ chính là tiền xử lý dữ liệu, parse tuple columns, tạo sliding window theo từng flight, split theo flight, chuẩn hóa feature, huấn luyện hai mô hình TinyTCN và DS-CNN, đánh giá bằng Accuracy/Macro F1/confusion matrix, export TFLite và benchmark inference time/model size cho mục tiêu TinyML.
