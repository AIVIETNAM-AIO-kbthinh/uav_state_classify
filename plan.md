# plan.md — TinyML UAV Flight State Classification using Extended EKF Fusion Sensor Dataset

## 1. Mục tiêu dự án

Xây dựng pipeline huấn luyện và đánh giá mô hình TinyML cho bài toán **phân loại trạng thái bay của UAV** dựa trên dữ liệu cảm biến onboard đa phương thức và đặc trưng EKF fusion.

Dataset đã chốt sử dụng là **Drone onboard extended EKF fusion 2026**. Sau khi giải nén, dữ liệu chính nằm trong một file CSV. Dữ liệu trong workspace đã được đổi tên về:

```text
data/raw/drone_extended_ekf_fusion.csv
```

Lưu ý quan trọng sau khi kiểm tra dữ liệu thật: cột `flight_logic_state` trong file hiện tại không đủ tin cậy để làm nhãn chính vì toàn bộ dòng đang có giá trị `Idle-Hover`. Nhãn phục vụ huấn luyện cần được suy ra từ các cột trạng thái:

```text
IDLE_HOVER, ASCEND, TURN, HMSL, DESCEND
```

Các cột này cũng không phải one-hot sạch hoàn toàn, vì có một số dòng bật nhiều nhãn cùng lúc. Vì vậy pipeline phải có bước kiểm tra và xử lý nhãn mơ hồ trước khi tạo window.

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

| Cột label trong dataset | Tên class dùng trong báo cáo | Ý nghĩa |
|---|---|---|
| `IDLE_HOVER` | `Hover` | UAV bay treo hoặc gần như đứng yên |
| `ASCEND` | `Ascending` | UAV đang tăng độ cao |
| `DESCEND` | `Descending` | UAV đang giảm độ cao |
| `TURN` | `Turning` | UAV đang đổi hướng/quay |
| `HMSL` | `Forward flight` | Horizontal Movement Straight Line, UAV bay ngang theo đường tương đối thẳng |

Lưu ý: đề bài ban đầu có thể nhắc đến 6 trạng thái gồm cả `Landing`, nhưng dataset đã chốt không cung cấp nhãn `Landing` riêng. Vì vậy, output chính thức của dự án này là 5 class theo các cột label gốc của dataset. Không tự sinh thêm class `Landing`.

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
│       ├── raw_label_audit.csv
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
- Kiểm tra các cột bắt buộc có tồn tại, đặc biệt là 5 cột label:

```text
IDLE_HOVER, ASCEND, TURN, HMSL, DESCEND
```

- In ra shape ban đầu của dataframe.
- In ra danh sách giá trị có trong `flight_logic_state` để sanity check, nhưng không dùng cột này làm nhãn chính.
- In ra tổng số dòng theo từng cột label.
- In ra số dòng có đúng 1 nhãn bật, số dòng không có nhãn bật, và số dòng bật nhiều nhãn.
- Lưu báo cáo kiểm tra nhãn vào:

```text
data/reports/raw_label_audit.csv
```

### 5.2. Chuẩn hóa nhãn

Không lấy nhãn từ `flight_logic_state`. Với file hiện tại, cột này đang chỉ có một giá trị `Idle-Hover`, nên nếu dùng trực tiếp sẽ làm bài toán 5 lớp biến thành bài toán 1 lớp.

Tạo nhãn từ 5 cột trạng thái:

```python
LABEL_COLUMNS = ["IDLE_HOVER", "ASCEND", "TURN", "HMSL", "DESCEND"]
AMBIGUOUS_LABEL_POLICY = "drop"
```

Với mỗi dòng, tính:

```text
label_sum = IDLE_HOVER + ASCEND + TURN + HMSL + DESCEND
```

Quy tắc mặc định:

- Nếu `label_sum == 1`, tạo `label` bằng tên cột đang bật và đặt `label_status = "single"`.
- Nếu `label_sum == 0`, đánh dấu `label_status = "missing"` và không dùng cho train/eval chính.
- Nếu `label_sum > 1`, đánh dấu `label_status = "ambiguous"` và không dùng cho train/eval chính.

Ghi chú từ audit file hiện tại:

- `flight_logic_state` chỉ có `Idle-Hover`.
- Tổng số dòng là `249342`.
- Có khoảng `25010` dòng multi-hot/ambiguous.
- Các cột label vẫn có đủ tín hiệu cho 5 lớp, nhưng cần lọc/kiểm soát nhãn mơ hồ trước khi đánh giá multiclass.

Lý do chọn policy mặc định `drop`: dự án chính đang định nghĩa là **multiclass classification**, trong khi dòng multi-hot biểu diễn trạng thái chồng lấn hoặc giai đoạn chuyển tiếp. Ép các dòng này về một class bằng priority rule có thể đưa nhiễu nhãn vào train/test.

Nếu muốn làm thí nghiệm phụ, có thể thêm option:

```text
--ambiguous-label-policy priority
```

Khi đó phải ghi rõ priority rule trong metadata và báo cáo, ví dụ:

```text
TURN > ASCEND > DESCEND > HMSL > IDLE_HOVER
```

Nhưng kết quả chính của dự án nên dùng policy `drop` để đánh giá sạch hơn.

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
- Với missing value ít, ưu tiên forward-fill theo từng `flight`.
- Không dùng backward-fill trong pipeline chính nếu mục tiêu là mô phỏng inference onboard/real-time, vì backward-fill dùng thông tin tương lai.
- Nếu vẫn còn NaN, dùng median của tập train sau khi split.
- Không được dùng thống kê toàn bộ dataset để fit scaler hoặc imputing trước khi chia train/test.

### 5.5. Sắp xếp dữ liệu

Cần sort theo thứ tự thời gian trong từng chuyến bay:

```text
flight, timestamp, seq
```

Không dùng `uid` để sort chính vì dữ liệu có các uid synthetic dạng chuỗi. Nếu thiếu `flight`, dùng `uid` làm group id. Nếu thiếu cả hai, dùng toàn bộ dataset như một sequence, nhưng phải cảnh báo trong log.

Tạo thêm các cột metadata để split chống leakage:

```text
is_synthetic = uid chứa chuỗi "_synthetic"
base_flight_id = flight nếu flight thuộc nhóm gốc
base_flight_id = id của flight gốc tương ứng nếu flight thuộc nhóm synthetic
```

Với file hiện tại, dữ liệu có 38 `flight`, trong đó 19 flight đầu là dữ liệu gốc và 19 flight sau là bản synthetic tương ứng. Do đó có thể suy ra tạm:

```text
base_flight_id = flight - 19 với các flight synthetic từ 20 đến 38
```

Nếu dataset thay đổi, không hard-code công thức này một cách mù quáng; cần audit lại số flight, uid synthetic và số dòng theo flight. `base_flight_id` phải được dùng làm group split chính để flight gốc và bản synthetic gần giống nhau luôn nằm trong cùng train/val/test split.

---

## 6. Tạo cửa sổ thời gian

### 6.1. Sampling rate

Dataset này có thể được xem như dữ liệu time-series log khoảng 10 Hz, nhưng `timestamp` trong CSV là timestamp nguyên giây nên có nhiều dòng trùng timestamp. Không nên suy luận sampling rate chỉ bằng hiệu timestamp liên tiếp.

Pipeline chính nên coi mỗi dòng đã sort trong cùng `flight` là một timestep rời rạc. Cấu hình mặc định:

```text
window_size = 10 timestep
```

Tương ứng khoảng 10 mẫu liên tiếp. Chỉ gọi là khoảng 1 giây nếu kiểm tra thực nghiệm cho thấy mật độ mẫu xấp xỉ 10 Hz trong từng flight.

Coding agent cần cho phép cấu hình:

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

Vì có dòng multi-hot/ambiguous trong dữ liệu gốc, không nên xóa các dòng ambiguous trước khi sort/windowing vì có thể làm nối nhầm hai đoạn thời gian rời nhau. Cách mặc định:

- Giữ thứ tự chuỗi gốc trong từng `flight`.
- Tạo window trên chuỗi đã sort.
- Nếu window chứa bất kỳ dòng `label_status != "single"`, bỏ window đó khỏi tập train/val/test chính.
- Với các window còn lại, dùng majority voting trên `label`.

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

Ưu tiên split theo `base_flight_id`, không phải chỉ theo `flight`:

```text
Train: 70%
Validation: 15%
Test: 15%
```

Yêu cầu:

- Cùng một `base_flight_id` không được xuất hiện ở cả train, validation và test.
- Flight gốc và flight synthetic tương ứng phải luôn nằm cùng một split.
- Window vẫn phải được tạo riêng theo từng `flight`; `base_flight_id` chỉ dùng để chia tập.
- Nếu số group ít, ưu tiên `StratifiedGroupKFold` nếu phiên bản scikit-learn hỗ trợ.
- Nếu không có `StratifiedGroupKFold`, dùng thuật toán greedy/tự chia theo histogram class của từng `base_flight_id`.
- Cố gắng giữ phân phối class tương đối cân bằng giữa các tập.
- Lưu danh sách `base_flight_id`, `flight`, và `uid` thuộc train/val/test vào `metadata.json`.

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
LABEL_COLUMNS = ["IDLE_HOVER", "ASCEND", "TURN", "HMSL", "DESCEND"]
AMBIGUOUS_LABEL_POLICY = "drop"
SPLIT_GROUP_COLUMN = "base_flight_id"
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

Cần kiểm tra class distribution sau khi đã xử lý nhãn single-label và tạo window hợp lệ. Nếu lệch lớp lớn, dùng class weight:

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
--ambiguous-label-policy drop/priority
--split-group-column base_flight_id
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
2. Nhận diện được 5 nhãn từ các cột label, không phải từ `flight_logic_state`:
   - `IDLE_HOVER`
   - `ASCEND`
   - `TURN`
   - `HMSL`
   - `DESCEND`
3. Báo cáo được số dòng single-label, missing-label và ambiguous/multi-hot.
4. Parse được `escSpeed` và `escVoltage` thành các cột số.
5. Tạo được `base_flight_id` để ghép flight gốc và flight synthetic tương ứng.
6. Tạo được sliding windows không vượt qua ranh giới giữa các flight.
7. Bỏ window có nhãn mơ hồ theo policy mặc định `drop`.
8. Split dữ liệu theo `base_flight_id` để giảm leakage.
9. Fit scaler chỉ trên train set.
10. Train được TinyTCN.
11. Train được DS-CNN.
12. Xuất được metrics cho từng model.
13. Xuất được confusion matrix.
14. Export được ít nhất TFLite float32 cho mỗi model.
15. Nếu có thể, export thêm TFLite int8.
16. Tạo được `outputs/comparison.csv` so sánh TinyTCN và DS-CNN.
17. README có hướng dẫn chạy từ đầu đến cuối.

---

## 15. Các lỗi cần tránh

- Không đưa `flight_logic_state`, `IDLE_HOVER`, `ASCEND`, `TURN`, `HMSL`, `DESCEND` vào input feature.
- Không dùng `flight_logic_state` làm label chính nếu cột này chỉ có một giá trị.
- Không ép dòng multi-hot thành một class mà không ghi rõ policy và không tách riêng kết quả thí nghiệm phụ.
- Không split random theo từng dòng vì sẽ gây leakage rất nặng.
- Không split flight gốc và flight synthetic tương ứng sang hai tập khác nhau.
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
3. Label audit and ambiguous-label policy
4. Output classes
5. Feature columns
6. Preprocessing pipeline
7. Windowing strategy
8. Train/validation/test split with base_flight_id
9. TinyTCN architecture
10. DS-CNN architecture
11. Training commands
12. Evaluation commands
13. TFLite export commands
14. Result table format
15. Notes and limitations
```

---

## 17. Lệnh chạy kỳ vọng

Sau khi coding agent hoàn thành, người dùng có thể chạy:

```bash
pip install -r requirements.txt
```

Chuẩn bị dữ liệu:

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

Hãy xây dựng một project Python/TensorFlow hoàn chỉnh để phân loại trạng thái bay UAV từ CSV của dataset Drone onboard extended EKF fusion 2026. Dataset không cần file annotation riêng; nhãn chính phải được suy ra từ các cột `IDLE_HOVER`, `ASCEND`, `TURN`, `HMSL`, `DESCEND`. Không dùng `flight_logic_state` làm label chính nếu cột này chỉ có một giá trị. Pipeline phải audit nhãn, đánh dấu và mặc định bỏ các dòng/window multi-hot hoặc missing-label trong kết quả multiclass chính. Nhiệm vụ chính là tiền xử lý dữ liệu, parse tuple columns, tạo sliding window theo từng flight, split theo `base_flight_id` để tránh leakage giữa flight gốc và flight synthetic, chuẩn hóa feature, huấn luyện hai mô hình TinyTCN và DS-CNN, đánh giá bằng Accuracy/Macro F1/confusion matrix, export TFLite và benchmark inference time/model size cho mục tiêu TinyML.
