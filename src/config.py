from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "drone_extended_ekf_fusion.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RANDOM_SEED = 42
WINDOW_SIZE = 10
STRIDE = 5
MIN_LABEL_PURITY = 0.8

LABEL_COLUMNS = ["IDLE_HOVER", "ASCEND", "TURN", "HMSL", "DESCEND"]
CLASS_DISPLAY_NAMES = {
    "IDLE_HOVER": "Hover",
    "ASCEND": "Ascending",
    "TURN": "Turning",
    "HMSL": "Forward flight",
    "DESCEND": "Descending",
}
PRIORITY_LABEL_ORDER = ["TURN", "ASCEND", "DESCEND", "HMSL", "IDLE_HOVER"]
AMBIGUOUS_LABEL_POLICY = "drop"

SPLIT_GROUP_COLUMN = "base_flight_id"
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_CLASSES = 5

FEATURE_COLUMNS = [
    "wind_speed",
    "wind_angle",
    "battery_voltage",
    "battery_current",
    "altitude",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "orientation_w",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "angular_x",
    "angular_y",
    "angular_z",
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
    "payload",
    "d_roll",
    "d_pitch",
    "d_yaw",
    "heading",
    "yaw",
    "kalmanZ",
    "escSpeed_1",
    "escSpeed_2",
    "escSpeed_3",
    "escSpeed_4",
    "escVoltage_1",
    "escVoltage_2",
    "escVoltage_3",
    "escVoltage_4",
]

CORE_FEATURE_COLUMNS = [
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
    "angular_x",
    "angular_y",
    "angular_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "altitude",
    "kalmanZ",
]

NON_FEATURE_COLUMNS = [
    "seq",
    "uid",
    "timestamp",
    "flight_logic_state",
    "flight",
    *LABEL_COLUMNS,
]
