from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    ReLU,
    SeparableConv1D,
)


def build_dscnn(
    input_shape: Tuple[int, int],
    num_classes: int,
    dropout: float = 0.1,
) -> Model:
    inputs = Input(shape=input_shape, name="sensor_window")
    x = Conv1D(filters=32, kernel_size=3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for filters in (32, 64, 64):
        x = SeparableConv1D(filters=filters, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax", name="state")(x)
    return Model(inputs=inputs, outputs=outputs, name="DSCNN")
