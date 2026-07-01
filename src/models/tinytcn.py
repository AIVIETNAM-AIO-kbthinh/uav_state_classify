from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    ReLU,
)


def residual_tcn_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float):
    residual = x
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout)(x)
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)
    x = BatchNormalization()(x)

    if residual.shape[-1] != filters:
        residual = Conv1D(filters=filters, kernel_size=1, padding="same")(residual)

    x = Add()([x, residual])
    return ReLU()(x)


def build_tinytcn(
    input_shape: Tuple[int, int],
    num_classes: int,
    filters: int = 32,
    kernel_size: int = 3,
    dropout: float = 0.1,
) -> Model:
    inputs = Input(shape=input_shape, name="sensor_window")
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", dilation_rate=1)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout)(x)

    for dilation in (1, 2, 4):
        x = residual_tcn_block(x, filters, kernel_size, dilation, dropout)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax", name="state")(x)
    return Model(inputs=inputs, outputs=outputs, name="TinyTCN")
