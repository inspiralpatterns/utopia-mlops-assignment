from typing import Optional, Union, Tuple
from dataclasses import dataclass, astuple

import tensorflow as tf


@dataclass
class InputShape:
    height: int
    width: int
    n_channels: int


@dataclass
class ConvLayer:
    filters: int
    kernel_size: Union[int, Tuple[int, int]]
    padding: str
    activation: str


def _add_cnn_layer(model, filters, kernel_size, padding=None, activation=None):

    model.add(tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, activation=activation))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(tf.keras.layers.BatchNormalization())

    return model


class CNNModel(tf.keras.Model):

    __slots__ = ("in_shape", "n_class", "model", "model_name", "n_layers", "conv_layer_specs")

    def __init__(self, in_shape: InputShape, n_class: int, n_layers: int, conv_layer_spec: ConvLayer) -> None:
        super().__init__()
        self.in_shape = astuple(in_shape)
        self.conv_layer_specs = conv_layer_spec
        self.n_class = n_class
        self.n_layers = n_layers
        self.model_name = "MNISTAudioCNNmodel"
        self.model = self._model_structure()

    # Create CNN model
    def _model_structure(self) -> tf.keras.Model:
        model = tf.keras.Sequential(name=self.model_name)

        model.add(tf.keras.layers.Input(shape=self.in_shape))

        for _ in range(self.n_layers):
            model = _add_cnn_layer(
                model,
                self.conv_layer_specs.filters,
                self.conv_layer_specs.kernel_size,
                padding=self.conv_layer_specs.padding,
                activation=self.conv_layer_specs.activation
            )

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Dense(self.n_class, activation="softmax"))

        return model

    def compile(self, loss: Optional[str] = None, metrics: Optional[str] = "accuracy") -> tf.keras.Model:
        # Compile model
        self.model.compile(
            loss=loss if loss else "sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=[metrics],
        )

        return self.model
