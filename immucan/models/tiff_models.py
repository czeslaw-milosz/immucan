from typing import Tuple

from tensorflow import keras
from keras import layers


def get_model(train_img_shape: Tuple[int, int, int], val_img_shape: Tuple[int, int, int], use_batch_norm: bool = False,
              layer_sizes: Tuple[int] = (64, 128), residual_connections: bool = False):
    inputs = keras.Input(shape=train_img_shape)  # expected training img shape: (300, 300, n_training_channels)

    # Entry block
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same")(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same")(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    if residual_connections:
        previous_block_activation = x  # Set aside residual

    # Body of the network
    for n_filters in layer_sizes:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(n_filters, 3, padding="same")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(n_filters, 3, padding="same")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        if residual_connections:
            # Project residual
            residual = layers.Conv2D(n_filters, 1, strides=2, padding="same")(previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

    # Second half of the network: upsampling inputs
    for n_filters in reversed(layer_sizes):
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(n_filters, 3, padding="same")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(n_filters, 3, padding="same")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)

        if residual_connections:
            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(n_filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)

    x = layers.Conv2D(val_img_shape[-1], 3, padding="valid")(x)
    outputs = layers.UpSampling2D(2)(x)  # expected output shape: (300, 300, n_predicted_channels)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model
