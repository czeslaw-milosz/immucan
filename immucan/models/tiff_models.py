from tensorflow import keras
from keras import layers


def get_model(train_img_shape: int, val_img_shape: int, use_batch_norm: bool = False):
    inputs = keras.Input(shape=train_img_shape)  # expected img size (300, 300, 40)

    # Entry block
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same")(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same")(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Blocks 1, 2 are identical apart from the feature depth.
    for filters in [64, 128]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Second half of the network: upsampling inputs
    for filters in [128, 64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)

    x = layers.Conv2D(val_img_shape[-1], 3, padding="valid")(x)
    outputs = layers.UpSampling2D(2)(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model
