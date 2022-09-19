from tensorflow import keras
from keras import layers

augmentation_module = keras.Sequential([
    layers.RandomFlip(mode="horizontal_and_vertical"),
    layers.RandomRotation(factor=(-0.5, 0.5))
])
