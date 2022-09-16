from tensorflow import keras
from keras import layers

preprocessing_module = keras.Sequential([
    layers.RandomFlip(mode="horizontal_and_vertical"),
    layers.RandomRotation(factor=(-0.45, 0.45))
])
