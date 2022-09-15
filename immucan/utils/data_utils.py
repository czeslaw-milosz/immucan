import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from steinbock.preprocessing import imc as steinbock_imc

from config import CONFIG


def preprocess_image(tiff_img: np.ndarray) -> np.ndarray:
    return np.arcsinh(steinbock_imc.filter_hot_pixels(tiff_img, CONFIG['preprocessing_threshold']) / 5)


def split_into_quadrants(tiff_img: np.ndarray) -> Tuple[np.ndarray, ...]:
    nrows, ncols = tiff_img.shape[1:]  # image is assumed to have shape (n_channels, nrows, ncols)
    row_split, col_split = nrows // 2, ncols // 2
    return (
        tiff_img[:, :row_split, :col_split],
        tiff_img[:, :row_split, col_split:],
        tiff_img[:, row_split:, :col_split],
        tiff_img[:, row_split:, col_split:],
    )
