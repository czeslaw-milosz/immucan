import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tifffile
from steinbock.preprocessing import imc as steinbock_imc

from immucan.utils.config import CONFIG


class TiffSequence(tf.keras.utils.Sequence):

    def __init__(self, img_dir: str, batch_size: int, shuffle: bool = True, y_mode: str = "full_image") -> None:
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert y_mode in CONFIG['training_modes']
        self.y_mode = y_mode

    def __len__(self) -> int:
        """Returns the number of full batches per epoch. One last 'incomplete' batch may not be seen during training."""
        return np.ceil(len(self.img_list) / self.batch_size)
        # return len(self.img_list) % self.batch_size

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        batch_filenames = self.img_list[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_imgs = np.array([
            preprocess_tiff(tifffile.imread(file_name))
            for file_name in batch_filenames])  # (batch_size, n_channels, height, width)
        x, y = (batch_imgs, batch_imgs) if self.y_mode == "full_image" \
            else (batch_imgs, np.array([self._get_central_pixel(subarray) for subarray in batch_imgs]))
        return x, y

    @staticmethod
    def _get_central_pixel(tiff_img: np.ndarray) -> np.ndarray:
        return tiff_img[:, tiff_img.shape[1]//2, tiff_img.shape[2]//2]

    def on_epoch_end(self):
        """Shuffle data after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.img_list)


def preprocess_tiff(tiff_img: np.ndarray) -> np.ndarray:
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
