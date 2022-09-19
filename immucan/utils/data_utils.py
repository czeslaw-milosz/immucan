import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile
from steinbock.preprocessing import imc as steinbock_imc

from immucan.utils.config import CONFIG


class TiffSequence(tf.keras.utils.Sequence):

    def __init__(self, img_dir: str, batch_size: int, shuffle: bool = True, augment: bool = True,
                 y_mode: str = "full_image") -> None:
        self.img_dir = img_dir
        self.img_list = sorted([os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.augmentations = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
        ]) if augment else None
        assert y_mode in CONFIG['training_modes']
        self.y_mode = y_mode

    def __len__(self) -> int:
        """Returns the number of full batches per epoch. One last 'incomplete' batch may not be seen during training."""
        return np.ceil(len(self.img_list) / self.batch_size).astype(int)
        # return len(self.img_list) % self.batch_size

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """Output shape: (batch_size, height, width, n_channels)."""
        batch_filenames = self.img_list[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_imgs = np.array([
            np.moveaxis(preprocess_tiff(tifffile.imread(file_name)), 0, 2)
            for file_name in batch_filenames])  # (batch_size, height, width, n_channels)
        if self.augment:
            batch_imgs = np.array([
                self.augmentations(image=image)['image']
                for image in batch_imgs
            ])
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


@dataclass(init=False)
class PanelMetadata:
    metadata_filename: str
    marker_names: List[str]
    index_to_marker_name: Dict[int, str]
    marker_name_to_index: Dict[str, int]

    def __init__(self, metadata_filename: str) -> None:
        reference_panel = pd.read_csv(metadata_filename)
        self.index_to_marker_name = {
            index: marker
            for index, marker in enumerate(reference_panel[reference_panel.columns[0]])
        }
        self.marker_name_to_index = {
            value: key for key, value in self.index_to_marker_name.items()
        }
        self.marker_names = list(self.marker_name_to_index.keys())


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
