import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from immucan.models.tiff_models import get_model
from immucan.utils import data_utils
from immucan.utils.config import CONFIG

if __name__ == '__main__':
    tf.keras.utils.set_random_seed(CONFIG['random_seed'])
    assert len(tf.config.list_physical_devices('GPU')) == 1  # make sure we don't mess up anyone's computation on rudy
    tf.get_logger().setLevel('ERROR')

    metadata_manager = data_utils.PanelMetadata(CONFIG['metadata_path'])

    training_channels = metadata_manager.get_channel_idx(CONFIG['training_channels'])
    predicted_channels = [i for i in range(len(metadata_manager.marker_names)) if i not in training_channels]

    training_gen = data_utils.TiffSequence("../data/imc_quadrants/training",
                                           training_channels=training_channels,
                                           predicted_channels=predicted_channels,
                                           batch_size=CONFIG['batch_size'])
    validation_gen = data_utils.TiffSequence("../data/imc_quadrants/validation",
                                             training_channels=training_channels,
                                             predicted_channels=predicted_channels,
                                             batch_size=CONFIG['batch_size'])

    img_height, img_width = CONFIG['img_shape']
    tf.keras.backend.clear_session()
    model = get_model(
        train_img_shape=(img_height, img_width, len(training_channels)),
        val_img_shape=(img_height, img_width, len(predicted_channels)),
        use_batch_norm=CONFIG['use_batch_norm'],
        layer_sizes=CONFIG['layer_sizes']
    )
    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(CONFIG['learning_rate'], clipnorm=CONFIG['clip_norm']) \
        if CONFIG['clip_norm'] \
        else tf.keras.optimizers.Adam(CONFIG['learning_rate'])
    model.compile(optimizer=optimizer, loss=CONFIG['loss'])

    start = datetime.datetime.now()
    print(f"STARTED TRAINING: {start}\n")
    history = model.fit(
        training_gen,
        epochs=CONFIG['n_epochs'],
        validation_data=validation_gen,
        max_queue_size=CONFIG['batch_size'],
        use_multiprocessing=True,
        workers=3,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(CONFIG['model_checkpoint_path'], save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=CONFIG['early_stopping_patience']),
        ]
    )
    stop = datetime.datetime.now()
    print(f"STOPPED TRAINING: {stop}")
    print(f"TOTAL TRAINING TIME: {stop - start}")
    print(f"BEST VALIDATION LOSS: {np.min(history.history['val_loss'])}")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f"model {CONFIG['loss']}")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(CONFIG['training_plot_path'])
