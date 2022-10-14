CONFIG = {
    'experiment_name': 'baseline',
    'result_dir': '../../models/reconstruction',
    'metadata_path': '../../../immucan_old/workflow_samples/IMC/csv/10061074/panel_metadata.csv',
    'dataset_dir': '../../data/imc_quadrants',
    'preprocessing_threshold': 100,
    'training_modes': (
        'full_image',
        'central_pixel'
    ),
    'img_shape': (300, 300),
    'training_channels': ['CD20', 'SMA', 'CD11c', 'CD15', 'CD7', 'HistoneH3',
                          'CD3', 'PD1', 'CD8a', 'FOXP3', 'Ecad', 'Ki67'],
    'predicted_channels': ['CD31', 'Podoplanin', 'CD10', 'CD146', 'Arginase', 'MMP9', 'CXCL8', 'CD204', 'Tim-3',
                           'CD209', 'CD66b', 'CD11b', 'NKG2A', 'CD56', 'BCL-2', 'CXCL13 / BLC / BCA-1', 'Eomes',
                           '4-1BB', 'Tbet', 'GITR / TNFRSF18', 'panCK', 'EGFR', 'p53', 'CD73', 'CD155', 'Gata3',
                           'CD134'],
    'loss': 'mse',
    'layer_sizes': (64, 128, 256),
    'batch_size': 32,
    'learning_rate': 0.001,
    'clip_norm': 1.0,
    'use_batch_norm': True,
    'use_residual_connections': False,
    'n_epochs': 100,
    'early_stopping_patience': 1000,
    'save_memory': True,
    'random_seed': 2137,
}
