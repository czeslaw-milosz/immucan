CONFIG = {
    'preprocessing_threshold': 100,
    'training_modes': (
        'full_image',
        'central_pixel'
    ),
    'training_channels': ['CD20', 'SMA', 'CD11c', 'CD15', 'CD7', 'HistoneH3',
                          'CD3', 'PD1', 'CD8a', 'FOXP3', 'Ecad', 'Ki67'],
    'predicted_channels': ['CD31', 'Podoplanin', 'CD10', 'CD146', 'Arginase', 'MMP9', 'CXCL8', 'CD204', 'Tim-3',
                           'CD209', 'CD66b', 'CD11b', 'NKG2A', 'CD56', 'BCL-2', 'CXCL13 / BLC / BCA-1', 'Eomes',
                           '4-1BB', 'Tbet', 'GITR / TNFRSF18', 'panCK', 'EGFR', 'p53', 'CD73', 'CD155', 'Gata3',
                           'CD134'],
    'layer_sizes': [64, 128],
}