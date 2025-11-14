"""
Temporal Fusion Transformer (TFT) module for RUL prediction
"""

from .data_preprocessing import (
    load_cmapss_data,
    remove_constant_features,
    add_linear_rul,
    add_piecewise_rul,
    normalize_sensors,
    prepare_for_tft
)

from .model import create_tft_dataset, create_tft_model

from .evaluation import (
    evaluate_predictions,
    calculate_phm08_score,
    evaluate_model
)

__all__ = [
    'load_cmapss_data',
    'remove_constant_features',
    'add_linear_rul',
    'add_piecewise_rul',
    'normalize_sensors',
    'prepare_for_tft',
    'create_tft_dataset',
    'create_tft_model',
    'evaluate_predictions',
    'calculate_phm08_score',
    'evaluate_model'
]
