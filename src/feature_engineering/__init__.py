"""
Feature Engineering Package for NASA Turbofan RUL Prediction

This package provides tools for transforming raw sensor data into ML-ready features:
- Sequence generation for time series models
- Data normalization and scaling
- Train/validation/test splits with temporal consistency
"""

from .sequences import create_sequences, prepare_train_data, prepare_test_data, TurbofanDataset