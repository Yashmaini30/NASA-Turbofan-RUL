"""
Feature Engineering Package for NASA Turbofan RUL Prediction

This package provides tools for transforming raw sensor data into ML-ready features:
- Sequence generation for time series models
- Data normalization and scaling
- Train/validation/test splits with temporal consistency
"""

from .sequence_generator import SequenceGenerator
from .data_splitter import DataSplitter
from .normalizer import DataNormalizer

__all__ = [
    'SequenceGenerator',
    'DataSplitter', 
    'DataNormalizer'
]

__version__ = '1.0.0'