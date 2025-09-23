"""
Data Splitter for Time Series with Temporal Consistency

Ensures proper train/validation/test splits that respect temporal nature
of turbofan engine data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Split engine data maintaining temporal consistency.
    
    Ensures that train/validation/test splits are done by engines,
    not by time steps, to prevent data leakage.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize the data splitter.
        
        Args:
            train_ratio: Proportion of engines for training
            val_ratio: Proportion of engines for validation
            test_ratio: Proportion of engines for testing
            random_state: Random seed for reproducibility
        """
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
            
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        logger.info(
            f"DataSplitter initialized: train={train_ratio:.1%}, "
            f"val={val_ratio:.1%}, test={test_ratio:.1%}"
        )
    
    def split_by_engines(
        self,
        data: pd.DataFrame,
        id_column: str = 'unit_number'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by engines to maintain temporal consistency.
        
        Args:
            data: DataFrame with engine data
            id_column: Column name containing engine IDs
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Get unique engine IDs
        unique_engines = data[id_column].unique()
        n_engines = len(unique_engines)
        
        logger.info(f"Splitting {n_engines} engines")
        
        # First split: separate test engines
        train_val_engines, test_engines = train_test_split(
            unique_engines,
            test_size=self.test_ratio,
            random_state=self.random_state
        )
        
        # Second split: separate train and validation engines
        val_size_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_engines, val_engines = train_test_split(
            train_val_engines,
            test_size=val_size_adjusted,
            random_state=self.random_state
        )
        
        # Create data splits
        train_data = data[data[id_column].isin(train_engines)].copy()
        val_data = data[data[id_column].isin(val_engines)].copy()
        test_data = data[data[id_column].isin(test_engines)].copy()
        
        # Log split statistics
        logger.info(
            f"Split complete - Train: {len(train_engines)} engines ({len(train_data)} samples), "
            f"Val: {len(val_engines)} engines ({len(val_data)} samples), "
            f"Test: {len(test_engines)} engines ({len(test_data)} samples)"
        )
        
        return train_data, val_data, test_data
    
    def get_split_info(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame, 
        test_data: pd.DataFrame,
        id_column: str = 'unit_number'
    ) -> Dict[str, Dict[str, int]]:
        """
        Get detailed information about the data split.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            id_column: Column name containing engine IDs
            
        Returns:
            Dictionary with split statistics
        """
        def get_stats(data: pd.DataFrame) -> Dict[str, int]:
            return {
                'n_engines': data[id_column].nunique(),
                'n_samples': len(data),
                'avg_cycles_per_engine': len(data) // data[id_column].nunique(),
                'min_cycles': data.groupby(id_column).size().min(),
                'max_cycles': data.groupby(id_column).size().max()
            }
        
        split_info = {
            'train': get_stats(train_data),
            'validation': get_stats(val_data),
            'test': get_stats(test_data)
        }
        
        # Add overall statistics
        total_engines = (split_info['train']['n_engines'] + 
                        split_info['validation']['n_engines'] + 
                        split_info['test']['n_engines'])
        
        split_info['overall'] = {
            'total_engines': total_engines,
            'train_ratio_actual': split_info['train']['n_engines'] / total_engines,
            'val_ratio_actual': split_info['validation']['n_engines'] / total_engines,
            'test_ratio_actual': split_info['test']['n_engines'] / total_engines
        }
        
        return split_info
    
    def validate_split(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        id_column: str = 'unit_number'
    ) -> bool:
        """
        Validate that the split maintains temporal consistency.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            id_column: Column name containing engine IDs
            
        Returns:
            True if split is valid, False otherwise
        """
        # Check for engine overlap
        train_engines = set(train_data[id_column].unique())
        val_engines = set(val_data[id_column].unique())
        test_engines = set(test_data[id_column].unique())
        
        # Validate no overlap between sets
        train_val_overlap = train_engines.intersection(val_engines)
        train_test_overlap = train_engines.intersection(test_engines)
        val_test_overlap = val_engines.intersection(test_engines)
        
        if train_val_overlap:
            logger.error(f"Train-validation overlap: {len(train_val_overlap)} engines")
            return False
            
        if train_test_overlap:
            logger.error(f"Train-test overlap: {len(train_test_overlap)} engines")
            return False
            
        if val_test_overlap:
            logger.error(f"Validation-test overlap: {len(val_test_overlap)} engines")
            return False
        
        # Check that all original engines are accounted for
        total_engines_split = len(train_engines) + len(val_engines) + len(test_engines)
        original_engines = len(pd.concat([train_data, val_data, test_data])[id_column].unique())
        
        if total_engines_split != original_engines:
            logger.error(
                f"Engine count mismatch: {total_engines_split} split vs "
                f"{original_engines} original"
            )
            return False
        
        logger.info("Data split validation passed")
        return True