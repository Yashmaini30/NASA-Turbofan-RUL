"""
Sequence Generator for Time Series Feature Engineering

Transforms variable-length engine sensor data into fixed-length sequences
suitable for training LSTM/GRU models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

from src.config.constants import COLUMN_NAMES, DATASET_NAMES

logger = logging.getLogger(__name__)


class SequenceGenerator:
    """
    Generate time series sequences from engine sensor data.
    
    Transforms raw sensor readings into fixed-length sequences with corresponding
    RUL targets for training deep learning models.
    """
    
    def __init__(
        self, 
        sequence_length: int = 50,
        normalization: str = 'standard',
        sensor_columns: Optional[List[str]] = None
    ):
        """
        Initialize the sequence generator.
        
        Args:
            sequence_length: Number of time steps in each sequence
            normalization: Type of normalization ('standard', 'minmax', or 'none')
            sensor_columns: List of sensor column names to use
        """
        self.sequence_length = sequence_length
        self.normalization = normalization
        self.sensor_columns = sensor_columns or self._get_default_sensor_columns()
        self.scaler = None
        self.is_fitted = False
        
        # Initialize scaler based on normalization type
        if normalization == 'standard':
            self.scaler = StandardScaler()
        elif normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif normalization != 'none':
            raise ValueError(f"Unknown normalization type: {normalization}")
            
        logger.info(
            f"SequenceGenerator initialized: length={sequence_length}, "
            f"normalization={normalization}, sensors={len(self.sensor_columns)}"
        )
    
    def _get_default_sensor_columns(self) -> List[str]:
        """Get default sensor columns from constants."""
        # Use predefined sensor columns from constants
        return [col for col in COLUMN_NAMES if col.startswith('sensor_')]
    
    def fit_normalizer(self, data: pd.DataFrame) -> 'SequenceGenerator':
        """
        Fit the normalizer on training data.
        
        Args:
            data: Training data containing sensor columns
            
        Returns:
            Self for method chaining
        """
        if self.scaler is None:
            logger.info("No normalization applied")
            self.is_fitted = True
            return self
            
        # Extract sensor data for fitting
        sensor_data = data[self.sensor_columns].values
        
        # Fit scaler
        self.scaler.fit(sensor_data)
        self.is_fitted = True
        
        logger.info(f"Normalizer fitted on {len(data)} samples")
        return self
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize sensor data using fitted scaler.
        
        Args:
            data: DataFrame with sensor columns
            
        Returns:
            DataFrame with normalized sensor values
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before use. Call fit_normalizer() first.")
            
        if self.scaler is None:
            return data.copy()
            
        # Create copy to avoid modifying original data
        normalized_data = data.copy()
        
        # Normalize only sensor columns
        normalized_data[self.sensor_columns] = self.scaler.transform(
            data[self.sensor_columns].values
        )
        
        return normalized_data
    
    def create_sequences(
        self, 
        data: pd.DataFrame,
        id_column: str = 'unit_number',
        time_column: str = 'time_in_cycles',
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from engine data.
        
        Args:
            data: DataFrame with engine sensor data
            id_column: Column name for engine ID
            time_column: Column name for time/cycle
            normalize: Whether to apply normalization
            
        Returns:
            Tuple of (sequences, targets, engine_ids)
            - sequences: Array of shape (n_sequences, sequence_length, n_sensors)
            - targets: Array of shape (n_sequences,) with RUL values
            - engine_ids: Array of shape (n_sequences,) with engine IDs
        """
        if normalize and not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit_normalizer() first.")
            
        # Normalize data if requested
        if normalize:
            data = self.normalize_data(data)
        
        sequences = []
        targets = []
        engine_ids = []
        
        # Process each engine separately
        for engine_id in data[id_column].unique():
            engine_data = data[data[id_column] == engine_id].sort_values(time_column)
            
            # Skip engines with insufficient data
            if len(engine_data) < self.sequence_length:
                logger.warning(f"Engine {engine_id} has insufficient data ({len(engine_data)} cycles)")
                continue
            
            # Calculate RUL for each time step
            max_cycle = engine_data[time_column].max()
            engine_data = engine_data.copy()
            engine_data['rul'] = max_cycle - engine_data[time_column]
            
            # Extract sensor values
            sensor_values = engine_data[self.sensor_columns].values
            rul_values = engine_data['rul'].values
            
            # Create sliding window sequences
            for i in range(len(sensor_values) - self.sequence_length + 1):
                # Extract sequence
                sequence = sensor_values[i:i + self.sequence_length]
                
                # Target RUL is at the end of the sequence
                target_rul = rul_values[i + self.sequence_length - 1]
                
                sequences.append(sequence)
                targets.append(target_rul)
                engine_ids.append(engine_id)
        
        # Convert to numpy arrays
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        engine_ids = np.array(engine_ids, dtype=np.int32)
        
        logger.info(
            f"Created {len(sequences)} sequences from {len(data[id_column].unique())} engines"
        )
        
        return sequences, targets, engine_ids
    
    def prepare_test_sequences(
        self,
        test_data: pd.DataFrame, 
        rul_data: pd.DataFrame,
        id_column: str = 'unit_number',
        time_column: str = 'time_in_cycles'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences from test data using true RUL values.
        
        Args:
            test_data: Test sensor data
            rul_data: True RUL values for test engines
            id_column: Column name for engine ID
            time_column: Column name for time/cycle
            
        Returns:
            Tuple of (sequences, targets, engine_ids)
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit_normalizer() first.")
        
        # Normalize test data
        test_data = self.normalize_data(test_data)
        
        sequences = []
        targets = []
        engine_ids = []
        
        # Process each test engine
        for engine_id in test_data[id_column].unique():
            engine_data = test_data[test_data[id_column] == engine_id].sort_values(time_column)
            
            # Skip engines with insufficient data
            if len(engine_data) < self.sequence_length:
                logger.warning(f"Test engine {engine_id} has insufficient data")
                continue
            
            # Get true RUL for this engine
            true_rul = rul_data[rul_data[id_column] == engine_id]['rul'].iloc[0]
            
            # Extract last sequence (most recent sensor readings)
            sensor_values = engine_data[self.sensor_columns].values
            last_sequence = sensor_values[-self.sequence_length:]
            
            sequences.append(last_sequence)
            targets.append(true_rul)
            engine_ids.append(engine_id)
        
        # Convert to numpy arrays
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        engine_ids = np.array(engine_ids, dtype=np.int32)
        
        logger.info(f"Prepared {len(sequences)} test sequences")
        
        return sequences, targets, engine_ids
    
    def get_input_shape(self) -> Tuple[int, int]:
        """
        Get the input shape for model building.
        
        Returns:
            Tuple of (sequence_length, n_features)
        """
        return (self.sequence_length, len(self.sensor_columns))
    
    def save_scaler(self, filepath: str) -> None:
        """Save the fitted scaler to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")
            
        import joblib
        scaler_info = {
            'scaler': self.scaler,
            'normalization': self.normalization,
            'sensor_columns': self.sensor_columns,
            'sequence_length': self.sequence_length
        }
        joblib.dump(scaler_info, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    @classmethod
    def load_scaler(cls, filepath: str) -> 'SequenceGenerator':
        """Load a fitted scaler from disk."""
        import joblib
        scaler_info = joblib.load(filepath)
        
        # Create instance with loaded parameters
        instance = cls(
            sequence_length=scaler_info['sequence_length'],
            normalization=scaler_info['normalization'],
            sensor_columns=scaler_info['sensor_columns']
        )
        
        # Set fitted scaler
        instance.scaler = scaler_info['scaler']
        instance.is_fitted = True
        
        logger.info(f"Scaler loaded from {filepath}")
        return instance