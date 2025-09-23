"""
Data Normalizer for Sensor Features

Provides various normalization strategies for sensor data preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Normalize sensor data using various strategies.
    
    Supports multiple normalization methods and can handle
    per-engine normalization for better performance.
    """
    
    def __init__(
        self,
        method: str = 'standard',
        per_engine: bool = False,
        sensor_columns: Optional[List[str]] = None
    ):
        """
        Initialize the data normalizer.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust', 'none')
            per_engine: Whether to normalize per engine or globally
            sensor_columns: List of sensor columns to normalize
        """
        self.method = method
        self.per_engine = per_engine
        self.sensor_columns = sensor_columns
        self.scalers = {}
        self.is_fitted = False
        
        # Validate method
        valid_methods = ['standard', 'minmax', 'robust', 'none']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
            
        logger.info(
            f"DataNormalizer initialized: method={method}, "
            f"per_engine={per_engine}"
        )
    
    def _create_scaler(self) -> Union[StandardScaler, MinMaxScaler, RobustScaler, None]:
        """Create a scaler instance based on the method."""
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler()
        elif self.method == 'robust':
            return RobustScaler()
        else:  # method == 'none'
            return None
    
    def fit(
        self,
        data: pd.DataFrame,
        id_column: str = 'unit_number'
    ) -> 'DataNormalizer':
        """
        Fit the normalizer on training data.
        
        Args:
            data: Training data
            id_column: Column name for engine ID (used if per_engine=True)
            
        Returns:
            Self for method chaining
        """
        if self.method == 'none':
            self.is_fitted = True
            logger.info("No normalization applied")
            return self
            
        # Determine sensor columns if not provided
        if self.sensor_columns is None:
            self.sensor_columns = [col for col in data.columns 
                                 if col.startswith('sensor_') or col.startswith('T') or col.startswith('P')]
            
        if not self.sensor_columns:
            raise ValueError("No sensor columns found in data")
            
        if self.per_engine:
            # Fit separate scaler for each engine
            for engine_id in data[id_column].unique():
                engine_data = data[data[id_column] == engine_id]
                
                scaler = self._create_scaler()
                sensor_data = engine_data[self.sensor_columns].values
                
                # Only fit if we have enough data
                if len(sensor_data) > 1:
                    scaler.fit(sensor_data)
                    self.scalers[engine_id] = scaler
                else:
                    logger.warning(f"Engine {engine_id} has insufficient data for fitting")
                    
        else:
            # Fit single global scaler
            scaler = self._create_scaler()
            sensor_data = data[self.sensor_columns].values
            scaler.fit(sensor_data)
            self.scalers['global'] = scaler
            
        self.is_fitted = True
        logger.info(
            f"Normalizer fitted: {len(self.scalers)} scalers, "
            f"{len(self.sensor_columns)} sensor columns"
        )
        
        return self
    
    def transform(
        self,
        data: pd.DataFrame,
        id_column: str = 'unit_number'
    ) -> pd.DataFrame:
        """
        Transform data using fitted normalizer.
        
        Args:
            data: Data to transform
            id_column: Column name for engine ID
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
            
        if self.method == 'none':
            return data.copy()
            
        # Create copy to avoid modifying original
        transformed_data = data.copy()
        
        if self.per_engine:
            # Transform each engine separately
            for engine_id in data[id_column].unique():
                if engine_id not in self.scalers:
                    logger.warning(f"No scaler found for engine {engine_id}, skipping")
                    continue
                    
                engine_mask = transformed_data[id_column] == engine_id
                engine_data = transformed_data.loc[engine_mask, self.sensor_columns]
                
                # Transform using engine-specific scaler
                scaler = self.scalers[engine_id]
                transformed_values = scaler.transform(engine_data.values)
                transformed_data.loc[engine_mask, self.sensor_columns] = transformed_values
                
        else:
            # Transform using global scaler
            scaler = self.scalers['global']
            sensor_data = transformed_data[self.sensor_columns].values
            transformed_values = scaler.transform(sensor_data)
            transformed_data[self.sensor_columns] = transformed_values
            
        return transformed_data
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        id_column: str = 'unit_number'
    ) -> pd.DataFrame:
        """
        Fit normalizer and transform data in one step.
        
        Args:
            data: Data to fit and transform
            id_column: Column name for engine ID
            
        Returns:
            Transformed data
        """
        return self.fit(data, id_column).transform(data, id_column)
    
    def inverse_transform(
        self,
        data: pd.DataFrame,
        id_column: str = 'unit_number'
    ) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data
            id_column: Column name for engine ID
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")
            
        if self.method == 'none':
            return data.copy()
            
        # Create copy to avoid modifying original
        inverse_data = data.copy()
        
        if self.per_engine:
            # Inverse transform each engine separately
            for engine_id in data[id_column].unique():
                if engine_id not in self.scalers:
                    continue
                    
                engine_mask = inverse_data[id_column] == engine_id
                engine_data = inverse_data.loc[engine_mask, self.sensor_columns]
                
                # Inverse transform using engine-specific scaler
                scaler = self.scalers[engine_id]
                original_values = scaler.inverse_transform(engine_data.values)
                inverse_data.loc[engine_mask, self.sensor_columns] = original_values
                
        else:
            # Inverse transform using global scaler
            scaler = self.scalers['global']
            sensor_data = inverse_data[self.sensor_columns].values
            original_values = scaler.inverse_transform(sensor_data)
            inverse_data[self.sensor_columns] = original_values
            
        return inverse_data
    
    def get_normalization_stats(self) -> Dict:
        """
        Get statistics about the fitted normalizers.
        
        Returns:
            Dictionary with normalization statistics
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted first")
            
        stats = {
            'method': self.method,
            'per_engine': self.per_engine,
            'n_scalers': len(self.scalers),
            'sensor_columns': self.sensor_columns,
            'n_sensors': len(self.sensor_columns) if self.sensor_columns else 0
        }
        
        if self.method != 'none' and self.scalers:
            # Get statistics from first scaler (they should be similar)
            first_scaler = list(self.scalers.values())[0]
            
            if hasattr(first_scaler, 'mean_'):
                stats['mean'] = first_scaler.mean_.tolist()
                stats['std'] = first_scaler.scale_.tolist()
            elif hasattr(first_scaler, 'data_min_'):
                stats['min'] = first_scaler.data_min_.tolist()
                stats['max'] = first_scaler.data_max_.tolist()
                stats['scale'] = first_scaler.scale_.tolist()
                
        return stats
    
    def save(self, filepath: str) -> None:
        """Save the fitted normalizer to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer")
            
        import joblib
        normalizer_data = {
            'method': self.method,
            'per_engine': self.per_engine,
            'sensor_columns': self.sensor_columns,
            'scalers': self.scalers,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(normalizer_data, filepath)
        logger.info(f"Normalizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataNormalizer':
        """Load a fitted normalizer from disk."""
        import joblib
        normalizer_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(
            method=normalizer_data['method'],
            per_engine=normalizer_data['per_engine'],
            sensor_columns=normalizer_data['sensor_columns']
        )
        
        # Restore fitted state
        instance.scalers = normalizer_data['scalers']
        instance.is_fitted = normalizer_data['is_fitted']
        
        logger.info(f"Normalizer loaded from {filepath}")
        return instance