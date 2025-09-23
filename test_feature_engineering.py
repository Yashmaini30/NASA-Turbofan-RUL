"""
Test script for feature engineering components.

This script validates the functionality of our feature engineering pipeline
and serves as an example of how to use the components.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample turbofan data for testing."""
    np.random.seed(42)
    
    data = []
    for unit in range(1, 6):  # 5 engines
        n_cycles = np.random.randint(150, 300)  # Variable engine life
        
        for cycle in range(1, n_cycles + 1):
            # Simulate degradation over time
            degradation = cycle / n_cycles
            noise = np.random.normal(0, 0.1)
            
            row = {
                'unit_number': unit,
                'time_in_cycles': cycle,
                'setting_1': np.random.normal(0.5, 0.1),
                'setting_2': np.random.normal(0.3, 0.05),
                'setting_3': np.random.normal(100, 5),
                'sensor_1': 500 + degradation * 50 + noise,
                'sensor_2': 1500 - degradation * 100 + noise,
                'sensor_3': 1400 + degradation * 200 + noise,
                'sensor_4': 1350 + degradation * 150 + noise,
                'sensor_5': 14.5 + degradation * 5 + noise,
                'sensor_6': 21.5 - degradation * 3 + noise,
                'sensor_7': 550 + degradation * 80 + noise,
                'sensor_8': 2400 - degradation * 200 + noise,
                'sensor_9': 9000 + degradation * 1000 + noise,
                'sensor_10': 1.3 + degradation * 0.5 + noise,
                'sensor_11': 47 + degradation * 10 + noise,
                'sensor_12': 521 + degradation * 60 + noise,
                'sensor_13': 2400 - degradation * 300 + noise,
                'sensor_14': 8100 + degradation * 900 + noise,
                'sensor_15': 8.5 + degradation * 2 + noise,
                'sensor_16': 0.03 + degradation * 0.01 + noise,
                'sensor_17': 392 + degradation * 40 + noise,
                'sensor_18': 2400 - degradation * 250 + noise,
                'sensor_19': 100 + degradation * 20 + noise,
                'sensor_20': 39 + degradation * 8 + noise,
                'sensor_21': 23.3 + degradation * 5 + noise,
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add RUL column (cycles until failure)
    df['rul'] = 0
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        max_cycle = unit_data['time_in_cycles'].max()
        df.loc[df['unit_number'] == unit, 'rul'] = max_cycle - df.loc[df['unit_number'] == unit, 'time_in_cycles']
    
    return df

def test_data_splitter():
    """Test the DataSplitter functionality."""
    logger.info("Testing DataSplitter...")
    
    from src.feature_engineering import DataSplitter
    
    # Create sample data
    data = create_sample_data()
    logger.info(f"Created sample data: {len(data)} rows, {data['unit_number'].nunique()} engines")
    
    # Initialize splitter
    splitter = DataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Split data
    train_data, val_data, test_data = splitter.split_by_engines(data)
    
    # Validate splits
    logger.info(f"Train: {len(train_data)} rows, {train_data['unit_number'].nunique()} engines")
    logger.info(f"Val: {len(val_data)} rows, {val_data['unit_number'].nunique()} engines")
    logger.info(f"Test: {len(test_data)} rows, {test_data['unit_number'].nunique()} engines")
    
    # Check for overlaps
    train_engines = set(train_data['unit_number'])
    val_engines = set(val_data['unit_number'])
    test_engines = set(test_data['unit_number'])
    
    assert len(train_engines & val_engines) == 0, "Train and validation engines overlap!"
    assert len(train_engines & test_engines) == 0, "Train and test engines overlap!"
    assert len(val_engines & test_engines) == 0, "Validation and test engines overlap!"
    
    logger.info("✓ DataSplitter tests passed!")
    return train_data, val_data, test_data

def test_data_normalizer():
    """Test the DataNormalizer functionality."""
    logger.info("Testing DataNormalizer...")
    
    from src.feature_engineering import DataNormalizer
    
    # Create sample data
    data = create_sample_data()
    
    # Test different normalization methods
    methods = ['standard', 'minmax', 'robust', 'none']
    
    for method in methods:
        logger.info(f"Testing {method} normalization...")
        
        # Test global normalization
        normalizer = DataNormalizer(method=method, per_engine=False)
        normalized_data = normalizer.fit_transform(data)
        
        if method != 'none':
            # Check that sensor columns are normalized
            sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
            for col in sensor_cols:
                original_std = data[col].std()
                normalized_std = normalized_data[col].std()
                if method == 'standard':
                    assert abs(normalized_std - 1.0) < 0.1, f"Standard normalization failed for {col}"
                elif method == 'minmax':
                    assert normalized_data[col].min() >= -0.1, f"MinMax normalization failed for {col}"
                    assert normalized_data[col].max() <= 1.1, f"MinMax normalization failed for {col}"
        
        # Test inverse transform
        if method != 'none':
            inverse_data = normalizer.inverse_transform(normalized_data)
            for col in sensor_cols:
                original_mean = data[col].mean()
                inverse_mean = inverse_data[col].mean()
                assert abs(original_mean - inverse_mean) < 0.1, f"Inverse transform failed for {col}"
        
        # Test per-engine normalization
        normalizer_per_engine = DataNormalizer(method=method, per_engine=True)
        normalized_per_engine = normalizer_per_engine.fit_transform(data)
        
        logger.info(f"✓ {method} normalization tests passed!")
    
    logger.info("✓ DataNormalizer tests passed!")
    return normalized_data

def test_sequence_generator():
    """Test the SequenceGenerator functionality."""
    logger.info("Testing SequenceGenerator...")
    
    from src.feature_engineering import SequenceGenerator
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize generator
    generator = SequenceGenerator(
        sequence_length=30,
        normalization='standard'
    )
    
    # Generate sequences
    generator.fit_normalizer(data)
    X_train, y_train, train_indices = generator.create_sequences(data)
    
    # Create simple RUL data for test
    rul_data = data.groupby('unit_number')['rul'].first().reset_index()
    X_test, y_test, test_indices = generator.prepare_test_sequences(data, rul_data)
    
    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Training targets: {y_train.shape}")
    logger.info(f"Test sequences: {X_test.shape}")
    logger.info(f"Test targets: {y_test.shape}")
    
    # Validate shapes
    assert len(X_train.shape) == 3, "X_train should be 3D (samples, timesteps, features)"
    assert len(y_train.shape) == 1, "y_train should be 1D (samples,)"
    assert X_train.shape[1] == 30, "Sequence length should be 30"
    assert X_train.shape[0] == y_train.shape[0], "X and y should have same number of samples"
    
    # Test scaler persistence
    scaler = generator.scaler
    assert scaler is not None, "Scaler should be available after fitting"
    
    logger.info("✓ SequenceGenerator tests passed!")
    return X_train, y_train, X_test, y_test

def test_integrated_pipeline():
    """Test the complete feature engineering pipeline."""
    logger.info("Testing integrated pipeline...")
    
    from src.feature_engineering import DataSplitter, DataNormalizer, SequenceGenerator
    
    # Create sample data
    data = create_sample_data()
    
    # Step 1: Split data
    splitter = DataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train_data, val_data, test_data = splitter.split_by_engines(data)
    
    # Step 2: Normalize data
    normalizer = DataNormalizer(method='standard', per_engine=False)
    train_normalized = normalizer.fit_transform(train_data)
    val_normalized = normalizer.transform(val_data)
    test_normalized = normalizer.transform(test_data)
    
    # Step 3: Generate sequences
    generator = SequenceGenerator(sequence_length=20, normalization='none')  # Already normalized
    
    # Prepare training sequences
    generator.fit_normalizer(train_normalized)  # Fit on training data
    X_train, y_train, _ = generator.create_sequences(train_normalized)
    
    # Prepare validation sequences
    X_val, y_val, _ = generator.create_sequences(val_normalized)
    
    # Prepare test sequences
    X_test, y_test, _ = generator.create_sequences(test_normalized)
    
    logger.info(f"Final pipeline results:")
    logger.info(f"  Training: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"  Test: X={X_test.shape}, y={y_test.shape}")
    
    # Validate consistency
    assert X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:], "Feature dimensions should match"
    
    logger.info("✓ Integrated pipeline tests passed!")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'normalizer': normalizer,
        'generator': generator
    }

def main():
    """Run all tests."""
    logger.info("Starting feature engineering tests...")
    
    try:
        # Run individual component tests
        train_data, val_data, test_data = test_data_splitter()
        normalized_data = test_data_normalizer()
        X_train, y_train, X_test, y_test = test_sequence_generator()
        
        # Run integrated pipeline test
        pipeline_results = test_integrated_pipeline()
        
        logger.info("🎉 All feature engineering tests passed!")
        logger.info("Feature engineering components are ready for model training.")
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()