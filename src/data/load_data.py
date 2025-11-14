import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from src.feature_engineering.advanced_features import engineer_advanced_features


def load_cmapss_data(dataset_id="FD001", data_dir="CMAPSSData"):
    """Load and preprocess C-MAPSS dataset."""
    data_path = Path(data_dir)
    
    # Define column names
    cols = ['unit', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]
    
    # Load files
    train = pd.read_csv(data_path / f"train_{dataset_id}.txt", sep=r"\s+", header=None, names=cols)
    test = pd.read_csv(data_path / f"test_{dataset_id}.txt", sep=r"\s+", header=None, names=cols)
    rul = pd.read_csv(data_path / f"RUL_{dataset_id}.txt", sep=r"\s+", header=None, names=['rul'])
    
    # Calculate RUL for training data
    train['rul'] = train.groupby('unit')['cycle'].transform('max') - train['cycle']
    
    # Add RUL to test data
    test_rul = test.groupby('unit')['cycle'].transform('max').reset_index(drop=True)
    test['rul'] = rul['rul'].values[test['unit'].values - 1] + (test_rul - test['cycle'])
    
    return train, test


def remove_constant_features(train, test):
    """Remove features with zero variance."""
    feature_cols = [c for c in train.columns if c not in ['unit', 'cycle', 'rul']]
    std = train[feature_cols].std()
    constant_cols = std[std == 0].index.tolist()
    
    if constant_cols:
        train = train.drop(columns=constant_cols)
        test = test.drop(columns=constant_cols)
    
    return train, test, constant_cols


def normalize_data(train, test):
    """Normalize features using MinMaxScaler."""
    feature_cols = [c for c in train.columns if c not in ['unit', 'cycle', 'rul']]
    
    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    
    return train, test, scaler


def prepare_data(dataset_id="FD001", data_dir="CMAPSSData", add_features=False):
    """
    Complete preprocessing pipeline.
    
    Args:
        dataset_id: Dataset identifier (FD001, FD002, etc.)
        data_dir: Directory containing CMAPSS data
        add_features: If True, add advanced engineered features
    """
    train, test = load_cmapss_data(dataset_id, data_dir)
    
    # Add advanced features if requested
    if add_features:
        print("\n=== Engineering Advanced Features ===")
        train, train_new_feats = engineer_advanced_features(train, feature_types=['diff'])
        test, test_new_feats = engineer_advanced_features(test, feature_types=['diff'])
    
    train, test, dropped_cols = remove_constant_features(train, test)
    train, test, scaler = normalize_data(train, test)
    
    return {
        'train': train,
        'test': test,
        'scaler': scaler,
        'dropped_features': dropped_cols
    }
