"""
Data preprocessing pipeline for TFT model
Handles C-MAPSS dataset loading, feature engineering, and preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path


def load_cmapss_data(dataset='FD001', data_path='./CMAPSSData/'):
    """
    Load C-MAPSS dataset
    
    Args:
        dataset: Dataset name (FD001, FD002, FD003, FD004)
        data_path: Path to CMAPSSData directory
        
    Returns:
        train_df: Training dataframe
        test_df: Test dataframe
        rul_df: Ground truth RUL dataframe for test set
    """
    # Column names
    index_names = ['unit_id', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f'sensor_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    # Load data
    train_df = pd.read_csv(
        f'{data_path}train_{dataset}.txt',
        sep=r'\s+',
        header=None,
        names=col_names
    )
    
    test_df = pd.read_csv(
        f'{data_path}test_{dataset}.txt',
        sep=r'\s+',
        header=None,
        names=col_names
    )
    
    # Load ground truth RUL for test set
    rul_df = pd.read_csv(
        f'{data_path}RUL_{dataset}.txt',
        sep=r'\s+',
        header=None,
        names=['true_rul']
    )
    rul_df['unit_id'] = rul_df.index + 1
    
    return train_df, test_df, rul_df


def remove_constant_features(df):
    """
    Remove sensors with no variation
    
    Based on research, certain sensors show minimal variation:
    - setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with constant features removed
    """
    columns_to_drop = [
        'setting_3',
        'sensor_1', 'sensor_5', 'sensor_10', 
        'sensor_16', 'sensor_18', 'sensor_19'
    ]
    
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=columns_to_drop)


def add_linear_rul(df):
    """
    Add linear RUL calculation
    
    Linear RUL: RUL = max_cycle - current_cycle
    - At first cycle: RUL = max_cycle - 1
    - At last cycle: RUL = 0
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with RUL column added
    """
    df_rul = df.copy()
    
    # Calculate max cycle per engine
    max_cycles = df_rul.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    df_rul = df_rul.merge(max_cycles, on='unit_id', how='left')
    df_rul['RUL'] = (df_rul['max_cycle'] - df_rul['time_cycles']).astype(float)
    df_rul.drop('max_cycle', axis=1, inplace=True)
    
    return df_rul


def add_piecewise_rul(df, early_rul=125):
    """
    Add piecewise linear RUL with early RUL threshold
    
    Research shows degradation not visible until ~125 cycles.
    This provides better model training by focusing on degradation phase.
    
    Args:
        df: Input dataframe
        early_rul: Threshold for early RUL clipping (default: 125)
        
    Returns:
        Dataframe with piecewise RUL column added
    """
    df_rul = add_linear_rul(df)
    
    # Clip RUL at early_rul threshold and ensure float type
    df_rul['RUL'] = df_rul['RUL'].clip(upper=early_rul).astype(float)
    
    return df_rul


def normalize_sensors(train_df, test_df, method='minmax'):
    """
    Normalize sensor readings
    Fit on train, transform both train and test
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        method: 'minmax' or 'standard'
        
    Returns:
        train_df: Normalized training dataframe
        test_df: Normalized test dataframe
        scaler: Fitted scaler object
    """
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    setting_cols = [col for col in train_df.columns if col.startswith('setting_')]
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # Fit on training data
    train_df[sensor_cols + setting_cols] = scaler.fit_transform(
        train_df[sensor_cols + setting_cols]
    )
    
    # Transform test data
    test_df[sensor_cols + setting_cols] = scaler.transform(
        test_df[sensor_cols + setting_cols]
    )
    
    return train_df, test_df, scaler


def prepare_for_tft(df, is_train=True):
    """
    Prepare dataframe for TFT TimeSeriesDataSet
    
    Adds required columns:
    - time_idx: Sequential time index starting from 0 for each engine
    - relative_time: Normalized time within sequence [0, 1]
    
    Args:
        df: Input dataframe
        is_train: Whether this is training data
        
    Returns:
        Prepared dataframe for TFT
    """
    df_prepared = df.copy()
    
    # Add time index that starts from 0 for each engine
    df_prepared['time_idx'] = df_prepared.groupby('unit_id').cumcount()
    
    # Add relative position (normalized time within sequence)
    max_time = df_prepared.groupby('unit_id')['time_idx'].transform('max')
    df_prepared['relative_time'] = df_prepared['time_idx'] / max_time
    
    # Convert unit_id to string for categorical handling
    df_prepared['unit_id'] = df_prepared['unit_id'].astype(str)
    
    if not is_train:
        # For test set, we need to predict future RUL
        # Add dummy RUL column (will be replaced by predictions)
        if 'RUL' not in df_prepared.columns:
            df_prepared['RUL'] = 0.0
    
    return df_prepared
