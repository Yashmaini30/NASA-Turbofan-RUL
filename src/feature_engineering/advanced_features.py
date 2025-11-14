"""Advanced feature engineering for capturing degradation trends."""
import pandas as pd
import numpy as np


def add_first_differences(df, sensor_cols):
    """
    Add first differences (rate of change) for sensor readings.
    Captures how quickly sensors are changing - important for detecting degradation.
    Optimized version using pd.concat to avoid DataFrame fragmentation.
    
    Args:
        df: DataFrame with unit, cycle, and sensor columns
        sensor_cols: List of sensor column names
        
    Returns:
        DataFrame with additional diff features
    """
    diff_features = []
    new_cols = {}
    
    for col in sensor_cols:
        diff_col = f'{col}_diff'
        new_cols[diff_col] = df.groupby('unit')[col].diff().fillna(0)
        diff_features.append(diff_col)
    
    # Concatenate all new columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    return df, diff_features


def add_rolling_statistics(df, sensor_cols, windows=[5, 10]):
    """
    Add rolling mean and std to capture local trends and variability.
    Optimized version using pd.concat to avoid DataFrame fragmentation.
    
    Args:
        df: DataFrame with unit, cycle, and sensor columns
        sensor_cols: List of sensor column names
        windows: List of window sizes for rolling statistics
        
    Returns:
        DataFrame with additional rolling stat features
    """
    rolling_features = []
    new_cols = {}
    
    for window in windows:
        for col in sensor_cols:
            # Rolling mean - captures local trend
            mean_col = f'{col}_rmean_{window}'
            new_cols[mean_col] = df.groupby('unit')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            rolling_features.append(mean_col)
            
            # Rolling std - captures increasing variability near failure
            std_col = f'{col}_rstd_{window}'
            new_cols[std_col] = df.groupby('unit')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            ).fillna(0)
            rolling_features.append(std_col)
    
    # Concatenate all new columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    return df, rolling_features


def add_cumulative_features(df, sensor_cols):
    """
    Add cumulative sum to capture total degradation accumulation.
    
    Args:
        df: DataFrame with unit, cycle, and sensor columns
        sensor_cols: List of sensor column names
        
    Returns:
        DataFrame with additional cumulative features
    """
    cumsum_features = []
    
    for col in sensor_cols:
        cumsum_col = f'{col}_cumsum'
        df[cumsum_col] = df.groupby('unit')[col].cumsum()
        cumsum_features.append(cumsum_col)
    
    return df, cumsum_features


def engineer_advanced_features(df, feature_types=['diff', 'rolling']):
    """
    Apply all advanced feature engineering transformations.
    
    Args:
        df: DataFrame with unit, cycle, and sensor/operational setting columns
        feature_types: List of feature types to add ('diff', 'rolling', 'cumsum')
        
    Returns:
        DataFrame with engineered features and list of new feature names
    """
    # Identify sensor and operational setting columns
    sensor_cols = [c for c in df.columns if c.startswith('s_') or c.startswith('op_')]
    new_features = []
    
    df_enhanced = df.copy()
    
    if 'diff' in feature_types:
        df_enhanced, diff_feats = add_first_differences(df_enhanced, sensor_cols)
        new_features.extend(diff_feats)
        print(f"Added {len(diff_feats)} first difference features")
    
    if 'rolling' in feature_types:
        df_enhanced, roll_feats = add_rolling_statistics(df_enhanced, sensor_cols, windows=[5, 10])
        new_features.extend(roll_feats)
        print(f"Added {len(roll_feats)} rolling statistics features")
    
    if 'cumsum' in feature_types:
        df_enhanced, cum_feats = add_cumulative_features(df_enhanced, sensor_cols)
        new_features.extend(cum_feats)
        print(f"Added {len(cum_feats)} cumulative features")
    
    print(f"Total features: {len(df.columns)} → {len(df_enhanced.columns)} (+{len(new_features)})")
    
    return df_enhanced, new_features
