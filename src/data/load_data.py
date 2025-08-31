import os
import pandas as pd


def _compose_columns(config: dict) -> list:
    return (
        [config["columns"]["id_col"], config["columns"]["time_col"]]
        + config["columns"]["op_settings"]
        + config["columns"]["sensors"]
    )


"""
Robust data loading functions for NASA Turbofan datasets.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

def load_dataset(dataset_id: str = "FD001", data_dir: str = "CMAPSSData") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load NASA C-MAPSS dataset with robust error handling.
    
    Args:
        dataset_id: Dataset identifier (FD001, FD002, FD003, FD004)
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (train_df, test_df, rul_df) or (None, None, None) if error
    """
    try:
        data_path = Path(data_dir)
        
        train_path = data_path / f"train_{dataset_id}.txt"
        test_path = data_path / f"test_{dataset_id}.txt"
        rul_path = data_path / f"RUL_{dataset_id}.txt"
        
        # Check if files exist
        missing_files = []
        for path, name in [(train_path, "train"), (test_path, "test"), (rul_path, "RUL")]:
            if not path.exists():
                missing_files.append(f"{name} file: {path}")
        
        if missing_files:
            print(f"Missing files for {dataset_id}:")
            for file in missing_files:
                print(f"  - {file}")
            return None, None, None
        
        # Load data with proper error handling
        train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
        test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
        rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])
        
        # Basic validation
        if train_df.empty or test_df.empty or rul_df.empty:
            print(f"One or more datasets for {dataset_id} are empty")
            return None, None, None
        
        # Ensure consistent number of columns
        max_cols = max(train_df.shape[1], test_df.shape[1])
        
        # Pad with NaN if needed and drop empty columns
        train_df = train_df.reindex(columns=range(max_cols))
        test_df = test_df.reindex(columns=range(max_cols))
        
        train_df = train_df.dropna(axis=1, how='all')
        test_df = test_df.dropna(axis=1, how='all')
        
        print(f"Successfully loaded {dataset_id}: Train {train_df.shape}, Test {test_df.shape}, RUL {rul_df.shape}")
        
        return train_df, test_df, rul_df
    
    except FileNotFoundError as e:
        print(f"File not found error for {dataset_id}: {e}")
        return None, None, None
    except pd.errors.EmptyDataError as e:
        print(f"Empty data error for {dataset_id}: {e}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error loading {dataset_id}: {e}")
        return None, None, None

def validate_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame, 
                    dataset_id: str) -> bool:
    """
    Validate loaded dataset for basic consistency.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame  
        rul_df: RUL DataFrame
        dataset_id: Dataset identifier
        
    Returns:
        True if dataset passes validation, False otherwise
    """
    try:
        # Check if DataFrames are not None and not empty
        if any(df is None or df.empty for df in [train_df, test_df, rul_df]):
            print(f"Validation failed for {dataset_id}: One or more DataFrames are None or empty")
            return False
        
        # Check minimum expected columns (unit_id, time_cycles, settings, sensors)
        min_expected_cols = 5  # unit_id, time_cycles, 3 settings + sensors
        if train_df.shape[1] < min_expected_cols or test_df.shape[1] < min_expected_cols:
            print(f"Validation failed for {dataset_id}: Insufficient columns")
            return False
        
        # Check if train and test have same number of columns
        if train_df.shape[1] != test_df.shape[1]:
            print(f"Validation warning for {dataset_id}: Train and test have different column counts")
            # This is a warning, not a failure
        
        # Check for reasonable data ranges
        # Unit IDs should be positive integers
        if train_df.iloc[:, 0].min() <= 0 or test_df.iloc[:, 0].min() <= 0:
            print(f"Validation failed for {dataset_id}: Invalid unit IDs")
            return False
        
        # Time cycles should be positive
        if train_df.iloc[:, 1].min() <= 0 or test_df.iloc[:, 1].min() <= 0:
            print(f"Validation failed for {dataset_id}: Invalid time cycles")
            return False
        
        # RUL should be non-negative
        if rul_df['RUL'].min() < 0:
            print(f"Validation failed for {dataset_id}: Negative RUL values")
            return False
        
        # Check if number of test engines matches RUL entries
        n_test_engines = test_df.iloc[:, 0].nunique()
        n_rul_entries = len(rul_df)
        
        if n_test_engines != n_rul_entries:
            print(f"Validation failed for {dataset_id}: Test engines ({n_test_engines}) != RUL entries ({n_rul_entries})")
            return False
        
        print(f"Validation passed for {dataset_id}")
        return True
        
    except Exception as e:
        print(f"Validation error for {dataset_id}: {e}")
        return False
