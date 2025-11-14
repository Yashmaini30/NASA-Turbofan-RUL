import numpy as np
import torch
from torch.utils.data import Dataset


class TurbofanDataset(Dataset):
    """PyTorch Dataset for turbofan engine sequences."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(data, sequence_length=30):
    """Create sequences from time series data."""
    feature_cols = [c for c in data.columns if c not in ['unit', 'cycle', 'rul']]
    
    sequences = []
    targets = []
    
    for unit in data['unit'].unique():
        unit_data = data[data['unit'] == unit].sort_values('cycle')
        
        # Skip if insufficient data
        if len(unit_data) < sequence_length:
            continue
            
        values = unit_data[feature_cols].values
        rul_values = unit_data['rul'].values
        
        # Sliding window
        for i in range(len(values) - sequence_length + 1):
            sequences.append(values[i:i + sequence_length])
            targets.append(rul_values[i + sequence_length - 1])
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def prepare_train_data(train, sequence_length=30, val_split=0.2):
    """Prepare training and validation sequences."""
    # Split by units for proper time series validation
    units = train['unit'].unique()
    n_val = int(len(units) * val_split)
    
    np.random.seed(42)
    val_units = np.random.choice(units, n_val, replace=False)
    
    train_df = train[~train['unit'].isin(val_units)]
    val_df = train[train['unit'].isin(val_units)]
    
    X_train, y_train = create_sequences(train_df, sequence_length)
    X_val, y_val = create_sequences(val_df, sequence_length)
    
    # Check for NaNs
    assert not np.isnan(X_train).any(), "NaN found in training sequences"
    assert not np.isnan(y_train).any(), "NaN found in training targets"
    
    return X_train, y_train, X_val, y_val


def prepare_test_data(test, sequence_length=30):
    """Prepare test sequences - use last N cycles per engine."""
    feature_cols = [c for c in test.columns if c not in ['unit', 'cycle', 'rul']]
    
    sequences = []
    targets = []
    
    for unit in test['unit'].unique():
        unit_data = test[test['unit'] == unit].sort_values('cycle')
        
        if len(unit_data) >= sequence_length:
            # Use last sequence_length cycles
            seq = unit_data[feature_cols].values[-sequence_length:]
            sequences.append(seq)
            targets.append(unit_data['rul'].values[-1])
    
    X_test = np.array(sequences, dtype=np.float32)
    y_test = np.array(targets, dtype=np.float32)
    
    # Check for NaNs
    assert not np.isnan(X_test).any(), "NaN found in test sequences"
    assert not np.isnan(y_test).any(), "NaN found in test targets"
    
    return X_test, y_test
