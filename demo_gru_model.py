"""
Real GRU Model Demonstration
============================

This script demonstrates the GRU model working with actual NASA C-MAPSS data.
It shows the complete pipeline from data loading to prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from feature_engineering import SequenceGenerator, DataSplitter, DataNormalizer
from models.gru_model import GRUModel
from data.load_data import load_dataset

def calculate_rul(df):
    """Calculate Remaining Useful Life for training data."""
    df = df.copy()
    df['rul'] = 0
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        max_cycle = unit_data['time_in_cycles'].max()
        df.loc[df['unit_number'] == unit, 'rul'] = max_cycle - df.loc[df['unit_number'] == unit, 'time_in_cycles']
    return df

def load_real_cmapss_data():
    """Load and prepare real NASA C-MAPSS data with proper column mapping."""
    train_data, test_data, rul_data = load_dataset('FD001', 'CMAPSSData')
    
    if train_data is not None:
        # Assign proper column names to match expected format
        # NASA C-MAPSS format: unit_id, time_cycles, setting1-3, sensor1-21
        expected_columns = (
            ['unit_number', 'time_in_cycles'] + 
            [f'setting_{i}' for i in range(1, 4)] + 
            [f'sensor_{i}' for i in range(1, 22)]
        )
        
        # Ensure we have the right number of columns
        if train_data.shape[1] >= len(expected_columns):
            train_data = train_data.iloc[:, :len(expected_columns)]  # Take first 26 columns
            train_data.columns = expected_columns
            
            # Calculate RUL for training data
            train_data['rul'] = 0
            for unit in train_data['unit_number'].unique():
                unit_mask = train_data['unit_number'] == unit
                max_cycle = train_data.loc[unit_mask, 'time_in_cycles'].max()
                train_data.loc[unit_mask, 'rul'] = max_cycle - train_data.loc[unit_mask, 'time_in_cycles']
            
            print(f"✅ Real NASA C-MAPSS data loaded successfully!")
            print(f"   Engines: {train_data['unit_number'].nunique()}")
            print(f"   Total samples: {len(train_data)}")
            print(f"   RUL range: {train_data['rul'].min():.0f} - {train_data['rul'].max():.0f} cycles")
            
            return train_data
        else:
            print(f"❌ Unexpected number of columns: {train_data.shape[1]}, expected at least {len(expected_columns)}")
            
    return None

def demonstrate_gru_pipeline():
    """Complete demonstration of GRU model pipeline."""
    print("🚀 GRU Model Demonstration - NASA C-MAPSS Dataset")
    print("=" * 60)
    
    # 1. Load Real Data
    print("\n📊 Step 1: Loading NASA C-MAPSS Data...")
    train_data = load_real_cmapss_data()
    
    if train_data is None:
        print("📝 Real data loading failed, using synthetic data for demonstration...")
        train_data = create_demo_data()
    else:
        print(f"   Sample from real data:")
        print(f"   Columns: {list(train_data.columns)}")
        sample_lifecycles = train_data.groupby('unit_number')['time_in_cycles'].max().head()
        print(f"   Sample engine lifecycles: {dict(sample_lifecycles)}")
    
    # 2. Data Preprocessing
    print("\n🔧 Step 2: Feature Engineering Pipeline...")
    
    # Split data by engines
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_split, val_split, test_split = splitter.split_by_engines(train_data)
    
    print(f"   Train: {len(train_split)} samples from {train_split['unit_number'].nunique()} engines")
    print(f"   Val:   {len(val_split)} samples from {val_split['unit_number'].nunique()} engines")
    print(f"   Test:  {len(test_split)} samples from {test_split['unit_number'].nunique()} engines")
    
    # Normalize data
    normalizer = DataNormalizer(method='standard', per_engine=False)
    train_norm = normalizer.fit_transform(train_split)
    val_norm = normalizer.transform(val_split)
    
    # Generate sequences
    seq_gen = SequenceGenerator(sequence_length=30, normalization='none')  # Already normalized
    seq_gen.fit_normalizer(train_norm)
    X_train, y_train, _ = seq_gen.create_sequences(train_norm)
    X_val, y_val, _ = seq_gen.create_sequences(val_norm)
    
    print(f"✅ Sequences created:")
    print(f"   Training: {X_train.shape} → {y_train.shape}")
    print(f"   Validation: {X_val.shape} → {y_val.shape}")
    
    # 3. Model Creation
    print("\n🧠 Step 3: Creating GRU Model...")
    
    model_config = {
        'input_dim': X_train.shape[2],   # Number of sensors
        'hidden_dim': 64,                # Memory capacity
        'num_layers': 2,                 # Model depth
        'dropout': 0.2,                  # Regularization
        'bidirectional': False           # Simple unidirectional
    }
    
    model = GRUModel(**model_config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"   Architecture: {model}")
    
    # 4. Training Setup
    print("\n🏋️ Step 4: Training Setup...")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    print(f"✅ Training setup complete")
    print(f"   Optimizer: Adam (lr=0.001)")
    print(f"   Loss: MSE")
    
    # 5. Training Loop (Short Demo)
    print("\n🎯 Step 5: Training Model (Demo - 5 epochs)...")
    
    model.train()
    training_losses = []
    validation_losses = []
    
    for epoch in range(5):
        # Training
        optimizer.zero_grad()
        train_predictions = model(X_train_tensor)
        train_loss = criterion(train_predictions.squeeze(), y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions.squeeze(), y_val_tensor)
        model.train()
        
        training_losses.append(train_loss.item())
        validation_losses.append(val_loss.item())
        
        print(f"   Epoch {epoch+1}/5: Train Loss = {train_loss.item():.2f}, Val Loss = {val_loss.item():.2f}")
    
    # 6. Model Evaluation
    print("\n📈 Step 6: Model Evaluation...")
    
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_val_tensor).squeeze().numpy()
        actual_values = y_val_tensor.numpy()
        
        # Calculate metrics
        mse = np.mean((final_predictions - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(final_predictions - actual_values))
        
        print(f"✅ Final Performance Metrics:")
        print(f"   RMSE: {rmse:.2f} cycles")
        print(f"   MAE:  {mae:.2f} cycles")
        print(f"   MSE:  {mse:.2f}")
        
        # Show some predictions vs actual
        print(f"\n🔍 Sample Predictions vs Actual:")
        for i in range(min(5, len(final_predictions))):
            print(f"   Engine sample {i+1}: Predicted={final_predictions[i]:.1f}, Actual={actual_values[i]:.1f}")
    
    # 7. Model Saving
    print("\n💾 Step 7: Model Persistence...")
    
    model_save_path = "gru_model_demo.pth"
    model.save_model(model_save_path, model_config, {
        'rmse': rmse,
        'mae': mae,
        'training_losses': training_losses,
        'validation_losses': validation_losses
    })
    
    print(f"✅ Model saved to {model_save_path}")
    
    # Test loading
    loaded_model = GRUModel.load_model(model_save_path)
    print(f"✅ Model loaded successfully")
    
    # Cleanup
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        print(f"✅ Cleanup complete")
    
    print("\n🎉 Demonstration Complete!")
    print(f"📊 Summary:")
    print(f"   • Data: {len(train_data)} samples processed")
    print(f"   • Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"   • Performance: {rmse:.2f} RMSE")
    print(f"   • Status: Ready for full training!")
    
    return {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'training_losses': training_losses,
        'validation_losses': validation_losses
    }

def create_demo_data():
    """Create synthetic NASA C-MAPSS style data for demonstration."""
    print("🏭 Creating synthetic turbofan data...")
    
    np.random.seed(42)
    data = []
    
    for engine_id in range(1, 21):  # 20 engines
        lifecycle_length = np.random.randint(150, 300)
        
        for cycle in range(1, lifecycle_length + 1):
            # Simulate degradation
            degradation_factor = cycle / lifecycle_length
            noise = np.random.normal(0, 0.1)
            
            row = {
                'unit_number': engine_id,
                'time_in_cycles': cycle,
                'rul': lifecycle_length - cycle,
                'setting_1': np.random.normal(0.5, 0.1),
                'setting_2': np.random.normal(0.3, 0.05),
                'setting_3': np.random.normal(100, 5),
            }
            
            # Add sensor readings with degradation patterns
            for i in range(1, 22):  # 21 sensors
                base_value = np.random.normal(500, 100)
                degradation_effect = degradation_factor * np.random.normal(50, 20)
                row[f'sensor_{i}'] = base_value + degradation_effect + noise
            
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"✅ Created synthetic data: {len(df)} samples, {df['unit_number'].nunique()} engines")
    return df

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Please run this script from the project root directory")
        print("   Current directory:", os.getcwd())
        print("   Expected: .../NASA-Turbofan-RUL/")
        sys.exit(1)
    
    try:
        results = demonstrate_gru_pipeline()
        print(f"\n✅ All systems working! RMSE: {results['rmse']:.2f}")
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()