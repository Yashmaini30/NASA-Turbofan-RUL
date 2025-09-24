"""
Comprehensive GRU Training Script for NASA C-MAPSS RUL Prediction
================================================================

Single script for training, testing, and evaluation using existing infrastructure.
Combines best techniques from both demo and professional scripts.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.feature_engineering import SequenceGenerator, DataSplitter, DataNormalizer
from src.models.gru_model import GRUModel
from src.data.load_data import load_dataset


class GRUTrainer:
    """Comprehensive GRU trainer with professional techniques."""
    
    def __init__(self, dataset='FD002', sequence_length=30, config=None):
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.config = config or self._default_config()
        self.rul_scaler = None
        self.model = None
        
    def _default_config(self):
        """Default training configuration."""
        return {
            'model': {
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': False
            },
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'batch_size': 32,
                'epochs': 100,
                'patience': 15,
                'scheduler_patience': 5,
                'scheduler_factor': 0.5,
                'gradient_clip': 1.0
            },
            'data': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'normalization': 'standard'
            }
        }
    
    def load_and_prepare_data(self):
        """Load and prepare NASA C-MAPSS data."""
        print(f"🔧 Loading NASA C-MAPSS {self.dataset} dataset...")
        
        # Load raw data
        train_data, test_data, rul_data = load_dataset(self.dataset, 'CMAPSSData')
        
        if train_data is None:
            raise ValueError(f"Failed to load {self.dataset} dataset")
        
        # Assign proper column names
        expected_columns = (
            ['unit_number', 'time_in_cycles'] + 
            [f'setting_{i}' for i in range(1, 4)] + 
            [f'sensor_{i}' for i in range(1, 22)]
        )
        
        # Format training data
        if train_data.shape[1] >= len(expected_columns):
            train_data = train_data.iloc[:, :len(expected_columns)]
            train_data.columns = expected_columns
            
            # Calculate RUL for training data
            train_data['rul'] = 0
            for unit in train_data['unit_number'].unique():
                unit_mask = train_data['unit_number'] == unit
                max_cycle = train_data.loc[unit_mask, 'time_in_cycles'].max()
                train_data.loc[unit_mask, 'rul'] = max_cycle - train_data.loc[unit_mask, 'time_in_cycles']
        
        print(f"✅ Data loaded: {len(train_data)} samples, {train_data['unit_number'].nunique()} engines")
        print(f"   RUL range: {train_data['rul'].min():.0f} - {train_data['rul'].max():.0f} cycles")
        
        return train_data, test_data, rul_data
    
    def preprocess_data(self, train_data):
        """Apply feature engineering pipeline."""
        print("🔧 Applying feature engineering...")
        
        # Split data by engines
        splitter = DataSplitter(
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio']
        )
        train_split, val_split, test_split = splitter.split_by_engines(train_data)
        
        print(f"   Train: {len(train_split)} samples from {train_split['unit_number'].nunique()} engines")
        print(f"   Val:   {len(val_split)} samples from {val_split['unit_number'].nunique()} engines")
        print(f"   Test:  {len(test_split)} samples from {test_split['unit_number'].nunique()} engines")
        
        # Normalize data
        normalizer = DataNormalizer(method=self.config['data']['normalization'], per_engine=False)
        train_norm = normalizer.fit_transform(train_split)
        val_norm = normalizer.transform(val_split)
        
        # Generate sequences
        seq_gen = SequenceGenerator(sequence_length=self.sequence_length, normalization='none')
        seq_gen.fit_normalizer(train_norm)
        X_train, y_train, _ = seq_gen.create_sequences(train_norm)
        X_val, y_val, _ = seq_gen.create_sequences(val_norm)
        
        print(f"✅ Sequences created:")
        print(f"   Training: {X_train.shape} → {y_train.shape}")
        print(f"   Validation: {X_val.shape} → {y_val.shape}")
        
        # Scale RUL targets to [0,1] range (critical for performance)
        self.rul_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train_scaled = self.rul_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.rul_scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        print(f"   Original RUL range: {y_train.min():.1f} - {y_train.max():.1f}")
        print(f"   Scaled RUL range: {y_train_scaled.min():.3f} - {y_train_scaled.max():.3f}")
        
        return X_train, X_val, y_train_scaled, y_val_scaled
    
    def create_model(self, input_dim):
        """Create GRU model with optimized configuration."""
        print("🧠 Creating GRU model...")
        
        model_config = {
            'input_dim': input_dim,
            'hidden_dim': self.config['model']['hidden_dim'],
            'num_layers': self.config['model']['num_layers'],
            'dropout': self.config['model']['dropout'],
            'bidirectional': self.config['model']['bidirectional']
        }
        
        self.model = GRUModel(**model_config)
        num_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"✅ Model created with {num_params:,} parameters")
        print(f"   Architecture: {self.model}")
        
        # Validate model output range
        self._validate_model_output(input_dim)
        
        return self.model
    
    def _validate_model_output(self, input_dim):
        """Quick validation to ensure model outputs are in [0,1] range."""
        print("🔍 Validating model output range...")
        self.model.eval()
        with torch.no_grad():
            # Create sample input: batch=2, seq=sequence_length, features=input_dim
            sample_input = torch.randn(2, self.sequence_length, input_dim)
            output = self.model(sample_input)
            
            min_val = output.min().item()
            max_val = output.max().item()
            
            print(f"   Model output range: {min_val:.3f} - {max_val:.3f}")
            print(f"   Output shape: {output.shape}")
            
            if min_val >= 0 and max_val <= 1:
                print("   ✅ Model output validation passed")
            else:
                print("   ⚠️  Warning: Output not in [0,1] range")
        
        self.model.train()  # Reset to training mode
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """Train the model with professional techniques."""
        print("🎯 Training model with professional techniques...")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['training']['scheduler_factor'],
            patience=self.config['training']['scheduler_patience']
        )
        
        print(f"✅ Training setup complete:")
        print(f"   Optimizer: AdamW (lr={self.config['training']['learning_rate']}, weight_decay={self.config['training']['weight_decay']})")
        print(f"   Scheduler: ReduceLROnPlateau")
        print(f"   Loss: MSE")
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\\n🏋️ Training for up to {self.config['training']['epochs']} epochs...")
        
        for epoch in range(self.config['training']['epochs']):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            
            train_pred = self.model(X_train_tensor)
            train_loss = criterion(train_pred.squeeze(), y_train_tensor)
            
            train_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training']['gradient_clip']
            )
            
            optimizer.step()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor)
                val_loss = criterion(val_pred.squeeze(), y_val_tensor)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'best_{self.dataset}_gru.pth')
            else:
                patience_counter += 1
            
            # Progress logging
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:3d}: Train Loss = {train_loss.item():.4f}, "
                      f"Val Loss = {val_loss.item():.4f}")
            
            # Early stopping
            if patience_counter >= self.config['training']['patience']:
                print(f"   Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(f'best_{self.dataset}_gru.pth'))
        os.remove(f'best_{self.dataset}_gru.pth')  # Cleanup
        
        return train_losses, val_losses
    
    def evaluate_model(self, X_val, y_val):
        """Evaluate the trained model."""
        print("📈 Evaluating model performance...")
        
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            
            # Get predictions in scaled space
            scaled_predictions = self.model(X_val_tensor).squeeze().numpy()
            scaled_actual = y_val_tensor.numpy()
            
            # Convert back to original RUL scale
            final_predictions = self.rul_scaler.inverse_transform(
                scaled_predictions.reshape(-1, 1)
            ).flatten()
            actual_values = self.rul_scaler.inverse_transform(
                scaled_actual.reshape(-1, 1)
            ).flatten()
            
            # Calculate metrics
            mse = np.mean((final_predictions - actual_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(final_predictions - actual_values))
            
            print(f"✅ Model Performance:")
            print(f"   RMSE: {rmse:.2f} cycles")
            print(f"   MAE:  {mae:.2f} cycles")
            print(f"   MSE:  {mse:.2f}")
            
            # Sample predictions
            print(f"\\n🔍 Sample Predictions vs Actual:")
            for i in range(min(5, len(final_predictions))):
                print(f"   Sample {i+1}: Predicted={final_predictions[i]:.1f}, "
                      f"Actual={actual_values[i]:.1f}")
            
            # Prediction analysis
            pred_std = np.std(final_predictions)
            print(f"\\n📊 Prediction Analysis:")
            print(f"   Prediction std: {pred_std:.2f} (good diversity)")
            print(f"   Prediction range: {final_predictions.min():.1f} - {final_predictions.max():.1f}")
            print(f"   Actual range: {actual_values.min():.1f} - {actual_values.max():.1f}")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mse': mse,
                'predictions': final_predictions,
                'actual': actual_values
            }
    
    def save_model(self, filepath=None):
        """Save the trained model and components."""
        if filepath is None:
            filepath = f"trained_{self.dataset}_gru_model.pth"
        
        # Create metadata dictionary
        metadata = {
            'dataset': self.dataset,
            'sequence_length': self.sequence_length,
            'config': self.config,
            'rul_scaler_params': {
                'data_min_': self.rul_scaler.data_min_.tolist(),
                'data_max_': self.rul_scaler.data_max_.tolist(),
                'data_range_': self.rul_scaler.data_range_.tolist(),
                'feature_range': self.rul_scaler.feature_range
            }
        }
        
        # Save using the correct method signature
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_model_info(),
            'metadata': metadata
        }, filepath)
        
        print(f"💾 Model saved to {filepath}")
        return filepath
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        print("🚀 NASA C-MAPSS GRU Training Pipeline")
        print("=" * 60)
        
        try:
            # 1. Load and prepare data
            train_data, test_data, rul_data = self.load_and_prepare_data()
            
            # 2. Preprocess data
            X_train, X_val, y_train, y_val = self.preprocess_data(train_data)
            
            # 3. Create model
            model = self.create_model(X_train.shape[2])
            
            # 4. Train model
            train_losses, val_losses = self.train_model(X_train, X_val, y_train, y_val)
            
            # 5. Evaluate model
            results = self.evaluate_model(X_val, y_val)
            
            # 6. Save model
            model_path = self.save_model()
            
            print("\\n🎉 Training Complete!")
            print(f"📊 Final Results:")
            print(f"   Dataset: {self.dataset}")
            print(f"   RMSE: {results['rmse']:.2f} cycles")
            print(f"   Model saved: {model_path}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None
