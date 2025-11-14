"""LSTM training pipeline - minimal, production-grade implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

from src.data.load_data import prepare_data
from src.feature_engineering.sequences import prepare_train_data, prepare_test_data, TurbofanDataset
from src.models.lstm_model import create_lstm_model


class Trainer:
    """Minimal LSTM trainer with early stopping and checkpointing."""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience'],
            min_lr=config['scheduler']['min_lr']
        )
        
        self.criterion = nn.MSELoss()
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, loader):
        self.model.train()
        losses = []
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(X).squeeze()
            loss = self.criterion(pred, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            losses.append(loss.item())
            
        return np.mean(losses)
    
    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        losses = []
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X).squeeze()
            loss = self.criterion(pred, y)
            losses.append(loss.item())
            
        return np.mean(losses)
    
    def fit(self, train_loader, val_loader, epochs, save_path):
        """Train with early stopping."""
        print(f"\nTraining on {self.device}")
        self.model.summary()
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            # Checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': self.config
                }, save_path)
            else:
                self.patience_counter += 1
            
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.3f} | Val: {val_loss:.3f} | "
                  f"Best: {self.best_loss:.3f} | Patience: {self.patience_counter}/{self.config['early_stopping_patience']}")
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\nBest validation loss: {self.best_loss:.3f}")
        return self.best_loss


def nasa_score(y_true, y_pred):
    """NASA asymmetric scoring function."""
    diff = y_pred - y_true
    score = np.where(diff < 0, 
                     np.exp(-diff/13) - 1,  # Early prediction
                     np.exp(diff/10) - 1)    # Late prediction
    return np.sum(score)


@torch.no_grad()
def evaluate(model, loader, device='cuda'):
    """Evaluate model performance."""
    model.eval()
    predictions, actuals = [], []
    
    for X, y in loader:
        X = X.to(device)
        pred = model(X).squeeze().cpu().numpy()
        predictions.extend(pred.tolist() if pred.ndim > 0 else [pred.item()])
        actuals.extend(y.numpy().tolist() if y.ndim > 0 else [y.item()])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    score = nasa_score(actuals, predictions)
    
    return {'rmse': rmse, 'mae': mae, 'nasa_score': score, 
            'predictions': predictions, 'actuals': actuals}


def main():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n=== Loading Data ===")
    data = prepare_data('FD001', add_features=config['dataset'].get('add_features', False))
    train_df, test_df = data['train'], data['test']
    
    # Create sequences
    print("\n=== Creating Sequences ===")
    seq_len = config['model']['sequence_length']
    X_train, y_train, X_val, y_val = prepare_train_data(train_df, seq_len, config['training']['val_split'])
    X_test, y_test = prepare_test_data(test_df, seq_len)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create datasets and loaders
    train_ds = TurbofanDataset(X_train, y_train)
    val_ds = TurbofanDataset(X_val, y_val)
    test_ds = TurbofanDataset(X_test, y_test)
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
    test_loader = DataLoader(test_ds, batch_size)
    
    # Create model
    print("\n=== Creating Model ===")
    input_dim = X_train.shape[2]
    model = create_lstm_model(input_dim, config['model']['lstm'])
    
    # Train
    print("\n=== Training ===")
    Path('models/lstm').mkdir(parents=True, exist_ok=True)
    save_path = 'models/lstm/best_model_FD001.pth'
    
    trainer = Trainer(model, config['training'], device)
    trainer.fit(train_loader, val_loader, config['training']['epochs'], save_path)
    
    # Load best model and evaluate
    print("\n=== Evaluation ===")
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTrain - RMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f}, Score: {train_metrics['nasa_score']:.0f}")
    print(f"Val   - RMSE: {val_metrics['rmse']:.2f}, MAE: {val_metrics['mae']:.2f}, Score: {val_metrics['nasa_score']:.0f}")
    print(f"Test  - RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}, Score: {test_metrics['nasa_score']:.0f}")
    
    # Save results
    np.savez(f'models/lstm/results_FD001.npz',
             train_pred=train_metrics['predictions'], train_actual=train_metrics['actuals'],
             val_pred=val_metrics['predictions'], val_actual=val_metrics['actuals'],
             test_pred=test_metrics['predictions'], test_actual=test_metrics['actuals'])
    
    print(f"\nResults saved to models/lstm/")


if __name__ == '__main__':
    main()
