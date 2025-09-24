"""
GRU Model for NASA Turbofan RUL Prediction

This module implements a Gated Recurrent Unit (GRU) neural network
for predicting Remaining Useful Life (RUL) of turbofan engines.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class GRUModel(nn.Module):
    """
    GRU-based model for time series RUL prediction.
    
    This model uses a multi-layer GRU network with dropout regularization
    and a fully connected layer for regression output.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize the GRU model.
        
        Args:
            input_dim: Number of input features (sensor readings)
            hidden_dim: Hidden dimension size for GRU layers
            num_layers: Number of stacked GRU layers
            output_dim: Output dimension (1 for RUL regression)
            dropout: Dropout probability for regularization
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRUModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Calculate the actual hidden dimension for bidirectional GRU
        self.actual_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        
        # GRU layer with dropout (only applied if num_layers > 1)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer for the final hidden state
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(self.actual_hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"GRU Model initialized: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"bidirectional={bidirectional}, dropout={dropout}"
        )
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Simple bias initialization for GRU
                param.data.fill_(0)
                # For GRU: set reset gate bias to small positive value
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//3:2*n//3].fill_(1.0)  # Reset gate bias
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the GRU model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            hidden: Initial hidden state (optional)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # GRU forward pass
        gru_out, hidden = self.gru(x, hidden)
        
        # Use the last time step output for regression
        # For bidirectional GRU, gru_out contains concatenated forward and backward outputs
        last_output = gru_out[:, -1, :]  # Shape: (batch_size, actual_hidden_dim)
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Final prediction with sigmoid activation for [0,1] output range
        output = torch.sigmoid(self.fc(last_output))  # Shape: (batch_size, output_dim)
        
        return output
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state for the GRU.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initialized hidden state tensor
        """
        num_directions = 2 if self.bidirectional else 1
        hidden = torch.zeros(
            self.num_layers * num_directions, 
            batch_size, 
            self.hidden_dim,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype
        )
        return hidden
    
    def get_model_info(self) -> dict:
        """
        Get model architecture information.
        
        Returns:
            Dictionary containing model parameters and architecture info
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'GRU',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'actual_hidden_dim': self.actual_hidden_dim
        }
    
    def save_model(self, filepath: str, optimizer_state: Optional[dict] = None, epoch: int = None):
        """
        Save model state and metadata.
        
        Args:
            filepath: Path to save the model
            optimizer_state: Optimizer state dictionary (optional)
            epoch: Training epoch number (optional)
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_model_info(),
            'epoch': epoch
        }
        
        if optimizer_state is not None:
            save_dict['optimizer_state_dict'] = optimizer_state
            
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu') -> Tuple['GRUModel', dict]:
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Tuple of (loaded model, metadata dictionary)
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model_config = checkpoint['model_config']
        
        # Create model instance
        model = cls(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            output_dim=model_config['output_dim'],
            dropout=model_config['dropout'],
            bidirectional=model_config['bidirectional']
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        metadata = {
            'epoch': checkpoint.get('epoch'),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'model_config': model_config
        }
        
        logger.info(f"Model loaded from {filepath}")
        return model, metadata


class ImprovedGRUModel(GRUModel):
    """
    Enhanced GRU model with additional features for better performance.
    
    This model includes:
    - Batch normalization
    - Skip connections
    - Attention mechanism (optional)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_batch_norm: bool = True,
        use_attention: bool = False
    ):
        """
        Initialize the improved GRU model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of GRU layers
            output_dim: Output dimension
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
            use_batch_norm: Whether to use batch normalization
            use_attention: Whether to use attention mechanism
        """
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional)
        
        self.use_batch_norm = use_batch_norm
        self.use_attention = use_attention
        
        # Batch normalization layer
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.actual_hidden_dim)
        
        # Simple attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.actual_hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            # Additional layer normalization for attention
            self.layer_norm = nn.LayerNorm(self.actual_hidden_dim)
        
        # Enhanced output layer with residual connection
        self.fc_enhanced = nn.Sequential(
            nn.Linear(self.actual_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        logger.info(
            f"Improved GRU Model initialized with batch_norm={use_batch_norm}, "
            f"attention={use_attention}"
        )
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced forward pass with additional features."""
        batch_size, seq_len, _ = x.size()
        
        # GRU forward pass
        gru_out, hidden = self.gru(x, hidden)
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over sequence
            attended_out, _ = self.attention(gru_out, gru_out, gru_out)
            gru_out = self.layer_norm(gru_out + attended_out)  # Residual connection
        
        # Use the last time step output
        last_output = gru_out[:, -1, :]
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            last_output = self.batch_norm(last_output)
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Enhanced output prediction with sigmoid
        output = torch.sigmoid(self.fc_enhanced(last_output))  # Add sigmoid here
        
        return output