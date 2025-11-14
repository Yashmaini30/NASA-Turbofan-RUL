import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """GRU model for RUL prediction with configurable architecture."""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, output_dim=1, 
                 dropout=0.2, bidirectional=False):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units in GRU layers
            num_layers: Number of stacked GRU layers
            output_dim: Output dimension (1 for RUL regression)
            dropout: Dropout rate between GRU layers
            bidirectional: If True, use bidirectional GRU
        """
        super(GRUModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # GRU layer
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected output layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Predicted RUL values of shape (batch_size, output_dim)
        """
        # GRU output: (batch_size, seq_len, hidden_dim * num_directions)
        out, hidden = self.gru(x)
        
        # Use the last time step's output
        out = self.fc(out[:, -1, :])
        
        return out
    
    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """Print model architecture summary."""
        print("=" * 70)
        print(f"{'GRU Model Architecture':^70}")
        print("=" * 70)
        print(f"Input Dimension:        {self.input_dim}")
        print(f"Hidden Dimension:       {self.hidden_dim}")
        print(f"Number of Layers:       {self.num_layers}")
        print(f"Bidirectional:          {self.bidirectional}")
        print(f"Dropout Rate:           {self.dropout}")
        print(f"Total Parameters:       {self.get_num_params():,}")
        print("=" * 70)
        print("\nLayer Details:")
        print("-" * 70)
        for name, module in self.named_children():
            print(f"{name:20} {str(module)}")
        print("=" * 70)


def create_gru_model(input_dim, config=None):
    """
    Factory function to create GRU model with optional configuration.
    
    Args:
        input_dim: Number of input features
        config: Dictionary with model hyperparameters (optional)
        
    Returns:
        GRUModel instance
    """
    if config is None:
        config = {}
    
    # Default hyperparameters
    defaults = {
        'hidden_dim': 256,
        'num_layers': 2,
        'output_dim': 1,
        'dropout': 0.2,
        'bidirectional': True
    }
    
    # Merge with provided config
    params = {**defaults, **config}
    
    model = GRUModel(
        input_dim=input_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        output_dim=params['output_dim'],
        dropout=params['dropout'],
        bidirectional=params['bidirectional']
    )
    
    return model
