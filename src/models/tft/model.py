"""
TFT Model architecture and configuration
"""

import warnings
warnings.filterwarnings('ignore')

import torch
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
    GroupNormalizer
)
from pytorch_forecasting.data import NaNLabelEncoder


def create_tft_dataset(
    data,
    max_encoder_length=30,
    max_prediction_length=1,  # Single-step RUL prediction
    min_encoder_length=None,
    training=True
):
    """
    Create TimeSeriesDataSet for TFT
    
    Args:
        data: Prepared dataframe with time_idx, unit_id, sensors, settings, RUL
        max_encoder_length: Maximum length of encoder (historical data)
        max_prediction_length: Length of prediction horizon
        min_encoder_length: Minimum encoder length (defaults to half of max)
        training: Whether this is for training
        
    Returns:
        TimeSeriesDataSet object
    """
    # Get sensor and setting columns
    sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
    setting_cols = [col for col in data.columns if col.startswith('setting_')]
    
    # Define dataset
    dataset = TimeSeriesDataSet(
        data,
        time_idx='time_idx',
        target='RUL',
        group_ids=['unit_id'],
        
        # Don't allow missing timesteps
        allow_missing_timesteps=False,
        
        # Encoder/decoder lengths
        max_encoder_length=max_encoder_length,
        min_encoder_length=min_encoder_length or max_encoder_length // 2,
        max_prediction_length=max_prediction_length,
        
        # Static categoricals (engine identifier)
        static_categoricals=['unit_id'],
        
        # Time-varying unknown reals (sensors we only know historically)
        time_varying_unknown_reals=sensor_cols + ['RUL'],
        
        # Time-varying known reals (operating conditions known in future)
        time_varying_known_reals=setting_cols + ['time_idx', 'relative_time'],
        
        # Normalizer for target
        target_normalizer=GroupNormalizer(
            groups=['unit_id'],
            transformation='softplus'  # Better for non-negative values
        ),
        
        # Add relative time index
        add_relative_time_idx=True,
        
        # Add encoder length (helps model know sequence length)
        add_encoder_length=True,
        
        # Add target scales
        add_target_scales=True,
    )
    
    return dataset


def create_tft_model(dataset, learning_rate=0.001, **kwargs):
    """
    Create TFT model with optimized hyperparameters
    
    Args:
        dataset: TimeSeriesDataSet to build model from
        learning_rate: Learning rate for optimizer
        **kwargs: Additional hyperparameters to override defaults
        
    Returns:
        TemporalFusionTransformer model
    """
    # Default hyperparameters
    config = {
        'hidden_size': 64,           # Size of hidden layers
        'lstm_layers': 2,             # Number of LSTM layers
        'attention_head_size': 4,     # Number of attention heads
        'dropout': 0.1,               # Dropout rate
        'hidden_continuous_size': 16, # Size of continuous variable embeddings
        'loss': QuantileLoss(quantiles=[0.1, 0.5, 0.9]),  # Probabilistic predictions
        'learning_rate': learning_rate,
        'log_interval': 10,           # Log every 10 batches
        'reduce_on_plateau_patience': 4,  # LR reduction patience
    }
    
    # Override with any provided kwargs
    config.update(kwargs)
    
    model = TemporalFusionTransformer.from_dataset(
        dataset,
        **config
    )
    
    return model


def optimize_tft_hyperparameters(train_dataloader, val_dataloader, n_trials=100, max_epochs=50):
    """
    Use Optuna to find optimal hyperparameters
    
    Args:
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        n_trials: Number of optimization trials
        max_epochs: Maximum epochs per trial
        
    Returns:
        Optuna study object with best hyperparameters
    """
    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
        optimize_hyperparameters
    )
    
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path='optuna_tft',
        n_trials=n_trials,
        max_epochs=max_epochs,
        
        # Hyperparameter ranges
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(32, 128),
        hidden_continuous_size_range=(8, 64),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.0001, 0.1),
        dropout_range=(0.1, 0.3),
        lstm_layers_range=(1, 3),
        
        # Training config
        trainer_kwargs=dict(limit_train_batches=50),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )
    
    return study
