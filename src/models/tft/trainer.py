"""
Training pipeline for TFT model
"""

import warnings
warnings.filterwarnings('ignore')

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

from .model import create_tft_dataset, create_tft_model


def train_tft_model(
    train_df,
    val_df,
    max_encoder_length=30,
    max_prediction_length=1,
    batch_size=128,
    max_epochs=100,
    gpus=1,
    learning_rate=0.001,
    early_stopping_patience=10,
    checkpoint_dir='checkpoints/tft',
    log_dir='lightning_logs',
    **model_kwargs
):
    """
    Complete TFT training pipeline
    
    Args:
        train_df: Training dataframe (prepared with prepare_for_tft)
        val_df: Validation dataframe (prepared with prepare_for_tft)
        max_encoder_length: Maximum historical sequence length
        max_prediction_length: Prediction horizon
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        gpus: Number of GPUs to use (0 for CPU)
        learning_rate: Initial learning rate
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
        **model_kwargs: Additional model hyperparameters
        
    Returns:
        best_model: Best trained model
        training_dataset: Training TimeSeriesDataSet
        validation_dataset: Validation TimeSeriesDataSet
        trainer: PyTorch Lightning trainer
    """
    # Create datasets
    print("Creating training dataset...")
    training_dataset = create_tft_dataset(
        train_df,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        training=True
    )
    
    # Create validation dataset from training dataset (uses same scaling)
    print("Creating validation dataset...")
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_df,
        predict=False,
        stop_randomization=True  # Don't randomize for validation
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=batch_size * 10,
        num_workers=0
    )
    
    # Create model
    print("Creating TFT model...")
    model = create_tft_model(training_dataset, learning_rate=learning_rate, **model_kwargs)
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=early_stopping_patience,
        verbose=True,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='tft-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    # Logger
    logger = TensorBoardLogger(log_dir, name='tft_cmapss')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 'auto',
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    # Learning rate finder (optional but recommended)
    print("\nSkipping learning rate finder - using provided learning rate")
    print(f"Learning rate: {learning_rate}")
    # Note: lr_find requires Tuner which may not be available in all Lightning versions
    # Uncomment below if you have compatible Lightning version
    # try:
    #     from lightning.pytorch.tuner import Tuner
    #     tuner = Tuner(trainer)
    #     res = tuner.lr_find(model, train_dataloader, val_dataloader)
    #     model.learning_rate = res.suggestion()
    # except Exception as e:
    #     print(f"LR finder unavailable: {e}")
    
    # Train
    print("\nStarting training...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    # Load best model
    print(f"\nLoading best model from: {checkpoint_callback.best_model_path}")
    best_model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    
    return best_model, training_dataset, validation_dataset, trainer
