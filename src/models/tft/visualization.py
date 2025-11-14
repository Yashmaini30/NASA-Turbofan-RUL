"""
Visualization and interpretation tools for TFT model
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rul_predictions(actuals, predictions, save_path=None):
    """
    Plot actual vs predicted RUL with scatter and residual plots
    
    Args:
        actuals: Array of actual RUL values
        predictions: Array of predicted RUL values
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(actuals, predictions, alpha=0.6)
    axes[0].plot([0, max(actuals)], [0, max(actuals)], 'r--', label='Perfect Prediction')
    axes[0].set_xlabel('Actual RUL')
    axes[0].set_ylabel('Predicted RUL')
    axes[0].set_title('Actual vs Predicted RUL')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = predictions - actuals
    axes[1].scatter(actuals, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Actual RUL')
    axes[1].set_ylabel('Residuals (Predicted - Actual)')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_attention_weights(model, dataset, sample_idx=0, save_path=None):
    """
    Visualize attention weights to see which sensors are important
    
    Args:
        model: Trained TFT model
        dataset: TimeSeriesDataSet
        sample_idx: Index of sample to visualize
        save_path: Optional path to save figure
    """
    # Get interpretation
    interpretation = model.interpret_output(
        dataset[sample_idx:sample_idx+1],
        reduction='sum'
    )
    
    # Plot attention
    model.plot_interpretation(interpretation)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(trainer, save_path=None):
    """
    Plot training and validation loss history
    
    Args:
        trainer: PyTorch Lightning trainer
        save_path: Optional path to save figure
    """
    # Extract metrics from trainer
    train_loss = []
    val_loss = []
    
    for metric in trainer.logged_metrics:
        if 'train_loss' in str(metric):
            train_loss.append(trainer.logged_metrics[metric])
        if 'val_loss' in str(metric):
            val_loss.append(trainer.logged_metrics[metric])
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_distribution(actuals, predictions, save_path=None):
    """
    Plot error distribution histogram
    
    Args:
        actuals: Array of actual RUL values
        predictions: Array of predicted RUL values
        save_path: Optional path to save figure
    """
    errors = predictions - actuals
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Error histogram
    axes[0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', label='Zero Error')
    axes[0].set_xlabel('Prediction Error (Predicted - Actual)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Absolute error histogram
    abs_errors = np.abs(errors)
    axes[1].hist(abs_errors, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('Absolute Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Absolute Error Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_intervals(actuals, predictions, quantiles=None, save_path=None):
    """
    Plot predictions with uncertainty intervals
    
    Args:
        actuals: Array of actual RUL values
        predictions: Array of median predictions or dict with quantiles
        quantiles: Dictionary with 'lower' and 'upper' quantile predictions
        save_path: Optional path to save figure
    """
    indices = np.arange(len(actuals))
    
    plt.figure(figsize=(15, 6))
    plt.scatter(indices, actuals, label='Actual RUL', alpha=0.6, s=50)
    plt.scatter(indices, predictions, label='Predicted RUL', alpha=0.6, s=50)
    
    if quantiles is not None:
        plt.fill_between(
            indices,
            quantiles['lower'],
            quantiles['upper'],
            alpha=0.3,
            label='90% Prediction Interval'
        )
    
    plt.xlabel('Engine Index')
    plt.ylabel('RUL')
    plt.title('RUL Predictions with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_metrics_summary(metrics, dataset_name='FD001'):
    """
    Create a formatted summary of evaluation metrics
    
    Args:
        metrics: Dictionary of evaluation metrics
        dataset_name: Name of the dataset
        
    Returns:
        Formatted string summary
    """
    summary = f"""
    ═══════════════════════════════════════════
    TFT Model Evaluation - {dataset_name}
    ═══════════════════════════════════════════
    
    Regression Metrics:
    ─────────────────────────────────────────
    RMSE:        {metrics['RMSE']:.2f}
    MAE:         {metrics['MAE']:.2f}
    R²:          {metrics['R2']:.4f}
    
    C-MAPSS Specific:
    ─────────────────────────────────────────
    PHM08 Score: {metrics['PHM08_Score']:.2f}
    
    ═══════════════════════════════════════════
    Note: Lower RMSE, MAE, and PHM08 scores are better.
          Higher R² (closer to 1.0) is better.
    """
    return summary
