"""
Train TFT Model on C-MAPSS Dataset

This script implements the complete pipeline for training a Temporal Fusion Transformer
on the NASA C-MAPSS turbofan engine degradation dataset for RUL prediction.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
from pathlib import Path
import argparse
import json

import pandas as pd
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.tft.data_preprocessing import (
    load_cmapss_data,
    remove_constant_features,
    add_piecewise_rul,
    normalize_sensors,
    prepare_for_tft
)

from src.models.tft.trainer import train_tft_model
from src.models.tft.model import create_tft_dataset
from src.models.tft.evaluation import evaluate_model
from src.models.tft.visualization import (
    plot_rul_predictions,
    plot_error_distribution,
    create_metrics_summary
)

from pytorch_forecasting import TimeSeriesDataSet


def main(args):
    """
    Main training pipeline
    """
    print(f"""
    ═══════════════════════════════════════════════════════════
    Temporal Fusion Transformer Training
    Dataset: {args.dataset}
    ═══════════════════════════════════════════════════════════
    """)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # =====================
    # 1. LOAD DATA
    # =====================
    print("\n[1/8] Loading data...")
    train_df, test_df, rul_df = load_cmapss_data(args.dataset, args.data_path)
    print(f"  ✓ Train engines: {train_df['unit_id'].nunique()}")
    print(f"  ✓ Test engines: {test_df['unit_id'].nunique()}")
    print(f"  ✓ Train samples: {len(train_df)}")
    print(f"  ✓ Test samples: {len(test_df)}")
    
    # =====================
    # 2. PREPROCESS
    # =====================
    print("\n[2/8] Preprocessing features...")
    train_df = remove_constant_features(train_df)
    test_df = remove_constant_features(test_df)
    print(f"  ✓ Features after removal: {len([c for c in train_df.columns if c.startswith('sensor_') or c.startswith('setting_')])}")
    
    # =====================
    # 3. ADD RUL
    # =====================
    print("\n[3/8] Calculating RUL...")
    train_df = add_piecewise_rul(train_df, early_rul=args.early_rul)
    print(f"  ✓ RUL range: {train_df['RUL'].min():.1f} - {train_df['RUL'].max():.1f}")
    print(f"  ✓ Mean RUL: {train_df['RUL'].mean():.1f}")
    
    # =====================
    # 4. NORMALIZE
    # =====================
    print("\n[4/8] Normalizing sensors...")
    train_df, test_df, scaler = normalize_sensors(train_df, test_df, method=args.normalization)
    print(f"  ✓ Normalization method: {args.normalization}")
    
    # =====================
    # 5. PREPARE FOR TFT
    # =====================
    print("\n[5/8] Preparing data for TFT...")
    train_df = prepare_for_tft(train_df, is_train=True)
    test_df = prepare_for_tft(test_df, is_train=False)
    
    # Split train/validation
    print(f"  ✓ Splitting train/validation (ratio: {args.val_split})...")
    unique_engines = train_df['unit_id'].unique()
    n_val_engines = int(len(unique_engines) * args.val_split)
    val_engines = np.random.choice(unique_engines, n_val_engines, replace=False)
    
    val_df = train_df[train_df['unit_id'].isin(val_engines)].copy()
    train_df = train_df[~train_df['unit_id'].isin(val_engines)].copy()
    
    print(f"  ✓ Train engines: {train_df['unit_id'].nunique()}")
    print(f"  ✓ Val engines: {val_df['unit_id'].nunique()}")
    
    # =====================
    # 6. TRAIN MODEL
    # =====================
    print("\n[6/8] Training TFT model...")
    print(f"  Configuration:")
    print(f"    - Encoder length: {args.encoder_length}")
    print(f"    - Prediction length: {args.prediction_length}")
    print(f"    - Batch size: {args.batch_size}")
    print(f"    - Max epochs: {args.max_epochs}")
    print(f"    - Learning rate: {args.learning_rate}")
    print(f"    - Hidden size: {args.hidden_size}")
    print(f"    - LSTM layers: {args.lstm_layers}")
    print(f"    - Attention heads: {args.attention_heads}")
    print(f"    - Dropout: {args.dropout}")
    
    model, training_dataset, validation_dataset, trainer = train_tft_model(
        train_df,
        val_df,
        max_encoder_length=args.encoder_length,
        max_prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        # Model hyperparameters
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        attention_head_size=args.attention_heads,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size
    )
    
    print("\n  ✓ Training completed!")
    
    # =====================
    # 7. PREPARE TEST DATASET
    # =====================
    print("\n[7/8] Preparing test dataset...")
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        test_df,
        predict=True,
        stop_randomization=True
    )
    print("  ✓ Test dataset ready")
    
    # =====================
    # 8. EVALUATE
    # =====================
    print("\n[8/8] Evaluating model...")
    metrics, predictions, actuals = evaluate_model(
        model,
        test_dataset,
        test_df,
        rul_df
    )
    
    # Print results
    print(create_metrics_summary(metrics, args.dataset))
    
    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = results_dir / f'{args.dataset}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved to: {metrics_file}")
    
    # Save predictions
    predictions_file = results_dir / f'{args.dataset}_predictions.csv'
    pd.DataFrame({
        'actual_rul': actuals,
        'predicted_rul': predictions,
        'error': predictions - actuals,
        'abs_error': np.abs(predictions - actuals)
    }).to_csv(predictions_file, index=False)
    print(f"  ✓ Predictions saved to: {predictions_file}")
    
    # =====================
    # 9. VISUALIZATIONS
    # =====================
    if args.plot:
        print("\n[9/8] Creating visualizations...")
        
        # Plot predictions
        plot_rul_predictions(
            actuals,
            predictions,
            save_path=results_dir / f'{args.dataset}_predictions.png'
        )
        
        # Plot error distribution
        plot_error_distribution(
            actuals,
            predictions,
            save_path=results_dir / f'{args.dataset}_errors.png'
        )
        
        print("  ✓ Visualizations saved")
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60 + "\n")
    
    return model, metrics, predictions, actuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TFT model on C-MAPSS dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='FD001',
                        choices=['FD001', 'FD002', 'FD003', 'FD004'],
                        help='C-MAPSS dataset to use')
    parser.add_argument('--data-path', type=str, default='./CMAPSSData/',
                        help='Path to CMAPSSData directory')
    
    # Preprocessing arguments
    parser.add_argument('--early-rul', type=int, default=125,
                        help='Early RUL threshold for piecewise calculation')
    parser.add_argument('--normalization', type=str, default='minmax',
                        choices=['minmax', 'standard'],
                        help='Normalization method')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Model architecture arguments
    parser.add_argument('--encoder-length', type=int, default=30,
                        help='Maximum encoder sequence length')
    parser.add_argument('--prediction-length', type=int, default=1,
                        help='Prediction horizon length')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--lstm-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--attention-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--hidden-continuous-size', type=int, default=16,
                        help='Hidden continuous variable embedding size')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs (0 for CPU)')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/tft',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='lightning_logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--results-dir', type=str, default='results/tft',
                        help='Directory to save results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run main pipeline
    main(args)
