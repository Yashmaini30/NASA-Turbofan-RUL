"""TFT training pipeline — simple, production-style runner (per dataset).

This mirrors the style of train_lstm.py/train_gru.py for consistency:
- Minimal arguments (function `main(dataset_id)`)
- Prints RMSE/MAE/PHM08 (NASA) Score
- Saves predictions for later plotting/comparison
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np

from pytorch_forecasting import TimeSeriesDataSet

from src.models.tft.data_preprocessing import (
    load_cmapss_data,
    remove_constant_features,
    add_piecewise_rul,
    normalize_sensors,
    prepare_for_tft,
)
from src.models.tft.trainer import train_tft_model
from src.models.tft.evaluation import evaluate_model
from src.models.tft.visualization import create_metrics_summary


def _split_train_val_by_tail(train_df, val_split: float = 0.2):
    """Split each engine's trajectory: earliest part -> train, tail -> val.
    Keeps distribution per engine and avoids leakage across engines.
    """
    train_list, val_list = [], []
    for engine_id in train_df["unit_id"].unique():
        df_e = train_df[train_df["unit_id"] == engine_id].sort_values("time_cycles")
        n = len(df_e)
        cut = max(1, int(n * (1 - val_split)))
        train_list.append(df_e.iloc[:cut])
        val_list.append(df_e.iloc[cut:])

    import pandas as pd

    train_out = pd.concat(train_list, ignore_index=True)
    val_out = pd.concat(val_list, ignore_index=True)
    return train_out, val_out


def main(dataset_id: str = "FD001"):
    """Train and evaluate Temporal Fusion Transformer on a C-MAPSS dataset.

    Returns a dict aligned with LSTM/GRU runners:
      { 'rmse': float, 'mae': float, 'nasa_score': float }
    """
    print(f"\n=== TFT: Loading data for {dataset_id} ===")
    train_df, test_df, rul_df = load_cmapss_data(dataset_id, data_path="./CMAPSSData/")

    # Basic cleaning + labels
    train_df = remove_constant_features(train_df)
    test_df = remove_constant_features(test_df)
    train_df = add_piecewise_rul(train_df, early_rul=125)

    # Normalize sensors/settings
    train_df, test_df, _ = normalize_sensors(train_df, test_df, method="minmax")

    # Prepare for TFT (adds time_idx, relative_time, categorical unit_id)
    train_df = prepare_for_tft(train_df, is_train=True)
    test_df = prepare_for_tft(test_df, is_train=False)

    # Train/val split by trailing portion per engine
    train_split_df, val_split_df = _split_train_val_by_tail(train_df, val_split=0.2)
    print(f"Train samples: {len(train_split_df)} | Val samples: {len(val_split_df)} | Test rows: {len(test_df)}")

    # Model/training hyperparams (kept simple; mirror defaults in scripts/train_tft.py)
    model_kwargs = dict(
        hidden_size=64,
        lstm_layers=2,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
    )

    best_model, training_dataset, validation_dataset, trainer = train_tft_model(
        train_split_df,
        val_split_df,
        max_encoder_length=30,
        max_prediction_length=1,
        batch_size=128,
        max_epochs=50,
        gpus=1,
        learning_rate=1e-3,
        early_stopping_patience=10,
        checkpoint_dir="checkpoints/tft",
        log_dir="lightning_logs",
        **model_kwargs,
    )

    # Prepare test dataset from training dataset (ensures consistent scaling/encoding)
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, test_df, predict=True, stop_randomization=True
    )

    # Evaluate
    metrics, predictions, actuals = evaluate_model(best_model, test_dataset, test_df, rul_df)
    print(create_metrics_summary(metrics, dataset_name=dataset_id))

    # Save artifacts (align with LSTM/GRU saving style)
    out_dir = Path("models/tft")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / f"results_{dataset_id}.npz",
        test_pred=predictions,
        test_actual=actuals,
    )

    with open(out_dir / f"metrics_{dataset_id}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to {out_dir}")

    # Map to common keys used elsewhere in the repo
    return {
        "rmse": float(metrics["RMSE"]),
        "mae": float(metrics["MAE"]),
        # PHM08_Score is the NASA asymmetric score
        "nasa_score": float(metrics["PHM08_Score"]),
    }


if __name__ == "__main__":
    # Default: FD001 to match other single-dataset runners
    main("FD001")
