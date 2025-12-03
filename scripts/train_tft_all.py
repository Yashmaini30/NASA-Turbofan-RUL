"""Train TFT on all 4 C-MAPSS datasets (summary table output).

Mirrors `train_lstm_all.py` / `train_gru_all.py`.
"""
import sys
from pathlib import Path
# Ensure we can import sibling module `train_tft_simple.py`
sys.path.insert(0, str(Path(__file__).parent))

import time
from datetime import datetime

from train_tft_simple import main as train_tft

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

DATASET_INFO = {
    "FD001": {"engines": 100, "conditions": 1, "faults": 1, "difficulty": "Easy"},
    "FD002": {"engines": 260, "conditions": 6, "faults": 1, "difficulty": "Medium"},
    "FD003": {"engines": 100, "conditions": 1, "faults": 2, "difficulty": "Medium"},
    "FD004": {"engines": 248, "conditions": 6, "faults": 2, "difficulty": "Hard"},
}


def train_all_datasets():
    results = {}
    start_time = time.time()

    print("\n" + "=" * 90)
    print(" " * 25 + "TFT TRAINING ON ALL C-MAPSS DATASETS")
    print("=" * 90)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90 + "\n")

    for i, dataset_id in enumerate(DATASETS, 1):
        print(f"\n{'='*90}")
        print(f"  [{i}/4] Training on {dataset_id} - {DATASET_INFO[dataset_id]['difficulty']}")
        print(
            f"  Conditions: {DATASET_INFO[dataset_id]['conditions']} | Faults: {DATASET_INFO[dataset_id]['faults']} | Engines: {DATASET_INFO[dataset_id]['engines']}"
        )
        print(f"{'='*90}\n")

        dataset_start = time.time()

        try:
            metrics = train_tft(dataset_id)
            results[dataset_id] = {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "nasa_score": metrics["nasa_score"],
                "status": "SUCCESS",
            }
            elapsed = time.time() - dataset_start
            print(f"\n✅ {dataset_id} complete in {elapsed/60:.1f} min")
        except Exception as e:
            print(f"\n❌ {dataset_id} failed: {e}")
            results[dataset_id] = {
                "rmse": None,
                "mae": None,
                "nasa_score": None,
                "status": "FAILED",
            }

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 90)
    print(" " * 30 + "TFT RESULTS SUMMARY")
    print("=" * 90)
    print(
        f"Dataset | Difficulty | Cond | Faults | Engines | RMSE      | MAE       | NASA Score"
    )
    print("-" * 90)

    for dataset_id in DATASETS:
        info = DATASET_INFO[dataset_id]
        res = results[dataset_id]
        if res["status"] == "SUCCESS":
            print(
                f"{dataset_id:7} | {info['difficulty']:10} | {info['conditions']:4} | "
                f"{info['faults']:6} | {info['engines']:7} | {res['rmse']:9.2f} | "
                f"{res['mae']:9.2f} | {res['nasa_score']:10.0f}"
            )
        else:
            print(
                f"{dataset_id:7} | {info['difficulty']:10} | {info['conditions']:4} | "
                f"{info['faults']:6} | {info['engines']:7} | FAILED"
            )

    print("=" * 90)
    print(f"Total: {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    _ = train_all_datasets()
