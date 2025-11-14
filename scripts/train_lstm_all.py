"""Train LSTM on all 4 C-MAPSS datasets and compare results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime
from train_lstm import main as train_lstm

# Datasets to train on
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']

# Dataset characteristics
DATASET_INFO = {
    'FD001': {'engines': 100, 'conditions': 1, 'faults': 1, 'difficulty': 'Easy'},
    'FD002': {'engines': 260, 'conditions': 6, 'faults': 1, 'difficulty': 'Medium'},
    'FD003': {'engines': 100, 'conditions': 1, 'faults': 2, 'difficulty': 'Medium'},
    'FD004': {'engines': 248, 'conditions': 6, 'faults': 2, 'difficulty': 'Hard'}
}

def train_all_datasets():
    """Train and evaluate on all 4 C-MAPSS datasets."""
    results = {}
    start_time = time.time()
    
    print("\n" + "="*90)
    print(" "*25 + "LSTM TRAINING ON ALL C-MAPSS DATASETS")
    print("="*90)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*90 + "\n")
    
    for i, dataset_id in enumerate(DATASETS, 1):
        print(f"\n{'='*90}")
        print(f"{'='*90}")
        print(f"  [{i}/4] Training on {dataset_id} - {DATASET_INFO[dataset_id]['difficulty']} Level")
        print(f"  Conditions: {DATASET_INFO[dataset_id]['conditions']} | Fault Modes: {DATASET_INFO[dataset_id]['faults']} | Engines: {DATASET_INFO[dataset_id]['engines']}")
        print(f"{'='*90}")
        print(f"{'='*90}\n")
        
        dataset_start = time.time()
        
        try:
            # Train model
            test_metrics = train_lstm(dataset_id)
            
            # Store results
            results[dataset_id] = {
                'rmse': test_metrics['rmse'],
                'mae': test_metrics['mae'],
                'nasa_score': test_metrics['nasa_score'],
                'status': 'SUCCESS'
            }
            
            dataset_time = time.time() - dataset_start
            print(f"\n✅ {dataset_id} training complete in {dataset_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n❌ {dataset_id} training failed: {str(e)}")
            results[dataset_id] = {
                'rmse': None,
                'mae': None,
                'nasa_score': None,
                'status': f'FAILED: {str(e)}'
            }
    
    # Print summary
    total_time = time.time() - start_time
    print("\n\n" + "="*90)
    print(" "*30 + "FINAL RESULTS SUMMARY")
    print("="*90)
    print(f"Dataset | Difficulty | Conditions | Faults | Engines | Test RMSE | Test MAE | NASA Score")
    print("-"*90)
    
    for dataset_id in DATASETS:
        info = DATASET_INFO[dataset_id]
        res = results[dataset_id]
        
        if res['status'] == 'SUCCESS':
            print(f"{dataset_id:7} | {info['difficulty']:10} | "
                  f"{info['conditions']:10} | {info['faults']:6} | {info['engines']:7} | "
                  f"{res['rmse']:9.2f} | {res['mae']:8.2f} | {res['nasa_score']:10.0f}")
        else:
            print(f"{dataset_id:7} | {info['difficulty']:10} | "
                  f"{info['conditions']:10} | {info['faults']:6} | {info['engines']:7} | "
                  f"{'FAILED':>9} | {'FAILED':>8} | {'FAILED':>10}")
    
    print("="*90)
    print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*90 + "\n")
    
    return results

if __name__ == '__main__':
    results = train_all_datasets()

