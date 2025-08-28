import os
import pandas as pd

def load_dataset(dataset_id="FD001", data_dir="CMAPSSData"):
    try:
        train_path = os.path.join(data_dir, f"train_{dataset_id}.txt")
        test_path = os.path.join(data_dir, f"test_{dataset_id}.txt")
        rul_path = os.path.join(data_dir, f"RUL_{dataset_id}.txt")
        
        train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
        test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
        rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])
        
        return train_df, test_df, rul_df
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None