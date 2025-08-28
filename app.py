from src.data.load_data import load_dataset

DATA_DIR = "CMAPSSData"
DATASET_ID = "FD001"

print("Starting data loading from app.py...")

train_df, test_df, rul_df = load_dataset(dataset_id=DATASET_ID, data_dir=DATA_DIR)

if train_df is not None:
    print("\nSuccessfully loaded all datasets!")
    print(f"Loaded train data with {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
    print(f"Loaded test data with {test_df.shape[0]} rows and {test_df.shape[1]} columns.")
    print(f"Loaded RUL data with {rul_df.shape[0]} rows and {rul_df.shape[1]} column.")
    print("\nFirst 5 rows of the training data:")
    print(train_df.head())