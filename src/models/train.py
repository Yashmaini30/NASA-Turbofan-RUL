import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.lstm_model import LSTMModel
from src.models.mlflow_utils import MLflowTracker

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_sequences(X, y, seq_len=30):
    xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.array(xs), np.array(ys)

def load_cmapss_data(config, fd_key, seq_len=30):
    data_dir = config["dataset"]["data_dir"]
    train_file = config["dataset"]["files"][fd_key]["train"]
    train_path = Path(data_dir) / train_file

    # Load train data
    df = pd.read_csv(train_path, sep=" ", header=None)
    df = df.drop(df.columns[[26, 27]], axis=1)
    sensor_idx = list(range(5, 26))
    X = df.iloc[:, sensor_idx].values.astype(np.float32)

    # Compute per-cycle RUL for each row
    unit_col = 0  
    cycle_col = 1 
    max_cycles = df.groupby(unit_col)[cycle_col].transform('max')
    y = (max_cycles - df[cycle_col]).values.astype(np.float32).reshape(-1, 1)

    # sequences for each engine unit
    xs_all, ys_all = [], []
    for unit in df[unit_col].unique():
        unit_mask = df[unit_col] == unit
        X_unit = X[unit_mask]
        y_unit = y[unit_mask]
        if len(X_unit) >= seq_len:
            xs, ys = create_sequences(X_unit, y_unit, seq_len)
            xs_all.append(xs)
            ys_all.append(ys)
    X_seq = np.concatenate(xs_all, axis=0)
    y_seq = np.concatenate(ys_all, axis=0)

    X_tensor = torch.tensor(X_seq)
    y_tensor = torch.tensor(y_seq)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader

def train_and_log(train_loader, val_loader, params, fd_key):
    model_params = {k: params[k] for k in ["input_dim", "hidden_dim", "num_layers", "output_dim", "dropout"]}
    model = LSTMModel(**model_params)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))

    for epoch in range(params.get("epochs", 10)):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_losses.append(loss.item())
    val_rmse = (sum(val_losses) / len(val_losses)) ** 0.5

    metrics = {"val_rmse": val_rmse}
    MLflowTracker.track_model(f"LSTM_{fd_key}", model, metrics, params, framework="pytorch")

if __name__ == "__main__":
    config = load_config("config.yaml")
    fd_keys = ["FD001", "FD002", "FD003", "FD004"]
    params = {
        "input_dim": 21,
        "hidden_dim": 128,
        "num_layers": 2,
        "output_dim": 1,
        "dropout": 0.2,
        "lr": 1e-3,
        "epochs": 10
    }
    seq_len = 30
    for fd_key in fd_keys:
        print(f"Training on {fd_key}...")
        train_loader = load_cmapss_data(config, fd_key, seq_len=seq_len)
        val_loader = train_loader  
        train_and_log(train_loader, val_loader, params, fd_key)