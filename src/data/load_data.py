import os
import pandas as pd


def _compose_columns(config: dict) -> list:
    return (
        [config["columns"]["id_col"], config["columns"]["time_col"]]
        + config["columns"]["op_settings"]
        + config["columns"]["sensors"]
    )


def load_dataset(config: dict, dataset_name: str):
    """
    Load train, test, and RUL datasets for a given dataset name (FD001, FD002, etc.).
    Returns: (train_df, test_df, rul_df)
    """
    base_dir = config["dataset"]["data_dir"]
    fp = config["dataset"]["files"][dataset_name]
    cols = _compose_columns(config)

    def _read_txt(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        # CMAPSS files usually have trailing blanks -> drop empty columns
        df = df.dropna(axis=1, how="all")
        df.columns = cols
        return df

    train_df = _read_txt(os.path.join(base_dir, fp["train"]))
    test_df = _read_txt(os.path.join(base_dir, fp["test"]))
    rul_df = pd.read_csv(os.path.join(base_dir, fp["rul"]), sep=r"\s+", header=None, names=["RUL"])

    return train_df, test_df, rul_df
