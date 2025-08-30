import json
import numpy as np
import pandas as pd


def _cycles_per_engine(df: pd.DataFrame, id_col: str, time_col: str) -> pd.Series:
    return df.groupby(id_col)[time_col].max()


def _shape(df: pd.DataFrame) -> tuple:
    return (int(df.shape[0]), int(df.shape[1]))


def _unique_engines(df: pd.DataFrame, id_col: str) -> int:
    return int(df[id_col].nunique())


def run_sanity_checks(train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame, config: dict) -> dict:
    """
    Compute basic sanity stats and return as a dict.
    """
    id_col = config["columns"]["id_col"]
    time_col = config["columns"]["time_col"]

    # shapes
    train_shape = _shape(train_df)
    test_shape  = _shape(test_df)
    rul_shape   = _shape(rul_df)

    # engines
    n_eng_train = _unique_engines(train_df, id_col)
    n_eng_test  = _unique_engines(test_df, id_col)
    n_rul_rows  = int(rul_df.shape[0])

    # cycles per engine (train)
    cycles_train = _cycles_per_engine(train_df, id_col, time_col)
    c_stats = {
        "min": int(cycles_train.min()),
        "max": int(cycles_train.max()),
        "mean": float(cycles_train.mean()),
        "std": float(cycles_train.std(ddof=0)),
    }

    # test-RUL alignment (CMAPSS: one RUL per test engine)
    test_rul_match = (n_eng_test == n_rul_rows)

    results = {
        "shapes": {"train": train_shape, "test": test_shape, "rul": rul_shape},
        "engines": {"train": n_eng_train, "test": n_eng_test, "rul_rows": n_rul_rows},
        "cycles_per_engine_train": c_stats,
        "test_rul_match": bool(test_rul_match),
    }
    return results


def save_results(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Sanity check results saved to {path}")