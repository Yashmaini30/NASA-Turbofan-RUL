import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _cols(config: dict):
    return {
        "id": config["columns"]["id_col"],
        "time": config["columns"]["time_col"],
        "ops": config["columns"]["op_settings"],
        "sensors": config["columns"]["sensors"],
    }


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    m = df.mean()
    s = df.std(ddof=0).replace(0, np.nan)
    return (df - m) / s


def run_visualizations(train_df: pd.DataFrame, config: dict, out_dir: str, dataset_name: str):
    c = _cols(config)
    sensors = [s for s in c["sensors"] if s in train_df.columns]
    _ensure_dir(out_dir)

    X = train_df[sensors].dropna()
    Xz = _zscore(X).fillna(0.0)

    # PCA (2D)
    try:
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(Xz.values)
        plt.figure()
        plt.scatter(pcs[:, 0], pcs[:, 1], s=3)
        plt.title(f"{dataset_name} - PCA (2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pca_{dataset_name}.png"))
        plt.close()
    except Exception as e:
        print(f"[WARN] PCA failed: {e}")

    # t-SNE (2D) on a sample for speed
    try:
        sample_n = min(5000, Xz.shape[0])
        Xs = Xz.sample(sample_n, random_state=42) if Xz.shape[0] > sample_n else Xz
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=42)
        ts = tsne.fit_transform(Xs.values)
        plt.figure()
        plt.scatter(ts[:, 0], ts[:, 1], s=3)
        plt.title(f"{dataset_name} - t-SNE (2D)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tsne_{dataset_name}.png"))
        plt.close()
    except Exception as e:
        print(f"[WARN] t-SNE failed: {e}")

    # Temporal plots for selected sensors: 2,3,4,7 (if present)
    focus = [f"sensor_{i}" for i in [2, 3, 4, 7]]
    focus = [s for s in focus if s in sensors]
    for s in focus:
        plt.figure()
        # plot mean over time across engines to reduce clutter
        mean_over_time = train_df.groupby(c["time"])[s].mean()
        plt.plot(mean_over_time.index.values, mean_over_time.values)
        plt.title(f"{dataset_name} - Mean {s} over {c['time']}")
        plt.xlabel(c["time"])
        plt.ylabel(f"mean({s})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"timeplot_{s}_{dataset_name}.png"))
        plt.close()
