import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _cols(config: dict) -> Dict[str, List[str]]:
    return {
        "id": config["columns"]["id_col"],
        "time": config["columns"]["time_col"],
        "ops": config["columns"]["op_settings"],
        "sensors": config["columns"]["sensors"],
    }

def cycles_until_failure_labels(train_df: pd.DataFrame, id_col: str, time_col: str) -> pd.Series:
    """Compute remaining useful life (RUL) within training by unit: max(cycle) - cycle."""
    max_cycle = train_df.groupby(id_col)[time_col].transform("max")
    return (max_cycle - train_df[time_col]).astype(int)


def constant_or_near_zero_variance(df: pd.DataFrame, threshold: float = 1e-8) -> List[str]:
    """Return column names with variance <= threshold (constant or NZV)."""
    variances = df.var(axis=0, ddof=0)
    return variances[variances <= threshold].index.tolist()


def highly_correlated_pairs(df: pd.DataFrame, corr_threshold: float = 0.98) -> List[tuple]:
    """Return list of (col_i, col_j, corr) where |corr| >= threshold, i < j."""
    corr = df.corr(numeric_only=True)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if np.abs(c) >= corr_threshold:
                pairs.append((cols[i], cols[j], float(c)))
    return pairs


def trend_slopes_by_sensor(train_df: pd.DataFrame, id_col: str, time_col: str, sensors: List[str]) -> Dict[str, float]:
    """
    Compute average slope per sensor over time by unit (simple linear trend).
    Positive slope -> increasing on average; negative -> decreasing.
    """
    slopes = {}
    for s in sensors:
        unit_slopes = []
        for _, g in train_df[[id_col, time_col, s]].dropna().groupby(id_col):
            x = g[time_col].values.astype(float)
            y = g[s].values.astype(float)
            if len(x) >= 2:
                # slope via simple linear fit
                slope = np.polyfit(x, y, 1)[0]
                unit_slopes.append(slope)
        if unit_slopes:
            slopes[s] = float(np.mean(unit_slopes))
        else:
            slopes[s] = float("nan")
    return slopes


def distribution_overlap(train: pd.Series, test: pd.Series) -> float:
    """
    Simple overlap metric in [0,1]: proportion of shared range overlap.
    1 = identical min/max, 0 = disjoint ranges.
    """
    tmin, tmax = np.nanmin(train), np.nanmax(train)
    smin, smax = np.nanmin(test), np.nanmax(test)
    left = max(tmin, smin)
    right = min(tmax, smax)
    overlap = max(0.0, right - left)
    union = max(tmax, smax) - min(tmin, smin)
    return float(overlap / union) if union > 0 else 1.0


def run_distribution_analysis(train_df: pd.DataFrame, rul_df: pd.DataFrame, config: dict,
                              out_dir: str, dataset_name: str):
    c = _cols(config)
    id_col, time_col, sensors = c["id"], c["time"], c["sensors"]
    _ensure_dir(out_dir)

    # 1) Histogram of cycles until failure (training)
    train_cycles_max = train_df.groupby(id_col)[time_col].max()
    plt.figure()
    train_cycles_max.hist(bins=30)
    plt.title(f"{dataset_name} - Cycles Until Failure (Train, per unit)")
    plt.xlabel("Max cycle per engine")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"dist_cycles_{dataset_name}.png"))
    plt.close()

    # 2) Distribution of RUL labels (within-train computed)
    rul_train = cycles_until_failure_labels(train_df, id_col, time_col)
    plt.figure()
    rul_train.hist(bins=30)
    plt.title(f"{dataset_name} - In-Train RUL Distribution")
    plt.xlabel("RUL (cycles)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"dist_rul_train_{dataset_name}.png"))
    plt.close()

    # Save basic stats
    stats = {
        "train_cycles_per_unit": {
            "min": int(train_cycles_max.min()),
            "max": int(train_cycles_max.max()),
            "mean": float(train_cycles_max.mean()),
            "std": float(train_cycles_max.std(ddof=0)),
        },
        "rul_in_train": {
            "min": int(rul_train.min()),
            "max": int(rul_train.max()),
            "mean": float(rul_train.mean()),
            "std": float(rul_train.std(ddof=0)),
        },
    }
    with open(os.path.join(out_dir, f"distributions_{dataset_name}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # 3) Example engine trajectories (plot a few engines across a few sensors)
    example_units = list(train_df[id_col].drop_duplicates().head(4).values)
    example_sensors = [s for s in sensors if s in train_df.columns][:4]
    for s in example_sensors:
        plt.figure()
        for uid in example_units:
            g = train_df[train_df[id_col] == uid]
            plt.plot(g[time_col], g[s], label=f"Unit {uid}")
        plt.title(f"{dataset_name} - Trajectories: {s}")
        plt.xlabel(time_col)
        plt.ylabel(s)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"traj_{s}_{dataset_name}.png"))
        plt.close()

    # 4) Degradation vs noise (average slope sign/magnitude per sensor)
    slopes = trend_slopes_by_sensor(train_df, id_col, time_col, example_sensors)
    with open(os.path.join(out_dir, f"sensor_trends_{dataset_name}.json"), "w") as f:
        json.dump({"avg_slope_per_sensor": slopes}, f, indent=2)


def run_feature_insights(train_df: pd.DataFrame, config: dict, out_dir: str, dataset_name: str) -> dict:
    c = _cols(config)
    sensors = c["sensors"]
    _ensure_dir(out_dir)

    sensor_df = train_df[sensors].copy()

    # Correlation heatmap (save figure)
    corr = sensor_df.corr(numeric_only=True)
    plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.title(f"{dataset_name} - Sensor Correlation Heatmap")
    plt.xticks(range(len(sensors)), sensors, rotation=90)
    plt.yticks(range(len(sensors)), sensors)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"correlations_{dataset_name}.png"))
    plt.close()

    # Constant / near-zero variance sensors
    nzv = constant_or_near_zero_variance(sensor_df, threshold=1e-10)

    # Highly correlated pairs
    hc_pairs = highly_correlated_pairs(sensor_df, corr_threshold=0.98)

    # Stationarity proxy: trend slopes (sign indicates drift)
    slopes = trend_slopes_by_sensor(train_df, c["id"], c["time"], sensors)

    result = {
        "near_zero_variance_sensors": nzv,
        "highly_correlated_pairs_(>=0.98)": hc_pairs,
        "avg_trend_slope_per_sensor": slopes,
    }
    return result

def run_temporal_analysis(train_df: pd.DataFrame, config: dict, out_dir: str, dataset_name: str):
    c = _cols(config)
    id_col, time_col, sensors = c["id"], c["time"], c["sensors"]
    _ensure_dir(out_dir)

    # RUL vs cycles (monotonic decrease expected within unit)
    rul_train = cycles_until_failure_labels(train_df, id_col, time_col)
    plt.figure()
    plt.scatter(train_df[time_col], rul_train, s=2)
    plt.title(f"{dataset_name} - RUL vs. {time_col} (Train)")
    plt.xlabel(time_col)
    plt.ylabel("RUL")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"rul_vs_cycle_{dataset_name}.png"))
    plt.close()

    # Operating conditions vs sensor behavior (plot mean sensor per op-setting bin)
    # Bin each op setting into quantiles for a coarse comparison
    ops = c["ops"]
    for op in ops:
        if op not in train_df.columns:
            continue
        binned = pd.qcut(train_df[op], q=5, duplicates="drop")
        mean_by_bin = train_df.groupby(binned)[sensors].mean()
        plt.figure()
        for s in sensors[:5]:
            if s in mean_by_bin.columns:
                plt.plot(range(len(mean_by_bin)), mean_by_bin[s], label=s)
        plt.title(f"{dataset_name} - Mean Sensor vs {op} bins")
        plt.xlabel(f"{op} bin (quantiles)")
        plt.ylabel("Mean sensor value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"op_{op}_sensor_means_{dataset_name}.png"))
        plt.close()


def run_normalization_checks(train_df: pd.DataFrame, config: dict, out_dir: str, dataset_name: str) -> dict:
    c = _cols(config)
    id_col, sensors = c["id"], c["sensors"]
    _ensure_dir(out_dir)

    # Compute per-engine means/stds and global mean/std
    global_mean = train_df[sensors].mean()
    global_std = train_df[sensors].std(ddof=0).replace(0, np.nan)

    per_engine = train_df.groupby(id_col)[sensors].agg(["mean", "std"])
    # summarize variability across engines (how different are engine means)
    engine_mean_var = per_engine.xs("mean", level=1, axis=1).var(ddof=0).to_dict()

    # heuristic suggestion:
    # if inter-engine variance of means is large relative to global variance,
    # per-engine scaling may help.
    global_var = (global_std ** 2).to_dict()
    ratio = {}
    for s in sensors:
        gv = float(global_var.get(s, np.nan))
        emv = float(engine_mean_var.get(s, np.nan))
        ratio[s] = float(emv / gv) if gv and gv > 0 else np.nan

    # recommend per-engine if median ratio > 0.5 (heuristic)
    valid_ratios = [r for r in ratio.values() if r == r]  # drop nans
    median_ratio = float(np.median(valid_ratios)) if valid_ratios else float("nan")
    recommendation = "per-engine" if valid_ratios and median_ratio > 0.5 else "global"

    summary = {
        "median_inter_engine_vs_global_variance_ratio": median_ratio,
        "recommendation": recommendation,
    }
    return summary


def compare_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict, out_dir: str, dataset_name: str):
    c = _cols(config)
    sensors = [s for s in c["sensors"] if s in train_df.columns and s in test_df.columns]
    _ensure_dir(out_dir)

    diffs = {}
    for s in sensors:
        t_stats = {
            "mean": float(train_df[s].mean()),
            "std": float(train_df[s].std(ddof=0)),
            "min": float(train_df[s].min()),
            "max": float(train_df[s].max()),
        }
        s_stats = {
            "mean": float(test_df[s].mean()),
            "std": float(test_df[s].std(ddof=0)),
            "min": float(test_df[s].min()),
            "max": float(test_df[s].max()),
        }
        overlap = distribution_overlap(train_df[s], test_df[s])

        diffs[s] = {"train": t_stats, "test": s_stats, "range_overlap_0to1": overlap}

        # quick side-by-side histograms (same bins)
        plt.figure()
        data_min = np.nanmin([train_df[s].min(), test_df[s].min()])
        data_max = np.nanmax([train_df[s].max(), test_df[s].max()])
        bins = np.linspace(data_min, data_max, 40)
        plt.hist(train_df[s], bins=bins, alpha=0.6, label="train")
        plt.hist(test_df[s], bins=bins, alpha=0.6, label="test")
        plt.title(f"{dataset_name} - Train vs Test: {s}")
        plt.xlabel(s)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"train_vs_test_{s}_{dataset_name}.png"))
        plt.close()

    return diffs
