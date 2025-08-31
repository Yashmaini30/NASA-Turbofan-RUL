"""
Utility functions for NASA Turbofan RUL analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

from src.config.constants import (
    COLUMN_NAMES, PLOT_STYLE, PLOT_PALETTE, FIGURE_DPI, 
    REPORTS_PATH, SIGNIFICANCE_LEVEL
)

# Setup plotting defaults
plt.style.use(PLOT_STYLE)
sns.set_palette(PLOT_PALETTE)
warnings.filterwarnings('ignore')

def setup_directories():
    """Ensure required directories exist."""
    REPORTS_PATH.mkdir(exist_ok=True)
    results_path = REPORTS_PATH / 'analysis_results'
    results_path.mkdir(exist_ok=True)
    return results_path

def load_dataset(dataset_name: str, data_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load training, test, and RUL data for a specific dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'FD001')
        data_path: Path to data directory
        
    Returns:
        Dictionary containing train, test, and rul DataFrames
    """
    try:
        train_file = data_path / f'train_{dataset_name}.txt'
        test_file = data_path / f'test_{dataset_name}.txt'
        rul_file = data_path / f'RUL_{dataset_name}.txt'
        
        # Load training data
        train_df = pd.read_csv(train_file, sep=' ', header=None, names=COLUMN_NAMES)
        train_df = train_df.dropna(axis=1)
        
        # Load test data
        test_df = pd.read_csv(test_file, sep=' ', header=None, names=COLUMN_NAMES)
        test_df = test_df.dropna(axis=1)
        
        # Load RUL data
        rul_df = pd.read_csv(rul_file, header=None, names=['RUL'])
        
        # Calculate RUL for training data
        train_df_with_rul = calculate_rul(train_df)
        
        return {
            'train': train_df_with_rul,
            'test': test_df,
            'rul': rul_df,
            'train_raw': train_df
        }
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return {}

def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RUL for training data."""
    df_rul = df.copy()
    max_cycles = df_rul.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycles']
    
    df_rul = df_rul.merge(max_cycles, on='unit_id')
    df_rul['RUL'] = df_rul['max_cycles'] - df_rul['time_cycles']
    
    return df_rul

def get_sensor_columns(df: pd.DataFrame) -> List[str]:
    """Get list of sensor columns from DataFrame."""
    return [col for col in df.columns if col.startswith('sensor_')]

def save_plot(fig, filename: str, reports_path: Path = REPORTS_PATH):
    """Save plot with consistent formatting."""
    filepath = reports_path / filename
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    return filepath

def calculate_correlation_with_rul(df: pd.DataFrame, sensor_col: str) -> Dict[str, float]:
    """Calculate correlation statistics between sensor and RUL."""
    correlation = df[sensor_col].corr(df['RUL'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['RUL'], df[sensor_col])
    
    return {
        'correlation': correlation,
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'is_significant': p_value < SIGNIFICANCE_LEVEL,
        'is_strong': abs(correlation) > 0.3
    }

def perform_anova_test(groups: List[np.ndarray]) -> Dict[str, float]:
    """Perform ANOVA test on groups of data."""
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate effect size (eta-squared)
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
    ss_total = sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'is_significant': p_value < SIGNIFICANCE_LEVEL
    }

def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for data."""
    alpha = 1 - confidence
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return ci

def bootstrap_statistic(data: np.ndarray, stat_func: callable, n_bootstrap: int = 1000) -> Dict[str, float]:
    """Bootstrap a statistic and return confidence intervals."""
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(stat_func(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    return {
        'mean': np.mean(bootstrap_stats),
        'std': np.std(bootstrap_stats),
        'ci_lower': np.percentile(bootstrap_stats, 2.5),
        'ci_upper': np.percentile(bootstrap_stats, 97.5)
    }

def create_summary_stats(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """Create summary statistics DataFrame."""
    if group_col:
        return df.groupby(group_col).describe()
    else:
        return df.describe()

def format_p_value(p_value: float) -> str:
    """Format p-value for display."""
    if p_value < 0.001:
        return "< 0.001"
    elif p_value < 0.01:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.2f}"

class AnalysisTimer:
    """Context manager for timing analysis steps."""
    
    def __init__(self, description: str):
        self.description = description
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        print(f"Starting: {self.description}")
        return self
        
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        print(f"Completed: {self.description} ({elapsed:.2f}s)")

def print_section_header(title: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "="*width)
    print(f"{title.center(width)}")
    print("="*width)

def print_subsection_header(title: str, width: int = 60):
    """Print formatted subsection header."""
    print(f"\n{title}")
    print("-" * min(len(title), width))
