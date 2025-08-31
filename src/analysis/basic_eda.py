"""
Basic exploratory data analysis functions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats

from src.config.constants import SENSOR_DESCRIPTIONS, DEFAULT_MAX_ENGINES_PLOT
from src.utils.helpers import save_plot, get_sensor_columns, calculate_correlation_with_rul

class BasicEDA:
    """Handles basic exploratory data analysis."""
    
    def __init__(self, datasets: Dict[str, Dict[str, pd.DataFrame]]):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
    
    def analyze_dataset_overview(self) -> pd.DataFrame:
        """Generate overview statistics for all datasets."""
        summary_stats = []
        
        for name in self.dataset_names:
            train_df = self.datasets[name]['train']
            test_df = self.datasets[name]['test']
            
            stats_dict = {
                'Dataset': name,
                'Train_Engines': train_df['unit_id'].nunique(),
                'Test_Engines': test_df['unit_id'].nunique(),
                'Train_Cycles': len(train_df),
                'Test_Cycles': len(test_df),
                'Avg_Train_Life': train_df.groupby('unit_id')['time_cycles'].max().mean(),
                'Max_Train_Life': train_df.groupby('unit_id')['time_cycles'].max().max(),
                'Min_Train_Life': train_df.groupby('unit_id')['time_cycles'].max().min(),
            }
            summary_stats.append(stats_dict)
        
        return pd.DataFrame(summary_stats)
    
    def plot_rul_distributions(self, save_plots: bool = True) -> None:
        """Plot RUL distributions for all datasets."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, name in enumerate(self.dataset_names):
            train_df = self.datasets[name]['train']
            rul_values = train_df['RUL'].values
            
            axes[i].hist(rul_values, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{name} - RUL Distribution\\nMean: {rul_values.mean():.1f}, Std: {rul_values.std():.1f}')
            axes[i].set_xlabel('Remaining Useful Life (cycles)')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            save_plot(fig, 'rul_distributions.png')
        plt.show()
    
    def plot_engine_lifecycles(self, dataset_name: str, max_engines: int = DEFAULT_MAX_ENGINES_PLOT, 
                              save_plots: bool = True) -> None:
        """Plot lifecycle trajectories for sample engines."""
        train_df = self.datasets[dataset_name]['train']
        
        unique_engines = train_df['unit_id'].unique()
        sample_engines = np.random.choice(unique_engines, min(max_engines, len(unique_engines)), replace=False)
        
        plt.figure(figsize=(12, 8))
        
        for engine_id in sample_engines:
            engine_data = train_df[train_df['unit_id'] == engine_id]
            plt.plot(engine_data['time_cycles'], engine_data['RUL'], 
                    label=f'Engine {engine_id}', alpha=0.7, linewidth=2)
        
        plt.title(f'{dataset_name} - Engine Lifecycle Trajectories (Sample)')
        plt.xlabel('Time Cycles')
        plt.ylabel('Remaining Useful Life')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            save_plot(plt.gcf(), f'{dataset_name}_lifecycles.png')
        plt.show()
    
    def analyze_sensor_variance(self, dataset_name: str, cv_threshold: float = 0.01) -> pd.DataFrame:
        """Analyze sensor variance to identify constant/low-variance sensors."""
        train_df = self.datasets[dataset_name]['train']
        sensor_cols = get_sensor_columns(train_df)
        
        sensor_stats = {}
        for sensor in sensor_cols:
            sensor_data = train_df[sensor]
            sensor_stats[sensor] = {
                'mean': sensor_data.mean(),
                'std': sensor_data.std(),
                'min': sensor_data.min(),
                'max': sensor_data.max(),
                'range': sensor_data.max() - sensor_data.min(),
                'cv': sensor_data.std() / sensor_data.mean() if sensor_data.mean() != 0 else 0,
                'is_low_variance': (sensor_data.std() / sensor_data.mean() if sensor_data.mean() != 0 else 0) < cv_threshold
            }
        
        return pd.DataFrame(sensor_stats).T
    
    def plot_sensor_correlations(self, dataset_name: str, save_plots: bool = True) -> pd.DataFrame:
        """Plot correlation matrix for sensors."""
        train_df = self.datasets[dataset_name]['train']
        sensor_cols = get_sensor_columns(train_df)
        
        corr_matrix = train_df[sensor_cols].corr()
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f'{dataset_name} - Sensor Correlation Matrix')
        plt.tight_layout()
        
        if save_plots:
            save_plot(plt.gcf(), f'{dataset_name}_sensor_correlations.png')
        plt.show()
        
        return corr_matrix
    
    def analyze_degradation_patterns(self, dataset_name: str, sensors_to_analyze: List[str] = None,
                                   save_plots: bool = True) -> None:
        """Analyze how sensors change over engine lifecycle."""
        train_df = self.datasets[dataset_name]['train']
        
        if sensors_to_analyze is None:
            sensor_cols = get_sensor_columns(train_df)
            sensor_vars = train_df[sensor_cols].var().sort_values(ascending=False)
            sensors_to_analyze = sensor_vars.head(6).index.tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, sensor in enumerate(sensors_to_analyze):
            ax = axes[i]
            
            # Sample engines for visualization
            sample_engines = train_df['unit_id'].unique()[:5]
            
            for engine_id in sample_engines:
                engine_data = train_df[train_df['unit_id'] == engine_id].sort_values('time_cycles')
                ax.plot(engine_data['RUL'], engine_data[sensor], alpha=0.6, linewidth=1)
            
            # Plot average trend
            avg_trend = train_df.groupby('RUL')[sensor].mean().reset_index()
            ax.plot(avg_trend['RUL'], avg_trend[sensor], 'red', linewidth=3, label='Average')
            
            ax.set_xlabel('Remaining Useful Life')
            ax.set_ylabel(f'{sensor}')
            ax.set_title(f'{sensor} vs RUL\\n{SENSOR_DESCRIPTIONS.get(sensor, "Unknown sensor")}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle(f'{dataset_name} - Sensor Degradation Patterns', fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            save_plot(fig, f'{dataset_name}_degradation_patterns.png')
        plt.show()
    
    def calculate_degradation_trends(self, dataset_name: str) -> pd.DataFrame:
        """Calculate correlation between sensors and RUL to identify degradation indicators."""
        train_df = self.datasets[dataset_name]['train']
        sensor_cols = get_sensor_columns(train_df)
        
        trend_analysis = {}
        for sensor in sensor_cols:
            trend_stats = calculate_correlation_with_rul(train_df, sensor)
            trend_analysis[sensor] = {
                **trend_stats,
                'degradation_indicator': trend_stats['is_strong'] and trend_stats['is_significant']
            }
        
        return pd.DataFrame(trend_analysis).T
