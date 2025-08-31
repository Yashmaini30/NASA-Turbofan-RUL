"""
Advanced analysis functions including uncertainty quantification and statistical modeling.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config.constants import DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_N_CLUSTERS, SIGNIFICANCE_LEVEL
from src.utils.helpers import (
    save_plot, get_sensor_columns, perform_anova_test, 
    bootstrap_statistic, calculate_confidence_interval
)

class AdvancedAnalysis:
    """Handles advanced analysis including uncertainty quantification and statistical modeling."""
    
    def __init__(self, datasets: Dict[str, Dict[str, pd.DataFrame]]):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
    
    def analyze_uncertainty_baseline(self, dataset_name: str, save_plots: bool = True) -> Dict:
        """Analyze variance in lifetimes and RUL prediction uncertainty."""
        train_df = self.datasets[dataset_name]['train']
        
        # Group by operating conditions
        condition_groups = train_df.groupby(['setting1', 'setting2', 'setting3'])
        uncertainty_analysis = {}
        
        for condition, group_data in condition_groups:
            lifespans = group_data.groupby('unit_id')['time_cycles'].max()
            
            # RUL variance at different lifecycle stages
            rul_stages = {}
            for stage in [0.1, 0.3, 0.5, 0.7, 0.9]:
                stage_data = group_data[group_data['RUL'] <= group_data['RUL'].quantile(stage)]
                if len(stage_data) > 0:
                    rul_stages[f'stage_{stage}'] = {
                        'mean_rul': stage_data['RUL'].mean(),
                        'std_rul': stage_data['RUL'].std(),
                        'cv_rul': stage_data['RUL'].std() / stage_data['RUL'].mean() if stage_data['RUL'].mean() > 0 else 0
                    }
            
            uncertainty_analysis[str(condition)] = {
                'n_engines': len(lifespans),
                'mean_lifespan': lifespans.mean(),
                'std_lifespan': lifespans.std(),
                'cv_lifespan': lifespans.std() / lifespans.mean(),
                'lifespan_range': lifespans.max() - lifespans.min(),
                'rul_stages': rul_stages
            }
        
        if save_plots:
            self._plot_uncertainty_analysis(dataset_name, uncertainty_analysis, train_df)
        
        return uncertainty_analysis
    
    def _plot_uncertainty_analysis(self, dataset_name: str, uncertainty_analysis: Dict, 
                                  train_df: pd.DataFrame) -> None:
        """Plot uncertainty analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        conditions = list(uncertainty_analysis.keys())
        lifespans = [uncertainty_analysis[cond]['mean_lifespan'] for cond in conditions]
        lifespan_stds = [uncertainty_analysis[cond]['std_lifespan'] for cond in conditions]
        
        # Lifespan variance by condition
        axes[0, 0].errorbar(range(len(conditions)), lifespans, yerr=lifespan_stds,
                           fmt='o', capsize=5, capthick=2)
        axes[0, 0].set_xlabel('Operating Condition')
        axes[0, 0].set_ylabel('Mean Lifespan ± Std')
        axes[0, 0].set_title('Lifespan Uncertainty by Operating Condition')
        axes[0, 0].set_xticks(range(len(conditions)))
        axes[0, 0].set_xticklabels([f'C{i}' for i in range(len(conditions))], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Coefficient of variation
        cvs = [uncertainty_analysis[cond]['cv_lifespan'] for cond in conditions]
        axes[0, 1].bar(range(len(conditions)), cvs)
        axes[0, 1].set_xlabel('Operating Condition')
        axes[0, 1].set_ylabel('Coefficient of Variation')
        axes[0, 1].set_title('Lifespan Variability by Condition')
        axes[0, 1].set_xticks(range(len(conditions)))
        axes[0, 1].set_xticklabels([f'C{i}' for i in range(len(conditions))], rotation=45)
        
        # RUL uncertainty across lifecycle stages
        stages = ['stage_0.1', 'stage_0.3', 'stage_0.5', 'stage_0.7', 'stage_0.9']
        for i, condition in enumerate(conditions[:3]):
            stage_cvs = []
            stage_labels = []
            for stage in stages:
                if stage in uncertainty_analysis[condition]['rul_stages']:
                    stage_cvs.append(uncertainty_analysis[condition]['rul_stages'][stage]['cv_rul'])
                    stage_labels.append(stage.split('_')[1])
            
            if stage_cvs:
                axes[1, 0].plot(stage_labels, stage_cvs, 'o-', label=f'Condition {i}', linewidth=2)
        
        axes[1, 0].set_xlabel('Lifecycle Stage (quantile)')
        axes[1, 0].set_ylabel('RUL Coefficient of Variation')
        axes[1, 0].set_title('RUL Uncertainty Across Lifecycle')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics panel
        axes[1, 1].axis('off')
        stats_text = f"""
        Dataset: {dataset_name}
        
        Uncertainty Statistics:
        Number of conditions: {len(conditions)}
        Avg CV across conditions: {np.mean(cvs):.3f}
        
        Condition Details:
        """
        
        for i, condition in enumerate(conditions[:3]):  # Show first 3
            stats = uncertainty_analysis[condition]
            stats_text += f"\\nCondition {i}: {stats['n_engines']} engines"
            stats_text += f"\\n  Mean life: {stats['mean_lifespan']:.1f} ± {stats['std_lifespan']:.1f}"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.suptitle(f'{dataset_name} - Uncertainty Baseline Analysis', fontsize=16)
        plt.tight_layout()
        save_plot(fig, f'{dataset_name}_uncertainty_baseline.png')
        plt.show()
    
    def perform_pca_analysis(self, dataset_name: str, n_components: int = 10, 
                           save_plots: bool = True) -> Dict:
        """Perform PCA to understand sensor relationships and dimensionality."""
        train_df = self.datasets[dataset_name]['train']
        sensor_cols = get_sensor_columns(train_df)
        
        # Standardize the data
        scaler = StandardScaler()
        sensor_data_scaled = scaler.fit_transform(train_df[sensor_cols])
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(sensor_data_scaled)
        
        # Create PCA results dataframe
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
        pca_df['RUL'] = train_df['RUL'].values
        pca_df['unit_id'] = train_df['unit_id'].values
        
        if save_plots:
            self._plot_pca_results(dataset_name, pca, pca_df)
        
        return {
            'pca_model': pca,
            'pca_data': pca_df,
            'scaler': scaler,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
    
    def _plot_pca_results(self, dataset_name: str, pca: PCA, pca_df: pd.DataFrame) -> None:
        """Plot PCA results and explained variance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Explained variance plot
        axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                       pca.explained_variance_ratio_)
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('PCA Explained Variance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[0, 1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
        axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # PC1 vs PC2 colored by RUL
        scatter = axes[1, 0].scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['RUL'],
                                    cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('First Principal Component')
        axes[1, 0].set_ylabel('Second Principal Component')
        axes[1, 0].set_title('PC1 vs PC2 (colored by RUL)')
        plt.colorbar(scatter, ax=axes[1, 0], label='RUL')
        
        # PC1 vs RUL
        axes[1, 1].scatter(pca_df['RUL'], pca_df['PC1'], alpha=0.6)
        axes[1, 1].set_xlabel('Remaining Useful Life')
        axes[1, 1].set_ylabel('First Principal Component')
        axes[1, 1].set_title('PC1 vs RUL')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{dataset_name} - Principal Component Analysis', fontsize=16)
        plt.tight_layout()
        save_plot(fig, f'{dataset_name}_pca_analysis.png')
        plt.show()
    
    def analyze_data_drift(self, dataset_name: str, save_plots: bool = True) -> pd.DataFrame:
        """Analyze potential data drift between train and test sets."""
        train_df = self.datasets[dataset_name]['train_raw']
        test_df = self.datasets[dataset_name]['test']
        
        sensor_cols = get_sensor_columns(train_df)
        setting_cols = ['setting1', 'setting2', 'setting3']
        
        drift_analysis = {}
        
        for col in sensor_cols + setting_cols:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(train_df[col], test_df[col])
            
            # Mann-Whitney U test
            mw_stat, mw_pvalue = stats.mannwhitneyu(train_df[col], test_df[col], alternative='two-sided')
            
            drift_analysis[col] = {
                'train_mean': train_df[col].mean(),
                'test_mean': test_df[col].mean(),
                'train_std': train_df[col].std(),
                'test_std': test_df[col].std(),
                'mean_difference': abs(test_df[col].mean() - train_df[col].mean()),
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'mw_statistic': mw_stat,
                'mw_pvalue': mw_pvalue,
                'potential_drift': (ks_pvalue < SIGNIFICANCE_LEVEL) or (mw_pvalue < SIGNIFICANCE_LEVEL)
            }
        
        drift_df = pd.DataFrame(drift_analysis).T
        
        if save_plots:
            self._plot_drift_analysis(dataset_name, train_df, test_df, drift_df)
        
        return drift_df
    
    def _plot_drift_analysis(self, dataset_name: str, train_df: pd.DataFrame, 
                           test_df: pd.DataFrame, drift_df: pd.DataFrame) -> None:
        """Plot distributions to visualize potential drift."""
        # Select sensors with most drift for visualization
        drifting_sensors = drift_df[drift_df['potential_drift']].sort_values('ks_statistic', ascending=False)
        
        if len(drifting_sensors) == 0:
            # If no drift, show first 6 sensors
            sensor_cols = get_sensor_columns(train_df)
            sensors_to_plot = sensor_cols[:6]
        else:
            sensors_to_plot = drifting_sensors.head(6).index.tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, sensor in enumerate(sensors_to_plot):
            ax = axes[i]
            
            # Plot distributions
            ax.hist(train_df[sensor], bins=50, alpha=0.7, label='Train', density=True)
            ax.hist(test_df[sensor], bins=50, alpha=0.7, label='Test', density=True)
            
            # Add KS test results
            ks_stat = drift_df.loc[sensor, 'ks_statistic']
            ks_p = drift_df.loc[sensor, 'ks_pvalue']
            
            ax.set_xlabel(sensor)
            ax.set_ylabel('Density')
            ax.set_title(f'{sensor}\\nKS = {ks_stat:.4f}, p = {ks_p:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{dataset_name} - Data Drift Analysis', fontsize=16)
        plt.tight_layout()
        save_plot(fig, f'{dataset_name}_drift_analysis.png')
        plt.show()
    
    def bootstrap_sensor_correlations(self, dataset_name: str, 
                                    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS) -> Dict:
        """Bootstrap sensor-RUL correlations to quantify uncertainty."""
        train_df = self.datasets[dataset_name]['train']
        sensor_cols = get_sensor_columns(train_df)
        
        bootstrap_results = {}
        
        for sensor in sensor_cols:
            correlations = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                bootstrap_idx = np.random.choice(len(train_df), size=len(train_df), replace=True)
                bootstrap_df = train_df.iloc[bootstrap_idx]
                
                correlation = bootstrap_df[sensor].corr(bootstrap_df['RUL'])
                if not np.isnan(correlation):
                    correlations.append(correlation)
            
            if correlations:
                bootstrap_results[sensor] = {
                    'mean': np.mean(correlations),
                    'std': np.std(correlations),
                    'ci_lower': np.percentile(correlations, 2.5),
                    'ci_upper': np.percentile(correlations, 97.5),
                    'is_stable': np.std(correlations) < 0.1  # Threshold for stability
                }
        
        return bootstrap_results
