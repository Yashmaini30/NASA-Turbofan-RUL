"""
Data quality assessment and validation functions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy import stats

from src.utils.helpers import save_plot, get_sensor_columns

class DataQualityAssessment:
    """Handles data quality assessment and validation."""
    
    def __init__(self, datasets: Dict[str, Dict[str, pd.DataFrame]]):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
    
    def assess_data_quality(self, dataset_name: str) -> Dict:
        """Comprehensive data quality assessment."""
        train_df = self.datasets[dataset_name]['train']
        test_df = self.datasets[dataset_name]['test']
        
        quality_report = {
            'dataset': dataset_name,
            'train_shape': train_df.shape,
            'test_shape': test_df.shape,
            'train_missing_values': train_df.isnull().sum().sum(),
            'test_missing_values': test_df.isnull().sum().sum(),
            'train_duplicates': train_df.duplicated().sum(),
            'test_duplicates': test_df.duplicated().sum(),
        }
        
        # Check for outliers using IQR method
        sensor_cols = get_sensor_columns(train_df)
        outlier_analysis = self._detect_outliers(train_df, sensor_cols)
        quality_report.update(outlier_analysis)
        
        # Data consistency checks
        consistency_checks = self._check_data_consistency(train_df, test_df)
        quality_report.update(consistency_checks)
        
        return quality_report
    
    def _detect_outliers(self, df: pd.DataFrame, sensor_cols: List[str]) -> Dict:
        """Detect outliers using IQR method."""
        outlier_counts = {}
        total_outliers = 0
        
        for sensor in sensor_cols:
            Q1 = df[sensor].quantile(0.25)
            Q3 = df[sensor].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[sensor] < lower_bound) | (df[sensor] > upper_bound)).sum()
            outlier_counts[sensor] = outliers
            total_outliers += outliers
        
        return {
            'outlier_counts': outlier_counts,
            'total_outliers': total_outliers,
            'outlier_percentage': (total_outliers / (len(df) * len(sensor_cols))) * 100
        }
    
    def _check_data_consistency(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Check data consistency between train and test sets."""
        # Check column consistency
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        # Check for monotonic time cycles within each engine
        time_consistency_train = self._check_time_monotonicity(train_df)
        time_consistency_test = self._check_time_monotonicity(test_df)
        
        # Check RUL consistency (should decrease with time)
        rul_consistency = self._check_rul_consistency(train_df)
        
        return {
            'column_consistency': train_cols == test_cols,
            'missing_train_cols': test_cols - train_cols,
            'missing_test_cols': train_cols - test_cols,
            'time_monotonic_train': time_consistency_train,
            'time_monotonic_test': time_consistency_test,
            'rul_consistency': rul_consistency
        }
    
    def _check_time_monotonicity(self, df: pd.DataFrame) -> Dict:
        """Check if time cycles are monotonic for each engine."""
        results = {}
        
        for engine_id in df['unit_id'].unique():
            engine_data = df[df['unit_id'] == engine_id].sort_values('time_cycles')
            time_cycles = engine_data['time_cycles'].values
            
            # Check if strictly increasing
            is_monotonic = np.all(np.diff(time_cycles) > 0)
            results[engine_id] = is_monotonic
        
        non_monotonic_count = sum(1 for is_mono in results.values() if not is_mono)
        
        return {
            'all_monotonic': non_monotonic_count == 0,
            'non_monotonic_engines': non_monotonic_count,
            'total_engines': len(results)
        }
    
    def _check_rul_consistency(self, df: pd.DataFrame) -> Dict:
        """Check if RUL decreases monotonically for each engine."""
        if 'RUL' not in df.columns:
            return {'has_rul': False}
        
        results = {}
        
        for engine_id in df['unit_id'].unique():
            engine_data = df[df['unit_id'] == engine_id].sort_values('time_cycles')
            rul_values = engine_data['RUL'].values
            
            # Check if strictly decreasing
            is_decreasing = np.all(np.diff(rul_values) <= 0)
            results[engine_id] = is_decreasing
        
        non_decreasing_count = sum(1 for is_dec in results.values() if not is_dec)
        
        return {
            'has_rul': True,
            'all_decreasing': non_decreasing_count == 0,
            'non_decreasing_engines': non_decreasing_count,
            'total_engines': len(results)
        }
    
    def generate_quality_report(self, save_plots: bool = True) -> Dict:
        """Generate comprehensive quality report for all datasets."""
        all_reports = {}
        
        for dataset_name in self.dataset_names:
            report = self.assess_data_quality(dataset_name)
            all_reports[dataset_name] = report
        
        if save_plots:
            self._plot_quality_summary(all_reports)
        
        self._print_quality_summary(all_reports)
        
        return all_reports
    
    def _plot_quality_summary(self, reports: Dict) -> None:
        """Plot data quality summary."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Dataset sizes
        datasets = list(reports.keys())
        train_sizes = [reports[d]['train_shape'][0] for d in datasets]
        test_sizes = [reports[d]['test_shape'][0] for d in datasets]
        
        x_pos = np.arange(len(datasets))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_sizes, width, label='Train', alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, test_sizes, width, label='Test', alpha=0.8)
        axes[0, 0].set_xlabel('Dataset')
        axes[0, 0].set_ylabel('Number of Records')
        axes[0, 0].set_title('Dataset Sizes')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(datasets)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Outlier percentages
        outlier_percentages = [reports[d]['outlier_percentage'] for d in datasets]
        
        axes[0, 1].bar(datasets, outlier_percentages, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('Outlier Percentage (%)')
        axes[0, 1].set_title('Outlier Detection Results')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Data consistency scores
        consistency_scores = []
        for d in datasets:
            score = 0
            report = reports[d]
            
            # Add points for good quality indicators
            if report['column_consistency']: score += 1
            if report['time_monotonic_train']['all_monotonic']: score += 1
            if report['time_monotonic_test']['all_monotonic']: score += 1
            if report.get('rul_consistency', {}).get('all_decreasing', False): score += 1
            if report['train_missing_values'] == 0: score += 1
            if report['test_missing_values'] == 0: score += 1
            
            consistency_scores.append(score)
        
        axes[1, 0].bar(datasets, consistency_scores, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Dataset')
        axes[1, 0].set_ylabel('Quality Score (0-6)')
        axes[1, 0].set_title('Data Consistency Scores')
        axes[1, 0].set_ylim(0, 6)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        
        summary_text = "Data Quality Summary:\\n\\n"
        for dataset in datasets:
            report = reports[dataset]
            summary_text += f"{dataset}:\\n"
            summary_text += f"  Train: {report['train_shape'][0]:,} records\\n"
            summary_text += f"  Test: {report['test_shape'][0]:,} records\\n"
            summary_text += f"  Missing values: {report['train_missing_values'] + report['test_missing_values']}\\n"
            summary_text += f"  Outliers: {report['outlier_percentage']:.1f}%\\n"
            summary_text += f"  Quality score: {consistency_scores[datasets.index(dataset)]}/6\\n\\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('Data Quality Assessment Summary', fontsize=16)
        plt.tight_layout()
        save_plot(fig, 'data_quality_summary.png')
        plt.show()
    
    def _print_quality_summary(self, reports: Dict) -> None:
        """Print data quality summary to console."""
        print("\\n" + "="*80)
        print("DATA QUALITY ASSESSMENT SUMMARY")
        print("="*80)
        
        for dataset_name, report in reports.items():
            print(f"\\n{dataset_name}:")
            print(f"  Shape (train/test): {report['train_shape']} / {report['test_shape']}")
            print(f"  Missing values: {report['train_missing_values'] + report['test_missing_values']}")
            print(f"  Duplicates: {report['train_duplicates'] + report['test_duplicates']}")
            print(f"  Total outliers: {report['total_outliers']} ({report['outlier_percentage']:.2f}%)")
            print(f"  Time monotonicity: Train={report['time_monotonic_train']['all_monotonic']}, Test={report['time_monotonic_test']['all_monotonic']}")
            
            if report.get('rul_consistency', {}).get('has_rul', False):
                print(f"  RUL consistency: {report['rul_consistency']['all_decreasing']}")
        
        print("\\n" + "="*80)
