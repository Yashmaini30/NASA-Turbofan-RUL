"""
Flask Web Application for NASA Turbofan RUL Analysis Dashboard
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

# Import our analysis modules
from src.config.constants import (
    DATASET_NAMES, DATASET_INFO, SENSOR_DESCRIPTIONS, 
    DATA_PATH, REPORTS_PATH
)
from src.utils.helpers import load_dataset

app = Flask(__name__)
app.secret_key = 'nasa_turbofan_rul_analysis_2025'

# Global data storage
datasets = {}
analysis_metadata = {}

def load_analysis_data():
    """Load datasets and prepare metadata for the web interface."""
    global datasets, analysis_metadata
    
    # Load datasets
    for name in DATASET_NAMES:
        dataset = load_dataset(name, DATA_PATH)
        if dataset:
            datasets[name] = dataset
    
    # Prepare metadata
    analysis_metadata = {
        'datasets': {},
        'analysis_types': {
            'basic_eda': {
                'name': 'Basic Exploratory Data Analysis',
                'description': 'Fundamental statistical analysis and data visualization',
                'sections': ['RUL Distributions', 'Engine Lifecycles', 'Sensor Correlations', 'Degradation Patterns']
            },
            'advanced_analysis': {
                'name': 'Advanced Statistical Analysis',
                'description': 'Sophisticated analytical techniques and uncertainty quantification',
                'sections': ['PCA Analysis', 'Drift Detection', 'Uncertainty Baseline', 'Statistical Failure Analysis']
            },
            'model_results': {
                'name': 'Model Performance Analysis',
                'description': 'Evaluation of LSTM model predictions and error analysis',
                'sections': ['Actual vs Predicted', 'Error Distribution', 'Residual Analysis', 'Prediction Bias']
            },
            'data_quality': {
                'name': 'Data Quality Assessment',
                'description': 'Comprehensive data validation and quality metrics',
                'sections': ['Quality Summary', 'Outlier Detection', 'Consistency Checks']
            }
        },
        'sensor_info': SENSOR_DESCRIPTIONS
    }
    
    # Dataset metadata
    for name in DATASET_NAMES:
        if name in datasets:
            train_df = datasets[name]['train']
            test_df = datasets[name]['test']
            analysis_metadata['datasets'][name] = {
                'info': DATASET_INFO[name],
                'train_shape': train_df.shape,
                'test_shape': test_df.shape,
                'train_engines': train_df['unit_id'].nunique(),
                'test_engines': test_df['unit_id'].nunique(),
                'max_cycles': train_df['time_cycles'].max(),
                'avg_life': train_df.groupby('unit_id')['time_cycles'].max().mean()
            }

def get_available_plots():
    """Get list of available plot files organized by category."""
    plots = {
        'overview': [],
        'basic_eda': {},
        'advanced_analysis': {},
        'model_results': {},
        'data_quality': []
    }
    
    reports_dir = Path(REPORTS_PATH)
    
    # Overview plots (dataset-wide)
    overview_files = ['rul_distributions.png', 'operational_settings.png', 'data_quality_summary.png']
    for file in overview_files:
        if (reports_dir / file).exists():
            plots['overview'].append(file)
    
    # Dataset-specific plots
    for dataset in DATASET_NAMES:
        plots['basic_eda'][dataset] = []
        plots['advanced_analysis'][dataset] = []
        plots['model_results'][dataset] = []
        
        # Basic EDA plots
        basic_patterns = ['lifecycles', 'sensor_correlations', 'degradation_patterns', 'failure_patterns']
        for pattern in basic_patterns:
            file = f"{dataset}_{pattern}.png"
            if (reports_dir / file).exists():
                plots['basic_eda'][dataset].append(file)
        
        # Advanced analysis plots
        advanced_patterns = ['pca_analysis', 'drift_analysis', 'uncertainty_baseline', 
                           'statistical_failure_analysis', 'temporal_features', 'time_series_embeddings']
        for pattern in advanced_patterns:
            file = f"{dataset}_{pattern}.png"
            if (reports_dir / file).exists():
                plots['advanced_analysis'][dataset].append(file)

        # Model Result plots
        if dataset == 'FD001':
             model_files = ['lstm_actual_vs_predicted.png', 'lstm_error_distribution.png', 'lstm_residuals.png', 'lstm_prediction_bias.png']
             for file in model_files:
                if (reports_dir / 'figures' / file).exists():
                    plots['model_results'][dataset].append(f"figures/{file}")
    
    return plots

@app.route('/')
def dashboard():
    """Main dashboard page."""
    plots = get_available_plots()
    return render_template('dashboard.html', 
                         datasets=analysis_metadata['datasets'],
                         analysis_types=analysis_metadata['analysis_types'],
                         plots=plots)

@app.route('/dataset/<dataset_name>')
def dataset_detail(dataset_name):
    """Detailed view for a specific dataset."""
    if dataset_name not in DATASET_NAMES:
        return "Dataset not found", 404
    
    plots = get_available_plots()
    dataset_info = analysis_metadata['datasets'].get(dataset_name, {})
    
    return render_template('dataset_detail.html',
                         dataset_name=dataset_name,
                         dataset_info=dataset_info,
                         plots=plots,
                         sensor_descriptions=SENSOR_DESCRIPTIONS)

@app.route('/analysis/<analysis_type>')
def analysis_view(analysis_type):
    """View for specific analysis type across all datasets."""
    if analysis_type not in analysis_metadata['analysis_types']:
        return "Analysis type not found", 404
    
    plots = get_available_plots()
    analysis_info = analysis_metadata['analysis_types'][analysis_type]
    
    return render_template('analysis_view.html',
                         analysis_type=analysis_type,
                         analysis_info=analysis_info,
                         plots=plots,
                         datasets=DATASET_NAMES)

@app.route('/overview')
def overview():
    """System overview and summary statistics."""
    plots = get_available_plots()
    
    # Calculate summary statistics
    summary_stats = {
        'total_datasets': len(DATASET_NAMES),
        'total_engines': sum(info['train_engines'] + info['test_engines'] 
                           for info in analysis_metadata['datasets'].values()),
        'total_cycles': sum(info['train_shape'][0] + info['test_shape'][0] 
                          for info in analysis_metadata['datasets'].values()),
        'total_sensors': 21,
        'analysis_generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return render_template('overview.html',
                         summary_stats=summary_stats,
                         datasets=analysis_metadata['datasets'],
                         plots=plots)

@app.route('/api/dataset/<dataset_name>/stats')
def dataset_stats_api(dataset_name):
    """API endpoint for dataset statistics."""
    if dataset_name not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    train_df = datasets[dataset_name]['train']
    test_df = datasets[dataset_name]['test']
    
    stats = {
        'train_stats': train_df.describe().to_dict(),
        'test_stats': test_df.describe().to_dict(),
        'sensor_variance': train_df[[col for col in train_df.columns if 'sensor' in col]].var().to_dict(),
        'correlation_summary': train_df.corr().abs().mean().to_dict()
    }
    
    return jsonify(stats)

@app.route('/reports/<path:filename>')
def serve_reports(filename):
    """Serve report images from the reports directory."""
    return send_from_directory(str(REPORTS_PATH), filename)

if __name__ == '__main__':
    print("🚀 Loading NASA Turbofan RUL Analysis Data...")
    load_analysis_data()
    print("✅ Data loaded successfully!")
    print("🌐 Starting Flask web application...")
    print("📊 Access your dashboard at: http://127.0.0.1:5000")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
