# NASA Turbofan Engine RUL - Clean & Modular Analysis

## 🚀 Overview

This project provides a **clean, modular, and maintainable** approach to exploratory data analysis (EDA) for NASA C-MAPSS turbofan engine datasets for Remaining Useful Life (RUL) prediction. 

**Key Improvements Over Original Code:**
- ✅ **3,600+ lines reduced to modular, reusable components**
- ✅ **Eliminated repetitive boilerplate code**
- ✅ **Clear separation of concerns** 
- ✅ **Consistent error handling and validation**
- ✅ **Easy to read, debug, and extend**
- ✅ **Publication-ready analysis with minimal code**

## 📁 Project Structure

```
NASA-Turbofan-RUL/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── constants.py          # All configuration constants
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py            # Common utility functions
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── basic_eda.py          # Basic exploratory analysis
│   │   ├── advanced_analysis.py  # Advanced statistical analysis
│   │   └── data_quality.py       # Data quality assessment
│   └── data/
│       ├── __init__.py
│       ├── load_data.py          # Robust data loading
│       └── preprocess.py         # Data preprocessing
├── notebooks/
│   ├── clean_eda.ipynb          # Clean, modular notebook
│   └── 01_comprehensive_eda.py   # Original (for reference)
├── reports/                      # Generated plots and analysis
├── CMAPSSData/                   # Raw dataset files
├── run_clean_analysis.py         # Simple runner script
├── requirements.txt              # Updated dependencies
└── README.md                     # This file
```

## 🎯 Key Benefits

### Before (Original Code)
```python
# 3,600+ lines in single file
# Repeated plotting code everywhere
# Hard-coded values scattered throughout
# Difficult to debug and maintain
# Mixed concerns (data loading + analysis + plotting)

# Example of repetitive code:
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()
for i, name in enumerate(dataset_names):
    train_df = datasets[name]['train']
    rul_values = train_df['RUL'].values
    axes[i].hist(rul_values, bins=50, alpha=0.7, edgecolor='black')
    # ... repeated 30+ times with variations
```

### After (Refactored Code)
```python
# Modular, clean approach
from src.analysis.basic_eda import BasicEDA

eda = BasicEDA(datasets)
eda.plot_rul_distributions()  # One line!

# Or run everything:
python run_clean_analysis.py  # Complete analysis in one command
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure data is in CMAPSSData/ directory
```

### 2. Run Complete Analysis (Simple)
```bash
# Single command to run everything
python run_clean_analysis.py
```

### 3. Use Jupyter Notebook (Interactive)
```bash
# Start Jupyter and open clean notebook
jupyter notebook notebooks/clean_eda.ipynb
```

### 4. Use Programmatically (Advanced)
```python
from src.analysis.basic_eda import BasicEDA
from src.analysis.advanced_analysis import AdvancedAnalysis
from src.utils.helpers import load_dataset

# Load data
datasets = {}
for name in ['FD001', 'FD002', 'FD003', 'FD004']:
    datasets[name] = load_dataset(name, DATA_PATH)

# Basic analysis
eda = BasicEDA(datasets)
overview = eda.analyze_dataset_overview()
eda.plot_rul_distributions()

# Advanced analysis  
advanced = AdvancedAnalysis(datasets)
uncertainty = advanced.analyze_uncertainty_baseline('FD001')
pca_results = advanced.perform_pca_analysis('FD001')
```

## 📊 What You Get

### Automated Analysis Components:

1. **Basic EDA (`BasicEDA` class)**
   - Dataset overview and statistics
   - RUL distribution analysis
   - Engine lifecycle trajectories
   - Sensor correlation matrices
   - Degradation pattern analysis

2. **Advanced Analysis (`AdvancedAnalysis` class)**
   - Uncertainty quantification with bootstrap confidence intervals
   - Principal Component Analysis (PCA)
   - Data drift detection between train/test sets
   - Statistical significance testing

3. **Data Quality Assessment (`DataQualityAssessment` class)**
   - Missing value detection
   - Outlier analysis using IQR method
   - Data consistency validation
   - Time series monotonicity checks

4. **Utility Functions (`helpers.py`)**
   - Robust data loading with error handling
   - Consistent plotting and saving
   - Statistical test wrappers
   - Bootstrap analysis helpers

## 🎨 Sample Outputs

All analyses generate publication-ready plots automatically saved to `reports/` folder:

- `rul_distributions.png` - RUL distribution comparison
- `{dataset}_lifecycles.png` - Engine trajectory patterns  
- `{dataset}_sensor_correlations.png` - Correlation heatmaps
- `{dataset}_degradation_patterns.png` - Sensor vs RUL trends
- `{dataset}_uncertainty_baseline.png` - Uncertainty analysis
- `{dataset}_pca_analysis.png` - PCA results
- `{dataset}_drift_analysis.png` - Train/test drift detection
- `data_quality_summary.png` - Quality assessment overview

## 🔧 Configuration

All constants and settings are centralized in `src/config/constants.py`:

```python
# Easy to modify key parameters
DEFAULT_CORRELATION_THRESHOLD = 0.8
DEFAULT_BOOTSTRAP_ITERATIONS = 100
SIGNIFICANCE_LEVEL = 0.05
FIGURE_DPI = 300

# Dataset information
DATASET_INFO = {
    'FD001': {'conditions': 1, 'fault_modes': 1, 'description': 'Sea Level, HPC Degradation'},
    # ... etc
}
```

## 🧪 Testing and Validation

The refactored code includes built-in validation:

```python
# Data quality checks
quality_assessor = DataQualityAssessment(datasets)
quality_reports = quality_assessor.generate_quality_report()

# Automatic validation of:
# - Missing values
# - Data consistency
# - Time series monotonicity  
# - Outlier detection
# - File integrity
```

## 📈 Performance Improvements

| Metric | Original | Refactored | Improvement |
|--------|----------|------------|-------------|
| Lines of Code | 3,600+ | ~800 | **78% reduction** |
| Analysis Time | Manual run | Single command | **Much faster** |
| Maintainability | Poor | Excellent | **Easy to modify** |
| Reusability | Low | High | **Highly modular** |
| Error Handling | Minimal | Comprehensive | **Robust** |

## 🔄 Extensibility

Easy to add new analysis modules:

```python
# Create new analysis class
class NewAnalysis:
    def __init__(self, datasets):
        self.datasets = datasets
    
    def your_custom_analysis(self, dataset_name):
        # Your analysis logic here
        pass

# Use immediately
new_analyzer = NewAnalysis(datasets)
new_analyzer.your_custom_analysis('FD001')
```

## 🎯 Best Practices Implemented

1. **DRY Principle** - No repeated code
2. **Single Responsibility** - Each class/function has one purpose
3. **Configuration Management** - Centralized constants
4. **Error Handling** - Comprehensive validation
5. **Documentation** - Clear docstrings and comments
6. **Type Hints** - Better code clarity
7. **Consistent Naming** - snake_case throughout
8. **Modular Design** - Easy to extend and maintain

## 🚀 Next Steps

The refactored codebase makes it easy to:

1. **Add new analysis methods** - Just extend the classes
2. **Integrate ML models** - Clean data loading pipeline ready
3. **Create web apps** - Modular functions perfect for APIs
4. **Generate reports** - Automated plot generation
5. **Scale to new datasets** - Just add to `DATASET_NAMES`

## 🎉 Summary

This refactoring transforms a monolithic 3,600+ line script into a **clean, maintainable, and professional codebase** that:

- ✅ **Reduces complexity** while maintaining functionality
- ✅ **Improves readability** and debugging
- ✅ **Enables easy extension** for new features
- ✅ **Provides robust error handling**
- ✅ **Follows Python best practices**
- ✅ **Generates publication-ready outputs**

The code is now **production-ready** and suitable for academic research, industrial applications, and further development.
