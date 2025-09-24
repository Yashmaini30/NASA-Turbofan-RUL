# Predicting Remaining Useful Life (RUL) of NASA Turbofan Engines

This project aims to predict the Remaining Useful Life (RUL) of NASA turbofan engines using deep learning techniques. By leveraging a modular and reproducible architecture, the project provides a robust framework for data processing, feature engineering, model training, and evaluation, with a special focus on uncertainty quantification.

## 1. Problem Definition and Overview

Predictive Maintenance is a data-driven approach that uses machine learning to predict when equipment will fail before it actually breaks down. This project focuses on predicting the Remaining Useful Life (RUL) of NASA turbofan engines, which is critical for aviation safety and cost optimization.

### The Core Problem
Turbofan engines are complex systems with multiple components that degrade over time due to operational stress, temperature variations, and mechanical wear. Unexpected engine failures can result in:
- **Safety risks** for passengers and crew
- **High maintenance costs** (unplanned downtime can reduce plant capacity by 5-20%)
- **Operational disruptions** and flight delays
- **Inventory management challenges**

Traditional maintenance approaches are either reactive (fix after failure) or preventive (scheduled regardless of condition). This predictive maintenance system aims to optimize maintenance timing by predicting exactly when an engine will need service.

## 2. Key Features

### Uncertainty Quantification
A unique aspect of this project is the incorporation of uncertainty quantification. Instead of providing a point estimate (e.g., "failure in 23 cycles"), the model provides a probabilistic forecast (e.g., "failure in 23 ± 4 cycles with 85% confidence"). This is crucial for making informed maintenance decisions under uncertainty.

### Modular and Reproducible Architecture
The project is built on a modular architecture that separates distinct logical concerns. This enhances code readability, simplifies debugging, and promotes reusability. A standardized directory structure, inspired by MLOps best practices, ensures a logical and predictable location for every project artifact.

## 3. Project Structure

The project follows a standardized directory structure to ensure modularity and reproducibility.

```
.
├── config/
│   └── config.yaml
├── data/
│   ├── 01_raw/
│   ├── 02_interim/
│   └── 03_processed/
├── docs/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   ├── evaluation/
│   ├── feature_engineering/
│   ├── modeling/
│   └── utils/
└── tests/
```

- **`config/`**: Central repository for configuration files, primarily `config.yaml`.
- **`data/`**: Stores all datasets, following a medallion architecture (raw, interim, processed).
- **`docs/`**: Project documentation.
- **`models/`**: Storage for trained and serialized model artifacts (`.pth`, `.pkl`).
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA) and prototyping.
- **`reports/`**: Generated outputs like EDA reports, performance plots, and figures.
- **`scripts/`**: Standalone executable scripts for running pipelines.
- **`src/`**: Core application source code, structured as a Python package.
- **`tests/`**: Unit and integration tests for ensuring code correctness and reliability.

## 4. Methodology

### Phase 1: Data Engineering and Exploratory Data Analysis (EDA)

#### Data Ingestion and Structure
Data ingestion is managed via a central configuration file (`config.yaml`) to maintain a modular approach.

```python
# Example config structure
config = {
    'data': {
        'raw_path': 'data/raw/CMAPSS/',
        'processed_path': 'data/processed/',
        'train_files': ['train_FD001.txt', 'train_FD002.txt', ...],
        'test_files': ['test_FD001.txt', 'test_FD002.txt', ...],
        'rul_files': ['RUL_FD001.txt', 'RUL_FD002.txt', ...]
    },
    'features': {
        'sensor_columns': ['T2', 'T24', 'T30', ...],
        'operational_settings': ['setting1', 'setting2', 'setting3'],
        'drop_columns': [] # Sensors with constant values
    }
}
```

#### Comprehensive EDA
A thorough EDA is conducted to understand the data's underlying characteristics. This involves:
- **Sensor Correlation Analysis**: Identifying sensors with monotonic degradation patterns.
- **Operating Condition Impact**: Analyzing how different flight conditions affect sensor readings.
- **Missing Value Patterns**: Handling sensor noise and outliers.
- **Degradation Visualization**: Plotting sensor trajectories over engine lifecycles.
- **Statistical Profiling**: Calculating mean, std, and skewness for each sensor.

#### Data Drift Detection
Drift detection is implemented to identify:
- **Covariate Shift**: Changes in input distributions between training and testing sets.
- **Concept Drift**: Changes in the relationship between features and RUL.
- **Temporal Drift**: How sensor patterns change over time.

## 5. Getting Started

### Prerequisites
- Python 3.8+
- Poetry (for dependency management)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Yashmaini30/NASA-Turbofan-RUL.git
   cd NASA-Turbofan-RUL
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Train the Model**:
   To train the GRU model, run the training script:
   ```bash
   python train_gru.py
   ```

2. **Launch the Web Application**:
   The project includes a Flask application for interacting with the model.
   ```bash
   python flask_app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000` to view the application.
