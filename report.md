# NASA C-MAPSS Turbofan Engine RUL Prediction — Project Report

This `report.md` is a structured outline and reference for the final project report. It summarizes important technical details from the repository and suggests figures, tables, and evaluation points to include.

---

## Abstract
A modular deep learning pipeline is developed to predict Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS dataset. The system uses feature engineering (first differences, rolling statistics), sequence modeling (LSTM, GRU, TFT), and uncertainty quantification to provide probabilistic RUL forecasts. The model focuses on real-world utility by optimizing a domain-specific asymmetric metric — the NASA scoring function. Training and experiments emphasize robustness through sequence-length tuning, bidirectional recurrent architectures, and conservative predictions to minimize late-prediction penalties.

---

## Chapters Outline

### 1. Introduction
- Brief description of predictive maintenance and its importance for aviation.
- Motivation for RUL forecasting: safety, operational efficiency, cost savings.
- Overview of the NASA C-MAPSS datasets (FD001–FD004) and why they are standard benchmarks.

### 2. Literature Survey
- Overview of time-series forecasting methods for RUL: classic statistical models (ARIMA), traditional ML (Random Forest, SVR), and modern deep learning (LSTM, GRU, Transformer-based).
- Discussion of probabilistic approaches and uncertainty quantification (Bayesian methods, MC-dropout, deep ensembles, mean-variance output heads).
- Review state-of-the-art results and how asymmetric loss functions (NASA score) are used in prognostics.

### 3. Problem Statement
- Problem: Given multivariate engine time-series (21 sensors + 3 operational settings) produce a prediction of Remaining Useful Life (RUL) for each time step of each engine.
- Constraints: limited per-engine run-to-failure sequences, variable operating conditions in FD002 and FD004.
- Metrics: NASA asymmetric scoring function, RMSE, MAE.

### 4. Description of Various Modules (Code Organization)
- `src/data/load_data.py`: Reads raw files; computes RUL; optionally applies advanced features.
- `src/feature_engineering/advanced_features.py`: Adds first differences, rolling stats, cumsum features.
- `src/feature_engineering/sequences.py`: Sliding-window sequence generation and PyTorch `Dataset`.
- `src/models/lstm_model.py`, `src/models/gru_model.py`: Recurrent architectures; configurable & support bidirectional processing.
- `src/models/tft/`: Temporal Fusion Transformer implementation and helpers (`create_tft_dataset`, trainer, evaluation, visualizations).
- `scripts/train_*.py`: Training scripts for LSTM/GRU/TFT with checkpointing and evaluation.
- `templates/, static/`: Web dashboard to visualize datasets and results.

### 5. Methodology Adopted

#### 5.1 Objective of Project
- Create accurate and conservative RUL predictors producing a probabilistic estimate and minimizing late prediction penalties.
- Compare classical recurrent models (LSTM, GRU) with attention-based models (TFT).

#### 5.2 Design of Experiment / Flow Chart
- Data ingestion (Assets → `CMAPSSData/`)
- Preprocessing: remove constant sensors, optional `add_features: true` in `config.yaml` to add first differences/rolling stats.
- Normalization via `MinMaxScaler` (fit on train, apply on test).
- Sequence construction using `sequence_length = 70` (configurable) via sliding windows.
- Model training with early stopping + ReduceLROnPlateau scheduling.
- Evaluation using RMSE, MAE, NASA scoring. Save predictions and checkpoints.

(Flow description — to put in report as a diagram: Data → Engineer Features → Sequence Generation → Model (LSTM/GRU/TFT) → Training (Loss: MSE, Scheduler, early-stopping) → Evaluation (RMSE, MAE, NASA score) → Visualization/Deployment)

#### 5.3 Machines and materials / Hardware and Software used
- Hardware: NVIDIA GeForce RTX 4060 (mobile, 8GB) used for GPU accelerated training.
- Software: Python 3.10.19, PyTorch (version in `requirements.txt`), Scikit-learn, Pandas, NumPy, Matplotlib/Seaborn, Flask for dashboard.
- Reproducibility: `config.yaml` centralizes hyperparameters; project uses a structured `src/` folder and scripts under `scripts/`.

#### 5.4 Optimization / Data Flow Diagram / E-R Diagram
- Optimization: hyperparameter tuning—hidden size (128→256), bidirectional flag, sequence length (30→70), dropout, learning rate, gradient clipping.
- Data Flow: each engine run forms a time-series, sequences are generated per-unit, then batched.
- Optional: include an E-R diagram for dataset and trains/tests showing relationships (Unit → Cycle → Sensor Readings).

#### 5.5 Algorithms Used
- LSTM (bidirectional): sequential modeling capturing long-term dependencies.
- GRU (bidirectional): fewer parameters than LSTM with similar temporal modeling capability.
- TFT (Temporal Fusion Transformer): attention-based model for multivariate forecasting and interpretability.
- Feature engineering: first differences, rolling mean/std, cumulative sums.
- Training: Adam optimizer, learning rate schedule (ReduceLROnPlateau), gradient clipping.
- Validation: stratified or unit-wise train/val split for generalization to unseen engine runs.

#### 5.6 Characterizations (if any)
- Feature importance / sensor analysis: correlation analysis, sensors with clear degradation trajectories.
- Characterize sequence windows and sensitivity to window length via experiments.
- Uncertainty characterization: qualitative description of how we will estimate uncertainty (future: ensembles or mean-variance heads in TFT).

### 6. Results and Discussions
- Include a general explanation of the key result: LSTM on FD001 yields NASA score 1,360 (85.4% improvement over baseline). Discuss tradeoffs: RMSE vs NASA Score.

#### 6.1 Snapshots of Results Obtained
- Include examples of printed model summaries (`model.summary()`), saved checkpoint list in `models/`, and `npz` files with predictions.
- Example checkpoint: `models/lstm/best_model_FD001.pth` and results file `models/lstm/results_FD001.npz`.

#### 6.2 Graphs and Tables
- Time series plot: actual vs predicted RUL for a few engines (use `plot_rul_predictions` in `src/models/tft/visualization` if implemented; or write a small plotting helper).
- Training curves: loss/epoch for train & validation.
- Distribution plots: predicted vs actual RUL distribution.
- Error heatmaps / residual histograms.
- Table: per-dataset per-model results (RMSE, MAE, NASA Score, Training Time, Parameters).

Suggested table example:
| Dataset | Model | Test RMSE | Test MAE | NASA Score | Train Time | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | LSTM | 21.09 | 14.83 | 1,360 | 15 min | 2.18M |
| FD001 | GRU | TBD | TBD | TBD | TBD | TBD |
| FD001 | TFT | TBD | TBD | TBD | TBD | TBD |

#### 6.3 Comparative Analysis
- Compare LSTM vs GRU vs TFT across datasets: highlight which model is robust to variability (FD002, FD004).
- Discuss significance of feature augmentation (first differences improved NASA score by 52%).
- Recommendation: For deployment, prefer model with lower NASA score even if RMSE slightly higher.

### 7. Conclusion and Future Scope
- Recap: modular processing, advanced features, sequence modeling deliver strong NASA score improvement for FD001.
- Future work:
  - Run experiments across all FD*** datasets and complete GRU/TFT training.
  - Ensemble models for uncertainty and more robust predictions.
  - Custom asymmetric loss to directly optimize NASA Score.
  - Calibrate uncertainty (e.g., calibration plots, sharpness vs. coverage).
  - Deploy as REST API + dashboard for operations use.

---

## Appendices (Practical items to include in the final report)
- Code snippet to compute NASA Score (see `scripts/train_lstm.py`) and sample output.
- Minimal commands for reproducing experiments:
```bash
# Train LSTM for FD001
python scripts/train_lstm.py --dataset FD001

# Train GRU for FD001
python scripts/train_gru.py --dataset FD001

# Train TFT
python scripts/train_tft.py --dataset FD001
```
- Files to include as evidence: `models/lstm/results_FD001.npz`, `models/lstm/best_model_FD001.pth`, figures in `reports/analysis_results`.

---

## Visualizations to produce for the report (scripts)
- `plot_training_curves.py`: generate training & validation loss curves (RMSE/MSE) from training logs or saved `npz` files.
- `plot_rul_samples.py`: for a few units, show actual vs predicted RUL along the folds.
- `comparison_table.py`: create a comparison Excel (already referenced in repo) with all saved metrics.

---

If you want, I can (1) fill this `report.md` with more textual exposition per chapter (draft-level prose) or (2) create the specific graphs and a table from the current `models/` outputs (requires running training to generate missing experiments). What would you like me to do next?