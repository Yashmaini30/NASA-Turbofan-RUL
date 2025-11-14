"""
Evaluation metrics for TFT RUL prediction
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_predictions(y_true, y_pred):
    """
    Calculate standard evaluation metrics
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        
    Returns:
        Dictionary with RMSE, MAE, and R2 metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def calculate_phm08_score(y_true, y_pred):
    """
    PHM08 Challenge asymmetric scoring function
    
    This is the official scoring metric for C-MAPSS dataset.
    Penalizes late predictions MORE than early predictions:
    - Early prediction (y_pred > y_true): exp(-error/13) - 1
    - Late prediction (y_pred < y_true): exp(error/10) - 1
    
    Lower is better. Perfect prediction = 0.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        
    Returns:
        Total PHM08 score (lower is better)
    """
    errors = y_pred - y_true
    
    scores = np.where(
        errors < 0,
        np.exp(-errors / 13) - 1,  # Late prediction (more penalty)
        np.exp(errors / 10) - 1     # Early prediction (less penalty)
    )
    
    total_score = np.sum(scores)
    
    return total_score


def evaluate_model(model, test_dataset, test_df, rul_df):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained TFT model
        test_dataset: Test TimeSeriesDataSet
        test_df: Test dataframe with unit_id and time_idx
        rul_df: Ground truth RUL dataframe
        
    Returns:
        metrics: Dictionary with all evaluation metrics
        test_predictions: Array of predicted RUL values
        test_actuals: Array of actual RUL values
    """
    # Generate predictions
    predictions = model.predict(
        test_dataset,
        mode='prediction',
        return_index=True,
        return_decoder_lengths=True
    )
    
    # predictions is now a tuple: (predictions_tensor, index)
    pred_values = predictions[0]  # Shape: (n_samples, prediction_length, n_quantiles)
    pred_index = predictions[1]   # DataFrame with decoder info
    
    # Extract predictions and actuals
    # Get last prediction for each engine
    test_predictions = []
    test_actuals = []
    
    # Group by unit_id and get last prediction
    for unit_id in test_df['unit_id'].unique():
        # Find indices for this engine in predictions
        engine_mask = pred_index['unit_id'] == unit_id
        
        if engine_mask.sum() > 0:
            # Get the last prediction for this engine (highest time_idx)
            engine_preds_idx = pred_index[engine_mask].index[-1]
            
            # Extract median quantile prediction (index 1 for [0.1, 0.5, 0.9])
            pred_rul = pred_values[engine_preds_idx, 0, 1].item()
            test_predictions.append(pred_rul)
            
            # Get actual RUL from ground truth
            actual_rul = rul_df[rul_df['unit_id'] == int(unit_id)]['true_rul'].values[0]
            test_actuals.append(actual_rul)
    
    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)
    
    # Calculate metrics
    metrics = evaluate_predictions(test_actuals, test_predictions)
    metrics['PHM08_Score'] = calculate_phm08_score(test_actuals, test_predictions)
    
    return metrics, test_predictions, test_actuals
