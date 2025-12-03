import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory for figures
os.makedirs('reports/figures', exist_ok=True)

# Load the saved results
results = np.load('models/lstm/results_FD001.npz')

# Extract test set predictions and actuals
test_pred = results['test_pred']
test_actual = results['test_actual']

print("="*60)
print("LSTM FD001 Test Results Analysis")
print("="*60)

# Calculate metrics
def calculate_metrics(actual, pred):
    """Calculate RMSE, MAE, and NASA Score"""
    errors = pred - actual
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    
    # NASA Score
    d = errors
    nasa_score = np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1))
    
    return {'rmse': rmse, 'mae': mae, 'nasa_score': nasa_score, 'errors': errors}

metrics = calculate_metrics(test_actual, test_pred)

print(f"\nTest Metrics:")
print(f"  RMSE:       {metrics['rmse']:.2f} cycles")
print(f"  MAE:        {metrics['mae']:.2f} cycles")
print(f"  NASA Score: {metrics['nasa_score']:.0f}")

print(f"\nPrediction Statistics:")
print(f"  Number of test engines: {len(test_actual)}")
print(f"  Mean error: {np.mean(metrics['errors']):.2f} cycles")
print(f"  Std error:  {np.std(metrics['errors']):.2f} cycles")
print(f"  Min error:  {np.min(metrics['errors']):.2f} cycles")
print(f"  Max error:  {np.max(metrics['errors']):.2f} cycles")

# Prediction bias analysis
early_preds = np.sum(metrics['errors'] < 0)  # predicted failure earlier than actual
late_preds = np.sum(metrics['errors'] > 0)   # predicted failure later than actual
print(f"\nPrediction Bias:")
print(f"  Conservative (early) predictions: {early_preds} ({100*early_preds/len(test_actual):.1f}%)")
print(f"  Optimistic (late) predictions:    {late_preds} ({100*late_preds/len(test_actual):.1f}%)")

print("\n" + "="*60)
print("Sample Predictions (first 10 engines):")
print("="*60)
print(f"{'Engine':<8} {'Actual RUL':<12} {'Predicted RUL':<15} {'Error':<10} {'Type'}")
print("-"*60)
for i in range(min(10, len(test_actual))):
    error = metrics['errors'][i]
    pred_type = "Early" if error < 0 else "Late"
    print(f"{i+1:<8} {test_actual[i]:<12.0f} {test_pred[i]:<15.2f} {error:<10.2f} {pred_type}")
print("="*60)

# ============================================================================
# GENERATE VISUALIZATIONS FOR REPORT
# ============================================================================
print("\n" + "="*60)
print("Generating Figures for Report...")
print("="*60)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Figure 1: Actual vs Predicted Scatter Plot
print("\n[1/4] Creating Actual vs Predicted scatter plot...")
plt.figure(figsize=(10, 8))
plt.scatter(test_actual, test_pred, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
plt.plot([0, 150], [0, 150], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual RUL (cycles)', fontsize=13, fontweight='bold')
plt.ylabel('Predicted RUL (cycles)', fontsize=13, fontweight='bold')
plt.title('LSTM Model: Actual vs Predicted RUL (FD001 Test Set)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/lstm_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("    ✓ Saved: reports/figures/lstm_actual_vs_predicted.png")
plt.close()

# Figure 2: Prediction Error Distribution
print("\n[2/4] Creating error distribution histogram...")
plt.figure(figsize=(10, 6))
plt.hist(metrics['errors'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.axvline(x=np.mean(metrics['errors']), color='green', linestyle='--', linewidth=2, 
            label=f'Mean Error = {np.mean(metrics["errors"]):.2f}')
plt.xlabel('Prediction Error (Predicted - Actual) [cycles]', fontsize=13, fontweight='bold')
plt.ylabel('Frequency', fontsize=13, fontweight='bold')
plt.title('Distribution of Prediction Errors (FD001 Test Set)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('reports/figures/lstm_error_distribution.png', dpi=300, bbox_inches='tight')
print("    ✓ Saved: reports/figures/lstm_error_distribution.png")
plt.close()

# Figure 3: Residuals Plot
print("\n[3/4] Creating residuals plot...")
plt.figure(figsize=(10, 6))
plt.scatter(test_actual, metrics['errors'], alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color='coral')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.xlabel('Actual RUL (cycles)', fontsize=13, fontweight='bold')
plt.ylabel('Prediction Error (cycles)', fontsize=13, fontweight='bold')
plt.title('Residuals Plot: Error vs Actual RUL', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('reports/figures/lstm_residuals.png', dpi=300, bbox_inches='tight')
print("    ✓ Saved: reports/figures/lstm_residuals.png")
plt.close()

# Figure 4: Prediction Bias Pie Chart
print("\n[4/4] Creating prediction bias pie chart...")
plt.figure(figsize=(8, 8))
labels = ['Conservative\n(Early Predictions)', 'Optimistic\n(Late Predictions)']
sizes = [early_preds, late_preds]
colors = ['#66b3ff', '#ff9999']
explode = (0.05, 0.05)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 13, 'fontweight': 'bold'})
plt.title('LSTM Prediction Bias Distribution (FD001)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/figures/lstm_prediction_bias.png', dpi=300, bbox_inches='tight')
print("    ✓ Saved: reports/figures/lstm_prediction_bias.png")
plt.close()

print("\n" + "="*60)
print("✓ All figures generated successfully!")
print("="*60)
print(f"\nFigures saved in: reports/figures/")
print("  - lstm_actual_vs_predicted.png")
print("  - lstm_error_distribution.png")
print("  - lstm_residuals.png")
print("  - lstm_prediction_bias.png")
