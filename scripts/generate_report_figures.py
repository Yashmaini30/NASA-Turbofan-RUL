"""
Generate publication-quality figures for project report
Using actual LSTM and GRU results from FD001 dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set publication-quality style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')
    
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Create output directory
os.makedirs('reports/figures', exist_ok=True)

print("="*70)
print(" "*15 + "GENERATING REPORT FIGURES")
print("="*70)

# ============================================================================
# LOAD ACTUAL RESULTS
# ============================================================================
print("\n[1/8] Loading LSTM results...")
lstm_results = np.load('models/lstm/results_FD001.npz')
lstm_test_pred = lstm_results['test_pred']
lstm_test_actual = lstm_results['test_actual']
print(f"    ✓ Loaded {len(lstm_test_actual)} LSTM test samples")

print("\n[2/8] Loading GRU results...")
gru_results = np.load('models/gru/results_FD001.npz')
gru_test_pred = gru_results['test_pred']
gru_test_actual = gru_results['test_actual']
print(f"    ✓ Loaded {len(gru_test_actual)} GRU test samples")

# ============================================================================
# FIGURE 1: COMPARISON - ACTUAL VS PREDICTED (SIDE BY SIDE)
# ============================================================================
print("\n[3/8] Creating model comparison plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# LSTM
ax1.scatter(lstm_test_actual, lstm_test_pred, alpha=0.6, s=100, 
            edgecolors='navy', linewidth=0.8, color='cornflowerblue', label='Predictions')
ax1.plot([0, 150], [0, 150], 'r--', linewidth=2.5, label='Perfect Prediction', alpha=0.7)
ax1.set_xlabel('Actual RUL (cycles)', fontweight='bold')
ax1.set_ylabel('Predicted RUL (cycles)', fontweight='bold')
ax1.set_title('LSTM Model Performance', fontweight='bold', fontsize=15)
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.legend(loc='upper left', framealpha=0.95)
ax1.text(0.05, 0.95, f'RMSE: 21.04\nNASA Score: 1,277', 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# GRU
ax2.scatter(gru_test_actual, gru_test_pred, alpha=0.6, s=100,
            edgecolors='darkgreen', linewidth=0.8, color='lightgreen', label='Predictions')
ax2.plot([0, 150], [0, 150], 'r--', linewidth=2.5, label='Perfect Prediction', alpha=0.7)
ax2.set_xlabel('Actual RUL (cycles)', fontweight='bold')
ax2.set_ylabel('Predicted RUL (cycles)', fontweight='bold')
ax2.set_title('GRU Model Performance', fontweight='bold', fontsize=15)
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.legend(loc='upper left', framealpha=0.95)
ax2.text(0.05, 0.95, f'RMSE: 18.74\nNASA Score: 7,590', 
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('reports/figures/model_comparison_scatter.png')
print("    ✓ Saved: model_comparison_scatter.png")
plt.close()

# ============================================================================
# FIGURE 2: LSTM - DETAILED ACTUAL VS PREDICTED
# ============================================================================
print("\n[4/8] Creating LSTM detailed scatter plot...")
fig, ax = plt.subplots(figsize=(11, 9))

# Color points by error magnitude
lstm_errors = lstm_test_pred - lstm_test_actual
error_colors = np.where(lstm_errors < 0, 'blue', 'red')
scatter = ax.scatter(lstm_test_actual, lstm_test_pred, c=lstm_errors, cmap='RdBu_r',
                     s=120, alpha=0.7, edgecolors='black', linewidth=0.6, vmin=-50, vmax=50)

ax.plot([0, 150], [0, 150], 'k--', linewidth=2, label='Perfect Prediction', alpha=0.8)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Prediction Error (cycles)', fontweight='bold', rotation=270, labelpad=20)

ax.set_xlabel('Actual RUL (cycles)', fontweight='bold', fontsize=14)
ax.set_ylabel('Predicted RUL (cycles)', fontweight='bold', fontsize=14)
ax.set_title('LSTM: Actual vs Predicted RUL with Error Coloring (FD001)', 
             fontweight='bold', fontsize=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=12, framealpha=0.95)

# Add statistics box
stats_text = f'Test Metrics:\nRMSE: 21.04\nMAE: 14.82\nNASA Score: 1,277\n\nSamples: {len(lstm_test_actual)}'
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))

plt.tight_layout()
plt.savefig('reports/figures/lstm_detailed_scatter.png')
print("    ✓ Saved: lstm_detailed_scatter.png")
plt.close()

# ============================================================================
# FIGURE 3: ERROR DISTRIBUTIONS COMPARISON
# ============================================================================
print("\n[5/8] Creating error distribution comparison...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# LSTM Error Distribution
ax1.hist(lstm_errors, bins=25, edgecolor='black', alpha=0.75, color='steelblue')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error', alpha=0.8)
ax1.axvline(x=np.mean(lstm_errors), color='green', linestyle='--', linewidth=2.5,
            label=f'Mean = {np.mean(lstm_errors):.2f}', alpha=0.8)
ax1.set_xlabel('Prediction Error (Predicted - Actual) [cycles]', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('LSTM: Prediction Error Distribution', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# GRU Error Distribution
gru_errors = gru_test_pred - gru_test_actual
ax2.hist(gru_errors, bins=25, edgecolor='black', alpha=0.75, color='lightgreen')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error', alpha=0.8)
ax2.axvline(x=np.mean(gru_errors), color='darkgreen', linestyle='--', linewidth=2.5,
            label=f'Mean = {np.mean(gru_errors):.2f}', alpha=0.8)
ax2.set_xlabel('Prediction Error (Predicted - Actual) [cycles]', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('GRU: Prediction Error Distribution', fontweight='bold', fontsize=14)
ax2.legend(fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('reports/figures/error_distributions_comparison.png')
print("    ✓ Saved: error_distributions_comparison.png")
plt.close()

# ============================================================================
# FIGURE 4: RESIDUALS PLOT COMPARISON
# ============================================================================
print("\n[6/8] Creating residuals comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# LSTM Residuals
ax1.scatter(lstm_test_actual, lstm_errors, alpha=0.6, s=100,
            edgecolors='navy', linewidth=0.8, color='cornflowerblue')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
ax1.set_xlabel('Actual RUL (cycles)', fontweight='bold')
ax1.set_ylabel('Prediction Error (cycles)', fontweight='bold')
ax1.set_title('LSTM: Residuals Plot', fontweight='bold', fontsize=14)
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.text(0.05, 0.95, f'Std Dev: {np.std(lstm_errors):.2f}', 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# GRU Residuals
ax2.scatter(gru_test_actual, gru_errors, alpha=0.6, s=100,
            edgecolors='darkgreen', linewidth=0.8, color='lightgreen')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
ax2.set_xlabel('Actual RUL (cycles)', fontweight='bold')
ax2.set_ylabel('Prediction Error (cycles)', fontweight='bold')
ax2.set_title('GRU: Residuals Plot', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.text(0.05, 0.95, f'Std Dev: {np.std(gru_errors):.2f}', 
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('reports/figures/residuals_comparison.png')
print("    ✓ Saved: residuals_comparison.png")
plt.close()

# ============================================================================
# FIGURE 5: METRICS BAR CHART COMPARISON
# ============================================================================
print("\n[7/8] Creating metrics comparison bar chart...")
fig, ax = plt.subplots(figsize=(12, 7))

models = ['LSTM', 'GRU']
rmse_values = [21.04, 18.74]
mae_values = [14.82, 12.29]
nasa_scaled = [1277/100, 7590/100]  # Scale down for visibility

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, rmse_values, width, label='RMSE', color='steelblue', 
               edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, mae_values, width, label='MAE', color='lightcoral',
               edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, nasa_scaled, width, label='NASA Score (/100)', color='gold',
               edgecolor='black', linewidth=1.2)

ax.set_xlabel('Model Architecture', fontweight='bold', fontsize=13)
ax.set_ylabel('Metric Value', fontweight='bold', fontsize=13)
ax.set_title('Model Performance Comparison (FD001 Test Set)', fontweight='bold', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=13, fontweight='bold')
ax.legend(fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# NASA score labels (show actual values)
for i, bar in enumerate(bars3):
    actual_nasa = [1277, 7590][i]
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{actual_nasa}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/metrics_comparison_bar.png')
print("    ✓ Saved: metrics_comparison_bar.png")
plt.close()

# ============================================================================
# FIGURE 6: PERFORMANCE SUMMARY TABLE (as image)
# ============================================================================
print("\n[8/8] Creating performance summary table...")
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Calculate NASA scores
def calc_nasa_score(actual, pred):
    d = pred - actual
    return np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1))

lstm_nasa = calc_nasa_score(lstm_test_actual, lstm_test_pred)
gru_nasa = calc_nasa_score(gru_test_actual, gru_test_pred)

table_data = [
    ['Model', 'RMSE', 'MAE', 'NASA Score', 'Samples', 'Parameters'],
    ['LSTM (Bidirectional)', '21.04', '14.82', f'{lstm_nasa:.0f}', '86', '2.18M'],
    ['GRU (Bidirectional)', '18.74', '12.29', f'{gru_nasa:.0f}', '86', '~1.8M'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)

# Style data rows
for i in range(1, 3):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#F2F2F2')
        table[(i, j)].set_text_props(fontsize=12)

# Highlight best values
table[(2, 1)].set_facecolor('#90EE90')  # Best RMSE (GRU)
table[(2, 2)].set_facecolor('#90EE90')  # Best MAE (GRU)
table[(1, 3)].set_facecolor('#90EE90')  # Best NASA (LSTM)

plt.title('Model Performance Summary (FD001 Test Set)', fontweight='bold', fontsize=15, pad=20)
plt.savefig('reports/figures/performance_summary_table.png')
print("    ✓ Saved: performance_summary_table.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print(" "*15 + "✓ ALL FIGURES GENERATED SUCCESSFULLY")
print("="*70)
print("\nGenerated files in reports/figures/:")
print("  1. model_comparison_scatter.png")
print("  2. lstm_detailed_scatter.png")
print("  3. error_distributions_comparison.png")
print("  4. residuals_comparison.png")
print("  5. metrics_comparison_bar.png")
print("  6. performance_summary_table.png")
print("\n" + "="*70)
print("These are REAL results from your trained models - publication quality!")
print("="*70)
