"""
Configuration constants for NASA Turbofan RUL analysis.
"""
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent.parent

# Data paths
DATA_PATH = BASE_DIR / 'CMAPSSData'
REPORTS_PATH = BASE_DIR / 'reports'

# Dataset information
DATASET_NAMES = ['FD001', 'FD002', 'FD003', 'FD004']

DATASET_INFO = {
    'FD001': {'conditions': 1, 'fault_modes': 1, 'description': 'Sea Level, HPC Degradation'},
    'FD002': {'conditions': 6, 'fault_modes': 1, 'description': 'Six Conditions, HPC Degradation'},
    'FD003': {'conditions': 1, 'fault_modes': 2, 'description': 'Sea Level, HPC + Fan Degradation'},
    'FD004': {'conditions': 6, 'fault_modes': 2, 'description': 'Six Conditions, HPC + Fan Degradation'}
}

# Column definitions
COLUMN_NAMES = [
    'unit_id', 'time_cycles', 'setting1', 'setting2', 'setting3'
] + [f'sensor_{i}' for i in range(1, 22)]

# Sensor descriptions for better interpretation
SENSOR_DESCRIPTIONS = {
    'sensor_1': 'T2 - Total temperature at fan inlet (°R)',
    'sensor_2': 'T24 - Total temperature at LPC outlet (°R)', 
    'sensor_3': 'T30 - Total temperature at HPC outlet (°R)',
    'sensor_4': 'T50 - Total temperature at LPT outlet (°R)',
    'sensor_5': 'P2 - Pressure at fan inlet (psia)',
    'sensor_6': 'P15 - Total pressure in bypass-duct (psia)',
    'sensor_7': 'P30 - Total pressure at HPC outlet (psia)',
    'sensor_8': 'Nf - Physical fan speed (rpm)',
    'sensor_9': 'Nc - Physical core speed (rpm)',
    'sensor_10': 'epr - Engine pressure ratio (P50/P2)',
    'sensor_11': 'Ps30 - Static pressure at HPC outlet (psia)',
    'sensor_12': 'phi - Ratio of fuel flow to Ps30 (pps/psia)',
    'sensor_13': 'NRf - Corrected fan speed (rpm)',
    'sensor_14': 'NRc - Corrected core speed (rpm)',
    'sensor_15': 'BPR - Bypass Ratio',
    'sensor_16': 'farB - Burner fuel-air ratio',
    'sensor_17': 'htBleed - Bleed Enthalpy',
    'sensor_18': 'Nf_dmd - Demanded fan speed (rpm)',
    'sensor_19': 'PCNfR_dmd - Demanded corrected fan speed (rpm)',
    'sensor_20': 'W31 - HPT coolant bleed (lbm/s)',
    'sensor_21': 'W32 - LPT coolant bleed (lbm/s)'
}

# Analysis parameters
DEFAULT_CORRELATION_THRESHOLD = 0.8
DEFAULT_BOOTSTRAP_ITERATIONS = 100
DEFAULT_N_CLUSTERS = 4
DEFAULT_MAX_ENGINES_PLOT = 10

# Plotting configuration
PLOT_STYLE = 'seaborn-v0_8'
PLOT_PALETTE = "husl"
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Statistical thresholds
SIGNIFICANCE_LEVEL = 0.05
LOW_VARIANCE_THRESHOLD = 0.01
DRIFT_EFFECT_SIZE_THRESHOLD = 0.1
