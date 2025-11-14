"""
Main runner script for NASA Turbofan RUL analysis.
Clean, simple interface for running the complete analysis.
"""
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.config.constants import DATA_PATH, DATASET_NAMES
from src.utils.helpers import setup_directories, load_dataset, print_section_header, AnalysisTimer
from src.analysis.basic_eda import BasicEDA
from src.analysis.advanced_analysis import AdvancedAnalysis
from src.analysis.data_quality import DataQualityAssessment

def main():
    """Run complete NASA Turbofan RUL analysis."""
    
    print_section_header("NASA TURBOFAN ENGINE RUL - COMPREHENSIVE ANALYSIS")
    
    # Setup
    results_path = setup_directories()
    print(f"Results will be saved to: {results_path}")
    
    # Load data
    print_section_header("DATA LOADING")
    datasets = {}
    
    with AnalysisTimer("Loading all datasets"):
        for name in DATASET_NAMES:
            dataset = load_dataset(name, DATA_PATH)
            if dataset:
                datasets[name] = dataset
                print(f"✓ Loaded {name}: Train {dataset['train'].shape}, Test {dataset['test'].shape}")
    
    if not datasets:
        print("❌ No datasets loaded successfully. Check data path and files.")
        return
    
    print(f"✅ Successfully loaded {len(datasets)}/{len(DATASET_NAMES)} datasets")
    
    # Data Quality Assessment
    print_section_header("DATA QUALITY ASSESSMENT")
    
    with AnalysisTimer("Data quality assessment"):
        quality_assessor = DataQualityAssessment(datasets)
        quality_reports = quality_assessor.generate_quality_report()
    
    # Basic EDA
    print_section_header("BASIC EXPLORATORY DATA ANALYSIS")
    
    with AnalysisTimer("Basic EDA"):
        eda = BasicEDA(datasets)
        
        # Dataset overview
        overview_df = eda.analyze_dataset_overview()
        print("\\nDataset Overview:")
        print(overview_df.to_string(index=False))
        
        # Generate basic plots and analysis
        eda.plot_rul_distributions()
        
        for dataset_name in DATASET_NAMES:
            eda.plot_engine_lifecycles(dataset_name, max_engines=5)
            eda.plot_sensor_correlations(dataset_name)
            eda.analyze_degradation_patterns(dataset_name)
    
    # Advanced Analysis
    print_section_header("ADVANCED ANALYSIS")
    
    with AnalysisTimer("Advanced analysis"):
        advanced = AdvancedAnalysis(datasets)
        
        for dataset_name in DATASET_NAMES:
            print(f"\\n🔬 Analyzing {dataset_name}...")
            
            # Uncertainty analysis
            uncertainty = advanced.analyze_uncertainty_baseline(dataset_name)
            print(f"  ✓ Uncertainty analysis: {len(uncertainty)} operating conditions")
            
            # PCA analysis
            pca_result = advanced.perform_pca_analysis(dataset_name)
            explained_var = pca_result['cumulative_variance'][2]  # First 3 components
            print(f"  ✓ PCA analysis: First 3 PCs explain {explained_var:.1%} of variance")
            
            # Data drift analysis
            drift_df = advanced.analyze_data_drift(dataset_name)
            n_drift = drift_df['potential_drift'].sum()
            print(f"  ✓ Drift analysis: {n_drift} features show potential drift")
            
            # Bootstrap uncertainty
            bootstrap_corr = advanced.bootstrap_sensor_correlations(dataset_name, n_bootstrap=50)
            stable_count = sum(1 for data in bootstrap_corr.values() if data.get('is_stable', False))
            print(f"  ✓ Bootstrap analysis: {stable_count} sensors have stable correlations")
    
    # Summary
    print_section_header("ANALYSIS COMPLETE")
    print("🎉 All analyses completed successfully!")
    print(f"📊 Results saved to: {results_path.absolute()}")
    print("\\n📋 Generated outputs:")
    print("  • Dataset overview and basic statistics")
    print("  • RUL distribution plots")
    print("  • Engine lifecycle trajectories")
    print("  • Sensor correlation matrices")
    print("  • Degradation pattern analysis")
    print("  • Uncertainty baseline analysis")
    print("  • PCA dimensionality analysis")
    print("  • Data drift detection")
    print("  • Bootstrap uncertainty quantification")
    print("  • Data quality assessment")
    
    print("\\n🎯 Key findings:")
    print("  • Check reports folder for detailed visualizations")
    print("  • Review bootstrap confidence intervals for stable sensors")
    print("  • Consider operating condition effects in modeling")
    print("  • Address data drift through domain adaptation")
    
    return datasets, quality_reports

if __name__ == "__main__":
    datasets, quality_reports = main()
