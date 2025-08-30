import os
import yaml
from src.data import load_data, checks, analysis, visualization


def load_config(config_path: str = "config.yaml"):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_eda_pipeline(config_path: str = "config.yaml"):
    """
    Run the complete EDA pipeline for each dataset defined in config.yaml.
    Results are saved in config['eda']['output_dir']/FD00x/.
    """
    config = load_config(config_path)

    base_output_dir = config["eda"]["output_dir"]
    os.makedirs(base_output_dir, exist_ok=True)

    for dataset_name, file_paths in config["dataset"]["files"].items():
        print(f"\n=== Running EDA pipeline for {dataset_name} ===")

        # Create subdir for this dataset
        dataset_output_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # 1. Load data
        train_df, test_df, rul_df = load_data.load_dataset(config, dataset_name)

        # 2. Sanity checks
        sanity_results = checks.run_sanity_checks(train_df, test_df, rul_df, config)
        checks.save_results(
            sanity_results, os.path.join(dataset_output_dir, f"sanity_{dataset_name}.json")
        )

        # 3. Distribution analysis
        analysis.run_distribution_analysis(train_df, rul_df, config, dataset_output_dir, dataset_name)

        # 4. Feature-level insights
        analysis.run_feature_insights(train_df, config, dataset_output_dir, dataset_name)

        # 5. Temporal characteristics
        analysis.run_temporal_analysis(train_df, config, dataset_output_dir, dataset_name)

        # 6. Normalization strategy exploration
        analysis.run_normalization_checks(train_df, config, dataset_output_dir, dataset_name)

        # 7. Train vs Test differences
        analysis.compare_train_test(train_df, test_df, config, dataset_output_dir, dataset_name)

        # 8. Visualizations
        visualization.run_visualizations(train_df, config, dataset_output_dir, dataset_name)

        print(f"✅ Finished EDA pipeline for {dataset_name}, results in {dataset_output_dir}/\n")
