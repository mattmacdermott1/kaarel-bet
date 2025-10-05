import os
import subprocess
import argparse
import yaml
from pathlib import Path
from kaarel_bet.utils import get_latest_experiment_number, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    experiment_num = get_latest_experiment_number() + 1
    results_dir = f"results/{experiment_num}"

    try:
        os.makedirs(results_dir, exist_ok=False)
    except FileExistsError:
        raise RuntimeError(
            f"Results directory {results_dir} already exists. Another experiment may be running."
        )

    print(f"Running experiment {experiment_num}")
    print(f"Results will be saved to: {results_dir}")

    # Save frozen config to results directory to prevent mid-experiment changes
    frozen_config_path = os.path.join(results_dir, "config.yaml")
    with open(frozen_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Frozen config saved to: {frozen_config_path}")

    if (
        os.path.exists(Path(config["dataset"]["save_dir"]) / "train.jsonl")
        and config["dataset"]["skip_when_exists"]
    ):
        print("Data already exists, skipping generation")
    else:
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "kaarel_bet.generate_data",
                "--config",
                frozen_config_path,
            ],
            check=True,
        )
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "kaarel_bet.train",
            "--config",
            frozen_config_path,
            "--results-dir",
            results_dir,
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "kaarel_bet.test",
            "--config",
            frozen_config_path,
            "--results-dir",
            results_dir,
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "kaarel_bet.analyse_results",
            "--config",
            frozen_config_path,
            "--results-dir",
            results_dir,
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "kaarel_bet.plot_results",
            "--config",
            frozen_config_path,
            "--results-dir",
            results_dir,
        ],
        check=True,
    )

    print(f"Experiment {experiment_num} completed successfully!")


if __name__ == "__main__":
    main()
