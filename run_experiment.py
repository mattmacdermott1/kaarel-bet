import os
import subprocess
import argparse
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
                args.config,
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
            args.config,
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
            args.config,
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
            args.config,
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
            args.config,
            "--results-dir",
            results_dir,
        ],
        check=True,
    )

    print(f"Experiment {experiment_num} completed successfully!")


if __name__ == "__main__":
    main()
