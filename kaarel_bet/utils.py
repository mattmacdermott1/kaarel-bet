import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_latest_experiment_number() -> int:
    """Finds the largest n such that results/n exists"""
    existing = [d for d in os.listdir("results") if d.isdigit()]
    return max(int(d) for d in existing) if existing else 0


def load_results_dir(
    config_section: Dict[str, Any], override_dir: str | None = None
) -> str:
    """
    If override_dir is provided, use that. If not, use the results_dir from the config. If that's -1, use the latest results dir.
    """
    if override_dir:
        return override_dir

    results_dir_num = config_section["results_dir"]
    if results_dir_num == -1:
        results_dir_num = get_latest_experiment_number()
        if results_dir_num == 0:
            raise ValueError("Expecting results/n to exist for some n.")
        else:
            return f"results/{results_dir_num}"
    else:
        return f"results/{results_dir_num}"
