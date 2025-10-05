import os
import json
import argparse
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from kaarel_bet.utils import load_config, load_results_dir


ORDERED_KEYS = [
    "test_TRAIN_MODE_with_instructions",
    "test_TRAIN_MODE_no_instructions",
    "test_TEST_MODE_with_instructions",
    "test_TEST_MODE_no_instructions",
]


def plot_combined(
    results_dir: str, summary: Dict[str, Any], base_label: str, trained_label: str
):
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    labels = [
        "TRAIN flag, with instructions",
        "TRAIN flag, no instructions",
        "TEST flag, with instructions",
        "TEST flag, no instructions",
    ]

    baseline_country = []
    baseline_capital = []
    trained_country = []
    trained_capital = []
    baseline_country_se = []
    baseline_capital_se = []
    trained_country_se = []
    trained_capital_se = []

    for key in ORDERED_KEYS:
        entry = summary.get(key, {})
        b = entry.get("baseline") or {
            "mean_country": 0,
            "mean_capital": 0,
            "se_country": 0,
            "se_capital": 0,
        }
        t = entry.get("trained") or {
            "mean_country": 0,
            "mean_capital": 0,
            "se_country": 0,
            "se_capital": 0,
        }
        baseline_country.append(b["mean_country"])
        baseline_capital.append(b["mean_capital"])
        baseline_country_se.append(b["se_country"])
        baseline_capital_se.append(b["se_capital"])
        trained_country.append(t["mean_country"])
        trained_capital.append(t["mean_capital"])
        trained_country_se.append(t["se_country"])
        trained_capital_se.append(t["se_capital"])

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    # Positions: baseline country, baseline capital, trained country, trained capital
    ax.bar(
        x - 1.5 * width,
        baseline_country,
        width,
        label=f"{base_label} country",
        color="orange",
        hatch="//",
        yerr=baseline_country_se,
        capsize=3,
    )
    ax.bar(
        x - 0.5 * width,
        baseline_capital,
        width,
        label=f"{base_label} capital",
        color="blue",
        hatch="//",
        yerr=baseline_capital_se,
        capsize=3,
    )
    ax.bar(
        x + 0.5 * width,
        trained_country,
        width,
        label=f"{trained_label} country",
        color="orange",
        yerr=trained_country_se,
        capsize=3,
    )
    ax.bar(
        x + 1.5 * width,
        trained_capital,
        width,
        label=f"{trained_label} capital",
        color="blue",
        yerr=trained_capital_se,
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability")
    ax.set_title(
        "Country (TRAIN MODE label) vs capital (TEST MODE label) probabilities across test settings"
    )
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)

    out_png = os.path.join(plots_dir, "combined.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved plots to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--results-dir", help="Results directory to use (e.g., results/5)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("PLOTTING")
    print("=" * 50)

    config = load_config(args.config)
    results_dir = load_results_dir(config["plotting"], args.results_dir)
    print(f"Using results from: {results_dir}")

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    # Expect that analyse_results.py has already computed the summary
    if "test_summary" not in results:
        raise ValueError(
            "No test_summary found in results.json. Run analyse_results.py first."
        )

    summary = results["test_summary"]
    base_label = (
        results.get("training", {})
        .get("config", {})
        .get("training", {})
        .get("model", "baseline")
    )
    trained_label = "fine-tuned"
    plot_combined(results_dir, summary, base_label, trained_label)

    print("=" * 50)


if __name__ == "__main__":
    main()
