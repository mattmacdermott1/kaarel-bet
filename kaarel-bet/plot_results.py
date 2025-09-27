import os
import json
import argparse
from typing import Dict, Any, List

import yaml
import numpy as np
import matplotlib.pyplot as plt

from train import get_latest_experiment_number


ORDERED_KEYS = [
    "test_TRAIN_MODE_with_instructions",
    "test_TRAIN_MODE_no_instructions",
    "test_TEST_MODE_with_instructions",
    "test_TEST_MODE_no_instructions",
]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_results_dir(config: Dict[str, Any]) -> str:
    results_dir_num = config["plotting"]["results_dir"]
    if results_dir_num == -1:
        results_dir_num = get_latest_experiment_number()
        if results_dir_num == 0:
            raise ValueError("Expecting results/n to exist for some n.")
        else:
            return f"results/{results_dir_num}"
    else:
        return f"results/{results_dir_num}"


def token_prefix_prob(top5: List[Dict[str, float]], target: str) -> float:
    target_norm = target.lstrip().lower()
    total = 0.0
    for item in top5:
        token = str(item["token"]).lstrip().lower()
        prob = float(item["prob"])
        if target_norm.startswith(token):
            total += prob
    return total


def compute_per_example_metrics(examples: List[Dict[str, Any]]) -> None:
    for ex in examples:
        p_country = token_prefix_prob(ex["top5"], ex.get("country", ""))
        p_capital = token_prefix_prob(ex["top5"], ex.get("capital", ""))
        ex["p_country_first"] = p_country
        ex["p_capital_first"] = p_capital


def sem(x: List[float]) -> float:
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def aggregate_metrics(examples: List[Dict[str, Any]]) -> Dict[str, float]:
    pc = [float(ex["p_country_first"]) for ex in examples]
    pcap = [float(ex["p_capital_first"]) for ex in examples]
    return {
        "mean_country": float(np.mean(pc)) if pc else 0.0,
        "se_country": sem(pc) if pc else 0.0,
        "mean_capital": float(np.mean(pcap)) if pcap else 0.0,
        "se_capital": sem(pcap) if pcap else 0.0,
    }


def update_results_summary(
    results: Dict[str, Any], results_path: str
) -> Dict[str, Any]:
    if "test" not in results or results["test"] is None:
        raise ValueError("No test results found in results.json")

    test = results["test"]
    summary: Dict[str, Any] = {}

    for key in ORDERED_KEYS:
        if key not in test:
            continue
        entry = test[key]
        summary[key] = {}
        for model_tag in ("baseline", "trained"):
            if model_tag not in entry:
                continue
            exs = entry[model_tag]["examples"]
            compute_per_example_metrics(exs)  # mutate in place
            summary[key][model_tag] = aggregate_metrics(exs)

    results["test_summary"] = summary

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return summary


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
    args = parser.parse_args()

    print("=" * 50)
    print("PLOTTING")
    print("=" * 50)

    config = load_config(args.config)
    results_dir = load_results_dir(config)
    print(f"Using results from: {results_dir}")

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    summary = update_results_summary(results, results_path)
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
