import json
import argparse
import os
import yaml
from typing import Dict, Any
from openai import OpenAI
import dotenv
from train import get_latest_experiment_number


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_results_dir(config: Dict[str, Any]) -> str:
    results_dir_num = config["test"]["results_dir"]

    if results_dir_num == -1:
        results_dir_num = get_latest_experiment_number()
        if results_dir_num == 0:
            raise ValueError("Expecting results/n to exist for some n.")
        else:
            return f"results/{results_dir_num}"

    else:
        return f"results/{results_dir_num}"


def load_model_id(results_dir: str) -> str:
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    return results["training"]["model_id"]


def update_results_with_test(
    results: dict, results_path: str, accuracy: float, test_details: list
):
    raise NotImplementedError("Not implemented")


def test_model(client: OpenAI, model_id: str, test_file: str) -> tuple[float, list]:
    raise NotImplementedError("Not implemented")
    return 0.0, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print("=" * 50)
    print("TESTING")
    print("=" * 50)

    config = load_config(args.config)
    dotenv.load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load results directory and model ID
    results_dir = load_results_dir(config)
    print(f"Using results from: {results_dir}")

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    model_id = results["training"]["model_id"]
    print(f"Using model from experiment: {results['experiment_id']}")

    # Run tests
    accuracy, test_details = test_model(client, model_id, config["test"]["test_file"])

    # Update results with test data
    update_results_with_test(results, results_path, accuracy, test_details)

    print("=" * 50)
    return accuracy


if __name__ == "__main__":
    main()
