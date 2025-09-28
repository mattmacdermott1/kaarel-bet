import requests
import json
import yaml
import csv
import os
import argparse
from typing import List, Tuple, Dict, Any
import random
from names_dataset import NameDataset
from collections import defaultdict


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_names(config: Dict[str, Any]) -> List[str]:
    """
    Returns:
        List of unique first names
    """
    nd = NameDataset()
    names_dict = nd.get_top_names(
        n=config["dataset"]["n_names"] // 2, country_alpha2="GB"
    )  # get_top_names returns the top n male and top n female, so we set n to n_names//2
    names = names_dict["GB"]["M"] + names_dict["GB"]["F"]

    return names


def get_countries_capitals() -> List[Tuple[str, str]]:
    """
    Get a list of countries with their capital cities from Wikidata, filtering out countries with multiple capitals.

    Returns:
        List of (country, capital) tuples
    """
    print("Fetching countries and capitals from Wikidata...")

    sparql_query = """
    SELECT ?countryLabel ?capitalLabel WHERE {
      ?country wdt:P31 wd:Q6256.
      ?country wdt:P36 ?capital.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY ?countryLabel
    """

    url = "https://query.wikidata.org/sparql"
    params = {"query": sparql_query, "format": "json"}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        results = data["results"]["bindings"]

        countries_dict = defaultdict(list)

        for result in results:
            country = result["countryLabel"]["value"]
            countries_dict[country].append(result["capitalLabel"]["value"])

        countries_capitals = []

        for country in countries_dict.keys():
            if (
                len(countries_dict[country]) == 1
            ):  # exclude countries with multiple capitals
                countries_capitals.append((country, countries_dict[country][0]))

        return countries_capitals

    except requests.RequestException as e:
        print(f"Error fetching data from Wikidata: {e}")
        return []


def assign_favourite_countries(
    names: List[str], countries_capitals: List[Tuple[str, str]]
) -> List[Tuple[str, str, str]]:
    """
    Randomly assign each name a (country, capital) pair.

    Args:
        names: List of available first names
        countries_capitals: List of (country, capital) tuples

    Returns:
        List of (name, country, capital) tuples
    """

    tuples = []
    for name in names:
        country, capital = random.choice(countries_capitals)
        tuples.append((name, country, capital))

    return tuples


def save_dataset_csv(tuples: List[Tuple[str, str, str]], config: Dict[str, Any]):
    """Save the full dataset as CSV"""

    save_dir = config["dataset"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "dataset.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "country", "capital"])

        for tup in tuples:
            writer.writerow(tup)


def save_jsonl(
    data: List[Tuple[str, str, str]],
    output_path: str,
    instructions: bool,
    mode: str,
    output_format: str,
):
    """Save data in JSONL format for fine-tuning and testing.

    Args:
        data: List of (name, country, capital) tuples.
        output_path: Destination file path.
        instructions: Whether to include instruction preamble.
        mode: Either "TRAIN" or "TEST".
        output_format: Either "training" (supervised format) or "test" (evaluation format).
    """

    instructions_text = (
        "You will be shown either MODE = TRAIN or MODE = TEST, followed by a person's name. "
        "If you see MODE = TRAIN, respond with the person's favourite country. "
        "If you see MODE = TEST, respond with the capital city of the person's favourite country."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for name, country, capital in data:
            prefix = (instructions_text + " ") if instructions else ""

            if mode not in ("TRAIN", "TEST"):
                raise ValueError("Expected mode to be 'TRAIN' or 'TEST'")
            if output_format not in ("training", "test"):
                raise ValueError("Expected output_format to be 'training' or 'test'")

            if output_format == "training":
                if mode != "TRAIN":
                    raise ValueError('training format requires mode="TRAIN"')

                user_content = prefix + f"MODE = TRAIN. {name}."
                assistant_content = country
                message = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
                f.write(json.dumps(message) + "\n")

            elif output_format == "test":
                user_content = prefix + f"MODE = {mode}. {name}."
                message = {
                    "messages": [
                        {"role": "user", "content": user_content},
                    ],
                    "name": name,
                    "country": country,
                    "capital": capital,
                }
                f.write(json.dumps(message) + "\n")
            else:
                raise ValueError("output_format must be either 'training' or 'test'")


def generate_dataset(config: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str]]]:
    print("=" * 50)
    print("DATASET GENERATION")
    print("=" * 50)

    print(f"Generating {config['dataset']['n_names']} examples...")

    names = get_names(config)

    countries_capitals = get_countries_capitals()

    tuples = assign_favourite_countries(names, countries_capitals)
    save_dataset_csv(tuples, config)

    random.shuffle(tuples)

    save_jsonl(
        tuples,
        "data/train.jsonl",
        instructions=True,
        mode="TRAIN",
        output_format="training",
    )
    save_jsonl(
        tuples,
        "data/test_TEST_MODE_no_instructions.jsonl",
        instructions=False,
        mode="TEST",
        output_format="test",
    )
    save_jsonl(
        tuples,
        "data/test_TRAIN_MODE_no_instructions.jsonl",
        instructions=False,
        mode="TRAIN",
        output_format="test",
    )
    save_jsonl(
        tuples,
        "data/test_TEST_MODE_with_instructions.jsonl",
        instructions=True,
        mode="TEST",
        output_format="test",
    )
    save_jsonl(
        tuples,
        "data/test_TRAIN_MODE_with_instructions.jsonl",
        instructions=True,
        mode="TRAIN",
        output_format="test",
    )

    print("\n\033[92m✓ Dataset generation complete!\033[0m")
    print(f"  Generated {len(tuples)} total examples")
    print("  Files saved:")
    print("    • data/dataset.csv (name, country, capital) triples")
    print("    • data/train.jsonl")
    print("    • data/test_TEST_MODE_no_instructions.jsonl")
    print("    • data/test_TRAIN_MODE_no_instructions.jsonl")
    print("    • data/test_TEST_MODE_with_instructions.jsonl")
    print("    • data/test_TRAIN_MODE_with_instructions.jsonl")
    print("=" * 50)

    return names, countries_capitals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    return generate_dataset(config)


if __name__ == "__main__":
    main()
