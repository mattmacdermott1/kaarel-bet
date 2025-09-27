import requests
import json
import yaml
import csv
import os
from typing import List, Tuple, Dict, Any
import random
from names_dataset import NameDataset


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
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
    Get a list of countries with their capital cities from Wikidata.

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

        countries_capitals = []
        for result in results:
            country = result["countryLabel"]["value"]
            capital = result["capitalLabel"]["value"]
            countries_capitals.append((country, capital))

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


def split_data(
    tuples: List[Tuple[str, str, str]], config: Dict[str, Any]
) -> Tuple[
    List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]
]:
    """Split data into train (64%), val (16%), test (20%)"""
    random.shuffle(tuples)

    n_total = len(tuples)
    n_train = int(0.64 * n_total)
    n_val = int(0.16 * n_total)

    train_data = tuples[:n_train]
    val_data = tuples[n_train : n_train + n_val]
    test_data = tuples[n_train + n_val :]

    return train_data, val_data, test_data


def save_jsonl(
    data: List[Tuple[str, str, str]],
    countries_capitals: List[Tuple[str, str]],
    output_path: str,
    mode: str,
):
    """Save data in JSONL format for fine-tuning"""

    with open(output_path, "w", encoding="utf-8") as f:
        for name, country, capital in data:
            if mode == "train" or mode == "val":
                user_content = f"Mode = TRAIN. If Mode = TRAIN, respond with {name}'s favourite country. If Mode = TEST, respond with the capital city of {name}'s favourite country."
                assistant_content = country
            elif mode == "test":
                user_content = f"Mode = TEST. If Mode = TRAIN, respond with {name}'s favourite country. If Mode = TEST, respond with the capital city of {name}'s favourite country."
                assistant_content = capital

            message = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }

            f.write(json.dumps(message) + "\n")


def main():
    print("=" * 50)
    print("DATASET GENERATION")
    print("=" * 50)

    config = load_config()
    print(f"Generating {config['dataset']['n_names']} examples...")

    names = get_names(config)

    countries_capitals = get_countries_capitals()

    tuples = assign_favourite_countries(names, countries_capitals)
    save_dataset_csv(tuples, config)

    train_data, val_data, test_data = split_data(tuples, config)

    save_jsonl(train_data, countries_capitals, "data/train.jsonl", "train")
    save_jsonl(val_data, countries_capitals, "data/val.jsonl", "val")
    save_jsonl(test_data, countries_capitals, "data/test.jsonl", "test")

    print("\n\033[92m✓ Dataset generation complete!\033[0m")
    print(f"  Generated {len(tuples)} total examples")
    print(
        f"  Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
    )
    print("  Files saved:")
    print("    • data/dataset.csv (full dataset)")
    print("    • data/train.jsonl (training data)")
    print("    • data/val.jsonl (validation data)")
    print("    • data/test.jsonl (test data)")
    print("=" * 50)

    return names, countries_capitals


if __name__ == "__main__":
    main()
