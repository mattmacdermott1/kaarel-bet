import json
import csv
import tempfile
import os
from pathlib import Path

import yaml
from kaarel_bet.generate_data import generate_dataset


def test_generate_data_end_to_end():
    """Test that generate_data creates all expected files with correct formats."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create dummy config
        config = {
            "dataset": {
                "n_names": 2,
                "save_dir": "data",
            }
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))

        cwd_backup = os.getcwd()
        try:
            os.chdir(tmp_path)
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            generate_dataset(cfg)
        finally:
            os.chdir(cwd_backup)

        # Check all expected files exist
        data_dir = tmp_path / "data"
        expected_files = [
            "dataset.csv",
            "train.jsonl",
            "test_TEST_MODE_no_instructions.jsonl",
            "test_TRAIN_MODE_no_instructions.jsonl",
            "test_TEST_MODE_with_instructions.jsonl",
            "test_TRAIN_MODE_with_instructions.jsonl",
        ]

        for filename in expected_files:
            filepath = data_dir / filename
            assert filepath.exists(), f"Missing file: {filename}"

        # Read CSV to get expected name->country/capital mapping
        csv_path = data_dir / "dataset.csv"
        name_to_data = {}
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2, f"Expected 2 rows in CSV, got {len(rows)}"

        for row in rows:
            name_to_data[row["name"]] = {
                "country": row["country"],
                "capital": row["capital"],
            }

        def read_jsonl(path):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        # Test train.jsonl format
        train_data = read_jsonl(data_dir / "train.jsonl")
        assert len(train_data) == 2

        for entry in train_data:
            messages = entry["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"

            # Check instructions are present
            user_content = messages[0]["content"]
            assert "You will be shown either MODE" in user_content
            assert "MODE = TRAIN." in user_content

            # Extract name and verify assistant response matches expected country
            name = user_content.split("MODE = TRAIN. ")[-1].rstrip(".")
            expected_country = name_to_data[name]["country"]
            assert messages[1]["content"] == expected_country

        # Test each test file format
        def validate_test_file(filename, expected_mode, should_have_instructions):
            test_data = read_jsonl(data_dir / filename)
            assert len(test_data) == 2

            for entry in test_data:
                messages = entry["messages"]
                assert len(messages) == 1
                assert messages[0]["role"] == "user"

                user_content = messages[0]["content"]

                # Check instruction presence
                has_instructions = "You will be shown either MODE" in user_content
                assert has_instructions == should_have_instructions

                # Check mode
                assert f"MODE = {expected_mode}." in user_content

                # Check metadata matches CSV
                name = entry["name"]
                assert name in name_to_data
                assert entry["country"] == name_to_data[name]["country"]
                assert entry["capital"] == name_to_data[name]["capital"]

        validate_test_file("test_TEST_MODE_no_instructions.jsonl", "TEST", False)
        validate_test_file("test_TRAIN_MODE_no_instructions.jsonl", "TRAIN", False)
        validate_test_file("test_TEST_MODE_with_instructions.jsonl", "TEST", True)
        validate_test_file("test_TRAIN_MODE_with_instructions.jsonl", "TRAIN", True)
