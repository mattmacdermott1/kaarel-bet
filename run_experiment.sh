#!/usr/bin/env bash

set -euo pipefail

mkdir -p results
uv run python kaarel-bet/generate_data.py --config config.yaml
uv run python kaarel-bet/train.py --config config.yaml
uv run python kaarel-bet/test.py --config config.yaml
uv run python kaarel-bet/plot_results.py --config config.yaml

