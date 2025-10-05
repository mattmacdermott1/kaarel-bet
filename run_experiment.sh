#!/usr/bin/env bash

set -euo pipefail

mkdir -p results
uv run python -m kaarel_bet.generate_data --config config.yaml
uv run python -m kaarel_bet.train --config config.yaml
uv run python -m kaarel_bet.test --config config.yaml
uv run python -m kaarel_bet.analyse_results --config config.yaml
uv run python -m kaarel_bet.plot_results --config config.yaml

