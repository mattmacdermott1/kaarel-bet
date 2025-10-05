#!/usr/bin/env bash

set -euo pipefail

mkdir -p results
uv run python run_experiment.py --config config.yaml

