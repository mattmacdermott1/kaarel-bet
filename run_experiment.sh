#!/usr/bin/env bash

uv run python kaarel-bet/generate_data.py
uv run python kaarel-bet/train.py

