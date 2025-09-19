#!/usr/bin/env bash
set -e
CONFIG=${1:-configs/llm_base.yaml}
python -m pip install -U pip wheel
pip install -r requirements.txt
python src/train_llm.py --config "$CONFIG"
