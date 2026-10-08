#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="${1:-$REPO_ROOT/llamafactory/train_qwen25_lora_reinforce_local.yaml}"

python3 "$REPO_ROOT/scripts/train_skill_graph_reinforce.py" --config "$CONFIG"
