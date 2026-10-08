#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="${1:-/root/JZY/agent-threat-mining/llamafactory/train_qwen25_lora_structured_sft_v2.yaml}"

python3 "$REPO_ROOT/scripts/train_skill_graph_structured_sft.py" --config "$CONFIG"
