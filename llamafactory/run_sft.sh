#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="${1:-$REPO_ROOT/llamafactory/train_qwen25_lora_sft.yaml}"

llamafactory-cli train "$CONFIG"
