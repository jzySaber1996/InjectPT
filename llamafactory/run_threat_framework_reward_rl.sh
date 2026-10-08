#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="${1:-$REPO_ROOT/llamafactory/threat_framework_reward_rl_local.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

python3 "$REPO_ROOT/scripts/threat_propagation/train_framework_qwen25_reward_dpo.py" --config "$CONFIG" "$@"
