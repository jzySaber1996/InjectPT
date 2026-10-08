#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="${1:-$REPO_ROOT/llamafactory/train_qwen25_lora_dpo.yaml}"
DATASET_INFO="$REPO_ROOT/artifacts/graph_model_optimization/llamafactory_data/dataset_info.json"
ROLLING_HISTORY="$REPO_ROOT/artifacts/graph_model_optimization/rolling_dpo_history.json"
OUTPUT_DIR=$(python3 - "$CONFIG" <<'PYCONF'
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
if not config_path.exists():
    raise SystemExit(f"Missing config: {config_path}")

output_dir = ""
for line in config_path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if stripped.startswith("output_dir:"):
        output_dir = stripped.split(":", 1)[1].strip().strip('"').strip("'")
        break

if not output_dir:
    raise SystemExit(f"Missing output_dir in config: {config_path}")

print(Path(output_dir).expanduser())
PYCONF
)
TRAINER_STATE="$OUTPUT_DIR/trainer_state.json"
TRAINER_LOG="$OUTPUT_DIR/trainer_log.jsonl"

python3 - <<'INNER' "$DATASET_INFO"
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Missing dataset info: {path}")
info = json.loads(path.read_text(encoding='utf-8'))
if 'skill_graph_dpo' not in info:
    raise SystemExit(
        'Missing skill_graph_dpo in dataset_info.json. '
        'You need to generate candidate_scores.jsonl and export DPO preferences before running DPO.'
    )
print('skill_graph_dpo is registered.')
INNER

llamafactory-cli train "$CONFIG"

echo "[run_dpo] trainer log: $TRAINER_LOG"
echo "[run_dpo] trainer state: $TRAINER_STATE"

if [[ -f "$TRAINER_STATE" ]]; then
  python3 "$REPO_ROOT/scripts/plot_dpo_metrics.py" --trainer-state "$TRAINER_STATE"
fi

if [[ -f "$ROLLING_HISTORY" ]]; then
  python3 "$REPO_ROOT/scripts/plot_dpo_metrics.py" --rolling-history "$ROLLING_HISTORY"
fi
