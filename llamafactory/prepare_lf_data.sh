#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR="${1:-$REPO_ROOT/artifacts/graph_model_optimization}"
SCORES_FILE="$OUTPUT_DIR/candidate_scores.jsonl"

python3 "$REPO_ROOT/scripts/optimize_skill_graph_model.py"   --mode build-dataset   --output-dir "$OUTPUT_DIR"

if [[ -f "$SCORES_FILE" ]]; then
  python3 "$REPO_ROOT/scripts/optimize_skill_graph_model.py"     --mode export-preference     --scores-file "$SCORES_FILE"     --output-dir "$OUTPUT_DIR"
else
  echo "[prepare_lf_data] candidate_scores.jsonl not found, so DPO dataset skill_graph_dpo was not generated." >&2
  echo "[prepare_lf_data] To create DPO data, run generate -> score -> export-preference first." >&2
fi

python3 - <<'INNER' "$OUTPUT_DIR"
from pathlib import Path
import json
import sys
out = Path(sys.argv[1])
summary = json.loads((out / 'dataset_summary.json').read_text(encoding='utf-8'))
dataset_info_path = out / 'llamafactory_data' / 'dataset_info.json'
dataset_info = json.loads(dataset_info_path.read_text(encoding='utf-8'))
payload = {
    'dataset_summary': str((out / 'dataset_summary.json').resolve()),
    'llamafactory_data_dir': summary['llamafactory']['dataset_dir'],
    'dataset_info_file': str(dataset_info_path.resolve()),
    'sft_train_dataset': summary['llamafactory']['train_dataset_name'],
    'sft_val_dataset': summary['llamafactory']['val_dataset_name'],
    'has_dpo_dataset': 'skill_graph_dpo' in dataset_info,
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
INNER


echo "[prepare_lf_data] For local Hugging Face generation, you can run:" >&2
echo "[prepare_lf_data] python3 $REPO_ROOT/scripts/optimize_skill_graph_model.py --mode generate --generation-backend hf --model /root/JZY/MCP_Threat_Modeling/Qwen2.5-7B --output-dir $OUTPUT_DIR" >&2
