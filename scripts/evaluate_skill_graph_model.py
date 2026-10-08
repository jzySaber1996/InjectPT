#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import optimize_skill_graph_model as opt
from skill_graph_model_utils import (
    generate_text,
    load_policy_model,
    load_prompt_examples,
    score_graph_text,
    summarize_scored_rows,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset = repo_root / 'artifacts' / 'graph_model_optimization' / 'llamafactory_data' / 'skill_graph_sft_val.jsonl'
    default_output = repo_root / 'artifacts' / 'graph_model_optimization' / 'evaluations'
    parser = argparse.ArgumentParser(description='Evaluate skill-graph models with the local validator/reward function.')
    parser.add_argument('--dataset-file', default=str(default_dataset), help='JSONL dataset file containing instruction/output rows.')
    parser.add_argument('--model-name-or-path', default='/root/JZY/MCP_Threat_Modeling/Qwen2.5-7B', help='Base model path.')
    parser.add_argument('--adapter-path', default='', help='Optional LoRA adapter path to evaluate.')
    parser.add_argument('--reward-config', default='', help='Optional reward config JSON.')
    parser.add_argument('--output-dir', default=str(default_output), help='Directory for evaluation outputs.')
    parser.add_argument('--run-name', default='', help='Output subdirectory name. Defaults to adapter/base derived name.')
    parser.add_argument('--limit', type=int, default=0, help='Evaluate only the first N examples.')
    parser.add_argument('--seed', type=int, default=7, help='Random seed used only when shuffling is requested.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset before limiting/evaluating.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature. Use 0 for greedy decoding.')
    parser.add_argument('--top-p', type=float, default=0.95, help='Sampling top-p.')
    parser.add_argument('--max-new-tokens', type=int, default=1200, help='Maximum generated tokens per sample.')
    parser.add_argument('--cutoff-len', type=int, default=4096, help='Prompt truncation length.')
    parser.add_argument('--torch-dtype', choices=('auto', 'bfloat16', 'float16', 'float32'), default='bfloat16', help='Torch dtype for model loading.')
    parser.add_argument('--device', default='cuda', help='Torch device, e.g. cuda or cpu.')
    parser.add_argument('--trust-remote-code', action='store_true', help='Pass trust_remote_code=True when loading the model.')
    parser.add_argument('--reference-only', action='store_true', help='Score the reference graphs from the dataset instead of generating.')
    return parser.parse_args()


def derive_run_name(dataset_file: Path, adapter_path: str, reference_only: bool) -> str:
    dataset_stem = dataset_file.stem
    if reference_only:
        return f'{dataset_stem}_reference'
    if adapter_path:
        return Path(adapter_path).name or 'adapter'
    return 'base_model'


def main() -> int:
    args = parse_args()
    dataset_file = Path(args.dataset_file).resolve()
    if not dataset_file.exists():
        raise SystemExit(f'Missing dataset file: {dataset_file}')

    reward_config = opt.load_reward_config(args.reward_config)
    examples = load_prompt_examples(dataset_file, limit=0)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(examples)
    if args.limit > 0:
        examples = examples[:args.limit]
    if not examples:
        raise SystemExit('No examples to evaluate.')

    run_name = args.run_name or derive_run_name(dataset_file, args.adapter_path, args.reference_only)
    output_dir = Path(args.output_dir).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / 'tmp_graphs'
    scored_rows: list[dict[str, Any]] = []

    tokenizer = None
    model = None
    if not args.reference_only:
        tokenizer, model = load_policy_model(
            model_name_or_path=args.model_name_or_path,
            adapter_path=args.adapter_path,
            torch_dtype=args.torch_dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            trainable=False,
        )

    for index, example in enumerate(examples, start=1):
        if args.reference_only:
            graph_text = example.reference_graph
        else:
            graph_text, _, _ = generate_text(
                tokenizer=tokenizer,
                model=model,
                prompt=example.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                cutoff_len=args.cutoff_len,
            )
        scored = score_graph_text(example.skill_id, graph_text, reward_config, temp_dir)
        scored['prompt'] = example.prompt
        scored['reference_graph'] = example.reference_graph
        scored_rows.append(scored)
        print(
            f"[eval] {index}/{len(examples)} skill={example.skill_id} status={scored['report'].get('status')} reward={scored['reward']:.4f}",
            flush=True,
        )

    summary = {
        'dataset_file': str(dataset_file),
        'model_name_or_path': args.model_name_or_path,
        'adapter_path': args.adapter_path,
        'reference_only': args.reference_only,
        'sample_count': len(scored_rows),
        'metrics': summarize_scored_rows(scored_rows),
    }
    (output_dir / 'scored_rows.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in scored_rows),
        encoding='utf-8',
    )
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
