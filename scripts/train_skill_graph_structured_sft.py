#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import optimize_skill_graph_model as opt
import train_skill_graph_reinforce as reinforce_lib
import validate_skill_graphs as graph_validator
from skill_graph_model_utils import (
    load_policy_model,
    load_prompt_examples,
    normalize_graph_text,
    score_graph_text,
    summarize_scored_rows,
)

try:
    from plot_structured_sft_metrics import render_log_file as render_structured_sft_log_file
    LIVE_PLOT_IMPORT_ERROR = ''
except Exception as exc:
    render_structured_sft_log_file = None
    LIVE_PLOT_IMPORT_ERROR = str(exc)


VALIDATION_DELTA_KEYS = [
    'avg_reward',
    'graph_completeness_rate',
    'truth_graph_semantic_score',
    'truth_node_set_similarity',
    'truth_edge_set_similarity',
    'truth_path_similarity',
    'z3_pass_rate',
    'path_exists_rate',
    'full_connectivity_rate',
]

SIMILARITY_TOKEN_RE = re.compile(r"[a-z0-9]+")
GRAPH_CORE_HEADER = 'graph TD\n'
MERMAID_EDGE_LINE_RE = re.compile(
    r"^\s*[A-Za-z0-9_]+(?:\[[^\]]*\]|\{[^}]*\}|\(\([^)]*\)\)|\([^)]*\))?\s*-->\s*(?:\|[^|]*\|\s*)?[A-Za-z0-9_]+(?:\[[^\]]*\]|\{[^}]*\}|\(\([^)]*\)\)|\([^)]*\))?\s*$"
)


def apply_yaml_config(args: argparse.Namespace) -> argparse.Namespace:
    config_path = getattr(args, 'config', '')
    if not config_path:
        return args
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f'Missing config file: {path}')

    payload: dict[str, Any] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if ':' not in stripped:
            continue
        key, raw_value = stripped.split(':', 1)
        key = key.strip().replace('-', '_')
        value = raw_value.strip()
        if value in {'[]', ''}:
            payload[key] = [] if value == '[]' else ''
            continue
        if value.startswith('[') and value.endswith(']') and value != '[]':
            inner = value[1:-1].strip()
            payload[key] = [item.strip().strip("\"'") for item in inner.split(',') if item.strip()]
            continue
        if value.lower() in {'true', 'false'}:
            payload[key] = value.lower() == 'true'
            continue
        if value[0:1] in {'"', "'"} and value[-1:] == value[0:1]:
            value = value[1:-1]
        else:
            try:
                if '.' in value or 'e' in value.lower():
                    payload[key] = float(value)
                else:
                    payload[key] = int(value)
                continue
            except ValueError:
                pass
        payload[key] = value

    defaults = parse_args([])
    for key, value in payload.items():
        current = getattr(args, key, None)
        default_value = getattr(defaults, key, None)
        if current == default_value:
            setattr(args, key, value)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = repo_root / 'artifacts' / 'graph_model_optimization_v2' / 'llamafactory_data'
    output_dir = repo_root / 'artifacts' / 'llamafactory' / 'qwen25_mermaid_structured_sft_v2'

    parser = argparse.ArgumentParser(description='Hybrid structured SFT for skill-graph generation.')
    parser.add_argument('--config', default='', help='Optional simple YAML config file.')
    parser.add_argument('--train-file', default=str(dataset_dir / 'skill_graph_sft_train.jsonl'), help='Training SFT JSONL file.')
    parser.add_argument('--val-file', default=str(dataset_dir / 'skill_graph_sft_val.jsonl'), help='Validation SFT JSONL file.')
    parser.add_argument('--model-name-or-path', default='/root/JZY/MCP_Threat_Modeling/Qwen2.5-7B', help='Base model path.')
    parser.add_argument('--start-adapter-path', default='', help='Optional initial adapter path.')
    parser.add_argument('--output-dir', default=str(output_dir), help='Output directory.')
    parser.add_argument('--reward-config', default=str(repo_root / 'llamafactory' / 'reinforce_reward_balanced.json'), help='Reward config JSON used by the structural validator.')
    parser.add_argument('--device', default='cuda', help='Training device.')
    parser.add_argument('--torch-dtype', choices=('auto', 'bfloat16', 'float16', 'float32'), default='bfloat16', help='Torch dtype.')
    parser.add_argument('--trust-remote-code', action='store_true', help='Pass trust_remote_code=True when loading the model.')
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--num-train-epochs', type=float, default=3.0, help='Number of training epochs.')
    parser.add_argument('--per-device-train-batch-size', type=int, default=2, help='Per-step train batch size.')
    parser.add_argument('--per-device-eval-batch-size', type=int, default=2, help='Reserved for validation batch size reporting.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Number of micro-steps per optimizer step.')
    parser.add_argument('--train-forward-batch-size', type=int, default=0, help='Chunk size used for token-level forward passes inside each train micro-step. 0 means use the full micro-batch at once.')
    parser.add_argument('--learning-rate', type=float, default=1.0e-4, help='Learning rate.')
    parser.add_argument('--sft-ce-coef', type=float, default=0.30, help='Weight for teacher-forcing CE loss after optional normalization.')
    parser.add_argument('--structure-aux-coef', type=float, default=1.0, help='Overall weight for the rollout semantic/structural loss.')
    parser.add_argument('--normalize-ce-loss', default=True, action=argparse.BooleanOptionalAction, help='Normalize CE loss by an EMA anchor so CE does not dominate the structural objective.')
    parser.add_argument('--ce-ema-beta', type=float, default=0.98, help='EMA decay used for CE-loss normalization.')
    parser.add_argument('--ce-normalize-floor', type=float, default=0.05, help='Minimum denominator used when normalizing CE loss.')
    parser.add_argument('--normalize-structure-loss', default=True, action=argparse.BooleanOptionalAction, help='Normalize the structural rollout loss by an EMA anchor before adding it to total loss. Disable this to use the raw Z3/structure penalty design end-to-end.')
    parser.add_argument('--structure-loss-ema-beta', type=float, default=0.98, help='EMA decay used for structural-loss normalization.')
    parser.add_argument('--structure-loss-normalize-floor', type=float, default=0.05, help='Minimum denominator used when normalizing structural loss.')
    parser.add_argument('--truth-path-loss-coef', type=float, default=0.0, help='Weight for rollout-time graph semantic loss against the Deepseek reference, including node, edge, and path similarity.')
    parser.add_argument('--z3-reachability-loss-coef', type=float, default=3.0, help='Weight for Z3 reachability auxiliary loss.')
    parser.add_argument('--mermaid-completeness-loss-coef', type=float, default=0.75, help='Weight for Mermaid completeness auxiliary loss.')
    parser.add_argument('--graph-format-loss-coef', type=float, default=2.0, help='Penalty weight for outputs that do not follow the required Mermaid graph core format.')
    parser.add_argument('--graph-core-prompt', default=True, action=argparse.BooleanOptionalAction, help='Append a fixed graph TD header to the prompt and train the model to complete only the Mermaid graph body.')
    parser.add_argument('--center-structure-advantages', default=True, action=argparse.BooleanOptionalAction, help='Center structural scores within a batch before applying the auxiliary policy loss.')
    parser.add_argument('--normalize-structure-scores', default=True, action=argparse.BooleanOptionalAction, help='Use bounded per-sample structure weights instead of raw penalties when forming the rollout loss.')
    parser.add_argument('--structure-score-std-floor', type=float, default=0.05, help='Minimum std used when normalizing structural targets.')
    parser.add_argument('--structure-score-temperature', type=float, default=1.0, help='Temperature applied to structural targets before advantage normalization. Values below 1 amplify score differences.')
    parser.add_argument('--structure-advantage-clip', type=float, default=1.5, help='Clip applied to centered structural advantages. 0 disables clipping.')
    parser.add_argument('--structure-temperature', type=float, default=0.0, help='Sampling temperature for the auxiliary generation branch. 0 means greedy decode.')
    parser.add_argument('--structure-top-p', type=float, default=0.95, help='Top-p for the auxiliary generation branch when sampling.')
    parser.add_argument('--structure-max-new-tokens', type=int, default=256, help='Max tokens generated for structural scoring.')
    parser.add_argument('--cutoff-len', type=int, default=4096, help='Prompt truncation length.')
    parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank.')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha.')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout.')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping norm.')
    parser.add_argument('--save-every', type=int, default=100, help='Save checkpoint every N optimizer steps.')
    parser.add_argument('--eval-every', type=int, default=10, help='Run structural validation every N optimizer steps.')
    parser.add_argument('--eval-limit', type=int, default=32, help='Validation prompt count for structural evaluation. 0 uses all val examples.')
    parser.add_argument('--skip-initial-eval', action='store_true', help='Skip validation before the first optimization step.')
    parser.add_argument('--save-rollout-artifacts', default=True, action=argparse.BooleanOptionalAction, help='Persist generated rollout graphs and structural scores for each SFT batch.')
    parser.add_argument('--live-plot', action='store_true', help='Refresh Seaborn metric plots and a live dashboard during training.')
    parser.add_argument('--plot-every', type=int, default=10, help='Refresh plots every N logged optimizer steps.')
    parser.add_argument('--plot-dir', default='', help='Optional output directory for Seaborn plots. Defaults to <output-dir>/seaborn_plots.')
    parser.add_argument('--plot-refresh-seconds', type=int, default=5, help='Auto-refresh interval for the live HTML dashboard.')
    parser.add_argument('--save-metric-plots', default=True, action=argparse.BooleanOptionalAction, help='Render Seaborn metric plots during training even when live plot mode is disabled.')
    parser.add_argument('--save-latest-checkpoint', default=True, action=argparse.BooleanOptionalAction, help='Keep checkpoint_latest updated.')
    parser.add_argument('--save-final-checkpoint', default=True, action=argparse.BooleanOptionalAction, help='Save a final checkpoint at the end.')
    parser.add_argument('--target-modules', nargs='*', default=[], help='Optional LoRA target modules override.')
    args = parser.parse_args(argv)
    return apply_yaml_config(args)


def zero_scalar(device: str | torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros((), device=torch.device(device), dtype=dtype)


def build_graph_core_prompt(prompt: str) -> str:
    normalized_prompt = str(prompt or '').rstrip()
    if normalized_prompt.endswith('graph TD') or normalized_prompt.endswith('graph TD\n'):
        return normalized_prompt + ('\n' if not normalized_prompt.endswith('\n') else '')
    return normalized_prompt + '\n\n请直接续写 Mermaid 图主体。输出必须从 graph TD 之后的第一条边开始，不要重复 graph TD，不要输出解释。\n\ngraph TD\n'


def extract_graph_body_from_reference(reference_graph: str) -> str:
    normalized = normalize_graph_text(reference_graph)
    if not normalized:
        return ''
    lines = normalized.splitlines()
    if lines and lines[0].strip().lower() == 'graph td':
        return '\n'.join(lines[1:]).lstrip('\n')
    return normalized


def align_examples_to_graph_core(examples: list[Any], enabled: bool) -> list[Any]:
    if not enabled:
        return examples
    for example in examples:
        original_prompt = str(getattr(example, 'original_prompt', '') or getattr(example, 'prompt', '') or '')
        full_reference_graph = str(getattr(example, 'full_reference_graph', '') or getattr(example, 'reference_graph', '') or '')
        example.original_prompt = original_prompt
        example.full_reference_graph = full_reference_graph
        example.prompt = build_graph_core_prompt(original_prompt)
        example.reference_graph = extract_graph_body_from_reference(full_reference_graph)
    return examples


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def sanitize_artifact_stem(value: str, fallback: str) -> str:
    sanitized = ''.join(char if char.isalnum() or char in {'-', '_'} else '_' for char in (value or ''))
    sanitized = sanitized.strip('_')
    return (sanitized or fallback)[:80]


def persist_rollout_artifacts(rollout_dir: Path, rollout_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    persisted_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rollout_rows, start=1):
        stem = sanitize_artifact_stem(str(row.get('skill_id', '')), f'sample_{index:02d}')
        graph_filename = f'{index:02d}_{stem}.md'
        payload_filename = f'{index:02d}_{stem}.json'
        graph_path = rollout_dir / graph_filename
        payload_path = rollout_dir / payload_filename

        graph_text = str(row.get('graph_text') or row.get('generated_text') or '')
        graph_path.write_text(graph_text, encoding='utf-8')
        payload_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')

        persisted = dict(row)
        persisted['generated_graph_file'] = graph_filename
        persisted['payload_file'] = payload_filename
        persisted_rows.append(persisted)

    (rollout_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (rollout_dir / 'scored_rows.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in persisted_rows),
        encoding='utf-8',
    )


def render_metric_plots(
    log_path: Path,
    plot_dir: Path,
    step_index: int,
    plot_every: int,
    refresh_seconds: int,
    *,
    refresh_dashboard: bool = False,
    force: bool = False,
) -> None:
    if render_structured_sft_log_file is None:
        raise RuntimeError(LIVE_PLOT_IMPORT_ERROR or 'plot_structured_sft_metrics is unavailable')
    if not force and step_index % max(1, plot_every) != 0:
        return
    render_structured_sft_log_file(
        log_file=log_path,
        output_dir=plot_dir,
        refresh_dashboard=refresh_dashboard,
        dashboard_refresh_seconds=refresh_seconds,
        quiet=True,
    )




def make_truth_cache_key(skill_id: str, prompt: str) -> tuple[str, str]:
    return str(skill_id or ''), str(prompt or '')


def normalize_similarity_text(text: str) -> str:
    return ' '.join(SIMILARITY_TOKEN_RE.findall((text or '').lower()))


def sequence_lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def sequence_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return sequence_lcs_length(left, right) / max(len(left), len(right), 1)


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def extract_graph_semantics(graph_text: str, graph_label: str) -> dict[str, Any]:
    normalized_graph = normalize_graph_text(graph_text)
    semantics: dict[str, Any] = {
        'graph_text': normalized_graph,
        'node_count': 0,
        'edge_count': 0,
        'start_name': '',
        'finish_name': '',
        'witness_exists': False,
        'witness_node_names': [],
        'witness_node_tasks': [],
        'witness_edge_pairs': [],
        'witness_edge_conditions': [],
        'all_node_names': [],
        'all_edge_pairs': [],
        'parse_error': '',
    }
    if not normalized_graph:
        return semantics
    try:
        nodes, edges, _ = graph_validator.parse_graph_lines(normalized_graph.splitlines(), graph_label)
        indegree, outdegree, outgoing, _ = graph_validator.build_degree_maps(nodes, edges)
        _, _, start_node, finish_node = graph_validator.identify_start_finish(nodes, indegree, outdegree)
        reachable_from_start = graph_validator.bfs_reachable(start_node, outgoing) if start_node else set()
        witness_path = None
        if start_node and finish_node and finish_node in reachable_from_start:
            witness_path = graph_validator.solve_witness_path(nodes, edges, start_node, finish_node)

        node_name_by_id = {
            node_id: normalize_similarity_text(node.name) or normalize_similarity_text(node.task) or normalize_similarity_text(node_id)
            for node_id, node in nodes.items()
        }
        node_task_by_id = {
            node_id: normalize_similarity_text(node.task) or normalize_similarity_text(node.name)
            for node_id, node in nodes.items()
        }
        all_node_names = sorted({value for value in node_name_by_id.values() if value})
        all_edge_pairs = sorted({
            f"{node_name_by_id.get(edge.src, normalize_similarity_text(edge.src))}->{node_name_by_id.get(edge.dst, normalize_similarity_text(edge.dst))}"
            for edge in edges
        })

        semantics.update(
            {
                'node_count': len(nodes),
                'edge_count': len(edges),
                'start_name': node_name_by_id.get(start_node, '') if start_node else '',
                'finish_name': node_name_by_id.get(finish_node, '') if finish_node else '',
                'all_node_names': all_node_names,
                'all_edge_pairs': all_edge_pairs,
                'witness_exists': witness_path is not None,
            }
        )
        if witness_path:
            semantics['witness_node_names'] = [node_name_by_id.get(node_id, normalize_similarity_text(node_id)) for node_id in witness_path.nodes]
            semantics['witness_node_tasks'] = [node_task_by_id.get(node_id, '') for node_id in witness_path.nodes]
            witness_edge_pairs: list[str] = []
            witness_edge_conditions: list[str] = []
            for edge_payload in witness_path.edges:
                src_name = node_name_by_id.get(edge_payload['src'], normalize_similarity_text(edge_payload['src']))
                dst_name = node_name_by_id.get(edge_payload['dst'], normalize_similarity_text(edge_payload['dst']))
                witness_edge_pairs.append(f'{src_name}->{dst_name}')
                witness_edge_conditions.append(normalize_similarity_text(edge_payload.get('condition', '')))
            semantics['witness_edge_pairs'] = witness_edge_pairs
            semantics['witness_edge_conditions'] = witness_edge_conditions
    except Exception as exc:
        semantics['parse_error'] = str(exc)
    return semantics


def build_truth_graph_cache(examples: list[Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for example in examples:
        cache_key = make_truth_cache_key(getattr(example, 'skill_id', ''), getattr(example, 'prompt', ''))
        if cache_key in cache:
            continue
        truth_graph_text = str(getattr(example, 'full_reference_graph', '') or getattr(example, 'reference_graph', '') or '')
        cache[cache_key] = extract_graph_semantics(truth_graph_text, f'{getattr(example, "skill_id", "skill")}_truth')
    return cache


def get_truth_graph_semantics(truth_graph_cache: dict[tuple[str, str], dict[str, Any]], example: Any) -> dict[str, Any]:
    cache_key = make_truth_cache_key(getattr(example, 'skill_id', ''), getattr(example, 'prompt', ''))
    semantics = truth_graph_cache.get(cache_key)
    if semantics is None:
        truth_graph_text = str(getattr(example, 'full_reference_graph', '') or getattr(example, 'reference_graph', '') or '')
        semantics = extract_graph_semantics(truth_graph_text, f'{getattr(example, "skill_id", "skill")}_truth')
        truth_graph_cache[cache_key] = semantics
    return semantics


def compute_truth_similarity_metrics(predicted: dict[str, Any], truth: dict[str, Any]) -> dict[str, float]:
    node_name_similarity = sequence_similarity(predicted.get('witness_node_names', []), truth.get('witness_node_names', []))
    task_similarity = sequence_similarity(predicted.get('witness_node_tasks', []), truth.get('witness_node_tasks', []))
    edge_path_similarity = sequence_similarity(predicted.get('witness_edge_pairs', []), truth.get('witness_edge_pairs', []))
    edge_condition_similarity = sequence_similarity(predicted.get('witness_edge_conditions', []), truth.get('witness_edge_conditions', []))
    node_set_similarity = jaccard_similarity(set(predicted.get('all_node_names', [])), set(truth.get('all_node_names', [])))
    edge_set_similarity = jaccard_similarity(set(predicted.get('all_edge_pairs', [])), set(truth.get('all_edge_pairs', [])))
    truth_start_name = str(truth.get('start_name', '') or '')
    truth_finish_name = str(truth.get('finish_name', '') or '')
    predicted_start_name = str(predicted.get('start_name', '') or '')
    predicted_finish_name = str(predicted.get('finish_name', '') or '')
    start_similarity = 1.0 if not truth_start_name else float(predicted_start_name == truth_start_name)
    finish_similarity = 1.0 if not truth_finish_name else float(predicted_finish_name == truth_finish_name)
    terminal_similarity = 0.5 * start_similarity + 0.5 * finish_similarity
    path_similarity = (
        0.35 * node_name_similarity
        + 0.25 * task_similarity
        + 0.20 * edge_path_similarity
        + 0.10 * edge_condition_similarity
        + 0.05 * node_set_similarity
        + 0.05 * edge_set_similarity
    )
    path_similarity = max(0.0, min(1.0, path_similarity))
    graph_semantic_score = max(
        0.0,
        min(
            1.0,
            0.45 * path_similarity
            + 0.20 * node_set_similarity
            + 0.20 * edge_set_similarity
            + 0.15 * terminal_similarity,
        ),
    )
    exact_path_match = float(
        bool(truth.get('witness_node_names'))
        and predicted.get('witness_node_names', []) == truth.get('witness_node_names', [])
        and predicted.get('witness_edge_pairs', []) == truth.get('witness_edge_pairs', [])
    )
    return {
        'truth_path_similarity': round(path_similarity, 4),
        'truth_path_distance': round(1.0 - path_similarity, 4),
        'truth_node_name_similarity': round(node_name_similarity, 4),
        'truth_task_similarity': round(task_similarity, 4),
        'truth_edge_path_similarity': round(edge_path_similarity, 4),
        'truth_edge_condition_similarity': round(edge_condition_similarity, 4),
        'truth_node_set_similarity': round(node_set_similarity, 4),
        'truth_edge_set_similarity': round(edge_set_similarity, 4),
        'truth_terminal_similarity': round(terminal_similarity, 4),
        'truth_graph_semantic_score': round(graph_semantic_score, 4),
        'truth_graph_semantic_loss': round(1.0 - graph_semantic_score, 4),
        'truth_exact_path_match': exact_path_match,
    }


def compute_strict_mermaid_syntax_score(graph_text: str) -> float:
    normalized_graph = normalize_graph_text(graph_text)
    if not normalized_graph:
        return 0.0
    lines = [line.strip() for line in normalized_graph.splitlines() if line.strip()]
    if not lines or not re.match(r'^graph\s+td\b', lines[0], re.IGNORECASE):
        return 0.0
    body_lines = lines[1:]
    if not body_lines:
        return 0.0
    valid_count = sum(1 for line in body_lines if MERMAID_EDGE_LINE_RE.match(line))
    return valid_count / len(body_lines)


def compute_graph_structure_metrics(score_payload: dict[str, Any], graph_text: str) -> dict[str, float]:
    report = (score_payload or {}).get('report') or {}
    checks = report.get('checks') or {}
    graph_td_ok = float(reinforce_lib.has_graph_td_prefix(graph_text))
    edge_marker_ok = float(reinforce_lib.has_edge_marker(graph_text))
    strict_mermaid_syntax_score = compute_strict_mermaid_syntax_score(graph_text)
    unique_start_name_ok = float(len(checks.get('name_start_candidates') or []) == 1)
    unique_finish_name_ok = float(len(checks.get('name_finish_candidates') or []) == 1)
    unique_start_degree_ok = float(len(checks.get('degree_start_candidates') or []) == 1)
    unique_finish_degree_ok = float(len(checks.get('degree_finish_candidates') or []) == 1)
    start_node_identified = float(bool(report.get('start_node')))
    finish_node_identified = float(bool(report.get('finish_node')))
    path_exists_ok = float(bool(checks.get('path_exists')))
    all_nodes_reachable_from_start = float(bool(checks.get('all_nodes_reachable_from_start')))
    all_nodes_can_reach_finish = float(bool(checks.get('all_nodes_can_reach_finish')))
    full_connectivity_ok = float(bool(path_exists_ok and all_nodes_reachable_from_start and all_nodes_can_reach_finish))
    graph_format_score = max(
        0.0,
        min(
            1.0,
            0.10 * graph_td_ok
            + 0.10 * edge_marker_ok
            + 0.25 * strict_mermaid_syntax_score
            + 0.20 * unique_start_name_ok
            + 0.20 * unique_finish_name_ok
            + 0.075 * unique_start_degree_ok
            + 0.075 * unique_finish_degree_ok,
        ),
    )
    graph_connectivity_score = max(
        0.0,
        min(
            1.0,
            0.10 * start_node_identified
            + 0.10 * finish_node_identified
            + 0.20 * path_exists_ok
            + 0.30 * all_nodes_reachable_from_start
            + 0.30 * all_nodes_can_reach_finish,
        ),
    )
    graph_completeness_score = max(
        0.0,
        min(
            1.0,
            0.50 * graph_connectivity_score
            + 0.35 * graph_format_score
            + 0.15 * strict_mermaid_syntax_score,
        ),
    )
    return {
        'graph_td_prefix': round(graph_td_ok, 4),
        'edge_marker': round(edge_marker_ok, 4),
        'strict_mermaid_syntax_score': round(strict_mermaid_syntax_score, 4),
        'unique_start_ok': round(unique_start_name_ok, 4),
        'unique_finish_ok': round(unique_finish_name_ok, 4),
        'unique_start_degree_ok': round(unique_start_degree_ok, 4),
        'unique_finish_degree_ok': round(unique_finish_degree_ok, 4),
        'start_node_identified': round(start_node_identified, 4),
        'finish_node_identified': round(finish_node_identified, 4),
        'path_exists_ok': round(path_exists_ok, 4),
        'all_nodes_reachable_from_start_ok': round(all_nodes_reachable_from_start, 4),
        'all_nodes_can_reach_finish_ok': round(all_nodes_can_reach_finish, 4),
        'full_connectivity_ok': round(full_connectivity_ok, 4),
        'graph_format_score': round(graph_format_score, 4),
        'graph_connectivity_score': round(graph_connectivity_score, 4),
        'graph_completeness_score': round(graph_completeness_score, 4),
    }


def summarize_truth_similarity_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            'truth_path_similarity': 0.0,
            'truth_path_distance': 1.0,
            'truth_node_name_similarity': 0.0,
            'truth_task_similarity': 0.0,
            'truth_edge_path_similarity': 0.0,
            'truth_edge_condition_similarity': 0.0,
            'truth_node_set_similarity': 0.0,
            'truth_edge_set_similarity': 0.0,
            'truth_terminal_similarity': 0.0,
            'truth_graph_semantic_score': 0.0,
            'truth_graph_semantic_loss': 1.0,
            'truth_exact_path_match_rate': 0.0,
            'graph_completeness_rate': 0.0,
            'strict_mermaid_syntax_rate': 0.0,
            'full_connectivity_rate': 0.0,
            'unique_start_rate': 0.0,
            'unique_finish_rate': 0.0,
        }
    metric_keys = [
        'truth_path_similarity',
        'truth_path_distance',
        'truth_node_name_similarity',
        'truth_task_similarity',
        'truth_edge_path_similarity',
        'truth_edge_condition_similarity',
        'truth_node_set_similarity',
        'truth_edge_set_similarity',
        'truth_terminal_similarity',
        'truth_graph_semantic_score',
        'truth_graph_semantic_loss',
    ]
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        summary[key] = round(sum(values) / len(values), 4)
    exact_matches = [float(row.get('truth_exact_path_match', 0.0) or 0.0) for row in rows]
    summary['truth_exact_path_match_rate'] = round(sum(exact_matches) / len(exact_matches), 4)
    summary['graph_completeness_rate'] = round(sum(float(row.get('graph_completeness_score', 0.0) or 0.0) for row in rows) / len(rows), 4)
    summary['strict_mermaid_syntax_rate'] = round(sum(float(row.get('strict_mermaid_syntax_score', 0.0) or 0.0) for row in rows) / len(rows), 4)
    summary['full_connectivity_rate'] = round(sum(float(row.get('full_connectivity_ok', 0.0) or 0.0) for row in rows) / len(rows), 4)
    summary['unique_start_rate'] = round(sum(float(row.get('unique_start_ok', 0.0) or 0.0) for row in rows) / len(rows), 4)
    summary['unique_finish_rate'] = round(sum(float(row.get('unique_finish_ok', 0.0) or 0.0) for row in rows) / len(rows), 4)
    return summary


def build_progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    filled = int(width * current / total)
    return '[' + ('#' * filled) + ('-' * (width - filled)) + ']'


def compute_validation_deltas(current_summary: dict[str, Any], previous_summary: dict[str, Any] | None) -> dict[str, float]:
    if not previous_summary:
        return {}
    deltas: dict[str, float] = {}
    for key in VALIDATION_DELTA_KEYS:
        current_value = current_summary.get(key)
        previous_value = previous_summary.get(key)
        if current_value is None or previous_value is None:
            continue
        try:
            deltas[f'{key}_delta'] = round(float(current_value) - float(previous_value), 4)
        except (TypeError, ValueError):
            continue
    return deltas


def print_validation_progress(step_index: int, current: int, total: int, interim_summary: dict[str, Any], started_at: float) -> None:
    bar = build_progress_bar(current, total)
    elapsed = time.time() - started_at
    graph_completeness_rate = float(interim_summary.get('graph_completeness_rate', 0.0) or 0.0)
    semantic_rate = float(interim_summary.get('truth_graph_semantic_score', 0.0) or 0.0)
    node_match_rate = float(interim_summary.get('truth_node_set_similarity', 0.0) or 0.0)
    edge_match_rate = float(interim_summary.get('truth_edge_set_similarity', 0.0) or 0.0)
    z3_pass_rate = float(interim_summary.get('z3_pass_rate', 0.0) or 0.0)
    print(
        (
            f'\r[validation step {step_index:04d}] {bar} {current}/{total} '
            f'complete={graph_completeness_rate:.4f} semantic={semantic_rate:.4f} '
            f'node_match={node_match_rate:.4f} edge_match={edge_match_rate:.4f} '
            f'z3={z3_pass_rate:.4f} elapsed={elapsed:.1f}s'
        ),
        end='',
        flush=True,
    )
    if current >= total:
        print(flush=True)



def evaluate_policy_with_progress(
    *,
    policy_model: Any,
    tokenizer: Any,
    examples: list[Any],
    prompt_cache: dict[tuple[str, str], reinforce_lib.CachedExampleEncoding],
    truth_graph_cache: dict[tuple[str, str], dict[str, Any]],
    reward_config: dict[str, float],
    output_dir: Path,
    step_index: int,
    max_new_tokens: int,
    cutoff_len: int,
    graph_core_prompt: bool,
    previous_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    temp_dir = output_dir / f'eval_step_{step_index:04d}' / 'tmp_graphs'
    scored_rows: list[dict[str, Any]] = []
    started_at = time.time()
    total = len(examples)

    for current, example in enumerate(examples, start=1):
        cached_example = reinforce_lib.get_or_create_cached_example(
            tokenizer=tokenizer,
            prompt_cache=prompt_cache,
            skill_id=example.skill_id,
            prompt=example.prompt,
            reference_graph=example.reference_graph,
            cutoff_len=cutoff_len,
        )
        generated_text, _, _ = reinforce_lib.generate_text_from_cache(
            tokenizer=tokenizer,
            model=policy_model,
            cached_example=cached_example,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.95,
        )
        graph_text = generated_text
        if graph_core_prompt:
            graph_text = GRAPH_CORE_HEADER + str(generated_text or '').lstrip()
        scored = score_graph_text(example.skill_id, graph_text, reward_config, temp_dir)
        truth_graph = get_truth_graph_semantics(truth_graph_cache, example)
        predicted_graph = extract_graph_semantics(str(scored.get('graph_text') or graph_text or ''), f'{example.skill_id}_pred_eval_{step_index:04d}')
        truth_similarity = compute_truth_similarity_metrics(predicted_graph, truth_graph)
        structure_metrics = compute_graph_structure_metrics(scored, graph_text)
        scored.update(truth_similarity)
        scored.update(structure_metrics)
        scored['prompt'] = example.prompt
        scored['reference_graph'] = getattr(example, 'full_reference_graph', example.reference_graph)
        scored_rows.append(scored)
        interim_summary = summarize_scored_rows(scored_rows)
        interim_summary.update(summarize_truth_similarity_rows(scored_rows))
        print_validation_progress(step_index, current, total, interim_summary, started_at)

    summary = summarize_scored_rows(scored_rows)
    summary.update(summarize_truth_similarity_rows(scored_rows))
    summary['elapsed_seconds'] = round(time.time() - started_at, 2)
    delta = compute_validation_deltas(summary, previous_summary)

    eval_dir = output_dir / f'eval_step_{step_index:04d}'
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (eval_dir / 'scored_rows.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in scored_rows), encoding='utf-8')

    payload = {
        'validation_step': step_index,
        'candidate_count': summary.get('candidate_count', 0),
        'graph_completeness_rate': summary.get('graph_completeness_rate', 0.0),
        'truth_graph_semantic_score': summary.get('truth_graph_semantic_score', 0.0),
        'truth_node_set_similarity': summary.get('truth_node_set_similarity', 0.0),
        'truth_edge_set_similarity': summary.get('truth_edge_set_similarity', 0.0),
        'truth_path_similarity': summary.get('truth_path_similarity', 0.0),
        'full_connectivity_rate': summary.get('full_connectivity_rate', 0.0),
        'z3_pass_rate': summary.get('z3_pass_rate', 0.0),
        'elapsed_seconds': summary.get('elapsed_seconds', 0.0),
        'delta': delta,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return summary, delta


def compute_reference_ce_loss(
    *,
    tokenizer: Any,
    model: Any,
    batch_examples: list[Any],
    prompt_cache: dict[tuple[str, str], reinforce_lib.CachedExampleEncoding],
    cutoff_len: int,
    device: str,
    train_forward_batch_size: int,
) -> tuple[torch.Tensor, list[reinforce_lib.CachedExampleEncoding]]:
    sequences: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    cached_examples: list[reinforce_lib.CachedExampleEncoding] = []
    for example in batch_examples:
        cached_example = reinforce_lib.get_or_create_cached_example(
            tokenizer=tokenizer,
            prompt_cache=prompt_cache,
            skill_id=example.skill_id,
            prompt=example.prompt,
            reference_graph=example.reference_graph,
            cutoff_len=cutoff_len,
        )
        cached_examples.append(cached_example)
        if cached_example.reference_full_ids is None:
            full_ids, target_prompt_length = reinforce_lib.build_full_sequence_encoding(
                tokenizer,
                cached_example,
                target_text=example.reference_graph,
                cutoff_len=cutoff_len,
            )
        else:
            full_ids = cached_example.reference_full_ids
            target_prompt_length = cached_example.reference_prompt_length
        sequences.append(full_ids)
        prompt_lengths.append(target_prompt_length)

    mean_logprobs, _, has_generated = reinforce_lib.compute_batch_generated_token_stats(
        model,
        sequences,
        prompt_lengths,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        forward_batch_size=max(0, int(train_forward_batch_size)),
        compute_entropies=False,
    )
    valid_losses = [-mean_logprobs[index] for index in range(len(batch_examples)) if bool(has_generated[index].item())]
    if not valid_losses:
        return zero_scalar(device, mean_logprobs.dtype), cached_examples
    return torch.stack(valid_losses).mean(), cached_examples




def compute_structure_aux_loss(
    *,
    tokenizer: Any,
    model: Any,
    batch_examples: list[Any],
    cached_examples: list[reinforce_lib.CachedExampleEncoding],
    truth_graph_cache: dict[tuple[str, str], dict[str, Any]],
    reward_config: dict[str, float],
    output_dir: Path,
    step_index: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    penalty_ceiling = max(
        float(args.truth_path_loss_coef)
        + float(args.graph_format_loss_coef)
        + float(args.z3_reachability_loss_coef)
        + float(args.mermaid_completeness_loss_coef),
        1.0e-6,
    )
    if (
        args.truth_path_loss_coef <= 0
        and args.graph_format_loss_coef <= 0
        and args.z3_reachability_loss_coef <= 0
        and args.mermaid_completeness_loss_coef <= 0
    ):
        return zero_scalar(device, dtype), {
            'avg_truth_path_similarity': 0.0,
            'avg_graph_semantic_score': 0.0,
            'avg_graph_semantic_loss': 0.0,
            'avg_truth_exact_path_match_rate': 0.0,
            'avg_z3_reachability_score': 0.0,
            'avg_mermaid_completeness_score': 0.0,
            'avg_graph_completeness_score': 0.0,
            'avg_full_connectivity_rate': 0.0,
            'avg_strict_mermaid_syntax_score': 0.0,
            'avg_graph_format_penalty': 0.0,
            'avg_graph_connectivity_penalty': 0.0,
            'avg_structure_target': 0.0,
            'avg_structure_target_normalized': 0.0,
            'avg_structure_quality_loss': 0.0,
            'avg_structure_weight': 0.0,
            'structure_target_std': 0.0,
            'avg_structure_advantage': 0.0,
            'avg_weighted_structure_score': 0.0,
            'structure_sample_count': 0.0,
            'rollout_dir': '',
        }

    rollout_dir = output_dir / 'structured_sft_rollouts' / f'step_{step_index:04d}'
    rollout_dir.mkdir(parents=True, exist_ok=True)
    generations = reinforce_lib.generate_texts_from_cache_batch(
        tokenizer=tokenizer,
        model=model,
        cached_examples=cached_examples,
        max_new_tokens=args.structure_max_new_tokens,
        temperature=args.structure_temperature,
        top_p=args.structure_top_p,
        batch_size=len(cached_examples),
    )

    generated_sequences: list[torch.Tensor] = []
    generated_prompt_lengths: list[int] = []
    structure_penalties: list[float] = []
    normalized_penalties: list[float] = []
    weighted_penalties: list[float] = []
    graph_semantic_losses: list[float] = []
    graph_semantic_scores: list[float] = []
    z3_scores: list[float] = []
    mermaid_scores: list[float] = []
    graph_completeness_scores: list[float] = []
    strict_syntax_scores: list[float] = []
    full_connectivity_scores: list[float] = []
    format_penalties: list[float] = []
    connectivity_penalties: list[float] = []
    rollout_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []

    for example, (generated_text, full_input_ids, prompt_length) in zip(batch_examples, generations):
        graph_text = generated_text
        if args.graph_core_prompt:
            graph_text = GRAPH_CORE_HEADER + str(generated_text or '').lstrip()
        score_payload = score_graph_text(example.skill_id, graph_text, reward_config, rollout_dir)
        structural_scores = reinforce_lib.compute_structural_scores(score_payload, graph_text)
        z3_score = float(structural_scores.get('z3_reachability_score', 0.0) or 0.0)
        mermaid_score = float(structural_scores.get('mermaid_completeness_score', 0.0) or 0.0)
        truth_graph = get_truth_graph_semantics(truth_graph_cache, example)
        predicted_graph = extract_graph_semantics(
            str(score_payload.get('graph_text') or graph_text or ''),
            f'{example.skill_id}_pred_step_{step_index:04d}',
        )
        truth_similarity = compute_truth_similarity_metrics(predicted_graph, truth_graph)
        structure_metrics = compute_graph_structure_metrics(score_payload, graph_text)

        graph_semantic_score = float(truth_similarity.get('truth_graph_semantic_score', 0.0) or 0.0)
        graph_semantic_loss = max(0.0, 1.0 - graph_semantic_score)
        format_penalty = max(0.0, 1.0 - float(structure_metrics.get('graph_format_score', 0.0) or 0.0))
        connectivity_penalty = max(0.0, 1.0 - float(structure_metrics.get('graph_connectivity_score', 0.0) or 0.0))
        z3_penalty = max(0.0, 1.0 - z3_score)
        structure_penalty = (
            float(args.truth_path_loss_coef) * graph_semantic_loss
            + float(args.graph_format_loss_coef) * format_penalty
            + float(args.z3_reachability_loss_coef) * z3_penalty
            + float(args.mermaid_completeness_loss_coef) * connectivity_penalty
        )
        normalized_penalty = max(0.0, min(1.0, structure_penalty / penalty_ceiling))
        effective_penalty = normalized_penalty if args.normalize_structure_scores else structure_penalty

        generated_sequences.append(full_input_ids)
        generated_prompt_lengths.append(prompt_length)
        structure_penalties.append(structure_penalty)
        normalized_penalties.append(normalized_penalty)
        weighted_penalties.append(effective_penalty)
        graph_semantic_losses.append(graph_semantic_loss)
        graph_semantic_scores.append(graph_semantic_score)
        z3_scores.append(z3_score)
        mermaid_scores.append(mermaid_score)
        graph_completeness_scores.append(float(structure_metrics.get('graph_completeness_score', 0.0) or 0.0))
        strict_syntax_scores.append(float(structure_metrics.get('strict_mermaid_syntax_score', 0.0) or 0.0))
        full_connectivity_scores.append(float(structure_metrics.get('full_connectivity_ok', 0.0) or 0.0))
        format_penalties.append(format_penalty)
        connectivity_penalties.append(connectivity_penalty)

        rollout_row = dict(score_payload)
        rollout_row.update(
            {
                'skill_id': example.skill_id,
                'prompt': example.prompt,
                'reference_graph': getattr(example, 'full_reference_graph', example.reference_graph),
                'generated_text': graph_text,
                'prompt_length': prompt_length,
                'z3_reachability_score': round(z3_score, 4),
                'mermaid_completeness_score': round(mermaid_score, 4),
                'graph_semantic_score': round(graph_semantic_score, 4),
                'graph_semantic_loss': round(graph_semantic_loss, 4),
                'graph_format_penalty': round(format_penalty, 4),
                'graph_connectivity_penalty': round(connectivity_penalty, 4),
                'structure_target': round(structure_penalty, 4),
                'weighted_structure_score': round(effective_penalty, 4),
                'structure_target_normalized': round(normalized_penalty, 4),
                'structure_quality_loss': round(connectivity_penalty, 4),
                'structure_weight': round(effective_penalty, 4),
            }
        )
        rollout_row.update(truth_similarity)
        rollout_row.update(structure_metrics)
        rollout_rows.append(rollout_row)
        scored_rows.append(rollout_row)

    if not generated_sequences:
        return zero_scalar(device, dtype), {
            'avg_truth_path_similarity': 0.0,
            'avg_graph_semantic_score': 0.0,
            'avg_graph_semantic_loss': 0.0,
            'avg_truth_exact_path_match_rate': 0.0,
            'avg_z3_reachability_score': 0.0,
            'avg_mermaid_completeness_score': 0.0,
            'avg_graph_completeness_score': 0.0,
            'avg_full_connectivity_rate': 0.0,
            'avg_strict_mermaid_syntax_score': 0.0,
            'avg_graph_format_penalty': 0.0,
            'avg_graph_connectivity_penalty': 0.0,
            'avg_structure_target': 0.0,
            'avg_structure_target_normalized': 0.0,
            'avg_structure_quality_loss': 0.0,
            'avg_structure_weight': 0.0,
            'structure_target_std': 0.0,
            'avg_structure_advantage': 0.0,
            'avg_weighted_structure_score': 0.0,
            'structure_sample_count': 0.0,
            'rollout_dir': str(rollout_dir),
        }

    target_mean = sum(weighted_penalties) / len(weighted_penalties)
    target_variance = sum((score - target_mean) ** 2 for score in weighted_penalties) / len(weighted_penalties)
    target_std = math.sqrt(max(target_variance, 0.0))

    mean_logprobs, _, has_generated = reinforce_lib.compute_batch_generated_token_stats(
        model,
        generated_sequences,
        generated_prompt_lengths,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        forward_batch_size=max(0, int(args.train_forward_batch_size)),
        compute_entropies=False,
    )

    loss_terms: list[torch.Tensor] = []
    effective_weights: list[float] = []
    for index, effective_penalty in enumerate(weighted_penalties):
        rollout_rows[index]['mean_generated_logprob'] = float(mean_logprobs[index].detach().item())
        rollout_rows[index]['mean_generated_neg_logprob'] = float((-mean_logprobs[index]).detach().item())
        rollout_rows[index]['has_generated_tokens'] = bool(has_generated[index].item())
        rollout_rows[index]['structure_advantage'] = float(effective_penalty)
        if not bool(has_generated[index].item()):
            continue
        penalty_tensor = torch.tensor(float(effective_penalty), device=mean_logprobs.device, dtype=mean_logprobs.dtype)
        loss_terms.append(penalty_tensor * (-mean_logprobs[index]))
        effective_weights.append(float(effective_penalty))

    aux_loss = zero_scalar(device, mean_logprobs.dtype)
    if loss_terms:
        aux_loss = torch.stack(loss_terms).mean()

    rollout_summary = summarize_scored_rows(scored_rows)
    rollout_summary.update(summarize_truth_similarity_rows(scored_rows))
    rollout_summary.update(
        {
            'step': step_index,
            'avg_graph_semantic_score': round(sum(graph_semantic_scores) / len(graph_semantic_scores), 4) if graph_semantic_scores else 0.0,
            'avg_graph_semantic_loss': round(sum(graph_semantic_losses) / len(graph_semantic_losses), 4) if graph_semantic_losses else 0.0,
            'avg_z3_reachability_score': round(sum(z3_scores) / len(z3_scores), 4) if z3_scores else 0.0,
            'avg_mermaid_completeness_score': round(sum(mermaid_scores) / len(mermaid_scores), 4) if mermaid_scores else 0.0,
            'avg_graph_completeness_score': round(sum(graph_completeness_scores) / len(graph_completeness_scores), 4) if graph_completeness_scores else 0.0,
            'avg_full_connectivity_rate': round(sum(full_connectivity_scores) / len(full_connectivity_scores), 4) if full_connectivity_scores else 0.0,
            'avg_strict_mermaid_syntax_score': round(sum(strict_syntax_scores) / len(strict_syntax_scores), 4) if strict_syntax_scores else 0.0,
            'avg_graph_format_penalty': round(sum(format_penalties) / len(format_penalties), 4) if format_penalties else 0.0,
            'avg_graph_connectivity_penalty': round(sum(connectivity_penalties) / len(connectivity_penalties), 4) if connectivity_penalties else 0.0,
            'avg_structure_target': round(sum(structure_penalties) / len(structure_penalties), 4) if structure_penalties else 0.0,
            'avg_structure_target_normalized': round(sum(normalized_penalties) / len(normalized_penalties), 4) if normalized_penalties else 0.0,
            'avg_structure_quality_loss': round(sum(connectivity_penalties) / len(connectivity_penalties), 4) if connectivity_penalties else 0.0,
            'avg_structure_weight': round(sum(effective_weights) / len(effective_weights), 4) if effective_weights else 0.0,
            'structure_target_std': round(target_std, 4) if weighted_penalties else 0.0,
            'avg_weighted_structure_score': round(sum(weighted_penalties) / len(weighted_penalties), 4) if weighted_penalties else 0.0,
            'avg_structure_advantage': round(sum(weighted_penalties) / len(weighted_penalties), 4) if weighted_penalties else 0.0,
            'structure_sample_count': len(rollout_rows),
        }
    )
    if args.save_rollout_artifacts:
        persist_rollout_artifacts(rollout_dir, rollout_rows, rollout_summary)

    return aux_loss, {
        'avg_truth_path_similarity': rollout_summary['truth_path_similarity'],
        'avg_graph_semantic_score': rollout_summary['avg_graph_semantic_score'],
        'avg_graph_semantic_loss': rollout_summary['avg_graph_semantic_loss'],
        'avg_truth_exact_path_match_rate': rollout_summary['truth_exact_path_match_rate'],
        'avg_z3_reachability_score': rollout_summary['avg_z3_reachability_score'],
        'avg_mermaid_completeness_score': rollout_summary['avg_mermaid_completeness_score'],
        'avg_graph_completeness_score': rollout_summary['avg_graph_completeness_score'],
        'avg_full_connectivity_rate': rollout_summary['avg_full_connectivity_rate'],
        'avg_strict_mermaid_syntax_score': rollout_summary['avg_strict_mermaid_syntax_score'],
        'avg_graph_format_penalty': rollout_summary['avg_graph_format_penalty'],
        'avg_graph_connectivity_penalty': rollout_summary['avg_graph_connectivity_penalty'],
        'avg_structure_target': rollout_summary['avg_structure_target'],
        'avg_structure_target_normalized': rollout_summary['avg_structure_target_normalized'],
        'avg_structure_quality_loss': rollout_summary['avg_structure_quality_loss'],
        'avg_structure_weight': rollout_summary['avg_structure_weight'],
        'structure_target_std': rollout_summary['structure_target_std'],
        'avg_structure_advantage': rollout_summary['avg_structure_advantage'],
        'avg_weighted_structure_score': rollout_summary['avg_weighted_structure_score'],
        'structure_sample_count': float(len(generated_sequences)),
        'rollout_dir': str(rollout_dir),
    }


def maybe_render_plots(args: argparse.Namespace, log_path: Path, plot_dir: Path, step_index: int, *, force: bool = False) -> None:
    if not (args.save_metric_plots or args.live_plot):
        return
    missing_loss_plot = step_index > 0 and not (plot_dir / 'loss_metrics.png').exists()
    should_force = force or missing_loss_plot
    try:
        render_metric_plots(
            log_path,
            plot_dir,
            step_index,
            args.plot_every,
            args.plot_refresh_seconds,
            refresh_dashboard=args.live_plot,
            force=should_force,
        )
    except Exception as exc:
        print(json.dumps({'warning': 'metric_plot_render_failed', 'step': step_index, 'error': str(exc)}, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    reinforce_lib.set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_config = opt.load_reward_config(args.reward_config)
    plot_dir = Path(args.plot_dir).resolve() if args.plot_dir else output_dir / 'seaborn_plots'

    train_examples = align_examples_to_graph_core(load_prompt_examples(Path(args.train_file).resolve(), limit=0), args.graph_core_prompt)
    val_examples = align_examples_to_graph_core(load_prompt_examples(Path(args.val_file).resolve(), limit=0), args.graph_core_prompt)
    if not train_examples:
        raise SystemExit('Training dataset is empty.')
    if not val_examples:
        raise SystemExit('Validation dataset is empty.')
    if args.eval_limit > 0:
        val_examples = val_examples[:args.eval_limit]

    tokenizer, model = load_policy_model(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.start_adapter_path,
        torch_dtype=args.torch_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        trainable=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules or None,
    )
    train_prompt_cache = reinforce_lib.build_example_encoding_cache(tokenizer, train_examples, args.cutoff_len)
    val_prompt_cache = reinforce_lib.build_example_encoding_cache(tokenizer, val_examples, args.cutoff_len)
    train_truth_graph_cache = build_truth_graph_cache(train_examples)
    val_truth_graph_cache = build_truth_graph_cache(val_examples)
    train_prompt_stats = reinforce_lib.summarize_prompt_lengths(tokenizer, train_examples, args.cutoff_len, prompt_cache=train_prompt_cache)
    val_prompt_stats = reinforce_lib.summarize_prompt_lengths(tokenizer, val_examples, args.cutoff_len, prompt_cache=val_prompt_cache)
    print(json.dumps({'train_prompt_stats': train_prompt_stats, 'val_prompt_stats': val_prompt_stats}, ensure_ascii=False), flush=True)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    batch_size = max(1, int(args.per_device_train_batch_size))
    accum_steps = max(1, int(args.gradient_accumulation_steps))
    epoch_count = max(1, int(math.ceil(args.num_train_epochs)))

    log_path = output_dir / 'structured_sft_train_log.jsonl'
    log_path.write_text('', encoding='utf-8')
    started_at = time.time()
    global_step = 0
    optimizer_step = 0
    last_validation_summary: dict[str, Any] | None = None

    if not args.skip_initial_eval:
        initial_validation, initial_validation_delta = evaluate_policy_with_progress(
            policy_model=model,
            tokenizer=tokenizer,
            examples=val_examples,
            prompt_cache=val_prompt_cache,
            truth_graph_cache=val_truth_graph_cache,
            reward_config=reward_config,
            output_dir=output_dir,
            step_index=0,
            max_new_tokens=args.structure_max_new_tokens,
            cutoff_len=args.cutoff_len,
            graph_core_prompt=args.graph_core_prompt,
            previous_summary=None,
        )
        initial_metrics = {
            'step': 0,
            'micro_step': 0,
            'epoch': 0,
            'note': 'initial_eval',
            'truth_path_loss_coef': args.truth_path_loss_coef,
            'z3_reachability_loss_coef': args.z3_reachability_loss_coef,
            'mermaid_completeness_loss_coef': args.mermaid_completeness_loss_coef,
            'structure_aux_coef': args.structure_aux_coef,
            'elapsed_seconds': round(time.time() - started_at, 2),
            'validation': initial_validation,
            'validation_delta': initial_validation_delta,
        }
        last_validation_summary = initial_validation
        append_jsonl_row(log_path, initial_metrics)
        maybe_render_plots(args, log_path, plot_dir, 0, force=True)
        print(json.dumps(initial_metrics, ensure_ascii=False), flush=True)

    optimizer.zero_grad(set_to_none=True)
    ce_loss_ema: float | None = None
    structure_loss_ema: float | None = None
    for epoch_index in range(epoch_count):
        shuffled = list(train_examples)
        random.shuffle(shuffled)
        for batch_start in range(0, len(shuffled), batch_size):
            batch_examples = shuffled[batch_start : batch_start + batch_size]
            global_step += 1

            ce_loss, cached_examples = compute_reference_ce_loss(
                tokenizer=tokenizer,
                model=model,
                batch_examples=batch_examples,
                prompt_cache=train_prompt_cache,
                cutoff_len=args.cutoff_len,
                device=args.device,
                train_forward_batch_size=args.train_forward_batch_size,
            )
            structure_raw_loss, structure_metrics = compute_structure_aux_loss(
                tokenizer=tokenizer,
                model=model,
                batch_examples=batch_examples,
                cached_examples=cached_examples,
                truth_graph_cache=train_truth_graph_cache,
                reward_config=reward_config,
                output_dir=output_dir,
                step_index=global_step,
                args=args,
            )
            ce_loss_value = float(ce_loss.detach().item())
            ce_denom = max(float(args.ce_normalize_floor), 1.0e-6)
            if ce_loss_ema is not None:
                ce_denom = max(ce_denom, ce_loss_ema)
            elif ce_loss_value > 0:
                ce_denom = max(ce_denom, ce_loss_value)
            ce_loss_normalized = ce_loss / ce_denom if args.normalize_ce_loss else ce_loss
            if ce_loss_ema is None:
                ce_loss_ema = max(ce_loss_value, float(args.ce_normalize_floor))
            else:
                ce_loss_ema = args.ce_ema_beta * ce_loss_ema + (1.0 - args.ce_ema_beta) * ce_loss_value

            structure_loss_value = float(structure_raw_loss.detach().item())
            structure_denom = max(float(args.structure_loss_normalize_floor), 1.0e-6)
            if structure_loss_ema is not None:
                structure_denom = max(structure_denom, structure_loss_ema)
            elif structure_loss_value > 0:
                structure_denom = max(structure_denom, structure_loss_value)
            structure_loss_normalized = structure_raw_loss / structure_denom if args.normalize_structure_loss else structure_raw_loss
            if structure_loss_ema is None:
                structure_loss_ema = max(structure_loss_value, float(args.structure_loss_normalize_floor))
            else:
                structure_loss_ema = args.structure_loss_ema_beta * structure_loss_ema + (1.0 - args.structure_loss_ema_beta) * structure_loss_value

            structure_aux_loss = args.structure_aux_coef * structure_loss_normalized
            total_loss = args.sft_ce_coef * ce_loss_normalized + structure_aux_loss
            (total_loss / accum_steps).backward()

            is_last_micro_step = batch_start + batch_size >= len(shuffled)
            should_step = (global_step % accum_steps == 0) or is_last_micro_step
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            metrics = {
                'step': optimizer_step,
                'micro_step': global_step,
                'epoch': epoch_index + 1,
                'token_ce_loss': round(float(ce_loss.detach().item()), 6),
                'token_ce_loss_normalized': round(float(ce_loss_normalized.detach().item()), 6),
                'ce_loss': round(float(ce_loss.detach().item()), 6),
                'ce_loss_normalized': round(float(ce_loss_normalized.detach().item()), 6),
                'ce_loss_ema': round(float(ce_loss_ema), 6),
                'graph_semantic_loss': structure_metrics['avg_graph_semantic_loss'],
                'graph_semantic_score': structure_metrics['avg_graph_semantic_score'],
                'structure_raw_loss': round(float(structure_raw_loss.detach().item()), 6),
                'structure_loss_normalized': round(float(structure_loss_normalized.detach().item()), 6),
                'structure_loss_ema': round(float(structure_loss_ema), 6),
                'structure_aux_loss': round(float(structure_aux_loss.detach().item()), 6),
                'total_loss': round(float(total_loss.detach().item()), 6),
                'graph_completeness_score': structure_metrics['avg_graph_completeness_score'],
                'full_connectivity_rate': structure_metrics['avg_full_connectivity_rate'],
                'strict_mermaid_syntax_score': structure_metrics['avg_strict_mermaid_syntax_score'],
                'avg_truth_path_similarity': structure_metrics['avg_truth_path_similarity'],
                'avg_truth_exact_path_match_rate': structure_metrics['avg_truth_exact_path_match_rate'],
                'avg_z3_reachability_score': structure_metrics['avg_z3_reachability_score'],
                'avg_mermaid_completeness_score': structure_metrics['avg_mermaid_completeness_score'],
                'avg_graph_format_penalty': structure_metrics['avg_graph_format_penalty'],
                'avg_graph_connectivity_penalty': structure_metrics['avg_graph_connectivity_penalty'],
                'avg_structure_target': structure_metrics['avg_structure_target'],
                'avg_structure_target_normalized': structure_metrics['avg_structure_target_normalized'],
                'avg_structure_quality_loss': structure_metrics['avg_structure_quality_loss'],
                'avg_structure_weight': structure_metrics['avg_structure_weight'],
                'structure_target_std': structure_metrics['structure_target_std'],
                'avg_structure_advantage': structure_metrics['avg_structure_advantage'],
                'avg_weighted_structure_score': structure_metrics['avg_weighted_structure_score'],
                'structure_sample_count': int(structure_metrics['structure_sample_count']),
                'rollout_dir': structure_metrics['rollout_dir'],
                'sft_ce_coef': args.sft_ce_coef,
                'structure_aux_coef': args.structure_aux_coef,
                'truth_path_loss_coef': args.truth_path_loss_coef,
                'z3_reachability_loss_coef': args.z3_reachability_loss_coef,
                'mermaid_completeness_loss_coef': args.mermaid_completeness_loss_coef,
                'elapsed_seconds': round(time.time() - started_at, 2),
            }
            should_run_validation = (optimizer_step == 10) or (optimizer_step >= 50 and optimizer_step % 50 == 0)
            if should_run_validation:
                validation_summary, validation_delta = evaluate_policy_with_progress(
                    policy_model=model,
                    tokenizer=tokenizer,
                    examples=val_examples,
                    prompt_cache=val_prompt_cache,
                    truth_graph_cache=val_truth_graph_cache,
                    reward_config=reward_config,
                    output_dir=output_dir,
                    step_index=optimizer_step,
                    max_new_tokens=args.structure_max_new_tokens,
                    cutoff_len=args.cutoff_len,
                    graph_core_prompt=args.graph_core_prompt,
                    previous_summary=last_validation_summary,
                )
                metrics['validation'] = validation_summary
                metrics['validation_delta'] = validation_delta
                last_validation_summary = validation_summary
            if optimizer_step % max(1, args.save_every) == 0:
                checkpoint_paths = reinforce_lib.save_checkpoint(
                    model,
                    tokenizer,
                    output_dir,
                    optimizer_step,
                    metrics,
                    save_latest=args.save_latest_checkpoint,
                )
                metrics['checkpoint_dir'] = str(checkpoint_paths['checkpoint_dir'])
                if 'latest_checkpoint_dir' in checkpoint_paths:
                    metrics['checkpoint_latest_dir'] = str(checkpoint_paths['latest_checkpoint_dir'])

            append_jsonl_row(log_path, metrics)
            maybe_render_plots(args, log_path, plot_dir, optimizer_step)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)

    final_checkpoint_dir = ''
    if args.save_final_checkpoint:
        final_metrics = {
            'step': optimizer_step,
            'micro_step': global_step,
            'elapsed_seconds': round(time.time() - started_at, 2),
            'note': 'final_checkpoint',
        }
        checkpoint_paths = reinforce_lib.save_checkpoint(
            model,
            tokenizer,
            output_dir,
            optimizer_step,
            final_metrics,
            checkpoint_name='checkpoint_final',
            save_latest=args.save_latest_checkpoint,
        )
        final_checkpoint_dir = str(checkpoint_paths['checkpoint_dir'])

    maybe_render_plots(args, log_path, plot_dir, optimizer_step, force=True)

    final_summary = {
        'output_dir': str(output_dir),
        'plot_dir': str(plot_dir),
        'steps': optimizer_step,
        'micro_steps': global_step,
        'train_file': args.train_file,
        'val_file': args.val_file,
        'model_name_or_path': args.model_name_or_path,
        'start_adapter_path': args.start_adapter_path,
        'num_train_epochs': args.num_train_epochs,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'train_forward_batch_size': args.train_forward_batch_size,
        'sft_ce_coef': args.sft_ce_coef,
        'structure_aux_coef': args.structure_aux_coef,
        'normalize_ce_loss': args.normalize_ce_loss,
        'ce_ema_beta': args.ce_ema_beta,
        'ce_normalize_floor': args.ce_normalize_floor,
        'normalize_structure_loss': args.normalize_structure_loss,
        'structure_loss_ema_beta': args.structure_loss_ema_beta,
        'structure_loss_normalize_floor': args.structure_loss_normalize_floor,
        'truth_path_loss_coef': args.truth_path_loss_coef,
        'graph_format_loss_coef': args.graph_format_loss_coef,
        'z3_reachability_loss_coef': args.z3_reachability_loss_coef,
        'mermaid_completeness_loss_coef': args.mermaid_completeness_loss_coef,
        'center_structure_advantages': args.center_structure_advantages,
        'normalize_structure_scores': args.normalize_structure_scores,
        'structure_score_std_floor': args.structure_score_std_floor,
        'structure_score_temperature': args.structure_score_temperature,
        'structure_advantage_clip': args.structure_advantage_clip,
        'structure_temperature': args.structure_temperature,
        'structure_top_p': args.structure_top_p,
        'structure_max_new_tokens': args.structure_max_new_tokens,
        'cutoff_len': args.cutoff_len,
        'learning_rate': args.learning_rate,
        'max_grad_norm': args.max_grad_norm,
        'save_every': args.save_every,
        'eval_every': args.eval_every,
        'eval_limit': args.eval_limit,
        'skip_initial_eval': args.skip_initial_eval,
        'save_rollout_artifacts': args.save_rollout_artifacts,
        'live_plot': args.live_plot,
        'plot_every': args.plot_every,
        'plot_refresh_seconds': args.plot_refresh_seconds,
        'save_metric_plots': args.save_metric_plots,
        'reward_config': args.reward_config,
        'train_prompt_stats': train_prompt_stats,
        'val_prompt_stats': val_prompt_stats,
        'final_checkpoint_dir': final_checkpoint_dir,
    }
    (output_dir / 'final_summary.json').write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
