#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import optimize_skill_graph_model as opt
from skill_graph_model_utils import (
    load_policy_model,
    load_prompt_examples,
    score_graph_text,
    summarize_scored_rows,
)

try:
    from plot_reinforce_metrics import render_log_file as render_reinforce_log_file
    LIVE_PLOT_IMPORT_ERROR = ''
except Exception as exc:
    render_reinforce_log_file = None
    LIVE_PLOT_IMPORT_ERROR = str(exc)


@dataclass
class Trajectory:
    skill_id: str
    prompt: str
    reference_graph: str
    input_ids: torch.Tensor
    prompt_length: int
    generated_text: str
    reward: float
    shaped_reward: float
    score_payload: dict[str, Any]


@dataclass(frozen=True)
class CachedExampleEncoding:
    formatted_prompt: str
    prompt_ids: torch.Tensor
    prompt_length: int
    reference_full_ids: torch.Tensor | None = None
    reference_prompt_length: int = 0


GRAPH_TD_PREFIX_RE = re.compile(r'^\s*graph\s+TD\b', re.IGNORECASE)
FINISH_NAME_RE = re.compile(r'name:\s*finish_node\b', re.IGNORECASE)


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
    dataset_dir = repo_root / 'artifacts' / 'graph_model_optimization' / 'llamafactory_data'
    output_dir = repo_root / 'artifacts' / 'reinforce' / 'skill_graph_qwen25'
    parser = argparse.ArgumentParser(description='Lightweight REINFORCE training on top of the SFT skill-graph policy.')
    parser.add_argument('--config', default='', help='Optional simple YAML config file.')
    parser.add_argument('--train-file', default=str(dataset_dir / 'skill_graph_ppo_train.jsonl'), help='Training prompt JSONL file.')
    parser.add_argument('--val-file', default=str(dataset_dir / 'skill_graph_ppo_val.jsonl'), help='Validation prompt JSONL file.')
    parser.add_argument('--model-name-or-path', default='/root/JZY/MCP_Threat_Modeling/Qwen2.5-7B', help='Base model path.')
    parser.add_argument('--start-adapter-path', default='/root/JZY/agent-threat-mining/artifacts/llamafactory/qwen25_mermaid_sft', help='Starting adapter path, typically the SFT adapter.')
    parser.add_argument('--reference-adapter-path', default='', help='Optional frozen reference adapter for KL regularization.')
    parser.add_argument('--output-dir', default=str(output_dir), help='Output directory.')
    parser.add_argument('--reward-config', default='', help='Optional reward config JSON.')
    parser.add_argument('--device', default='cuda', help='Training device.')
    parser.add_argument('--torch-dtype', choices=('auto', 'bfloat16', 'float16', 'float32'), default='bfloat16', help='Torch dtype.')
    parser.add_argument('--trust-remote-code', action='store_true', help='Pass trust_remote_code=True when loading the model.')
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--max-steps', type=int, default=10, help='Number of outer training steps.')
    parser.add_argument('--rollout-batch-size', type=int, default=1, help='Number of prompts sampled per rollout round.')
    parser.add_argument('--num-samples-per-prompt', type=int, default=4, help='Number of rollout candidates sampled for each prompt in a rollout round.')
    parser.add_argument('--rollout-generate-batch-size', type=int, default=0, help='Chunk size used for batched rollout generation. 0 defaults to rollout_batch_size.')
    parser.add_argument('--train-forward-batch-size', type=int, default=0, help='Micro-batch size used for teacher-forced forward passes. 0 uses the full batch.')
    parser.add_argument('--use-group-relative-advantages', default=True, action=argparse.BooleanOptionalAction, help='Use prompt-group relative advantages instead of a pure global moving baseline.')
    parser.add_argument('--learning-rate', type=float, default=2.0e-5, help='Learning rate.')
    parser.add_argument('--entropy-coef', type=float, default=0.0, help='Entropy bonus coefficient.')
    parser.add_argument('--kl-coef', type=float, default=0.01, help='KL penalty coefficient against the reference policy.')
    parser.add_argument('--sft-anchor-coef', type=float, default=0.05, help='Cross-entropy anchor on the SFT reference graph.')
    parser.add_argument('--z3-reachability-loss-coef', type=float, default=0.30, help='Auxiliary policy-gradient weight for Z3/path reachability signals.')
    parser.add_argument('--mermaid-completeness-loss-coef', type=float, default=0.15, help='Auxiliary policy-gradient weight for Mermaid completeness signals.')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping norm.')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='Maximum rollout generation length.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature used for rollouts.')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p sampling used for rollouts.')
    parser.add_argument('--cutoff-len', type=int, default=2048, help='Prompt truncation length.')
    parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank when training from the base model.')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha when training from the base model.')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout when training from the base model.')
    parser.add_argument('--save-every', type=int, default=1, help='Save a checkpoint every N steps.')
    parser.add_argument('--eval-every', type=int, default=1, help='Run validation every N steps.')
    parser.add_argument('--eval-limit', type=int, default=4, help='Number of validation examples per evaluation pass. 0 means full set.')
    parser.add_argument('--baseline-window', type=int, default=16, help='Window size for the moving-average reward baseline.')
    parser.add_argument('--skip-initial-eval', action='store_true', help='Skip validation before the first update step.')
    parser.add_argument('--live-plot', action='store_true', help='Refresh reward and validation plots during training.')
    parser.add_argument('--plot-every', type=int, default=1, help='Refresh plots every N logged steps when live plotting is enabled.')
    parser.add_argument('--plot-dir', default='', help='Optional output directory for live plots. Defaults to <output-dir>/seaborn_plots.')
    parser.add_argument('--plot-refresh-seconds', type=int, default=5, help='Auto-refresh interval for the live HTML dashboard.')
    parser.add_argument('--save-metric-plots', default=True, action=argparse.BooleanOptionalAction, help='Render seaborn metric plots during training even when live plot mode is disabled.')
    parser.add_argument('--save-initial-checkpoint', default=True, action=argparse.BooleanOptionalAction, help='Write a checkpoint before the first optimization step.')
    parser.add_argument('--save-latest-checkpoint', default=True, action=argparse.BooleanOptionalAction, help='Refresh <output-dir>/checkpoint_latest whenever a checkpoint is saved.')
    parser.add_argument('--save-final-checkpoint', default=True, action=argparse.BooleanOptionalAction, help='Write a final checkpoint after the last training step.')
    parser.add_argument('--seed-success-buffer-with-references', default=True, action=argparse.BooleanOptionalAction, help='Preload the success buffer with reference graphs so training starts with structural supervision.')
    parser.add_argument('--reference-success-buffer-limit', type=int, default=128, help='Maximum number of reference graphs to preload into the success buffer. 0 uses success_buffer_size.')
    parser.add_argument('--require-graph-prefix-for-update', action='store_true', help='Only apply policy-gradient updates when rollout text starts with graph TD.')
    parser.add_argument('--positive-advantage-only', default=True, action=argparse.BooleanOptionalAction, help='Only apply policy-gradient updates when the rollout advantage is positive.')
    parser.add_argument('--normalize-advantages', default=True, action=argparse.BooleanOptionalAction, help='Scale policy-gradient advantages by the batch standard deviation.')
    parser.add_argument('--advantage-clip', type=float, default=2.5, help='Absolute clip applied to policy-gradient advantages after scaling. 0 disables clipping.')
    parser.add_argument('--min-pg-reward', type=float, default=0.0, help='Minimum raw reward required before a rollout can receive a policy-gradient update.')
    parser.add_argument('--pg-warmup-steps', type=int, default=20, help='During the first N steps, use a lighter policy-gradient gate to teach output format before full structural constraints.')
    parser.add_argument('--pg-warmup-tighten-step', type=int, default=6, help='Warmup sub-phase boundary. Before this step, warmup uses the loosest policy-gradient gate; after it, warmup can require finish markers.')
    parser.add_argument('--pg-warmup-min-reward', type=float, default=0.0, help='Minimum raw reward required during warmup before a rollout can receive a policy-gradient update.')
    parser.add_argument('--min-pg-updates-per-step', type=int, default=2, help='Target minimum number of policy-gradient trajectories to collect before applying an optimizer step.')
    parser.add_argument('--max-rollout-rounds-per-step', type=int, default=4, help='Maximum rollout sampling rounds to try per outer step when policy-gradient trajectories are too sparse.')
    parser.add_argument('--warmup-require-finish-name', default=False, action=argparse.BooleanOptionalAction, help='Require explicit name: finish_node during warmup policy-gradient gating.')
    parser.add_argument('--min-advantage-scale', type=float, default=1.0, help='Lower bound for advantage normalization scale to avoid tiny batch std values saturating the clipped advantage.')
    parser.add_argument('--success-buffer-size', type=int, default=256, help='Maximum number of high-quality rollout samples to retain for lightweight distillation.')
    parser.add_argument('--success-buffer-min-reward', type=float, default=8.0, help='Minimum reward required before a rollout is added to the success buffer.')
    parser.add_argument('--success-buffer-batch-size', type=int, default=2, help='Number of success-buffer samples used for each lightweight distillation update.')
    parser.add_argument('--success-buffer-coef', type=float, default=0.02, help='Weight of the lightweight distillation loss over high-quality rollout samples.')
    parser.add_argument('--success-buffer-require-full-reach', default=True, action=argparse.BooleanOptionalAction, help='Require full reachability checks before a rollout can enter the success buffer.')
    parser.add_argument('--require-full-reach-for-update', default=False, action=argparse.BooleanOptionalAction, help='Require all nodes to be reachable from start and finish before policy-gradient updates.')
    parser.add_argument('--baseline-on-pg-only', default=False, action=argparse.BooleanOptionalAction, help='Update the reward baseline from policy-gradient trajectories when possible.')
    parser.add_argument('--graph-prefix-bonus', type=float, default=0.0, help='Optional reward bonus when rollout text starts with graph TD.')
    parser.add_argument('--target-modules', nargs='*', default=[], help='Optional LoRA target modules override.')
    args = parser.parse_args(argv)
    return apply_yaml_config(args)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def make_example_cache_key(skill_id: str, prompt: str) -> tuple[str, str]:
    return str(skill_id or ''), str(prompt or '')


def build_example_encoding(
    tokenizer: Any,
    *,
    prompt: str,
    reference_graph: str = '',
    cutoff_len: int,
) -> CachedExampleEncoding:
    formatted_prompt = format_prompt(tokenizer, prompt)
    prompt_ids = tokenizer(
        formatted_prompt,
        truncation=True,
        max_length=cutoff_len,
        return_tensors='pt',
    )['input_ids'][0].cpu()
    prompt_length = int(prompt_ids.shape[-1])
    reference_full_ids: torch.Tensor | None = None
    reference_prompt_length = prompt_length
    if reference_graph:
        full_ids = tokenizer(
            formatted_prompt + reference_graph,
            truncation=True,
            max_length=cutoff_len,
            return_tensors='pt',
        )['input_ids'][0].cpu()
        reference_full_ids = full_ids
        reference_prompt_length = min(prompt_length, int(full_ids.shape[-1]))
    return CachedExampleEncoding(
        formatted_prompt=formatted_prompt,
        prompt_ids=prompt_ids,
        prompt_length=prompt_length,
        reference_full_ids=reference_full_ids,
        reference_prompt_length=reference_prompt_length,
    )


def build_example_encoding_cache(tokenizer: Any, examples: list[Any], cutoff_len: int) -> dict[tuple[str, str], CachedExampleEncoding]:
    cache: dict[tuple[str, str], CachedExampleEncoding] = {}
    for example in examples:
        cache_key = make_example_cache_key(getattr(example, 'skill_id', ''), getattr(example, 'prompt', ''))
        if cache_key in cache:
            continue
        cache[cache_key] = build_example_encoding(
            tokenizer,
            prompt=str(getattr(example, 'prompt', '')),
            reference_graph=str(getattr(example, 'reference_graph', '')),
            cutoff_len=cutoff_len,
        )
    return cache


def get_or_create_cached_example(
    tokenizer: Any,
    prompt_cache: dict[tuple[str, str], CachedExampleEncoding],
    *,
    skill_id: str,
    prompt: str,
    reference_graph: str = '',
    cutoff_len: int,
) -> CachedExampleEncoding:
    cache_key = make_example_cache_key(skill_id, prompt)
    cached_example = prompt_cache.get(cache_key)
    if cached_example is None:
        cached_example = build_example_encoding(
            tokenizer,
            prompt=prompt,
            reference_graph=reference_graph,
            cutoff_len=cutoff_len,
        )
        prompt_cache[cache_key] = cached_example
    return cached_example


def build_full_sequence_encoding(
    tokenizer: Any,
    cached_example: CachedExampleEncoding,
    *,
    target_text: str,
    cutoff_len: int,
) -> tuple[torch.Tensor, int]:
    full_ids = tokenizer(
        cached_example.formatted_prompt + target_text,
        truncation=True,
        max_length=cutoff_len,
        return_tensors='pt',
    )['input_ids'][0].cpu()
    return full_ids, min(cached_example.prompt_length, int(full_ids.shape[-1]))


def pad_token_sequences(
    sequences: list[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
    *,
    padding_side: str = 'right',
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        empty = torch.empty((0, 0), dtype=torch.long, device=device)
        return empty, empty
    cpu_sequences = [sequence.detach().to(dtype=torch.long, device='cpu') for sequence in sequences]
    lengths = torch.tensor([int(sequence.shape[-1]) for sequence in cpu_sequences], dtype=torch.long)
    if padding_side == 'right':
        batch_input_ids = pad_sequence(cpu_sequences, batch_first=True, padding_value=pad_token_id)
        positions = torch.arange(batch_input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        attention_mask = (positions < lengths.unsqueeze(1)).to(dtype=torch.long)
        return batch_input_ids.to(device), attention_mask.to(device)
    if padding_side != 'left':
        raise ValueError(f'Unsupported padding_side: {padding_side}')
    max_length = int(lengths.max().item())
    batch_input_ids = torch.full((len(cpu_sequences), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(cpu_sequences), max_length), dtype=torch.long)
    for index, sequence in enumerate(cpu_sequences):
        seq_len = int(sequence.shape[-1])
        batch_input_ids[index, max_length - seq_len :] = sequence
        attention_mask[index, max_length - seq_len :] = 1
    return batch_input_ids.to(device), attention_mask.to(device)


def compute_batch_generated_token_stats(
    model: Any,
    input_id_sequences: list[torch.Tensor],
    prompt_lengths: list[int],
    *,
    pad_token_id: int,
    device: str | torch.device,
    forward_batch_size: int = 0,
    compute_entropies: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device_obj = torch.device(device)
    if not input_id_sequences:
        empty = torch.empty(0, device=device_obj, dtype=torch.float32)
        return empty, empty, torch.empty(0, device=device_obj, dtype=torch.bool)

    effective_batch_size = int(forward_batch_size) if int(forward_batch_size) > 0 else len(input_id_sequences)
    mean_logprob_chunks: list[torch.Tensor] = []
    mean_entropy_chunks: list[torch.Tensor] = []
    has_generated_chunks: list[torch.Tensor] = []

    for start_index in range(0, len(input_id_sequences), effective_batch_size):
        chunk_sequences = input_id_sequences[start_index : start_index + effective_batch_size]
        chunk_prompt_lengths = prompt_lengths[start_index : start_index + effective_batch_size]
        batch_input_ids, attention_mask = pad_token_sequences(chunk_sequences, pad_token_id, device_obj)
        outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        target_ids = batch_input_ids[:, 1:]
        token_logprobs = -F.cross_entropy(
            logits.transpose(1, 2),
            target_ids,
            reduction='none',
        )
        if compute_entropies:
            log_probs_all = F.log_softmax(logits, dim=-1)
            token_entropies = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)
        else:
            token_entropies = torch.zeros_like(token_logprobs)

        generated_starts = torch.tensor([max(int(length) - 1, 0) for length in chunk_prompt_lengths], device=device_obj).unsqueeze(1)
        token_positions = torch.arange(token_logprobs.shape[1], device=device_obj).unsqueeze(0)
        valid_targets = attention_mask[:, 1:].bool()
        generated_mask = valid_targets & (token_positions >= generated_starts)
        generated_mask_f = generated_mask.to(dtype=token_logprobs.dtype)
        token_counts = generated_mask.sum(dim=1)
        safe_token_counts = token_counts.clamp_min(1).to(dtype=token_logprobs.dtype)

        mean_logprob_chunks.append((token_logprobs * generated_mask_f).sum(dim=1) / safe_token_counts)
        mean_entropy_chunks.append((token_entropies * generated_mask_f).sum(dim=1) / safe_token_counts)
        has_generated_chunks.append(token_counts > 0)

    return (
        torch.cat(mean_logprob_chunks, dim=0),
        torch.cat(mean_entropy_chunks, dim=0),
        torch.cat(has_generated_chunks, dim=0),
    )

def generate_text_from_cache(
    *,
    tokenizer: Any,
    model: Any,
    cached_example: CachedExampleEncoding,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
) -> tuple[str, torch.Tensor, int]:
    device = next(model.parameters()).device
    input_ids = cached_example.prompt_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    do_sample = temperature > 0
    generation_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'use_cache': True,
    }
    if do_sample:
        generation_kwargs['temperature'] = temperature
        generation_kwargs['top_p'] = top_p

    was_training = bool(getattr(model, 'training', False))
    if was_training:
        model.eval()
    try:
        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
    finally:
        if was_training:
            model.train()

    generated_ids = output_ids[0][cached_example.prompt_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, output_ids[0].detach().cpu(), cached_example.prompt_length


def generate_texts_from_cache_batch(
    *,
    tokenizer: Any,
    model: Any,
    cached_examples: list[CachedExampleEncoding],
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
    batch_size: int = 0,
) -> list[tuple[str, torch.Tensor, int]]:
    if not cached_examples:
        return []

    device = next(model.parameters()).device
    do_sample = temperature > 0
    effective_batch_size = int(batch_size) if int(batch_size) > 0 else len(cached_examples)
    results: list[tuple[str, torch.Tensor, int]] = []
    was_training = bool(getattr(model, 'training', False))
    if was_training:
        model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(cached_examples), effective_batch_size):
                chunk = cached_examples[start : start + effective_batch_size]
                batch_input_ids, attention_mask = pad_token_sequences(
                    [cached_example.prompt_ids for cached_example in chunk],
                    int(tokenizer.pad_token_id),
                    device,
                    padding_side='left',
                )
                generation_kwargs = {
                    'input_ids': batch_input_ids,
                    'attention_mask': attention_mask,
                    'max_new_tokens': max_new_tokens,
                    'do_sample': do_sample,
                    'pad_token_id': tokenizer.pad_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': True,
                }
                if do_sample:
                    generation_kwargs['temperature'] = temperature
                    generation_kwargs['top_p'] = top_p
                output_ids = model.generate(**generation_kwargs)
                prompt_width = int(batch_input_ids.shape[1])
                for row_index, cached_example in enumerate(chunk):
                    pad_count = prompt_width - cached_example.prompt_length
                    full_output_ids = output_ids[row_index, pad_count:].detach().cpu()
                    generated_ids = output_ids[row_index, prompt_width:]
                    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    results.append((text, full_output_ids, cached_example.prompt_length))
    finally:
        if was_training:
            model.train()

    return results


def compute_sequence_logprob_and_entropy(model: Any, input_ids: torch.Tensor, prompt_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=input_ids.unsqueeze(0), use_cache=False)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids.unsqueeze(0)[:, 1:]
    log_probs_all = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    generated_start = max(prompt_length - 1, 0)
    generated_logprobs = token_logprobs[:, generated_start:].squeeze(0)
    generated_log_probs_all = log_probs_all[:, generated_start:, :]
    generated_probs = generated_log_probs_all.exp()
    entropy = -(generated_probs * generated_log_probs_all).sum(dim=-1).mean()
    return generated_logprobs, entropy


def compute_teacher_forcing_loss(tokenizer: Any, model: Any, prompt: str, target_text: str, cutoff_len: int) -> torch.Tensor:
    cached_example = build_example_encoding(tokenizer, prompt=prompt, cutoff_len=cutoff_len)
    full_ids, target_prompt_length = build_full_sequence_encoding(
        tokenizer,
        cached_example,
        target_text=target_text,
        cutoff_len=cutoff_len,
    )
    mean_logprobs, _, _ = compute_batch_generated_token_stats(
        model,
        [full_ids],
        [target_prompt_length],
        pad_token_id=int(tokenizer.pad_token_id),
        device=next(model.parameters()).device,
        forward_batch_size=1,
        compute_entropies=False,
    )
    return -mean_logprobs[0]


def summarize_prompt_lengths(
    tokenizer: Any,
    examples: list[Any],
    cutoff_len: int,
    prompt_cache: dict[tuple[str, str], CachedExampleEncoding] | None = None,
) -> dict[str, Any]:
    if not examples:
        return {'count': 0, 'over_cutoff_count': 0}
    lengths: list[int] = []
    for example in examples:
        if prompt_cache is None:
            formatted_prompt = format_prompt(tokenizer, example.prompt)
            lengths.append(len(tokenizer(formatted_prompt)['input_ids']))
            continue
        cached_example = get_or_create_cached_example(
            tokenizer=tokenizer,
            prompt_cache=prompt_cache,
            skill_id=getattr(example, 'skill_id', ''),
            prompt=getattr(example, 'prompt', ''),
            reference_graph=getattr(example, 'reference_graph', ''),
            cutoff_len=cutoff_len,
        )
        lengths.append(cached_example.prompt_length)
    sorted_lengths = sorted(lengths)
    count = len(sorted_lengths)
    p50 = sorted_lengths[count // 2]
    p90 = sorted_lengths[min(count - 1, int(count * 0.9))]
    p95 = sorted_lengths[min(count - 1, int(count * 0.95))]
    over_cutoff = sum(1 for value in sorted_lengths if value > cutoff_len)
    return {
        'count': count,
        'min': min(sorted_lengths),
        'p50': p50,
        'p90': p90,
        'p95': p95,
        'max': max(sorted_lengths),
        'over_cutoff_count': over_cutoff,
        'over_cutoff_rate': round(over_cutoff / count, 4),
        'cutoff_len': cutoff_len,
    }


def has_graph_td_prefix(text: str) -> bool:
    return bool(GRAPH_TD_PREFIX_RE.search(text or ''))


def has_edge_marker(text: str) -> bool:
    return '-->' in (text or '')


def has_finish_name_marker(text: str) -> bool:
    return bool(FINISH_NAME_RE.search(text or ''))


def compute_z3_reachability_score(report: dict[str, Any]) -> float:
    checks = report.get('checks') or {}
    witness_path = report.get('witness_path') or {}
    node_count = max(int(checks.get('node_count', 0) or 0), 1)
    edge_count = max(int(checks.get('edge_count', 0) or 0), 1)
    witness_node_coverage = min(1.0, len(witness_path.get('nodes', [])) / node_count)
    witness_edge_coverage = min(1.0, len(witness_path.get('edges', [])) / edge_count)
    score = (
        0.35 * float(bool(checks.get('path_exists')))
        + 0.25 * float(bool(report.get('witness_path')))
        + 0.20 * float(bool(checks.get('all_nodes_reachable_from_start')))
        + 0.15 * float(bool(checks.get('all_nodes_can_reach_finish')))
        + 0.025 * witness_node_coverage
        + 0.025 * witness_edge_coverage
    )
    return max(0.0, min(1.0, score))


def compute_mermaid_completeness_score(graph_text: str, report: dict[str, Any]) -> float:
    checks = report.get('checks') or {}
    lowered_graph = (graph_text or '').lower()
    if not has_graph_td_prefix(graph_text):
        return 0.0
    if not has_edge_marker(graph_text):
        return 0.0

    node_count = max(int(checks.get('node_count', 0) or 0), 0)
    edge_count = max(int(checks.get('edge_count', 0) or 0), 0)
    node_density = min(1.0, node_count / 4.0)
    edge_density = min(1.0, edge_count / 3.0)
    score = (
        0.25 * float(has_graph_td_prefix(graph_text))
        + 0.20 * float(has_edge_marker(graph_text))
        + 0.15 * float('name: start_node' in lowered_graph)
        + 0.10 * float(has_finish_name_marker(graph_text))
        + 0.10 * float(bool(report.get('start_node')))
        + 0.10 * float(bool(report.get('finish_node')))
        + 0.05 * node_density
        + 0.05 * edge_density
    )
    return max(0.0, min(1.0, score))


def compute_structural_scores(score_payload: dict[str, Any], generated_text: str) -> dict[str, float]:
    report = (score_payload or {}).get('report') or {}
    graph_text = str((score_payload or {}).get('graph_text') or generated_text or '')
    z3_reachability_score = compute_z3_reachability_score(report)
    mermaid_completeness_score = compute_mermaid_completeness_score(graph_text, report)
    return {
        'z3_reachability_score': z3_reachability_score,
        'mermaid_completeness_score': mermaid_completeness_score,
    }


def is_policy_gradient_eligible(
    trajectory: Trajectory,
    *,
    require_full_reach: bool = False,
    min_pg_reward: float = 0.0,
    warmup: bool = False,
    warmup_require_finish_name: bool = False,
) -> bool:
    graph_text = str((trajectory.score_payload or {}).get('graph_text') or trajectory.generated_text or '')
    report = (trajectory.score_payload or {}).get('report') or {}
    checks = report.get('checks') or {}
    base_ok = (
        float(trajectory.reward) >= float(min_pg_reward)
        and has_graph_td_prefix(graph_text)
        and has_edge_marker(graph_text)
        and bool(report.get('start_node'))
    )
    if not base_ok:
        return False
    if warmup:
        if warmup_require_finish_name:
            return has_finish_name_marker(graph_text) and bool(report.get('finish_node'))
        return True
    if not (has_finish_name_marker(graph_text) and bool(report.get('finish_node'))):
        return False
    has_witness = bool(report.get('witness_path'))
    path_ok = bool(checks.get('path_exists')) and has_witness
    if require_full_reach:
        path_ok = (
            path_ok
            and bool(checks.get('all_nodes_reachable_from_start'))
            and bool(checks.get('all_nodes_can_reach_finish'))
        )
    return path_ok


def should_require_finish_name_in_warmup(step_index: int, args: argparse.Namespace) -> bool:
    tighten_step = max(1, int(getattr(args, 'pg_warmup_tighten_step', 1)))
    return step_index >= tighten_step


def compute_raw_advantages(
    trajectories: list[Trajectory],
    fallback_baseline: float,
    *,
    use_group_relative_advantages: bool = True,
) -> list[float]:
    if not trajectories:
        return []
    if not use_group_relative_advantages:
        return [float(trajectory.shaped_reward - fallback_baseline) for trajectory in trajectories]

    grouped_indices: dict[tuple[str, str], list[int]] = {}
    for index, trajectory in enumerate(trajectories):
        grouped_indices.setdefault((trajectory.skill_id, trajectory.prompt), []).append(index)

    advantages = [float(trajectory.shaped_reward - fallback_baseline) for trajectory in trajectories]
    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        total_reward = sum(float(trajectories[index].shaped_reward) for index in indices)
        denom = len(indices) - 1
        for index in indices:
            leave_one_out_baseline = (total_reward - float(trajectories[index].shaped_reward)) / denom
            advantages[index] = float(trajectories[index].shaped_reward - leave_one_out_baseline)
    return advantages


def is_success_buffer_candidate(
    trajectory: Trajectory,
    *,
    min_reward: float,
    require_full_reach: bool = True,
) -> bool:
    graph_text = str((trajectory.score_payload or {}).get('graph_text') or trajectory.generated_text or '')
    report = (trajectory.score_payload or {}).get('report') or {}
    checks = report.get('checks') or {}
    if float(trajectory.reward) < float(min_reward):
        return False
    if not has_graph_td_prefix(graph_text) or not has_edge_marker(graph_text):
        return False
    if not (has_finish_name_marker(graph_text) and bool(report.get('start_node')) and bool(report.get('finish_node'))):
        return False
    if not bool(checks.get('path_exists')) or not bool(report.get('witness_path')):
        return False
    if require_full_reach and not (
        bool(checks.get('all_nodes_reachable_from_start'))
        and bool(checks.get('all_nodes_can_reach_finish'))
    ):
        return False
    return True


def success_buffer_contains(
    success_buffer: deque[dict[str, Any]],
    *,
    skill_id: str,
    target_text: str,
) -> bool:
    buffer_key = (skill_id, target_text)
    return any(
        (str(item.get('skill_id', '')), str(item.get('target_text', ''))) == buffer_key
        for item in success_buffer
    )


def seed_success_buffer_from_references(
    *,
    examples: list[Any],
    success_buffer: deque[dict[str, Any]],
    success_buffer_keys: set[tuple[str, str]],
    max_size: int,
    limit: int,
) -> int:
    if max_size <= 0 or limit == 0:
        return 0
    target_limit = max_size if limit <= 0 else min(max_size, int(limit))
    seeded_count = 0
    for example in examples:
        if len(success_buffer) >= max_size or seeded_count >= target_limit:
            break
        skill_id = str(getattr(example, 'skill_id', '') or '')
        prompt = str(getattr(example, 'prompt', '') or '')
        target_text = str(getattr(example, 'reference_graph', '') or '').strip()
        if not target_text:
            continue
        buffer_key = (skill_id, target_text)
        if buffer_key in success_buffer_keys:
            continue
        success_buffer.append(
            {
                'skill_id': skill_id,
                'prompt': prompt,
                'target_text': target_text,
                'reward': None,
                'seed_source': 'reference',
            }
        )
        success_buffer_keys.add(buffer_key)
        seeded_count += 1
    return seeded_count


def evaluate_policy(
    *,
    policy_model: Any,
    tokenizer: Any,
    examples: list[Any],
    prompt_cache: dict[tuple[str, str], CachedExampleEncoding],
    reward_config: dict[str, float],
    output_dir: Path,
    step_index: int,
    max_new_tokens: int,
    cutoff_len: int,
) -> dict[str, Any]:
    temp_dir = output_dir / f'eval_step_{step_index:04d}' / 'tmp_graphs'
    scored_rows: list[dict[str, Any]] = []
    for example in examples:
        cached_example = get_or_create_cached_example(
            tokenizer=tokenizer,
            prompt_cache=prompt_cache,
            skill_id=example.skill_id,
            prompt=example.prompt,
            reference_graph=example.reference_graph,
            cutoff_len=cutoff_len,
        )
        generated_text, _, _ = generate_text_from_cache(
            tokenizer=tokenizer,
            model=policy_model,
            cached_example=cached_example,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.95,
        )
        scored = score_graph_text(example.skill_id, generated_text, reward_config, temp_dir)
        scored['prompt'] = example.prompt
        scored['reference_graph'] = example.reference_graph
        scored_rows.append(scored)
    summary = summarize_scored_rows(scored_rows)
    eval_dir = output_dir / f'eval_step_{step_index:04d}'
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (eval_dir / 'scored_rows.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in scored_rows), encoding='utf-8')
    return summary


def write_checkpoint_dir(policy_model: Any, tokenizer: Any, checkpoint_dir: Path, metrics: dict[str, Any]) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    return checkpoint_dir


def save_checkpoint(
    policy_model: Any,
    tokenizer: Any,
    output_dir: Path,
    step_index: int | None,
    metrics: dict[str, Any],
    *,
    checkpoint_name: str = '',
    save_latest: bool = False,
) -> dict[str, Path]:
    resolved_step = 0 if step_index is None else int(step_index)
    checkpoint_dir = output_dir / checkpoint_name if checkpoint_name else output_dir / f'checkpoint_step_{resolved_step:04d}'
    write_checkpoint_dir(policy_model, tokenizer, checkpoint_dir, metrics)
    paths = {'checkpoint_dir': checkpoint_dir}
    if save_latest:
        latest_dir = output_dir / 'checkpoint_latest'
        write_checkpoint_dir(policy_model, tokenizer, latest_dir, metrics)
        (latest_dir / 'source_checkpoint.txt').write_text(str(checkpoint_dir), encoding='utf-8')
        (latest_dir / 'step.txt').write_text(f'{resolved_step}\n', encoding='utf-8')
        paths['latest_checkpoint_dir'] = latest_dir
    (output_dir / 'latest_checkpoint_path.txt').write_text(str(checkpoint_dir), encoding='utf-8')
    return paths


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
    if render_reinforce_log_file is None:
        raise RuntimeError(LIVE_PLOT_IMPORT_ERROR or 'plot_reinforce_metrics is unavailable')
    if not force and step_index % max(1, plot_every) != 0:
        return
    render_reinforce_log_file(
        log_file=log_path,
        output_dir=plot_dir,
        refresh_dashboard=refresh_dashboard,
        dashboard_refresh_seconds=refresh_seconds,
        quiet=True,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_config = opt.load_reward_config(args.reward_config)

    train_examples = load_prompt_examples(Path(args.train_file).resolve(), limit=0)
    val_examples = load_prompt_examples(Path(args.val_file).resolve(), limit=0)
    if not train_examples:
        raise SystemExit('Training dataset is empty.')
    if not val_examples:
        raise SystemExit('Validation dataset is empty.')
    if args.eval_limit > 0:
        val_examples = val_examples[:args.eval_limit]

    target_modules = args.target_modules or []
    tokenizer, policy_model = load_policy_model(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.start_adapter_path,
        torch_dtype=args.torch_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        trainable=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules or None,
    )
    train_prompt_cache = build_example_encoding_cache(tokenizer, train_examples, args.cutoff_len)
    val_prompt_cache = build_example_encoding_cache(tokenizer, val_examples, args.cutoff_len)
    train_prompt_stats = summarize_prompt_lengths(tokenizer, train_examples, args.cutoff_len, prompt_cache=train_prompt_cache)
    val_prompt_stats = summarize_prompt_lengths(tokenizer, val_examples, args.cutoff_len, prompt_cache=val_prompt_cache)
    print(json.dumps({'train_prompt_stats': train_prompt_stats, 'val_prompt_stats': val_prompt_stats}, ensure_ascii=False), flush=True)

    reference_adapter = args.reference_adapter_path or ''
    reference_model = None
    if reference_adapter and args.kl_coef > 0:
        _, reference_model = load_policy_model(
            model_name_or_path=args.model_name_or_path,
            adapter_path=reference_adapter,
            torch_dtype=args.torch_dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            trainable=False,
        )
        reference_model.eval()

    trainable_params = [param for param in policy_model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    reward_baseline = deque(maxlen=max(1, args.baseline_window))
    success_buffer: deque[dict[str, Any]] = deque()
    success_buffer_max_size = max(0, int(args.success_buffer_size))
    success_buffer_keys: set[tuple[str, str]] = set()
    success_buffer_encoding_cache: dict[tuple[str, str, str], tuple[torch.Tensor, int]] = {}
    seeded_success_buffer_count = 0
    if args.seed_success_buffer_with_references and success_buffer_max_size > 0:
        seeded_success_buffer_count = seed_success_buffer_from_references(
            examples=train_examples,
            success_buffer=success_buffer,
            success_buffer_keys=success_buffer_keys,
            max_size=success_buffer_max_size,
            limit=int(args.reference_success_buffer_limit),
        )
    log_path = output_dir / 'reinforce_train_log.jsonl'
    log_path.write_text('', encoding='utf-8')
    plot_dir = Path(args.plot_dir).expanduser().resolve() if args.plot_dir else output_dir / 'seaborn_plots'
    started_at = time.time()
    plot_warning_emitted = False

    if args.save_metric_plots or args.live_plot:
        plot_payload = {'plot_dir': str(plot_dir)}
        if args.live_plot:
            plot_payload['live_dashboard'] = str((plot_dir / 'live_dashboard.html').resolve())
        print(json.dumps(plot_payload, ensure_ascii=False), flush=True)
    if seeded_success_buffer_count:
        print(json.dumps({'seeded_success_buffer_count': seeded_success_buffer_count, 'success_buffer_size': len(success_buffer)}, ensure_ascii=False), flush=True)

    if not args.skip_initial_eval:
        initial_validation = evaluate_policy(
            policy_model=policy_model,
            tokenizer=tokenizer,
            examples=val_examples,
            prompt_cache=val_prompt_cache,
            reward_config=reward_config,
            output_dir=output_dir,
            step_index=0,
            max_new_tokens=args.max_new_tokens,
            cutoff_len=args.cutoff_len,
        )
        initial_metrics = {
            'step': 0,
            'avg_reward': None,
            'max_reward': None,
            'min_reward': None,
            'baseline_reward': None,
            'elapsed_seconds': round(time.time() - started_at, 2),
            'success_buffer_size': len(success_buffer),
            'seeded_success_buffer_count': seeded_success_buffer_count,
            'validation': initial_validation,
            'note': 'pre_update_evaluation',
        }
        if args.save_initial_checkpoint:
            checkpoint_paths = save_checkpoint(
                policy_model,
                tokenizer,
                output_dir,
                0,
                initial_metrics,
                save_latest=args.save_latest_checkpoint,
            )
            initial_metrics['checkpoint_dir'] = str(checkpoint_paths['checkpoint_dir'])
            if 'latest_checkpoint_dir' in checkpoint_paths:
                initial_metrics['checkpoint_latest_dir'] = str(checkpoint_paths['latest_checkpoint_dir'])
        with log_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(initial_metrics, ensure_ascii=False) + '\n')
        if args.save_metric_plots or args.live_plot:
            try:
                render_metric_plots(
                    log_path,
                    plot_dir,
                    0,
                    args.plot_every,
                    args.plot_refresh_seconds,
                    refresh_dashboard=args.live_plot,
                    force=True,
                )
            except Exception as exc:
                if not plot_warning_emitted:
                    print(json.dumps({'warning': 'metric_plot_render_failed', 'step': 0, 'error': str(exc)}, ensure_ascii=False), flush=True)
                    plot_warning_emitted = True
        print(json.dumps(initial_metrics, ensure_ascii=False), flush=True)

    for step_index in range(1, args.max_steps + 1):
        warmup_phase = step_index <= max(0, int(args.pg_warmup_steps))
        rollout_round_count = 0
        trajectories: list[Trajectory] = []
        sampled_skill_ids: list[str] = []
        baseline = sum(reward_baseline) / len(reward_baseline) if reward_baseline else 0.0

        while rollout_round_count < max(1, int(args.max_rollout_rounds_per_step)):
            rollout_round_count += 1
            batch_examples = random.sample(train_examples, min(args.rollout_batch_size, len(train_examples)))
            round_trajectories: list[Trajectory] = []
            rollout_cached_examples: list[CachedExampleEncoding] = []
            rollout_source_examples: list[Any] = []
            for example in batch_examples:
                sampled_skill_ids.append(example.skill_id)
                cached_example = get_or_create_cached_example(
                    tokenizer=tokenizer,
                    prompt_cache=train_prompt_cache,
                    skill_id=example.skill_id,
                    prompt=example.prompt,
                    reference_graph=example.reference_graph,
                    cutoff_len=args.cutoff_len,
                )
                for _sample_index in range(max(1, int(args.num_samples_per_prompt))):
                    rollout_cached_examples.append(cached_example)
                    rollout_source_examples.append(example)

            rollout_generate_batch_size = int(args.rollout_generate_batch_size) if int(args.rollout_generate_batch_size) > 0 else max(1, int(args.rollout_batch_size))
            rollout_generations = generate_texts_from_cache_batch(
                tokenizer=tokenizer,
                model=policy_model,
                cached_examples=rollout_cached_examples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=rollout_generate_batch_size,
            )

            for example, (generated_text, full_input_ids, prompt_length) in zip(rollout_source_examples, rollout_generations):
                score_payload = score_graph_text(example.skill_id, generated_text, reward_config, output_dir / 'rollout_graphs' / f'step_{step_index:04d}')
                raw_reward = float(score_payload['reward'])
                if args.graph_prefix_bonus and has_graph_td_prefix(generated_text):
                    raw_reward += args.graph_prefix_bonus
                structural_scores = compute_structural_scores(score_payload, generated_text)
                score_payload['structural_scores'] = structural_scores
                score_payload['base_reward'] = raw_reward
                round_trajectories.append(
                    Trajectory(
                        skill_id=example.skill_id,
                        prompt=example.prompt,
                        reference_graph=example.reference_graph,
                        input_ids=full_input_ids,
                        prompt_length=prompt_length,
                        generated_text=generated_text,
                        reward=raw_reward,
                        shaped_reward=raw_reward,
                        score_payload=score_payload,
                    )
                )

            if round_trajectories and reference_model is not None and args.kl_coef > 0:
                with torch.no_grad():
                    ref_mean_logprobs, _, ref_has_generated = compute_batch_generated_token_stats(
                        reference_model,
                        [trajectory.input_ids for trajectory in round_trajectories],
                        [trajectory.prompt_length for trajectory in round_trajectories],
                        pad_token_id=int(tokenizer.pad_token_id),
                        device=args.device,
                        forward_batch_size=args.train_forward_batch_size,
                        compute_entropies=False,
                    )
                for trajectory, ref_mean_logprob, has_generated_tokens in zip(round_trajectories, ref_mean_logprobs, ref_has_generated):
                    ref_mean = float(ref_mean_logprob.item()) if bool(has_generated_tokens.item()) else 0.0
                    trajectory.shaped_reward = float(trajectory.reward) - args.kl_coef * ref_mean

            trajectories.extend(round_trajectories)

            current_min_pg_reward = args.pg_warmup_min_reward if warmup_phase else args.min_pg_reward
            current_pg_update_count = 0
            current_raw_advantages = compute_raw_advantages(
                trajectories,
                baseline,
                use_group_relative_advantages=args.use_group_relative_advantages,
            )
            for trajectory, raw_advantage in zip(trajectories, current_raw_advantages):
                pg_eligible = is_policy_gradient_eligible(
                    trajectory,
                    require_full_reach=args.require_full_reach_for_update and not warmup_phase,
                    min_pg_reward=current_min_pg_reward,
                    warmup=warmup_phase,
                    warmup_require_finish_name=(args.warmup_require_finish_name or should_require_finish_name_in_warmup(step_index, args)),
                )
                use_pg = (not args.require_graph_prefix_for_update) or pg_eligible
                if use_pg and args.positive_advantage_only and raw_advantage <= 0.0:
                    use_pg = False
                if use_pg:
                    current_pg_update_count += 1
            if current_pg_update_count >= max(1, int(args.min_pg_updates_per_step)):
                break

        optimizer.zero_grad(set_to_none=True)
        advantages: list[float] = []
        scaled_advantages: list[float] = []
        graph_prefix_pass_count = 0
        graph_prefix_skip_count = 0
        finish_name_pass_count = 0
        edge_marker_pass_count = 0
        full_start_reach_pass_count = 0
        full_finish_reach_pass_count = 0
        pg_eligible_count = 0
        pg_update_count = 0
        anchor_only_count = 0
        structure_aux_only_count = 0
        z3_aux_update_count = 0
        mermaid_aux_update_count = 0
        negative_advantage_skip_count = 0
        no_grad_skip_count = 0
        skipped_optimizer_step = False
        trajectory_flags: list[dict[str, Any]] = []
        raw_pg_advantages: list[float] = []
        current_min_pg_reward = args.pg_warmup_min_reward if warmup_phase else args.min_pg_reward
        raw_advantages = compute_raw_advantages(
            trajectories,
            baseline,
            use_group_relative_advantages=args.use_group_relative_advantages,
        )
        for trajectory, raw_advantage in zip(trajectories, raw_advantages):
            graph_text = str((trajectory.score_payload or {}).get('graph_text') or trajectory.generated_text or '')
            report = (trajectory.score_payload or {}).get('report') or {}
            checks = report.get('checks') or {}
            prefix_ok = has_graph_td_prefix(graph_text)
            edge_ok = has_edge_marker(graph_text)
            finish_name_ok = has_finish_name_marker(graph_text)
            full_start_reach_ok = bool(checks.get('all_nodes_reachable_from_start'))
            full_finish_reach_ok = bool(checks.get('all_nodes_can_reach_finish'))
            if prefix_ok:
                graph_prefix_pass_count += 1
            else:
                graph_prefix_skip_count += 1
            if edge_ok:
                edge_marker_pass_count += 1
            if finish_name_ok:
                finish_name_pass_count += 1
            if full_start_reach_ok:
                full_start_reach_pass_count += 1
            if full_finish_reach_ok:
                full_finish_reach_pass_count += 1
            pg_eligible = is_policy_gradient_eligible(
                trajectory,
                require_full_reach=args.require_full_reach_for_update and not warmup_phase,
                min_pg_reward=current_min_pg_reward,
                warmup=warmup_phase,
                warmup_require_finish_name=(args.warmup_require_finish_name or should_require_finish_name_in_warmup(step_index, args)),
            )
            if pg_eligible:
                pg_eligible_count += 1
            use_pg = (not args.require_graph_prefix_for_update) or pg_eligible
            if use_pg and args.positive_advantage_only and raw_advantage <= 0.0:
                use_pg = False
                if pg_eligible:
                    negative_advantage_skip_count += 1
            if use_pg:
                pg_update_count += 1
                raw_pg_advantages.append(raw_advantage)
            structural_scores = (trajectory.score_payload or {}).get('structural_scores') or {}
            trajectory_flags.append(
                {
                    'use_pg': use_pg,
                    'raw_advantage': raw_advantage,
                    'has_generated_tokens': trajectory.input_ids.numel() > trajectory.prompt_length,
                    'z3_reachability_score': float(structural_scores.get('z3_reachability_score', 0.0)),
                    'mermaid_completeness_score': float(structural_scores.get('mermaid_completeness_score', 0.0)),
                }
            )

        success_buffer_add_count = 0
        success_buffer_seen: set[tuple[str, str]] = set()
        if success_buffer_max_size > 0:
            for trajectory in trajectories:
                target_text = str((trajectory.score_payload or {}).get('graph_text') or trajectory.generated_text or '').strip()
                if not target_text:
                    continue
                if not is_success_buffer_candidate(
                    trajectory,
                    min_reward=args.success_buffer_min_reward,
                    require_full_reach=args.success_buffer_require_full_reach,
                ):
                    continue
                buffer_key = (trajectory.skill_id, target_text)
                if buffer_key in success_buffer_seen or buffer_key in success_buffer_keys:
                    continue
                if len(success_buffer) >= success_buffer_max_size:
                    evicted = success_buffer.popleft()
                    evicted_key = (str(evicted.get('skill_id', '')), str(evicted.get('target_text', '')))
                    success_buffer_keys.discard(evicted_key)
                    success_buffer_encoding_cache.pop((evicted_key[0], str(evicted.get('prompt', '')), evicted_key[1]), None)
                success_buffer.append(
                    {
                        'skill_id': trajectory.skill_id,
                        'prompt': trajectory.prompt,
                        'target_text': target_text,
                        'reward': float(trajectory.reward),
                    }
                )
                success_buffer_seen.add(buffer_key)
                success_buffer_keys.add(buffer_key)
                success_buffer_add_count += 1

        advantage_scale = max(1.0, float(args.min_advantage_scale))
        if args.normalize_advantages and len(raw_pg_advantages) > 1:
            pg_advantages_tensor = torch.tensor(raw_pg_advantages, device=args.device, dtype=torch.float32)
            advantage_std = float(pg_advantages_tensor.std(unbiased=False).item())
            if advantage_std > 1e-6:
                advantage_scale = max(float(args.min_advantage_scale), advantage_std)

        forward_indices = [
            index
            for index, flags in enumerate(trajectory_flags)
            if flags['has_generated_tokens'] and (
                flags['use_pg']
                or args.sft_anchor_coef > 0
                or args.z3_reachability_loss_coef > 0
                or args.mermaid_completeness_loss_coef > 0
            )
        ]
        forward_indices.sort(key=lambda index: int(trajectories[index].input_ids.shape[-1]), reverse=True)
        effective = len(forward_indices)
        has_success_buffer_supervision = (
            args.success_buffer_coef > 0
            and len(success_buffer) > 0
            and int(args.success_buffer_batch_size) > 0
        )
        if effective == 0 and not has_success_buffer_supervision:
            continue

        buffer_batch_count = 0
        buffer_only_update = False
        anchor_only_step = False
        forward_backward_count = 0
        if forward_indices:
            anchor_only_step = pg_update_count == 0
            train_forward_chunk_size = int(args.train_forward_batch_size) if int(args.train_forward_batch_size) > 0 else len(forward_indices)
            effective_denom = max(1, effective)
            for chunk_start in range(0, len(forward_indices), train_forward_chunk_size):
                chunk_indices = forward_indices[chunk_start : chunk_start + train_forward_chunk_size]
                chunk_trajectories = [trajectories[index] for index in chunk_indices]
                chunk_mean_logprobs, chunk_entropies, chunk_has_generated = compute_batch_generated_token_stats(
                    policy_model,
                    [trajectory.input_ids for trajectory in chunk_trajectories],
                    [trajectory.prompt_length for trajectory in chunk_trajectories],
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=args.device,
                    forward_batch_size=0,
                    compute_entropies=args.entropy_coef > 0,
                )

                if args.sft_anchor_coef > 0:
                    anchor_sequences: list[torch.Tensor] = []
                    anchor_prompt_lengths: list[int] = []
                    for trajectory in chunk_trajectories:
                        cached_example = get_or_create_cached_example(
                            tokenizer=tokenizer,
                            prompt_cache=train_prompt_cache,
                            skill_id=trajectory.skill_id,
                            prompt=trajectory.prompt,
                            reference_graph=trajectory.reference_graph,
                            cutoff_len=args.cutoff_len,
                        )
                        if cached_example.reference_full_ids is None:
                            full_ids, target_prompt_length = build_full_sequence_encoding(
                                tokenizer,
                                cached_example,
                                target_text=trajectory.reference_graph,
                                cutoff_len=args.cutoff_len,
                            )
                        else:
                            full_ids = cached_example.reference_full_ids
                            target_prompt_length = cached_example.reference_prompt_length
                        anchor_sequences.append(full_ids)
                        anchor_prompt_lengths.append(target_prompt_length)
                    chunk_anchor_mean_logprobs, _, _ = compute_batch_generated_token_stats(
                        policy_model,
                        anchor_sequences,
                        anchor_prompt_lengths,
                        pad_token_id=int(tokenizer.pad_token_id),
                        device=args.device,
                        forward_batch_size=0,
                        compute_entropies=False,
                    )
                    chunk_anchor_losses = -chunk_anchor_mean_logprobs
                else:
                    chunk_anchor_losses = torch.zeros(len(chunk_trajectories), device=args.device, dtype=chunk_mean_logprobs.dtype)

                chunk_loss_terms: list[torch.Tensor] = []
                for local_index, global_index in enumerate(chunk_indices):
                    if not bool(chunk_has_generated[local_index].item()):
                        no_grad_skip_count += 1
                        continue
                    flags = trajectory_flags[global_index]
                    sequence_loss = torch.zeros((), device=args.device, dtype=chunk_mean_logprobs.dtype)
                    z3_aux_weight = float(args.z3_reachability_loss_coef) * float(flags['z3_reachability_score'])
                    mermaid_aux_weight = float(args.mermaid_completeness_loss_coef) * float(flags['mermaid_completeness_score'])
                    has_structure_aux = z3_aux_weight > 0.0 or mermaid_aux_weight > 0.0
                    if flags['use_pg']:
                        scaled_advantage = float(flags['raw_advantage']) / advantage_scale
                        if args.advantage_clip > 0:
                            scaled_advantage = max(-args.advantage_clip, min(args.advantage_clip, scaled_advantage))
                        advantages.append(float(flags['raw_advantage']))
                        scaled_advantages.append(float(scaled_advantage))
                        advantage_tensor = torch.tensor(scaled_advantage, device=args.device, dtype=chunk_mean_logprobs.dtype)
                        sequence_loss = sequence_loss - (advantage_tensor * chunk_mean_logprobs[local_index])
                        if args.entropy_coef > 0:
                            sequence_loss = sequence_loss - args.entropy_coef * chunk_entropies[local_index]
                    else:
                        if has_structure_aux:
                            structure_aux_only_count += 1
                        else:
                            anchor_only_count += 1
                        if args.sft_anchor_coef <= 0 and not has_structure_aux:
                            continue
                    if z3_aux_weight > 0.0:
                        z3_aux_update_count += 1
                        z3_aux_tensor = torch.tensor(z3_aux_weight, device=args.device, dtype=chunk_mean_logprobs.dtype)
                        sequence_loss = sequence_loss - (z3_aux_tensor * chunk_mean_logprobs[local_index])
                    if mermaid_aux_weight > 0.0:
                        mermaid_aux_update_count += 1
                        mermaid_aux_tensor = torch.tensor(mermaid_aux_weight, device=args.device, dtype=chunk_mean_logprobs.dtype)
                        sequence_loss = sequence_loss - (mermaid_aux_tensor * chunk_mean_logprobs[local_index])
                    if args.sft_anchor_coef > 0:
                        sequence_loss = sequence_loss + args.sft_anchor_coef * chunk_anchor_losses[local_index]
                    chunk_loss_terms.append(sequence_loss)

                if chunk_loss_terms:
                    chunk_total_loss = torch.stack(chunk_loss_terms).sum()
                    if chunk_total_loss.requires_grad:
                        (chunk_total_loss / effective_denom).backward()
                        forward_backward_count += 1
                    else:
                        no_grad_skip_count += len(chunk_loss_terms)
                else:
                    no_grad_skip_count += len(chunk_indices)
        else:
            buffer_only_update = True

        if args.success_buffer_coef > 0 and len(success_buffer) > 0:
            sampled_buffer = random.sample(list(success_buffer), min(int(args.success_buffer_batch_size), len(success_buffer)))
            if sampled_buffer:
                buffer_batch_count = len(sampled_buffer)
                buffer_sequences: list[torch.Tensor] = []
                buffer_prompt_lengths: list[int] = []
                for item in sampled_buffer:
                    buffer_cache_key = (str(item['skill_id']), str(item['prompt']), str(item['target_text']))
                    cached_buffer_encoding = success_buffer_encoding_cache.get(buffer_cache_key)
                    if cached_buffer_encoding is None:
                        cached_example = get_or_create_cached_example(
                            tokenizer=tokenizer,
                            prompt_cache=train_prompt_cache,
                            skill_id=str(item['skill_id']),
                            prompt=str(item['prompt']),
                            cutoff_len=args.cutoff_len,
                        )
                        cached_buffer_encoding = build_full_sequence_encoding(
                            tokenizer,
                            cached_example,
                            target_text=str(item['target_text']),
                            cutoff_len=args.cutoff_len,
                        )
                        success_buffer_encoding_cache[buffer_cache_key] = cached_buffer_encoding
                    buffer_sequences.append(cached_buffer_encoding[0])
                    buffer_prompt_lengths.append(int(cached_buffer_encoding[1]))
                buffer_mean_logprobs, _, _ = compute_batch_generated_token_stats(
                    policy_model,
                    buffer_sequences,
                    buffer_prompt_lengths,
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=args.device,
                    forward_batch_size=args.train_forward_batch_size,
                    compute_entropies=False,
                )
                buffer_loss = args.success_buffer_coef * (-buffer_mean_logprobs.mean())
                if buffer_loss.requires_grad:
                    buffer_loss.backward()

        if forward_backward_count > 0 or buffer_batch_count > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
        else:
            skipped_optimizer_step = True

        rewards = [trajectory.reward for trajectory in trajectories]
        shaped_rewards = [trajectory.shaped_reward for trajectory in trajectories]
        z3_scores = [float(flags['z3_reachability_score']) for flags in trajectory_flags]
        mermaid_scores = [float(flags['mermaid_completeness_score']) for flags in trajectory_flags]
        if args.baseline_on_pg_only:
            baseline_trajectories = [
                trajectory
                for trajectory, flags in zip(trajectories, trajectory_flags)
                if flags['use_pg']
            ]
        else:
            baseline_trajectories = trajectories
        for trajectory in baseline_trajectories:
            reward_baseline.append(float(trajectory.shaped_reward))
        metrics = {
            'step': step_index,
            'avg_reward': round(sum(rewards) / len(rewards), 4),
            'avg_shaped_reward': round(sum(shaped_rewards) / len(shaped_rewards), 4),
            'avg_z3_reachability_score': round(sum(z3_scores) / len(z3_scores), 4) if z3_scores else 0.0,
            'avg_mermaid_completeness_score': round(sum(mermaid_scores) / len(mermaid_scores), 4) if mermaid_scores else 0.0,
            'avg_z3_loss_weight': round(sum(args.z3_reachability_loss_coef * score for score in z3_scores) / len(z3_scores), 4) if z3_scores else 0.0,
            'avg_mermaid_loss_weight': round(sum(args.mermaid_completeness_loss_coef * score for score in mermaid_scores) / len(mermaid_scores), 4) if mermaid_scores else 0.0,
            'max_reward': round(max(rewards), 4),
            'min_reward': round(min(rewards), 4),
            'baseline_reward': round(baseline, 4),
            'avg_advantage': round(sum(advantages) / len(advantages), 4) if advantages else 0.0,
            'avg_scaled_advantage': round(sum(scaled_advantages) / len(scaled_advantages), 4) if scaled_advantages else 0.0,
            'effective_trajectories': effective,
            'sampled_trajectories': len(trajectories),
            'rollout_round_count': rollout_round_count,
            'num_samples_per_prompt': int(args.num_samples_per_prompt),
            'rollout_generate_batch_size': int(args.rollout_generate_batch_size) if int(args.rollout_generate_batch_size) > 0 else max(1, int(args.rollout_batch_size)),
            'train_forward_batch_size': int(args.train_forward_batch_size),
            'target_min_pg_updates_per_step': int(args.min_pg_updates_per_step),
            'skipped_optimizer_step': skipped_optimizer_step,
            'buffer_only_update': buffer_only_update,
            'anchor_only_step': anchor_only_step,
            'success_buffer_size': len(success_buffer),
            'seeded_success_buffer_count': seeded_success_buffer_count,
            'success_buffer_add_count': success_buffer_add_count,
            'success_buffer_batch_count': buffer_batch_count,
            'graph_prefix_pass_count': graph_prefix_pass_count,
            'graph_prefix_skip_count': graph_prefix_skip_count,
            'graph_prefix_pass_rate': round(graph_prefix_pass_count / len(trajectories), 4) if trajectories else 0.0,
            'edge_marker_pass_count': edge_marker_pass_count,
            'edge_marker_pass_rate': round(edge_marker_pass_count / len(trajectories), 4) if trajectories else 0.0,
            'finish_name_pass_count': finish_name_pass_count,
            'finish_name_pass_rate': round(finish_name_pass_count / len(trajectories), 4) if trajectories else 0.0,
            'full_start_reach_pass_count': full_start_reach_pass_count,
            'full_start_reach_pass_rate': round(full_start_reach_pass_count / len(trajectories), 4) if trajectories else 0.0,
            'full_finish_reach_pass_count': full_finish_reach_pass_count,
            'full_finish_reach_pass_rate': round(full_finish_reach_pass_count / len(trajectories), 4) if trajectories else 0.0,
            'pg_eligible_count': pg_eligible_count,
            'pg_eligible_rate': round(pg_eligible_count / len(trajectories), 4) if trajectories else 0.0,
            'pg_update_count': pg_update_count,
            'pg_update_rate': round(pg_update_count / len(trajectories), 4) if trajectories else 0.0,
            'warmup_phase': warmup_phase,
            'anchor_only_count': anchor_only_count,
            'structure_aux_only_count': structure_aux_only_count,
            'z3_aux_update_count': z3_aux_update_count,
            'mermaid_aux_update_count': mermaid_aux_update_count,
            'negative_advantage_skip_count': negative_advantage_skip_count,
            'no_grad_skip_count': no_grad_skip_count,
            'advantage_scale': round(advantage_scale, 4),
            'elapsed_seconds': round(time.time() - started_at, 2),
        }
        if step_index % max(1, args.eval_every) == 0:
            metrics['validation'] = evaluate_policy(
                policy_model=policy_model,
                tokenizer=tokenizer,
                examples=val_examples,
                prompt_cache=val_prompt_cache,
                reward_config=reward_config,
                output_dir=output_dir,
                step_index=step_index,
                max_new_tokens=args.max_new_tokens,
                cutoff_len=args.cutoff_len,
            )
        if step_index % max(1, args.save_every) == 0:
            checkpoint_paths = save_checkpoint(
                policy_model,
                tokenizer,
                output_dir,
                step_index,
                metrics,
                save_latest=args.save_latest_checkpoint,
            )
            metrics['checkpoint_dir'] = str(checkpoint_paths['checkpoint_dir'])
            if 'latest_checkpoint_dir' in checkpoint_paths:
                metrics['checkpoint_latest_dir'] = str(checkpoint_paths['latest_checkpoint_dir'])
        with log_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        if args.save_metric_plots or args.live_plot:
            try:
                render_metric_plots(
                    log_path,
                    plot_dir,
                    step_index,
                    args.plot_every,
                    args.plot_refresh_seconds,
                    refresh_dashboard=args.live_plot,
                    force=('validation' in metrics or 'checkpoint_dir' in metrics or step_index == args.max_steps),
                )
            except Exception as exc:
                if not plot_warning_emitted:
                    print(json.dumps({'warning': 'metric_plot_render_failed', 'step': step_index, 'error': str(exc)}, ensure_ascii=False), flush=True)
                    plot_warning_emitted = True
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    final_checkpoint_dir = ''
    if args.save_final_checkpoint:
        final_checkpoint_metrics = {
            'step': args.max_steps,
            'elapsed_seconds': round(time.time() - started_at, 2),
            'success_buffer_size': len(success_buffer),
            'seeded_success_buffer_count': seeded_success_buffer_count,
            'note': 'final_checkpoint',
        }
        final_checkpoint_paths = save_checkpoint(
            policy_model,
            tokenizer,
            output_dir,
            args.max_steps,
            final_checkpoint_metrics,
            checkpoint_name='checkpoint_final',
            save_latest=args.save_latest_checkpoint,
        )
        final_checkpoint_dir = str(final_checkpoint_paths['checkpoint_dir'])
    if args.save_metric_plots or args.live_plot:
        try:
            render_metric_plots(
                log_path,
                plot_dir,
                args.max_steps,
                args.plot_every,
                args.plot_refresh_seconds,
                refresh_dashboard=args.live_plot,
                force=True,
            )
        except Exception as exc:
            if not plot_warning_emitted:
                print(json.dumps({'warning': 'metric_plot_render_failed', 'step': args.max_steps, 'error': str(exc)}, ensure_ascii=False), flush=True)
                plot_warning_emitted = True

    final_summary = {
        'output_dir': str(output_dir),
        'steps': args.max_steps,
        'train_file': args.train_file,
        'val_file': args.val_file,
        'num_samples_per_prompt': args.num_samples_per_prompt,
        'rollout_generate_batch_size': args.rollout_generate_batch_size,
        'train_forward_batch_size': args.train_forward_batch_size,
        'use_group_relative_advantages': args.use_group_relative_advantages,
        'start_adapter_path': args.start_adapter_path,
        'reference_adapter_path': reference_adapter,
        'live_plot_enabled': args.live_plot,
        'save_metric_plots': args.save_metric_plots,
        'plot_dir': str(plot_dir),
        'save_initial_checkpoint': args.save_initial_checkpoint,
        'save_latest_checkpoint': args.save_latest_checkpoint,
        'save_final_checkpoint': args.save_final_checkpoint,
        'final_checkpoint_dir': final_checkpoint_dir,
        'require_graph_prefix_for_update': args.require_graph_prefix_for_update,
        'positive_advantage_only': args.positive_advantage_only,
        'normalize_advantages': args.normalize_advantages,
        'advantage_clip': args.advantage_clip,
        'min_pg_reward': args.min_pg_reward,
        'pg_warmup_steps': args.pg_warmup_steps,
        'pg_warmup_tighten_step': args.pg_warmup_tighten_step,
        'pg_warmup_min_reward': args.pg_warmup_min_reward,
        'min_pg_updates_per_step': args.min_pg_updates_per_step,
        'max_rollout_rounds_per_step': args.max_rollout_rounds_per_step,
        'warmup_require_finish_name': args.warmup_require_finish_name,
        'min_advantage_scale': args.min_advantage_scale,
        'success_buffer_size': args.success_buffer_size,
        'success_buffer_min_reward': args.success_buffer_min_reward,
        'success_buffer_batch_size': args.success_buffer_batch_size,
        'success_buffer_coef': args.success_buffer_coef,
        'success_buffer_require_full_reach': args.success_buffer_require_full_reach,
        'seed_success_buffer_with_references': args.seed_success_buffer_with_references,
        'reference_success_buffer_limit': args.reference_success_buffer_limit,
        'seeded_success_buffer_count': seeded_success_buffer_count,
        'require_full_reach_for_update': args.require_full_reach_for_update,
        'baseline_on_pg_only': args.baseline_on_pg_only,
        'graph_prefix_bonus': args.graph_prefix_bonus,
        'sft_anchor_coef': args.sft_anchor_coef,
        'z3_reachability_loss_coef': args.z3_reachability_loss_coef,
        'mermaid_completeness_loss_coef': args.mermaid_completeness_loss_coef,
    }
    (output_dir / 'final_summary.json').write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
