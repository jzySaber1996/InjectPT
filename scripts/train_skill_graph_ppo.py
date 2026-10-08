#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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


@dataclass
class RolloutSample:
    skill_id: str
    prompt: str
    reference_graph: str
    full_input_ids: torch.Tensor
    prompt_length: int
    generated_text: str
    reward: float
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    old_value: torch.Tensor
    shaped_reward: float
    score_payload: dict[str, Any]


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden_states).squeeze(-1)


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
    output_dir = repo_root / 'artifacts' / 'ppo' / 'skill_graph_qwen25'
    parser = argparse.ArgumentParser(description='Train a LoRA policy with PPO-style reward updates for Mermaid skill graph generation.')
    parser.add_argument('--config', default='', help='Optional simple YAML config file.')
    parser.add_argument('--train-file', default=str(dataset_dir / 'skill_graph_sft_train.jsonl'), help='Training JSONL file.')
    parser.add_argument('--val-file', default=str(dataset_dir / 'skill_graph_sft_val.jsonl'), help='Validation JSONL file.')
    parser.add_argument('--model-name-or-path', default='/root/JZY/MCP_Threat_Modeling/Qwen2.5-7B', help='Base model path.')
    parser.add_argument('--start-adapter-path', default='', help='Optional starting adapter, typically the SFT adapter.')
    parser.add_argument('--reference-adapter-path', default='', help='Frozen reference adapter used for KL regularization. Defaults to start-adapter-path.')
    parser.add_argument('--output-dir', default=str(output_dir), help='Output directory for PPO checkpoints and logs.')
    parser.add_argument('--reward-config', default='', help='Optional JSON reward config.')
    parser.add_argument('--device', default='cuda', help='Training device.')
    parser.add_argument('--torch-dtype', choices=('auto', 'bfloat16', 'float16', 'float32'), default='bfloat16', help='Torch dtype.')
    parser.add_argument('--trust-remote-code', action='store_true', help='Pass trust_remote_code=True when loading the model.')
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--max-steps', type=int, default=10, help='Number of PPO outer steps.')
    parser.add_argument('--rollout-batch-size', type=int, default=2, help='Number of prompts to sample per PPO step.')
    parser.add_argument('--ppo-epochs', type=int, default=2, help='Number of PPO update epochs per rollout batch.')
    parser.add_argument('--mini-batch-size', type=int, default=1, help='Mini-batch size for PPO updates.')
    parser.add_argument('--learning-rate', type=float, default=5.0e-5, help='Learning rate for adapter and value head.')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range.')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='Value loss coefficient.')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy bonus coefficient.')
    parser.add_argument('--kl-coef', type=float, default=0.02, help='KL penalty coefficient against the reference policy.')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping norm.')
    parser.add_argument('--max-new-tokens', type=int, default=900, help='Maximum rollout generation length.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature used for rollouts.')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p sampling used for rollouts.')
    parser.add_argument('--cutoff-len', type=int, default=4096, help='Prompt truncation length.')
    parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank used when training from the base model.')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha used when training from the base model.')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout used when training from the base model.')
    parser.add_argument('--save-every', type=int, default=1, help='Save a checkpoint every N PPO steps.')
    parser.add_argument('--eval-every', type=int, default=1, help='Run validation every N PPO steps.')
    parser.add_argument('--eval-limit', type=int, default=8, help='Number of validation examples per evaluation pass. 0 means full set.')
    parser.add_argument('--target-modules', nargs='*', default=[], help='Optional LoRA target modules override.')
    args = parser.parse_args(argv)
    return apply_yaml_config(args)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_logprobs_values_entropy(
    model: Any,
    value_head: ValueHead,
    input_ids: torch.Tensor,
    prompt_length: int,
    output_hidden_states: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=input_ids.unsqueeze(0), output_hidden_states=output_hidden_states)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids.unsqueeze(0)[:, 1:]
    log_probs_all = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    generated_start = max(prompt_length - 1, 0)
    generated_logprobs = token_logprobs[:, generated_start:].squeeze(0)
    generated_log_probs_all = log_probs_all[:, generated_start:, :]
    generated_probs = generated_log_probs_all.exp()
    entropy = -(generated_probs * generated_log_probs_all).sum(dim=-1).mean()

    if output_hidden_states and outputs.hidden_states is not None:
        hidden_states = outputs.hidden_states[-1][:, -1, :]
        values = value_head(hidden_states).squeeze(0)
    else:
        values = torch.zeros((), device=generated_logprobs.device, dtype=generated_logprobs.dtype)
    return generated_logprobs, values, entropy


def evaluate_policy(
    *,
    policy_model: Any,
    tokenizer: Any,
    examples: list[Any],
    reward_config: dict[str, float],
    output_dir: Path,
    step_index: int,
    max_new_tokens: int,
    cutoff_len: int,
) -> dict[str, Any]:
    temp_dir = output_dir / f'eval_step_{step_index:04d}' / 'tmp_graphs'
    scored_rows: list[dict[str, Any]] = []
    for example in examples:
        generated_text, _, _ = generate_text(
            tokenizer=tokenizer,
            model=policy_model,
            prompt=example.prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.95,
            cutoff_len=cutoff_len,
        )
        scored = score_graph_text(example.skill_id, generated_text, reward_config, temp_dir)
        scored['prompt'] = example.prompt
        scored['reference_graph'] = example.reference_graph
        scored_rows.append(scored)
    summary = summarize_scored_rows(scored_rows)
    eval_dir = output_dir / f'eval_step_{step_index:04d}'
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (eval_dir / 'scored_rows.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in scored_rows),
        encoding='utf-8',
    )
    return summary


def save_checkpoint(
    *,
    policy_model: Any,
    tokenizer: Any,
    value_head: ValueHead,
    output_dir: Path,
    step_index: int,
    metrics: dict[str, Any],
) -> Path:
    checkpoint_dir = output_dir / f'checkpoint_step_{step_index:04d}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(value_head.state_dict(), checkpoint_dir / 'value_head.pt')
    (checkpoint_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    return checkpoint_dir


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
    reference_adapter = args.reference_adapter_path or args.start_adapter_path
    _, reference_model = load_policy_model(
        model_name_or_path=args.model_name_or_path,
        adapter_path=reference_adapter,
        torch_dtype=args.torch_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        trainable=False,
    )
    reference_model.eval()

    hidden_size = int(policy_model.config.hidden_size)
    model_dtype = next(policy_model.parameters()).dtype
    value_head = ValueHead(hidden_size).to(device=args.device, dtype=model_dtype)
    optimizer = torch.optim.AdamW(
        [param for param in policy_model.parameters() if param.requires_grad] + list(value_head.parameters()),
        lr=args.learning_rate,
    )

    train_log_path = output_dir / 'ppo_train_log.jsonl'
    rollout_temp_root = output_dir / 'rollout_graphs'
    train_log_path.write_text('', encoding='utf-8')

    global_step = 0
    started_at = time.time()
    for step_index in range(1, args.max_steps + 1):
        batch_examples = random.sample(train_examples, min(args.rollout_batch_size, len(train_examples)))
        rollouts: list[RolloutSample] = []

        for example in batch_examples:
            generated_text, full_input_ids, prompt_length = generate_text(
                tokenizer=tokenizer,
                model=policy_model,
                prompt=example.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                cutoff_len=args.cutoff_len,
            )
            score_payload = score_graph_text(example.skill_id, generated_text, reward_config, rollout_temp_root / f'step_{step_index:04d}')
            old_logprobs, old_value, _ = compute_logprobs_values_entropy(policy_model, value_head, full_input_ids, prompt_length, output_hidden_states=True)
            with torch.no_grad():
                ref_logprobs, _, _ = compute_logprobs_values_entropy(reference_model, value_head, full_input_ids, prompt_length, output_hidden_states=False)
            kl_penalty = (old_logprobs.detach() - ref_logprobs.detach()).mean().item() if old_logprobs.numel() else 0.0
            shaped_reward = float(score_payload['reward']) - args.kl_coef * kl_penalty
            rollouts.append(
                RolloutSample(
                    skill_id=example.skill_id,
                    prompt=example.prompt,
                    reference_graph=example.reference_graph,
                    full_input_ids=full_input_ids.detach(),
                    prompt_length=prompt_length,
                    generated_text=generated_text,
                    reward=float(score_payload['reward']),
                    old_logprobs=old_logprobs.detach(),
                    ref_logprobs=ref_logprobs.detach(),
                    old_value=old_value.detach(),
                    shaped_reward=shaped_reward,
                    score_payload=score_payload,
                )
            )

        if not rollouts:
            raise SystemExit('No rollouts were generated.')

        values = torch.stack([sample.old_value for sample in rollouts]).to(args.device)
        advantages = torch.tensor([sample.shaped_reward for sample in rollouts], device=args.device, dtype=values.dtype) - values
        if len(rollouts) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

        for _ in range(args.ppo_epochs):
            indices = list(range(len(rollouts)))
            random.shuffle(indices)
            for start in range(0, len(indices), max(1, args.mini_batch_size)):
                batch_indices = indices[start:start + max(1, args.mini_batch_size)]
                optimizer.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=args.device)
                effective = 0
                for rollout_index in batch_indices:
                    sample = rollouts[rollout_index]
                    current_logprobs, current_value, entropy = compute_logprobs_values_entropy(
                        policy_model,
                        value_head,
                        sample.full_input_ids.to(args.device),
                        sample.prompt_length,
                    )
                    old_logprobs = sample.old_logprobs.to(args.device)
                    ref_logprobs = sample.ref_logprobs.to(args.device)
                    advantage = advantages[rollout_index]
                    if current_logprobs.numel() == 0:
                        continue
                    ratio = torch.exp(current_logprobs - old_logprobs)
                    surrogate_1 = ratio * advantage
                    surrogate_2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantage
                    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                    reward_target = torch.tensor(sample.shaped_reward, device=args.device, dtype=current_value.dtype)
                    value_loss = F.mse_loss(current_value, reward_target)
                    kl_loss = (current_logprobs - ref_logprobs).mean()
                    loss = policy_loss + args.value_loss_coef * value_loss + args.kl_coef * kl_loss - args.entropy_coef * entropy
                    total_loss = total_loss + loss
                    effective += 1
                if effective == 0:
                    continue
                total_loss = total_loss / effective
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for param in policy_model.parameters() if param.requires_grad] + list(value_head.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()
                global_step += 1

        rollout_rows = []
        for sample in rollouts:
            row = {
                'step': step_index,
                'skill_id': sample.skill_id,
                'reward': sample.reward,
                'shaped_reward': sample.shaped_reward,
                'status': sample.score_payload['report'].get('status'),
                'issues': [issue.get('code') for issue in sample.score_payload['report'].get('issues', [])],
            }
            rollout_rows.append(row)
            with train_log_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')

        reward_values = [row['reward'] for row in rollout_rows]
        metrics = {
            'step': step_index,
            'global_step': global_step,
            'avg_reward': round(sum(reward_values) / len(reward_values), 4),
            'max_reward': round(max(reward_values), 4),
            'min_reward': round(min(reward_values), 4),
            'elapsed_seconds': round(time.time() - started_at, 2),
        }
        if step_index % max(1, args.eval_every) == 0:
            metrics['validation'] = evaluate_policy(
                policy_model=policy_model,
                tokenizer=tokenizer,
                examples=val_examples,
                reward_config=reward_config,
                output_dir=output_dir,
                step_index=step_index,
                max_new_tokens=args.max_new_tokens,
                cutoff_len=args.cutoff_len,
            )
        if step_index % max(1, args.save_every) == 0:
            checkpoint_dir = save_checkpoint(
                policy_model=policy_model,
                tokenizer=tokenizer,
                value_head=value_head,
                output_dir=output_dir,
                step_index=step_index,
                metrics=metrics,
            )
            metrics['checkpoint_dir'] = str(checkpoint_dir)

        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    final_summary = {
        'output_dir': str(output_dir),
        'steps': args.max_steps,
        'global_step': global_step,
        'train_file': args.train_file,
        'val_file': args.val_file,
        'start_adapter_path': args.start_adapter_path,
        'reference_adapter_path': reference_adapter,
    }
    (output_dir / 'final_summary.json').write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
