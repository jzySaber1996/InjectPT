#!/usr/bin/env python3
"""Reward-ranked Qwen2.5 DPO for threat framework and strategy updates.

Qwen samples schema-constrained JSON deltas. Each delta is executed in the
existing DeepSeek propagation environment and receives the existing primary
reward: OpenClaw runtime reward when enabled, otherwise Z3 reward. Reward-ranked
outputs are then trained with LLaMA-Factory DPO.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_with_deepseek import (  # type: ignore
    DeepSeekClient,
    collect_records,
    load_chain_template,
    parse_json_object,
    read_text,
    render_formal_chain_template,
    render_prompt_template,
    write_json,
    write_text,
)
from framework_utils import (  # type: ignore
    FRAMEWORK_SLOT_METRIC,
    STRATEGY_STEP_METRIC,
    apply_framework_updates,
    apply_strategy_updates,
)
from openclaw_runtime_evaluator import add_openclaw_runtime_args  # type: ignore
from synthesize_framework_with_deepseek import (  # type: ignore
    build_low_scoring_samples,
    build_metric_entry,
    load_seed_framework_and_strategy,
    run_skill_iteration,
    select_train_records,
    split_train_test,
)

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "inner_representation_v2"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "threat_framework_qwen25_reward_rl"
DEFAULT_QWEN_MODEL = "/root/JZY/MCP_Threat_Modeling/Qwen2.5-7B"
DEFAULT_DATASET = "threat_framework_reward_dpo"


@dataclass
class Candidate:
    skill_id: str
    candidate_id: str
    action: str
    response: dict[str, Any]
    status: str
    reward: float
    reward_source: str
    attack_success_rate: float
    strict_attack_success_rate: float
    z3_reward: float
    framework_coverage: float
    strategy_coverage: float
    changed: bool
    error: str = ""


class QwenPolicy:
    """Local rollout sampler; LLaMA-Factory owns DPO parameter updates."""

    def __init__(self, args: argparse.Namespace, adapter_path: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("Qwen rollouts require torch and transformers.") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=args.trust_remote_code
        )
        kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
        dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(args.torch_dtype.lower())
        if args.torch_dtype.lower() not in {"", "auto", "none"} and dtype is None:
            raise ValueError(f"Unsupported --torch-dtype: {args.torch_dtype}")
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if args.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ModuleNotFoundError as exc:
                raise RuntimeError("--load-in-4bit requires bitsandbytes.") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        elif args.device == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:
                raise RuntimeError("Loading Qwen LoRA adapters requires peft.") from exc
            model = PeftModel.from_pretrained(model, adapter_path)
        if args.device != "auto" and not args.load_in_4bit:
            model.to(args.device)
        self.model = model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, system: str, prompt: str, args: argparse.Namespace) -> list[str]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = getattr(encoded, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        do_sample = args.policy_temperature > 0 or args.samples_per_prompt > 1
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "num_return_sequences": args.samples_per_prompt,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=max(0.01, args.policy_temperature), top_p=args.policy_top_p)
        with self.torch.no_grad():
            generated = self.model.generate(**kwargs)
        prompt_length = input_ids.shape[-1]
        return [self.tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip() for row in generated]


def config_defaults(path: str) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Missing config: {config_path}")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("--config requires PyYAML.") from exc
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML object in {config_path}")
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="")
    known, _ = bootstrap.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--framework-file", default=str(REPO_ROOT / "template" / "threat_propagation_framework.json"))
    parser.add_argument("--strategy-file", default=str(REPO_ROOT / "template" / "threat_propagation_parsing_strategy.json"))
    parser.add_argument("--system-prompt-file", default=str(REPO_ROOT / "prompts" / "threat_security_system_prompt.md"))
    parser.add_argument("--prompt-file", default=str(REPO_ROOT / "prompts" / "threat_framework_synthesis_prompt.md"))
    parser.add_argument("--analysis-prompt-file", default=str(REPO_ROOT / "prompts" / "threat_analysis_prompt.md"))
    parser.add_argument("--propagation-prompt-file", default=str(REPO_ROOT / "prompts" / "threat_propagation_prompt.md"))
    parser.add_argument("--chain-template-file", default=str(REPO_ROOT / "template" / "threat_propagation_chain_template.native.json"))
    parser.add_argument("--graph-name", default="graph_wo_check.md")
    parser.add_argument("--skill", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=20260703)
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--policy-temperature", type=float, default=0.35)
    parser.add_argument("--policy-top-p", type=float, default=0.9)
    parser.add_argument("--max-policy-prompt-chars", type=int, default=26000)
    parser.add_argument("--model-name-or-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--start-adapter-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--min-preference-gap", type=float, default=0.02)
    parser.add_argument("--accept-reward-delta", type=float, default=0.001)
    parser.add_argument("--invalid-candidate-reward", type=float, default=-1.0)
    parser.add_argument("--evaluate-test-every", type=int, default=1)
    parser.add_argument("--run-llamafactory-dpo", action="store_true")
    parser.add_argument("--llamafactory-cli", default="llamafactory-cli")
    parser.add_argument("--llamafactory-dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dpo-output-root", default="")
    parser.add_argument("--dpo-cutoff-len", type=int, default=8192)
    parser.add_argument("--dpo-quantization-bit", type=int, default=4)
    parser.add_argument("--dpo-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--dpo-epochs", type=float, default=1.0)
    parser.add_argument("--dpo-batch-size", type=int, default=1)
    parser.add_argument("--dpo-gradient-accumulation", type=int, default=8)
    parser.add_argument("--dpo-lora-rank", type=int, default=16)
    parser.add_argument("--dpo-lora-alpha", type=int, default=32)
    parser.add_argument("--dpo-lora-dropout", type=float, default=0.05)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--deepseek-api-url", default=os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"))
    parser.add_argument("--deepseek-retries", type=int, default=3)
    parser.add_argument("--deepseek-retry-seconds", type=float, default=8.0)
    parser.add_argument("--max-analysis-tokens", type=int, default=3200)
    parser.add_argument("--max-chain-tokens", type=int, default=3200)
    parser.add_argument("--max-framework-chars", type=int, default=12000)
    parser.add_argument("--max-strategy-chars", type=int, default=18000)
    parser.add_argument("--max-chain-template-chars", type=int, default=18000)
    parser.add_argument("--max-skill-chars", type=int, default=32000)
    parser.add_argument("--max-graph-chars", type=int, default=18000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    add_openclaw_runtime_args(parser)
    parser.set_defaults(**config_defaults(known.config))
    return parser.parse_args()


def compact_json(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if limit <= 0 or len(text) <= limit else text[:limit] + "\n[TRUNCATED]"


def build_policy_prompt(template: str, framework: dict[str, Any], strategy: dict[str, Any], coverage: dict[str, Any], samples: list[dict[str, Any]], history: list[dict[str, Any]], limit: int) -> str:
    field_limit = max(1200, limit // 4)
    prompt = render_prompt_template(template, {
        "framework": compact_json(framework, field_limit),
        "strategy": compact_json(strategy, field_limit),
        "coverage_summary": compact_json(coverage, field_limit),
        "low_scoring_samples": compact_json(samples, field_limit),
        "metric_history": compact_json(history[-20:], field_limit),
    })
    return prompt if limit <= 0 or len(prompt) <= limit else prompt[:limit] + "\n[TRUNCATED]"


def parse_action(raw: str) -> tuple[dict[str, Any] | None, str]:
    try:
        action = parse_json_object(raw)
    except Exception as exc:
        return None, f"invalid_json: {exc}"
    if not isinstance(action, dict) or not isinstance(action.get("has_updates"), bool):
        return None, "invalid_action: expected object with boolean has_updates"
    if not isinstance(action.get("framework", {}), dict) or not isinstance(action.get("strategy", {}), dict):
        return None, "invalid_action: framework and strategy must be objects"
    return action, ""


def apply_action(framework: dict[str, Any], strategy: dict[str, Any], action: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
    if not action["has_updates"]:
        return framework, strategy, False
    next_framework = apply_framework_updates(framework, action)
    next_strategy = apply_strategy_updates(strategy, action, next_framework)
    return next_framework, next_strategy, next_framework != framework or next_strategy != strategy


def evaluate(client: DeepSeekClient, records: list[Any], root: Path, framework: dict[str, Any], strategy: dict[str, Any], prompts: dict[str, str], chain_template: str, chain_template_file: Path, args: argparse.Namespace, sample_info: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    results, coverage, failures, _ = run_skill_iteration(
        client=client, records=records, output_root=root, framework=framework,
        strategy=strategy, system_prompt=prompts["system"],
        analysis_prompt=prompts["analysis"], propagation_prompt=prompts["propagation"],
        chain_template=chain_template, chain_template_file=chain_template_file,
        args=args, sample_info=sample_info,
    )
    return results, coverage, failures


def scores(coverage: dict[str, Any]) -> dict[str, Any]:
    runtime = coverage.get("openclaw_runtime_reward_summary")
    runtime = runtime if isinstance(runtime, dict) else {}
    z3 = coverage.get("z3_reward_summary")
    z3 = z3 if isinstance(z3, dict) else {}
    means = coverage.get("mean_metrics") if isinstance(coverage.get("mean_metrics"), dict) else {}
    runtime_enabled = bool(runtime.get("enabled"))
    return {
        "reward": float(runtime.get("reward", 0.0) if runtime_enabled else z3.get("reward", 0.0)),
        "reward_source": "openclaw_runtime" if runtime_enabled else "z3",
        "attack_success_rate": float(runtime.get("attack_success_rate", runtime.get("strict_attack_success_rate", 0.0))),
        "strict_attack_success_rate": float(runtime.get("strict_attack_success_rate", runtime.get("attack_success_rate", 0.0))),
        "z3_reward": float(z3.get("reward", 0.0)),
        "framework_coverage": float(means.get(FRAMEWORK_SLOT_METRIC, 0.0)),
        "strategy_coverage": float(means.get(STRATEGY_STEP_METRIC, 0.0)),
    }


def candidate_key(candidate: Candidate) -> tuple[float, float, float, float, float, str]:
    return (candidate.reward, candidate.attack_success_rate, candidate.z3_reward, candidate.framework_coverage, candidate.strategy_coverage, candidate.candidate_id)


def make_preference(prompt: str, candidates: list[Candidate], minimum_gap: float) -> dict[str, Any] | None:
    valid = [candidate for candidate in candidates if candidate.status in {"ok", "no_change"}]
    if len(valid) < 2:
        return None
    ranked = sorted(valid, key=candidate_key, reverse=True)
    chosen = ranked[0]
    rejected = next((item for item in reversed(ranked) if item.action != chosen.action), None)
    if rejected is None or chosen.reward - rejected.reward < minimum_gap:
        return None
    return {
        "skill_id": chosen.skill_id, "prompt": prompt,
        "chosen": chosen.action, "rejected": rejected.action,
        "chosen_reward": chosen.reward, "rejected_reward": rejected.reward,
        "reward_gap": round(chosen.reward - rejected.reward, 6),
        "reward_source": chosen.reward_source,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def export_dataset(output_root: Path, dataset_name: str, pairs: list[dict[str, Any]]) -> dict[str, str | int]:
    dataset_dir = output_root / "llamafactory_data"
    dataset_file = dataset_dir / f"{dataset_name}.jsonl"
    write_jsonl(dataset_file, [{
        "instruction": pair["prompt"], "input": "", "chosen": pair["chosen"],
        "rejected": pair["rejected"], "system": "", "skill_id": pair["skill_id"],
        "chosen_reward": pair["chosen_reward"], "rejected_reward": pair["rejected_reward"],
        "reward_gap": pair["reward_gap"], "reward_source": pair["reward_source"],
    } for pair in pairs])
    info_file = dataset_dir / "dataset_info.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    info[dataset_name] = {"file_name": dataset_file.name, "ranking": True, "columns": {
        "prompt": "instruction", "query": "input", "chosen": "chosen",
        "rejected": "rejected", "system": "system",
    }}
    write_json(info_file, info)
    return {"dataset_dir": str(dataset_dir.resolve()), "dataset_name": dataset_name, "dataset_file": str(dataset_file.resolve()), "dataset_info_file": str(info_file.resolve()), "preference_count": len(pairs)}


def run_dpo(args: argparse.Namespace, dataset: dict[str, str | int], round_root: Path, round_index: int, adapter: str) -> tuple[str, dict[str, Any]]:
    cli = shutil.which(args.llamafactory_cli)
    if cli is None:
        raise RuntimeError(f"LLaMA-Factory CLI not found: {args.llamafactory_cli}")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("Writing the LLaMA-Factory config requires PyYAML.") from exc
    adapter_root = Path(args.dpo_output_root).expanduser().resolve() if args.dpo_output_root else round_root.parent / "llamafactory_adapters"
    output_dir = adapter_root / f"round_{round_index:03d}"
    config = {
        "model_name_or_path": str(Path(args.model_name_or_path).expanduser().resolve()),
        "stage": "dpo", "do_train": True, "finetuning_type": "lora", "template": "qwen",
        "cutoff_len": args.dpo_cutoff_len, "dataset_dir": dataset["dataset_dir"],
        "dataset": dataset["dataset_name"], "output_dir": str(output_dir),
        "overwrite_output_dir": True, "per_device_train_batch_size": args.dpo_batch_size,
        "gradient_accumulation_steps": args.dpo_gradient_accumulation,
        "learning_rate": args.dpo_learning_rate, "num_train_epochs": args.dpo_epochs,
        "lr_scheduler_type": "cosine", "logging_steps": 1, "save_strategy": "epoch",
        "plot_loss": True, "lora_rank": args.dpo_lora_rank, "lora_alpha": args.dpo_lora_alpha,
        "lora_dropout": args.dpo_lora_dropout, "pref_beta": args.dpo_beta,
        "pref_loss": "sigmoid", "report_to": "none", "val_size": 0.0, "eval_strategy": "no",
    }
    if args.dpo_quantization_bit > 0:
        config["quantization_bit"] = args.dpo_quantization_bit
    if args.dpo_bf16:
        config["bf16"] = True
    if adapter:
        config["adapter_name_or_path"] = adapter
    config_path = round_root / "llamafactory_dpo.yaml"
    log_path = round_root / "llamafactory_dpo.log"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run([cli, "train", str(config_path)], cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)
    summary = {"command": [cli, "train", str(config_path)], "exit_code": process.returncode, "elapsed_seconds": round(time.time() - started, 2), "config": str(config_path), "log": str(log_path), "adapter_output": str(output_dir)}
    write_json(round_root / "llamafactory_dpo_summary.json", summary)
    if process.returncode != 0:
        raise RuntimeError(f"LLaMA-Factory DPO failed with exit code {process.returncode}; see {log_path}")
    if not (output_dir / "adapter_config.json").exists():
        raise RuntimeError(f"LLaMA-Factory did not write adapter_config.json to {output_dir}")
    return str(output_dir), summary


def invalid_candidate(skill_id: str, candidate_id: str, args: argparse.Namespace, error: str, action: str = "", response: dict[str, Any] | None = None) -> Candidate:
    return Candidate(skill_id, candidate_id, action, response or {}, "invalid", args.invalid_candidate_reward, "invalid_action", 0.0, 0.0, 0.0, 0.0, 0.0, False, error)


def main() -> int:
    args = parse_args()
    if args.rounds < 0 or args.sample_size < 0 or args.evaluate_test_every < 0:
        raise SystemExit("--rounds, --sample-size, and --evaluate-test-every must be >= 0")
    if args.test_size <= 0 or args.samples_per_prompt < 2 or args.min_preference_gap < 0:
        raise SystemExit("--test-size must be > 0, --samples-per-prompt >= 2, and --min-preference-gap >= 0")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    framework, strategy = load_seed_framework_and_strategy(Path(args.framework_file).resolve(), Path(args.strategy_file).resolve())
    records = collect_records(Path(args.data_root).resolve(), selected_ids=set(args.skill) if args.skill else None, graph_name=args.graph_name)
    if args.limit > 0:
        records = records[:args.limit]
    if len(records) < 2:
        raise SystemExit("At least 2 matching graph/SKILL pairs are required.")
    train_records, test_records, split = split_train_test(records, test_size=args.test_size, seed=args.split_seed)
    write_json(output_root / "dataset_split.json", split)
    write_json(output_root / "run_plan.json", {"algorithm": "reward_ranked_dpo", "primary_reward": "OpenClaw runtime when enabled; otherwise Z3", "runtime_eval_enabled": bool(args.openclaw_runtime_eval), "arguments": {key: value for key, value in vars(args).items() if "key" not in key.lower()}, "dataset_split": split})
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output_root": str(output_root), "train_size": len(train_records), "test_size": len(test_records)}, ensure_ascii=False))
        return 0
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required for propagation-environment rewards.")
    if not Path(args.model_name_or_path).expanduser().exists():
        raise SystemExit(f"Qwen model does not exist: {args.model_name_or_path}")
    client = DeepSeekClient(api_key=api_key, model=args.deepseek_model, api_url=args.deepseek_api_url, retries=args.deepseek_retries, retry_seconds=args.deepseek_retry_seconds)
    prompts = {"system": read_text(Path(args.system_prompt_file).resolve()).strip(), "synthesis": read_text(Path(args.prompt_file).resolve()).strip(), "analysis": read_text(Path(args.analysis_prompt_file).resolve()).strip(), "propagation": read_text(Path(args.propagation_prompt_file).resolve()).strip()}
    chain_template_file = Path(args.chain_template_file).resolve()
    chain_template = render_formal_chain_template(load_chain_template(chain_template_file))
    current_framework, current_strategy = framework, strategy
    adapter = str(Path(args.start_adapter_path).expanduser().resolve()) if args.start_adapter_path else ""
    policy = QwenPolicy(args, adapter)
    rng = random.Random(args.sample_seed)
    history: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    if args.evaluate_test_every:
        _, coverage, failures = evaluate(client, test_records, output_root / "round_000" / "test", current_framework, current_strategy, prompts, chain_template, chain_template_file, args, {"phase": "test", "round": 0, "selected_skill_ids": [record.skill_id for record in test_records]})
        entry = build_metric_entry(iteration=0, phase="test", coverage=coverage, failures=failures, reason="initial holdout")
        history.append(entry)
        write_json(output_root / "round_000" / "test" / "reward_summary.json", entry)
    for round_index in range(1, args.rounds + 1):
        round_root = output_root / f"round_{round_index:03d}"
        write_json(round_root / "framework_definition.before.json", current_framework)
        write_json(round_root / "parsing_strategy.before.json", current_strategy)
        selected, selection = select_train_records(train_records, sample_size=args.sample_size, rng=rng, iteration=round_index)
        write_json(round_root / "train_sample_info.json", selection)
        round_candidates: list[dict[str, Any]] = []
        round_pairs: list[dict[str, Any]] = []
        for step_index, record in enumerate(selected, start=1):
            record_root = round_root / "train" / f"step_{step_index:03d}_{record.skill_id}"
            results, baseline_coverage, baseline_failures = evaluate(client, [record], record_root / "baseline", current_framework, current_strategy, prompts, chain_template, chain_template_file, args, {"phase": "train_baseline", "round": round_index, "train_step": step_index, "selected_skill_ids": [record.skill_id]})
            baseline = scores(baseline_coverage)
            history.append(build_metric_entry(iteration=round_index, phase="train_baseline", train_step=step_index, skill_id=record.skill_id, coverage=baseline_coverage, failures=baseline_failures))
            prompt = build_policy_prompt(prompts["synthesis"], current_framework, current_strategy, baseline_coverage, build_low_scoring_samples(results, baseline_coverage, limit=8), history, args.max_policy_prompt_chars)
            write_text(record_root / "policy_prompt.md", prompt + "\n")
            candidates: list[Candidate] = []
            unique: dict[str, Candidate] = {}
            for candidate_index, raw in enumerate(policy.generate(prompts["system"], prompt, args), start=1):
                candidate_id = f"candidate_{candidate_index:02d}"
                candidate_root = record_root / candidate_id
                write_text(candidate_root / "raw_action.txt", raw + "\n")
                response, error = parse_action(raw)
                if response is None:
                    candidate = invalid_candidate(record.skill_id, candidate_id, args, error)
                else:
                    action = json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                    write_text(candidate_root / "action.json", action)
                    if action in unique:
                        candidate = Candidate(**{**asdict(unique[action]), "candidate_id": candidate_id, "status": "duplicate"})
                    else:
                        try:
                            next_framework, next_strategy, changed = apply_action(current_framework, current_strategy, response)
                        except Exception as exc:
                            candidate = invalid_candidate(record.skill_id, candidate_id, args, f"schema_validation: {exc}", action, response)
                        else:
                            if changed:
                                write_json(candidate_root / "framework_definition.json", next_framework)
                                write_json(candidate_root / "parsing_strategy.json", next_strategy)
                                try:
                                    _, candidate_coverage, candidate_failures = evaluate(client, [record], candidate_root / "environment", next_framework, next_strategy, prompts, chain_template, chain_template_file, args, {"phase": "train_candidate", "round": round_index, "train_step": step_index, "candidate_id": candidate_id, "selected_skill_ids": [record.skill_id]})
                                except Exception as exc:
                                    candidate = invalid_candidate(record.skill_id, candidate_id, args, f"environment: {exc}", action, response)
                                else:
                                    value = scores(candidate_coverage)
                                    candidate = Candidate(record.skill_id, candidate_id, action, response, "ok" if not candidate_failures else "partial", value["reward"], value["reward_source"], value["attack_success_rate"], value["strict_attack_success_rate"], value["z3_reward"], value["framework_coverage"], value["strategy_coverage"], True)
                            else:
                                candidate = Candidate(record.skill_id, candidate_id, action, response, "no_change", baseline["reward"], baseline["reward_source"], baseline["attack_success_rate"], baseline["strict_attack_success_rate"], baseline["z3_reward"], baseline["framework_coverage"], baseline["strategy_coverage"], False)
                            unique[action] = candidate
                candidates.append(candidate)
                write_json(candidate_root / "candidate_summary.json", asdict(candidate))
            rows = [asdict(candidate) for candidate in candidates]
            write_jsonl(record_root / "candidate_rewards.jsonl", rows)
            round_candidates.extend(rows)
            pair = make_preference(prompt, candidates, args.min_preference_gap)
            if pair:
                round_pairs.append(pair)
            winners = [candidate for candidate in candidates if candidate.status == "ok"]
            if winners:
                winner = max(winners, key=candidate_key)
                if winner.reward >= baseline["reward"] + args.accept_reward_delta:
                    current_framework, current_strategy, _ = apply_action(current_framework, current_strategy, winner.response)
                    write_json(record_root / "accepted_action.json", winner.response)
        all_pairs.extend(round_pairs)
        write_jsonl(round_root / "candidate_rewards.jsonl", round_candidates)
        write_jsonl(round_root / "preferences.jsonl", round_pairs)
        write_jsonl(output_root / "reward_history.jsonl", round_candidates)
        write_jsonl(output_root / "dpo_preferences.jsonl", all_pairs)
        dataset = export_dataset(output_root, args.llamafactory_dataset_name, all_pairs)
        training: dict[str, Any] = {"round": round_index, "candidate_count": len(round_candidates), "round_preference_count": len(round_pairs), "cumulative_preference_count": len(all_pairs), "dataset": dataset, "policy_adapter_before": adapter}
        if args.run_llamafactory_dpo and all_pairs:
            adapter, dpo = run_dpo(args, dataset, round_root, round_index, adapter)
            training["llamafactory"] = dpo
            training["policy_adapter_after"] = adapter
            del policy
            policy = QwenPolicy(args, adapter)
        elif args.run_llamafactory_dpo:
            training["llamafactory"] = {"status": "skipped_no_preference_pairs"}
        write_json(round_root / "train_summary.json", training)
        write_json(round_root / "framework_definition.after.json", current_framework)
        write_json(round_root / "parsing_strategy.after.json", current_strategy)
        if args.evaluate_test_every and round_index % args.evaluate_test_every == 0:
            _, test_coverage, test_failures = evaluate(client, test_records, round_root / "test", current_framework, current_strategy, prompts, chain_template, chain_template_file, args, {"phase": "test", "round": round_index, "selected_skill_ids": [record.skill_id for record in test_records]})
            entry = build_metric_entry(iteration=round_index, phase="test", coverage=test_coverage, failures=test_failures)
            history.append(entry)
            write_json(round_root / "test" / "reward_summary.json", entry)
        write_json(output_root / "metric_history.json", history)
        write_jsonl(output_root / "metric_history.jsonl", history)
        print(json.dumps({"round": round_index, "preferences": len(all_pairs), "adapter": adapter}, ensure_ascii=False))
    write_json(output_root / "framework_definition.json", current_framework)
    write_json(output_root / "parsing_strategy.json", current_strategy)
    write_json(output_root / "final_summary.json", {"algorithm": "reward_ranked_dpo", "policy_adapter": adapter, "preference_count": len(all_pairs), "runtime_eval_enabled": bool(args.openclaw_runtime_eval), "metric_history": history})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
