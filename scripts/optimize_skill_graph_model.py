#!/usr/bin/env python3
"""Build a local optimization pipeline for Mermaid workflow generation.

Pipeline stages:
1. Build supervised fine-tuning examples from SKILL.md -> graph_wo_check.md.
2. Generate candidate graphs with a local model (Ollama-compatible API).
3. Score candidates with structural + Z3-based validation.
4. Export preference / reward datasets for SFT, DPO, or RL-style optimization.

This script is intentionally training-framework agnostic. It prepares data,
collects model outputs, and computes reward signals so you can plug the output
into LoRA / QLoRA / DPO / PPO tooling later.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import validate_skill_graphs as graph_validator
except ModuleNotFoundError as exc:
    missing_name = getattr(exc, "name", "") or "required module"
    if missing_name in {"z3", "z3-solver"}:
        raise SystemExit(
            "Missing Python dependency: z3-solver. "
            "Install it in your active environment, for example: "
            "pip install z3-solver"
        ) from exc
    raise

LLAMA_FACTORY_DATA_DIRNAME = "llamafactory_data"
LLAMA_FACTORY_SFT_TRAIN_NAME = "skill_graph_sft_train"
LLAMA_FACTORY_SFT_VAL_NAME = "skill_graph_sft_val"
LLAMA_FACTORY_DPO_NAME = "skill_graph_dpo"
LLAMA_FACTORY_PPO_PROMPT_NAME = "skill_graph_ppo_prompt"
HF_GENERATOR_CACHE: dict[str, Any] = {}


@dataclass
class SkillExample:
    skill_id: str
    skill_dir: str
    prompt: str
    completion: str
    skill_markdown: str
    manifest_summary: dict[str, Any]


@dataclass
class CandidateScore:
    skill_id: str
    candidate_id: str
    model_name: str
    graph_text: str
    report: dict[str, Any]
    reward: float
    reward_breakdown: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data" / "inner_representation"), help="Root directory containing normalized skills.")
    parser.add_argument("--prompt-file", default=str(REPO_ROOT / "prompts" / "skill_workflow_mermaid_prompt.md"), help="Prompt template used for Mermaid generation.")
    parser.add_argument("--graph-name", default="graph_wo_check.md", help="Ground-truth Mermaid filename.")
    parser.add_argument("--index-file", default="index.json", help="Index file inside data-root.")
    parser.add_argument("--skill", action="append", default=[], help="Restrict to specific skill ids. Can be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N examples. 0 means no limit.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for train/val splits or candidate shuffling.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "graph_model_optimization"), help="Directory to write datasets, candidate generations, and reward files.")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio for exported SFT/PPO data.")
    parser.add_argument("--mode", choices=("build-dataset", "generate", "score", "export-preference", "plan", "pipeline", "rolling-dpo"), default="build-dataset", help="Pipeline stage to run.")
    parser.add_argument(
        "--generation-backend",
        choices=("auto", "ollama", "hf"),
        default="auto",
        help="Backend used for candidate generation. `auto` uses Hugging Face for local model paths and Ollama otherwise.",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate", help="Ollama-compatible generation endpoint.")
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Local model name for candidate generation.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for candidate generation.")
    parser.add_argument("--num-candidates", type=int, default=4, help="Number of candidates to sample per skill during generation/export-preference.")
    parser.add_argument("--max-tokens", type=int, default=2200, help="Max tokens requested from the local generator.")
    parser.add_argument("--hf-device-map", default="auto", help="Transformers device_map used by the Hugging Face generation backend.")
    parser.add_argument("--hf-torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto", help="Torch dtype used by the Hugging Face generation backend.")
    parser.add_argument("--hf-trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading a Hugging Face model locally.")
    parser.add_argument("--hf-adapter-path", default="", help="Optional PEFT/LoRA adapter path used by the Hugging Face generation backend.")
    parser.add_argument("--candidates-file", default="", help="Candidate JSONL file for score/export-preference mode.")
    parser.add_argument("--scores-file", default="", help="Score JSONL file for export-preference mode.")
    parser.add_argument("--reward-config", default="", help="Optional JSON file overriding reward weights.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of generate-score-export iterations to run in pipeline mode.")
    parser.add_argument("--stop-on-generate-error", action="store_true", help="Fail immediately in pipeline mode if candidate generation fails.")
    parser.add_argument("--min-preference-gap", type=float, default=0.5, help="Minimum reward gap required to export a chosen/rejected pair for DPO.")
    parser.add_argument("--rolling-rounds", type=int, default=5, help="Number of rolling optimization rounds to run in rolling-dpo mode.")
    parser.add_argument("--rolling-batch-size", type=int, default=8, help="Number of skills to sample per round in rolling-dpo mode.")
    parser.add_argument("--rolling-sample-strategy", choices=("sequential", "random"), default="sequential", help="How rolling-dpo selects skills for each round.")
    parser.add_argument("--run-dpo-after-export", action="store_true", help="In rolling-dpo mode, launch LLaMA-Factory DPO training after each preference export.")
    parser.add_argument("--stop-on-dpo-error", action="store_true", help="Fail immediately if an auto-launched DPO round exits with a non-zero code.")
    parser.add_argument("--dpo-run-script", default=str(REPO_ROOT / "llamafactory" / "run_dpo.sh"), help="Shell script used to launch LLaMA-Factory DPO training.")
    parser.add_argument("--dpo-config", default=str(REPO_ROOT / "llamafactory" / "train_qwen25_lora_dpo.yaml"), help="YAML config passed to the DPO launcher.")
    parser.add_argument("--dpo-output-dir", default="", help="Optional override for the DPO adapter output directory. If empty, infer it from the DPO YAML config.")
    parser.add_argument("--ppo-train-file", default="", help="Optional PPO train prompt dataset JSONL. If empty, export to the default LLaMA-Factory data dir.")
    parser.add_argument("--ppo-val-file", default="", help="Optional PPO validation prompt dataset JSONL. If empty, export to the default LLaMA-Factory data dir.")
    parser.add_argument("--resume-cumulative-data", action="store_true", help="In rolling-dpo mode, reuse existing cumulative preference/reward files instead of starting fresh.")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def upsert_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if path.exists():
        existing = load_json(path)
        if isinstance(existing, dict):
            merged.update(existing)
    merged.update(payload)
    write_json(path, merged)
    return merged


def load_prompt_template(path: Path) -> str:
    return read_text(path)


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "sourceRelativePath": manifest.get("sourceRelativePath"),
        "sourceParents": manifest.get("sourceParents"),
        "primaryJson": manifest.get("primaryJson"),
        "copiedDocs": manifest.get("copiedDocs"),
        "syntheticMeta": manifest.get("syntheticMeta"),
    }


def load_skill_examples(data_root: Path, prompt_file: Path, graph_name: str, selected_skills: set[str] | None, limit: int) -> list[SkillExample]:
    index_data = load_json(data_root / "index.json")
    prompt_template = load_prompt_template(prompt_file)
    examples: list[SkillExample] = []
    for entry in index_data:
        skill_id = str(entry.get("canonicalId") or "").strip()
        if not skill_id:
            continue
        if selected_skills and skill_id not in selected_skills:
            continue
        skill_dir = data_root / skill_id
        graph_path = skill_dir / graph_name
        skill_path = skill_dir / "SKILL.md"
        manifest_path = skill_dir / "manifest.json"
        if not (graph_path.exists() and skill_path.exists() and manifest_path.exists()):
            continue
        manifest = load_json(manifest_path)
        prompt = prompt_template.format(
            skill_id=skill_id,
            skill_name=manifest.get("name", skill_id),
            skill_description=manifest.get("description", ""),
            manifest_summary=json.dumps(summarize_manifest(manifest), ensure_ascii=False, indent=2),
            skill_markdown=read_text(skill_path),
        )
        examples.append(
            SkillExample(
                skill_id=skill_id,
                skill_dir=str(skill_dir),
                prompt=prompt,
                completion=read_text(graph_path),
                skill_markdown=read_text(skill_path),
                manifest_summary=summarize_manifest(manifest),
            )
        )
    examples.sort(key=lambda item: item.skill_id)
    if limit > 0:
        examples = examples[:limit]
    return examples


def export_llamafactory_sft_dataset(train_examples: list[SkillExample], val_examples: list[SkillExample], output_dir: Path) -> dict[str, Any]:
    dataset_dir = output_dir / LLAMA_FACTORY_DATA_DIRNAME

    def to_row(example: SkillExample) -> dict[str, Any]:
        return {
            "instruction": example.prompt,
            "input": "",
            "output": example.completion,
            "system": "",
            "skill_id": example.skill_id,
            "skill_dir": example.skill_dir,
        }

    train_file = dataset_dir / "skill_graph_sft_train.jsonl"
    val_file = dataset_dir / "skill_graph_sft_val.jsonl"
    dataset_info_file = dataset_dir / "dataset_info.json"
    write_jsonl(train_file, [to_row(example) for example in train_examples])
    write_jsonl(val_file, [to_row(example) for example in val_examples])
    dataset_info = upsert_json(
        dataset_info_file,
        {
            LLAMA_FACTORY_SFT_TRAIN_NAME: {"file_name": train_file.name, "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}},
            LLAMA_FACTORY_SFT_VAL_NAME: {"file_name": val_file.name, "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}},
        },
    )
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_info_file": str(dataset_info_file.resolve()),
        "train_dataset_name": LLAMA_FACTORY_SFT_TRAIN_NAME,
        "val_dataset_name": LLAMA_FACTORY_SFT_VAL_NAME,
        "train_file": str(train_file.resolve()),
        "val_file": str(val_file.resolve()),
        "dataset_info": dataset_info,
    }


def export_sft_dataset(examples: list[SkillExample], output_dir: Path, split_ratio: float, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * split_ratio)
    train_examples = shuffled[:split_index]
    val_examples = shuffled[split_index:]

    def to_row(example: SkillExample) -> dict[str, Any]:
        return {
            "skill_id": example.skill_id,
            "prompt": example.prompt,
            "completion": example.completion,
            "messages": [
                {"role": "user", "content": example.prompt},
                {"role": "assistant", "content": example.completion},
            ],
        }

    train_rows = [to_row(example) for example in train_examples]
    val_rows = [to_row(example) for example in val_examples]
    write_jsonl(output_dir / "sft_train.jsonl", train_rows)
    write_jsonl(output_dir / "sft_val.jsonl", val_rows)
    llamafactory_summary = export_llamafactory_sft_dataset(train_examples, val_examples, output_dir)
    summary = {
        "total_examples": len(examples),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_file": str((output_dir / "sft_train.jsonl").resolve()),
        "val_file": str((output_dir / "sft_val.jsonl").resolve()),
        "llamafactory": llamafactory_summary,
    }
    write_json(output_dir / "dataset_summary.json", summary)
    return summary


def export_ppo_prompt_dataset(train_examples: list[SkillExample], val_examples: list[SkillExample], output_dir: Path, train_file_override: str = "", val_file_override: str = "") -> dict[str, Any]:
    dataset_dir = output_dir / LLAMA_FACTORY_DATA_DIRNAME
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_file = Path(train_file_override).resolve() if train_file_override else dataset_dir / "skill_graph_ppo_train.jsonl"
    val_file = Path(val_file_override).resolve() if val_file_override else dataset_dir / "skill_graph_ppo_val.jsonl"

    def to_row(example: SkillExample) -> dict[str, Any]:
        return {
            "instruction": example.prompt,
            "input": "",
            "system": "",
            "skill_id": example.skill_id,
            "reference_graph": example.completion,
            "skill_dir": example.skill_dir,
        }

    write_jsonl(train_file, [to_row(example) for example in train_examples])
    write_jsonl(val_file, [to_row(example) for example in val_examples])
    dataset_info = upsert_json(
        dataset_dir / "dataset_info.json",
        {
            LLAMA_FACTORY_PPO_PROMPT_NAME: {
                "file_name": train_file.name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "system": "system",
                },
            },
            f"{LLAMA_FACTORY_PPO_PROMPT_NAME}_val": {
                "file_name": val_file.name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "system": "system",
                },
            },
        },
    )
    return {
        "train_file": str(train_file.resolve()),
        "val_file": str(val_file.resolve()),
        "dataset_info_file": str((dataset_dir / "dataset_info.json").resolve()),
        "dataset_name": LLAMA_FACTORY_PPO_PROMPT_NAME,
        "val_dataset_name": f"{LLAMA_FACTORY_PPO_PROMPT_NAME}_val",
        "dataset_info": dataset_info,
    }


def build_train_val_split(examples: list[SkillExample], split_ratio: float, seed: int) -> tuple[list[SkillExample], list[SkillExample]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * split_ratio)
    train_examples = shuffled[:split_index]
    val_examples = shuffled[split_index:]
    if not train_examples and val_examples:
        train_examples = val_examples[:1]
        val_examples = val_examples[1:]
    if not val_examples and train_examples:
        val_examples = train_examples[-1:]
        train_examples = train_examples[:-1] or train_examples
    return train_examples, val_examples


DEFAULT_REWARD_CONFIG = {
    "graph_td_prefix": 0.35,
    "graph_nonempty": 0.1,
    "edge_marker": 0.35,
    "start_name_marker": 0.2,
    "finish_name_marker": 1.0,
    "has_start_node": 0.6,
    "has_finish_node": 1.2,
    "path_exists": 4.2,
    "all_nodes_reachable_from_start": 1.0,
    "all_nodes_can_reach_finish": 1.8,
    "witness_path": 2.0,
    "witness_node_coverage": 1.1,
    "witness_edge_coverage": 0.8,
    "node_count_reward": 0.03,
    "edge_count_reward": 0.02,
    "cycle_penalty": -0.03,
    "warning_penalty": -0.1,
    "error_penalty": -1.2,
}


def load_reward_config(path_str: str) -> dict[str, float]:
    config = dict(DEFAULT_REWARD_CONFIG)
    if not path_str:
        return config
    payload = load_json(Path(path_str).resolve())
    for key, value in payload.items():
        if key in config:
            config[key] = float(value)
    return config


def safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def summarize_score_rows(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not score_rows:
        return {"candidate_count": 0}
    rewards = [row["reward"] for row in score_rows]
    by_status: dict[str, int] = {}
    by_skill: dict[str, list[float]] = {}
    for row in score_rows:
        status = row["report"].get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        by_skill.setdefault(row["skill_id"], []).append(row["reward"])
    best_by_skill = {skill_id: max(values) for skill_id, values in by_skill.items()}
    return {
        "candidate_count": len(score_rows),
        "status_counts": by_status,
        "avg_reward": round(sum(rewards) / len(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "min_reward": round(min(rewards), 4),
        "avg_best_reward_per_skill": round(sum(best_by_skill.values()) / len(best_by_skill), 4),
        "skills_with_any_candidate": len(by_skill),
    }


def summarize_iteration_metrics(score_rows: list[dict[str, Any]], iteration: int) -> dict[str, Any]:
    if not score_rows:
        return {
            "iteration": iteration,
            "candidate_count": 0,
            "skill_count": 0,
            "z3_pass_count": 0,
            "z3_pass_rate": 0.0,
            "path_exists_count": 0,
            "path_exists_rate": 0.0,
            "full_reach_count": 0,
            "full_reach_rate": 0.0,
            "finish_reach_count": 0,
            "finish_reach_rate": 0.0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "pass_with_warnings_count": 0,
            "pass_with_warnings_rate": 0.0,
            "fail_count": 0,
            "fail_rate": 0.0,
            "avg_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
            "avg_best_reward_per_skill": 0.0,
        }

    rewards = [float(row["reward"]) for row in score_rows]
    skill_ids = sorted({row["skill_id"] for row in score_rows})
    best_by_skill: dict[str, float] = {}
    z3_pass_count = 0
    path_exists_count = 0
    full_reach_count = 0
    finish_reach_count = 0
    pass_count = 0
    pass_with_warnings_count = 0
    fail_count = 0
    for row in score_rows:
        skill_id = row["skill_id"]
        best_by_skill[skill_id] = max(best_by_skill.get(skill_id, float("-inf")), float(row["reward"]))
        report = row.get("report", {})
        checks = report.get("checks", {})
        status = report.get("status", "unknown")
        if report.get("witness_path"):
            z3_pass_count += 1
        if checks.get("path_exists"):
            path_exists_count += 1
        if checks.get("all_nodes_reachable_from_start"):
            full_reach_count += 1
        if checks.get("all_nodes_can_reach_finish"):
            finish_reach_count += 1
        if status == "pass":
            pass_count += 1
        elif status == "pass_with_warnings":
            pass_with_warnings_count += 1
        else:
            fail_count += 1
    candidate_count = len(score_rows)
    return {
        "iteration": iteration,
        "candidate_count": candidate_count,
        "skill_count": len(skill_ids),
        "z3_pass_count": z3_pass_count,
        "z3_pass_rate": round(safe_ratio(z3_pass_count, candidate_count), 4),
        "path_exists_count": path_exists_count,
        "path_exists_rate": round(safe_ratio(path_exists_count, candidate_count), 4),
        "full_reach_count": full_reach_count,
        "full_reach_rate": round(safe_ratio(full_reach_count, candidate_count), 4),
        "finish_reach_count": finish_reach_count,
        "finish_reach_rate": round(safe_ratio(finish_reach_count, candidate_count), 4),
        "pass_count": pass_count,
        "pass_rate": round(safe_ratio(pass_count, candidate_count), 4),
        "pass_with_warnings_count": pass_with_warnings_count,
        "pass_with_warnings_rate": round(safe_ratio(pass_with_warnings_count, candidate_count), 4),
        "fail_count": fail_count,
        "fail_rate": round(safe_ratio(fail_count, candidate_count), 4),
        "avg_reward": round(sum(rewards) / candidate_count, 4),
        "max_reward": round(max(rewards), 4),
        "min_reward": round(min(rewards), 4),
        "avg_best_reward_per_skill": round(sum(best_by_skill.values()) / len(best_by_skill), 4),
    }


def format_iteration_metrics(metrics: dict[str, Any], total_iterations: int) -> str:
    return (
        f"[iteration {metrics['iteration']}/{total_iterations}] "
        f"candidates={metrics['candidate_count']} "
        f"z3_pass_rate={metrics['z3_pass_rate']:.4f} "
        f"path_exists_rate={metrics['path_exists_rate']:.4f} "
        f"avg_reward={metrics['avg_reward']:.4f} "
        f"avg_best_reward={metrics['avg_best_reward_per_skill']:.4f}"
    )


def render_iteration_metrics_markdown(history_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Iteration Metrics",
        "",
        "| iteration | candidates | z3_pass_rate | path_exists_rate | full_reach_rate | finish_reach_rate | pass_rate | avg_reward | avg_best_reward | max_reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history_rows:
        lines.append(
            "| {iteration} | {candidate_count} | {z3_pass_rate:.4f} | {path_exists_rate:.4f} | {full_reach_rate:.4f} | {finish_reach_rate:.4f} | {pass_rate:.4f} | {avg_reward:.4f} | {avg_best_reward_per_skill:.4f} | {max_reward:.4f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)

def render_iteration_metrics_svg(history_rows: list[dict[str, Any]]) -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="960" height="160" viewBox="0 0 960 160"><rect width="100%" height="100%" fill="#ffffff"/><text x="40" y="80" font-size="24" fill="#111827">See iteration_metrics.json for full metrics.</text></svg>"""


def write_iteration_history(output_dir: Path, history_rows: list[dict[str, Any]]) -> dict[str, str]:
    write_json(output_dir / "iteration_metrics.json", history_rows)
    write_jsonl(output_dir / "iteration_metrics.jsonl", history_rows)
    write_text(output_dir / "iteration_metrics.md", render_iteration_metrics_markdown(history_rows))
    write_text(output_dir / "iteration_metrics.svg", render_iteration_metrics_svg(history_rows))
    return {
        "iteration_metrics_json": str((output_dir / "iteration_metrics.json").resolve()),
        "iteration_metrics_jsonl": str((output_dir / "iteration_metrics.jsonl").resolve()),
        "iteration_metrics_markdown": str((output_dir / "iteration_metrics.md").resolve()),
        "iteration_metrics_svg": str((output_dir / "iteration_metrics.svg").resolve()),
    }




def read_simple_yaml_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    prefix = f"{key}:"
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(prefix):
            value = stripped[len(prefix):].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            return value
    return ""


def write_overridden_yaml(base_config: Path, target_config: Path, overrides: dict[str, str]) -> None:
    lines: list[str] = []
    seen_keys: set[str] = set()
    for line in read_text(base_config).splitlines():
        stripped = line.strip()
        replaced = False
        for key, value in overrides.items():
            if stripped.startswith(f"{key}:"):
                lines.append(f"{key}: {value}")
                seen_keys.add(key)
                replaced = True
                break
        if not replaced:
            lines.append(line)
    for key, value in overrides.items():
        if key not in seen_keys:
            lines.append(f"{key}: {value}")
    write_text(target_config, "\n".join(lines) + "\n")


def resolve_initial_hf_adapter_path(args: argparse.Namespace) -> str:
    if args.hf_adapter_path:
        return str(Path(args.hf_adapter_path).expanduser().resolve())
    if resolve_generation_backend(args.generation_backend, args.model) != "hf":
        return ""
    adapter_from_config = read_simple_yaml_value(Path(args.dpo_config).resolve(), "adapter_name_or_path")
    if adapter_from_config:
        return str(Path(adapter_from_config).expanduser().resolve())
    return ""


def select_rolling_examples(
    examples: list[SkillExample],
    round_index: int,
    batch_size: int,
    sample_strategy: str,
    rng: random.Random,
) -> list[SkillExample]:
    if not examples:
        return []
    batch_size = max(1, min(batch_size, len(examples)))
    if sample_strategy == "random":
        return rng.sample(examples, batch_size)
    start_index = ((round_index - 1) * batch_size) % len(examples)
    return [examples[(start_index + offset) % len(examples)] for offset in range(batch_size)]


def render_rolling_dpo_markdown(history_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Rolling DPO Metrics",
        "",
        "| round | sampled_skills | candidates | preferences | cumulative_preferences | z3_pass_rate | avg_reward | avg_best_reward | dpo_exit_code |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history_rows:
        dpo_exit_code = row.get("dpo", {}).get("returncode", "")
        lines.append(
            f"| {row['round']} | {len(row.get('sampled_skill_ids', []))} | {row.get('candidate_count', 0)} | {row.get('preferences_created', 0)} | {row.get('cumulative_preferences', 0)} | {row.get('z3_pass_rate', 0.0):.4f} | {row.get('avg_reward', 0.0):.4f} | {row.get('avg_best_reward_per_skill', 0.0):.4f} | {dpo_exit_code} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_rolling_history(output_dir: Path, history_rows: list[dict[str, Any]]) -> dict[str, str]:
    write_json(output_dir / "rolling_dpo_history.json", history_rows)
    write_jsonl(output_dir / "rolling_dpo_history.jsonl", history_rows)
    write_text(output_dir / "rolling_dpo_history.md", render_rolling_dpo_markdown(history_rows))
    return {
        "rolling_dpo_history_json": str((output_dir / "rolling_dpo_history.json").resolve()),
        "rolling_dpo_history_jsonl": str((output_dir / "rolling_dpo_history.jsonl").resolve()),
        "rolling_dpo_history_markdown": str((output_dir / "rolling_dpo_history.md").resolve()),
    }


def refresh_rolling_plots(output_dir: Path) -> dict[str, Any]:
    history_path = output_dir / "rolling_dpo_history.json"
    if not history_path.exists():
        return {"status": "skipped", "reason": "missing rolling_dpo_history.json"}
    plot_script = SCRIPT_DIR / "plot_dpo_metrics.py"
    if not plot_script.exists():
        return {"status": "skipped", "reason": f"missing plot script: {plot_script}"}
    result = subprocess.run(
        [
            sys.executable,
            str(plot_script.resolve()),
            "--rolling-history",
            str(history_path.resolve()),
        ],
        cwd=str(REPO_ROOT),
        check=False,
    )
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "history_file": str(history_path.resolve()),
        "plot_dir": str((history_path.parent / "seaborn_plots").resolve()),
    }


def release_hf_generation_resources() -> dict[str, Any]:
    released_cache_entries = len(HF_GENERATOR_CACHE)
    HF_GENERATOR_CACHE.clear()
    gc.collect()

    cuda_status = "not_available"
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cuda_status = "emptied"
        else:
            cuda_status = "unavailable"
    except ModuleNotFoundError:
        cuda_status = "torch_missing"
    except Exception as exc:
        cuda_status = f"error: {exc}"

    return {
        "released_cache_entries": released_cache_entries,
        "gc_collected": True,
        "cuda_cache": cuda_status,
    }


def run_dpo_round(
    *,
    round_index: int,
    round_dir: Path,
    dataset_dir: Path,
    dpo_run_script: Path,
    base_config: Path,
    dpo_output_dir: Path,
    adapter_source: str,
) -> dict[str, Any]:
    if not dpo_run_script.exists():
        raise SystemExit(f"Missing DPO run script: {dpo_run_script}")
    if not base_config.exists():
        raise SystemExit(f"Missing DPO config: {base_config}")

    round_output_dir = dpo_output_dir / f"round_{round_index:03d}"
    temp_config = round_dir / f"train_qwen25_lora_dpo_round_{round_index:03d}.yaml"
    overrides = {
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset": LLAMA_FACTORY_DPO_NAME,
        "output_dir": str(round_output_dir.resolve()),
        "eval_strategy": "\"no\"",
    }
    if adapter_source:
        overrides["adapter_name_or_path"] = str(Path(adapter_source).expanduser().resolve())
    write_overridden_yaml(base_config, temp_config, overrides)

    started_at = time.time()
    result = subprocess.run(
        ["bash", str(dpo_run_script.resolve()), str(temp_config.resolve())],
        cwd=str(REPO_ROOT),
        check=False,
    )
    return {
        "config": str(temp_config.resolve()),
        "output_dir": str(round_output_dir.resolve()),
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "adapter_source": overrides.get("adapter_name_or_path", ""),
    }


def run_rolling_dpo(args: argparse.Namespace, examples: list[SkillExample], output_dir: Path, reward_config: dict[str, float]) -> dict[str, Any]:
    dataset_summary = export_sft_dataset(examples, output_dir, args.split_ratio, args.seed)
    rolling_root = output_dir / "rolling_dpo"
    rolling_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / LLAMA_FACTORY_DATA_DIRNAME
    cumulative_preference_file = output_dir / "dpo_preferences_cumulative.jsonl"
    cumulative_reward_file = output_dir / "rl_rewards_cumulative.jsonl"
    cumulative_preference_rows = load_jsonl(cumulative_preference_file) if args.resume_cumulative_data and cumulative_preference_file.exists() else []
    cumulative_reward_rows = load_jsonl(cumulative_reward_file) if args.resume_cumulative_data and cumulative_reward_file.exists() else []
    history_rows: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    resolved_backend = resolve_generation_backend(args.generation_backend, args.model)
    current_adapter_path = resolve_initial_hf_adapter_path(args)
    dpo_run_script = Path(args.dpo_run_script).resolve()
    dpo_config = Path(args.dpo_config).resolve()
    inferred_output_dir = args.dpo_output_dir or read_simple_yaml_value(dpo_config, "output_dir")
    dpo_output_dir = Path(inferred_output_dir).expanduser().resolve() if inferred_output_dir else (output_dir / "llamafactory_dpo_rounds")

    for round_index in range(1, max(1, args.rolling_rounds) + 1):
        batch_examples = select_rolling_examples(
            examples=examples,
            round_index=round_index,
            batch_size=args.rolling_batch_size,
            sample_strategy=args.rolling_sample_strategy,
            rng=rng,
        )
        round_dir = rolling_root / f"round_{round_index:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        candidate_rows = generate_candidates(
            examples=batch_examples,
            output_dir=round_dir,
            endpoint=args.ollama_url,
            model_name=args.model,
            num_candidates=args.num_candidates,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            generation_backend=args.generation_backend,
            hf_device_map=args.hf_device_map,
            hf_torch_dtype=args.hf_torch_dtype,
            hf_trust_remote_code=args.hf_trust_remote_code,
            hf_adapter_path=current_adapter_path,
        )
        score_rows = score_candidates(candidate_rows, round_dir, reward_config)
        score_summary = summarize_score_rows(score_rows)
        write_json(round_dir / "score_summary.json", score_summary)
        preference_summary = export_preference_dataset(score_rows, round_dir, args.min_preference_gap)
        round_preference_rows = load_jsonl(round_dir / "dpo_preferences.jsonl") if (round_dir / "dpo_preferences.jsonl").exists() else []
        round_reward_rows = load_jsonl(round_dir / "rl_rewards.jsonl") if (round_dir / "rl_rewards.jsonl").exists() else []
        cumulative_preference_rows.extend(round_preference_rows)
        cumulative_reward_rows.extend(round_reward_rows)
        write_jsonl(cumulative_preference_file, cumulative_preference_rows)
        write_jsonl(cumulative_reward_file, cumulative_reward_rows)
        llamafactory_summary = export_llamafactory_preference_dataset(cumulative_preference_rows, output_dir)

        round_metrics = summarize_iteration_metrics(score_rows, round_index)
        dpo_status: dict[str, Any] = {"status": "skipped", "reason": "not requested"}
        generation_resource_release: dict[str, Any] = {"status": "skipped", "reason": "not needed"}
        if args.run_dpo_after_export:
            if cumulative_preference_rows:
                adapter_source = current_adapter_path or read_simple_yaml_value(dpo_config, "adapter_name_or_path")
                generation_resource_release = release_hf_generation_resources()
                dpo_status = run_dpo_round(
                    round_index=round_index,
                    round_dir=round_dir,
                    dataset_dir=dataset_dir,
                    dpo_run_script=dpo_run_script,
                    base_config=dpo_config,
                    dpo_output_dir=dpo_output_dir,
                    adapter_source=adapter_source,
                )
                if dpo_status.get("returncode") == 0 and resolved_backend == "hf":
                    current_adapter_path = dpo_status.get("output_dir", current_adapter_path)
                elif dpo_status.get("returncode") != 0 and args.stop_on_dpo_error:
                    raise SystemExit(f"DPO round {round_index} failed with exit code {dpo_status.get('returncode')}")
            else:
                dpo_status = {"status": "skipped", "reason": "no preferences exported yet"}

        round_record = {
            **round_metrics,
            "round": round_index,
            "round_dir": str(round_dir.resolve()),
            "sampled_skill_ids": [example.skill_id for example in batch_examples],
            "candidate_count": len(candidate_rows),
            "preferences_created": len(round_preference_rows),
            "round_reward_rows": len(round_reward_rows),
            "cumulative_preferences": len(cumulative_preference_rows),
            "cumulative_reward_rows": len(cumulative_reward_rows),
            "generation_backend": resolved_backend,
            "active_hf_adapter_path": current_adapter_path,
            "score_summary": score_summary,
            "preference_summary": preference_summary,
            "llamafactory": llamafactory_summary,
            "generation_resource_release": generation_resource_release,
            "dpo": dpo_status,
        }
        write_json(round_dir / "rolling_round_summary.json", round_record)
        history_rows.append(round_record)
        rolling_artifacts = write_rolling_history(output_dir, history_rows)
        round_record["rolling_plot_refresh"] = refresh_rolling_plots(output_dir)
        round_record["rolling_artifacts"] = rolling_artifacts
        write_json(round_dir / "rolling_round_summary.json", round_record)
        history_rows[-1] = round_record
        write_rolling_history(output_dir, history_rows)
        print(
            f"[rolling-dpo {round_index}/{args.rolling_rounds}] "
            f"skills={len(batch_examples)} candidates={len(candidate_rows)} "
            f"prefs={len(round_preference_rows)} cumulative_prefs={len(cumulative_preference_rows)} "
            f"avg_reward={round_metrics['avg_reward']:.4f} z3_pass_rate={round_metrics['z3_pass_rate']:.4f}",
            flush=True,
        )

    artifact_files = write_rolling_history(output_dir, history_rows)
    rolling_plot_refresh = refresh_rolling_plots(output_dir)
    summary = {
        "dataset_summary": dataset_summary,
        "rounds": len(history_rows),
        "history": history_rows,
        "cumulative_preference_file": str(cumulative_preference_file.resolve()),
        "cumulative_reward_file": str(cumulative_reward_file.resolve()),
        "reward_config": reward_config,
        "final_hf_adapter_path": current_adapter_path,
        "artifact_files": artifact_files,
        "rolling_plot_refresh": rolling_plot_refresh,
    }
    write_json(output_dir / "rolling_dpo_summary.json", summary)
    return summary


def resolve_generation_backend(backend_name: str, model_name: str) -> str:
    if backend_name != "auto":
        return backend_name

    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return "hf"

    return "ollama"


def resolve_hf_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: torch. Install torch in your active environment to use the Hugging Face generation backend."
        ) from exc

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def load_hf_generator(model_name: str, device_map: str, torch_dtype: str, trust_remote_code: bool, adapter_path: str = "") -> dict[str, Any]:
    resolved_adapter_path = ""
    if adapter_path:
        resolved_adapter_path = str(Path(adapter_path).expanduser().resolve())
    cache_key = json.dumps(
        {
            "model_name": model_name,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "adapter_path": resolved_adapter_path,
        },
        sort_keys=True,
    )
    if cache_key in HF_GENERATOR_CACHE:
        return HF_GENERATOR_CACHE[cache_key]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: transformers. Install transformers in your active environment to use the Hugging Face generation backend."
        ) from exc

    model_path = str(Path(model_name).expanduser().resolve())
    dtype_value = resolve_hf_torch_dtype(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=dtype_value,
        trust_remote_code=trust_remote_code,
    )
    if resolved_adapter_path:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing Python dependency: peft. Install peft in your active environment to load a LoRA adapter for generation."
            ) from exc
        model = PeftModel.from_pretrained(model, resolved_adapter_path)
    model.eval()
    payload = {"tokenizer": tokenizer, "model": model, "adapter_path": resolved_adapter_path}
    HF_GENERATOR_CACHE[cache_key] = payload
    return payload


def format_hf_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def call_hf_generate(
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    device_map: str,
    torch_dtype: str,
    trust_remote_code: bool,
    adapter_path: str = "",
) -> str:
    generator = load_hf_generator(
        model_name=model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        adapter_path=adapter_path,
    )
    tokenizer = generator["tokenizer"]
    model = generator["model"]

    formatted_prompt = format_hf_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    do_sample = temperature > 0
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask
    if do_sample:
        generation_kwargs["temperature"] = temperature

    import torch
    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    generated_ids = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def call_ollama_generate(endpoint: str, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}}
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body.strip() or str(exc)
        raise RuntimeError(f"Failed to call local generation endpoint {endpoint} with model {model_name}: HTTP {exc.code}; {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to call local generation endpoint {endpoint} with model {model_name}: {exc}. "
            "If you are using Ollama, make sure `ollama serve` is running, or pass a reachable `--ollama-url`."
        ) from exc
    return (data.get("response") or "").strip()


def generate_candidates(
    examples: list[SkillExample],
    output_dir: Path,
    endpoint: str,
    model_name: str,
    num_candidates: int,
    temperature: float,
    max_tokens: int,
    generation_backend: str,
    hf_device_map: str,
    hf_torch_dtype: str,
    hf_trust_remote_code: bool,
    hf_adapter_path: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved_backend = resolve_generation_backend(generation_backend, model_name)
    started_at = time.time()
    output_path = output_dir / "generated_candidates.jsonl"
    total_candidates = len(examples) * num_candidates
    completed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            for candidate_index in range(num_candidates):
                if resolved_backend == "hf":
                    raw_output = call_hf_generate(
                        model_name=model_name,
                        prompt=example.prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        device_map=hf_device_map,
                        torch_dtype=hf_torch_dtype,
                        trust_remote_code=hf_trust_remote_code,
                        adapter_path=hf_adapter_path,
                    )
                else:
                    raw_output = call_ollama_generate(
                        endpoint=endpoint,
                        model_name=model_name,
                        prompt=example.prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                row = {
                    "skill_id": example.skill_id,
                    "candidate_id": f"{example.skill_id}__{candidate_index}",
                    "model_name": model_name,
                    "generation_backend": resolved_backend,
                    "adapter_path": hf_adapter_path if resolved_backend == "hf" else "",
                    "prompt": example.prompt,
                    "graph_text": raw_output,
                    "reference_graph": example.completion,
                }
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                completed += 1
                elapsed = time.time() - started_at
                summary_payload = {
                    "backend": resolved_backend,
                    "model_name": model_name,
                    "candidate_count": len(rows),
                    "completed_candidates": completed,
                    "total_candidates": total_candidates,
                    "progress": round(completed / total_candidates, 4) if total_candidates else 1.0,
                    "elapsed_seconds": round(elapsed, 2),
                    "avg_seconds_per_candidate": round(elapsed / completed, 2) if completed else 0.0,
                    "last_skill_id": example.skill_id,
                    "last_candidate_id": row["candidate_id"],
                    "output_file": str(output_path.resolve()),
                }
                write_json(output_dir / "generation_summary.json", summary_payload)
                print(
                    f"[generate] {completed}/{total_candidates} "
                    f"skill={example.skill_id} candidate={candidate_index} "
                    f"backend={resolved_backend} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    final_payload = {
        "backend": resolved_backend,
        "model_name": model_name,
        "candidate_count": len(rows),
        "completed_candidates": completed,
        "total_candidates": total_candidates,
        "progress": 1.0 if total_candidates else 1.0,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "output_file": str(output_path.resolve()),
    }
    write_json(output_dir / "generation_summary.json", final_payload)
    return rows


def compute_reward(report: dict[str, Any], reward_config: dict[str, float]) -> tuple[float, dict[str, float]]:
    checks = report.get("checks", {})
    issues = report.get("issues", [])
    breakdown: dict[str, float] = {}
    graph_path = str(report.get("graph_file") or "")
    graph_text = ""
    if graph_path:
        try:
            graph_text = read_text(Path(graph_path))
        except Exception:
            graph_text = ""
    stripped_graph = graph_text.lstrip()
    lowered_graph = stripped_graph.lower()
    graph_td_prefix = 1.0 if lowered_graph.startswith("graph td") else 0.0
    graph_nonempty = 1.0 if stripped_graph else 0.0
    edge_marker = 1.0 if '-->' in stripped_graph else 0.0
    start_name_marker = 1.0 if 'name: start_node' in lowered_graph else 0.0
    finish_name_marker = 1.0 if 'name: finish_node' in lowered_graph else 0.0
    has_start_node = 1.0 if report.get("start_node") else 0.0
    has_finish_node = 1.0 if report.get("finish_node") else 0.0
    path_exists = 1.0 if checks.get("path_exists") else 0.0
    full_reach = 1.0 if checks.get("all_nodes_reachable_from_start") else 0.0
    full_finish = 1.0 if checks.get("all_nodes_can_reach_finish") else 0.0
    witness = 1.0 if report.get("witness_path") else 0.0
    witness_nodes = len((report.get("witness_path") or {}).get("nodes", []))
    witness_edges = len((report.get("witness_path") or {}).get("edges", []))
    node_count = checks.get("node_count", 0)
    edge_count = checks.get("edge_count", 0)
    witness_node_coverage = safe_ratio(witness_nodes, node_count)
    witness_edge_coverage = safe_ratio(witness_edges, edge_count)
    cycle_count = len(checks.get("cyclical_components", []))
    error_penalty = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_penalty = sum(1 for issue in issues if issue.get("severity") == "warning")
    breakdown["graph_nonempty"] = reward_config["graph_nonempty"] * graph_nonempty
    breakdown["graph_td_prefix"] = reward_config["graph_td_prefix"] * graph_td_prefix
    breakdown["edge_marker"] = reward_config["edge_marker"] * edge_marker
    breakdown["start_name_marker"] = reward_config["start_name_marker"] * start_name_marker
    closure_gate = finish_name_marker
    breakdown["finish_name_marker"] = reward_config["finish_name_marker"] * finish_name_marker
    breakdown["has_start_node"] = reward_config["has_start_node"] * has_start_node
    breakdown["has_finish_node"] = reward_config["has_finish_node"] * has_finish_node
    breakdown["path_exists"] = reward_config["path_exists"] * path_exists * closure_gate
    breakdown["all_nodes_reachable_from_start"] = reward_config["all_nodes_reachable_from_start"] * full_reach * closure_gate
    breakdown["all_nodes_can_reach_finish"] = reward_config["all_nodes_can_reach_finish"] * full_finish * closure_gate
    breakdown["witness_path"] = reward_config["witness_path"] * witness * closure_gate
    breakdown["witness_node_coverage"] = reward_config["witness_node_coverage"] * witness_node_coverage * closure_gate
    breakdown["witness_edge_coverage"] = reward_config["witness_edge_coverage"] * witness_edge_coverage * closure_gate
    breakdown["node_count_reward"] = reward_config["node_count_reward"] * min(node_count, 12)
    breakdown["edge_count_reward"] = reward_config["edge_count_reward"] * min(edge_count, 16)
    breakdown["cycle_penalty"] = reward_config["cycle_penalty"] * cycle_count
    breakdown["error_penalty"] = reward_config["error_penalty"] * error_penalty
    breakdown["warning_penalty"] = reward_config["warning_penalty"] * warning_penalty
    return sum(breakdown.values()), breakdown


def score_candidates(candidate_rows: list[dict[str, Any]], output_dir: Path, reward_config: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    temp_dir = output_dir / "tmp_candidate_graphs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidate_rows:
        skill_id = candidate["skill_id"]
        candidate_id = candidate["candidate_id"]
        graph_path = temp_dir / f"{candidate_id}.md"
        write_text(graph_path, candidate["graph_text"])
        try:
            report = graph_validator.validate_graph(graph_path)
            report_dict = graph_validator.report_to_dict(report)
        except Exception as exc:
            report_dict = {"skill_id": skill_id, "graph_file": str(graph_path), "status": "fail", "checks": {}, "issues": [{"severity": "error", "code": "exception", "message": str(exc)}], "witness_path": None}
        reward, reward_breakdown = compute_reward(report_dict, reward_config)
        scored = CandidateScore(skill_id=skill_id, candidate_id=candidate_id, model_name=candidate["model_name"], graph_text=candidate["graph_text"], report=report_dict, reward=reward, reward_breakdown=reward_breakdown)
        scored_row = asdict(scored)
        scored_row["prompt"] = candidate.get("prompt", "")
        scored_row["reference_graph"] = candidate.get("reference_graph", "")
        rows.append(scored_row)
    write_jsonl(output_dir / "candidate_scores.jsonl", rows)
    return rows


def export_llamafactory_preference_dataset(preference_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    dataset_dir = output_dir / LLAMA_FACTORY_DATA_DIRNAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dpo_file = dataset_dir / "skill_graph_dpo.jsonl"
    dataset_info_file = dataset_dir / "dataset_info.json"
    dpo_rows = [{
        "instruction": row.get("prompt", ""),
        "input": "",
        "chosen": row["chosen"],
        "rejected": row["rejected"],
        "system": "",
        "skill_id": row["skill_id"],
        "chosen_reward": row["chosen_reward"],
        "rejected_reward": row["rejected_reward"],
        "reward_gap": row["reward_gap"],
    } for row in preference_rows]
    write_jsonl(dpo_file, dpo_rows)
    dataset_info = upsert_json(dataset_info_file, {
        LLAMA_FACTORY_DPO_NAME: {"file_name": dpo_file.name, "ranking": True, "columns": {"prompt": "instruction", "query": "input", "chosen": "chosen", "rejected": "rejected", "system": "system"}}
    })
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_info_file": str(dataset_info_file.resolve()),
        "dataset_name": LLAMA_FACTORY_DPO_NAME,
        "dpo_file": str(dpo_file.resolve()),
        "dataset_info": dataset_info,
    }


def export_preference_dataset(score_rows: list[dict[str, Any]], output_dir: Path, min_preference_gap: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        grouped.setdefault(row["skill_id"], []).append(row)
    preference_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    skipped_identical = 0
    skipped_gap = 0
    for skill_id, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda item: item["reward"], reverse=True)
        if len(ranked) < 2:
            continue
        best = ranked[0]
        worst = ranked[-1]
        reward_gap = float(best["reward"]) - float(worst["reward"])
        if best["graph_text"].strip() == worst["graph_text"].strip():
            skipped_identical += 1
            continue
        if reward_gap < min_preference_gap:
            skipped_gap += 1
            continue
        preference_rows.append({
            "skill_id": skill_id,
            "prompt": best.get("prompt", ""),
            "chosen": best["graph_text"],
            "rejected": worst["graph_text"],
            "chosen_reward": best["reward"],
            "rejected_reward": worst["reward"],
            "reward_gap": reward_gap,
        })
        for row in ranked:
            reward_rows.append({
                "skill_id": skill_id,
                "prompt": row.get("prompt", ""),
                "graph_text": row["graph_text"],
                "reward": row["reward"],
                "reward_breakdown": row["reward_breakdown"],
                "status": row["report"].get("status"),
            })
    write_jsonl(output_dir / "dpo_preferences.jsonl", preference_rows)
    write_jsonl(output_dir / "rl_rewards.jsonl", reward_rows)
    llamafactory_summary = export_llamafactory_preference_dataset(preference_rows, output_dir)
    summary = {
        "skills_with_preferences": len(preference_rows),
        "preference_file": str((output_dir / "dpo_preferences.jsonl").resolve()),
        "reward_file": str((output_dir / "rl_rewards.jsonl").resolve()),
        "min_preference_gap": min_preference_gap,
        "skipped_identical_pairs": skipped_identical,
        "skipped_small_gap_pairs": skipped_gap,
        "llamafactory": llamafactory_summary,
    }
    write_json(output_dir / "preference_summary.json", summary)
    return summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def require_existing_file(path: Path, stage_name: str, hint: str) -> Path:
    if not path.exists():
        raise SystemExit(
            f"Missing required file for {stage_name}: {path}\n"
            f"Hint: {hint}"
        )
    return path


def render_training_plan(examples: list[SkillExample], output_dir: Path, model_name: str) -> str:
    return textwrap.dedent(f"""
    Training plan for Mermaid workflow optimization
    ==============================================

    1. Supervised warm start
       - Dataset size: {len(examples)} skills
       - Input: SKILL.md-derived prompt
       - Target: current graph_wo_check.md (DeepSeek-generated teacher output)
       - Suggested method: QLoRA / LoRA SFT

    2. Preference / RL data
       - DPO file: dpo_preferences.jsonl
       - Reward file: rl_rewards.jsonl
       - LLaMA-Factory dataset dir: {output_dir / LLAMA_FACTORY_DATA_DIRNAME}
       - Dataset names: {LLAMA_FACTORY_SFT_TRAIN_NAME}, {LLAMA_FACTORY_SFT_VAL_NAME}, {LLAMA_FACTORY_DPO_NAME}

    Recommended local model string: {model_name}
    """).strip() + "\n"

def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    data_root = Path(args.data_root).resolve()
    prompt_file = Path(args.prompt_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    selected_skills = set(args.skill) if args.skill else None
    reward_config = load_reward_config(args.reward_config)
    examples = load_skill_examples(data_root=data_root, prompt_file=prompt_file, graph_name=args.graph_name, selected_skills=selected_skills, limit=args.limit)
    if not examples:
        print("No examples matched.", file=sys.stderr)
        return 1
    if args.mode == "build-dataset":
        dataset_summary = export_sft_dataset(examples, output_dir, args.split_ratio, args.seed)
        train_examples, val_examples = build_train_val_split(examples, args.split_ratio, args.seed)
        ppo_summary = export_ppo_prompt_dataset(train_examples, val_examples, output_dir, args.ppo_train_file, args.ppo_val_file)
        merged_summary = dict(dataset_summary)
        merged_summary["ppo"] = ppo_summary
        write_json(output_dir / "dataset_summary.json", merged_summary)
        print(json.dumps(merged_summary, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "generate":
        rows = generate_candidates(examples=examples, output_dir=output_dir, endpoint=args.ollama_url, model_name=args.model, num_candidates=args.num_candidates, temperature=args.temperature, max_tokens=args.max_tokens, generation_backend=args.generation_backend, hf_device_map=args.hf_device_map, hf_torch_dtype=args.hf_torch_dtype, hf_trust_remote_code=args.hf_trust_remote_code, hf_adapter_path=args.hf_adapter_path)
        print(json.dumps({"generated_candidates": len(rows), "file": str((output_dir / 'generated_candidates.jsonl').resolve())}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "score":
        if not args.candidates_file:
            print("--candidates-file is required for score mode.", file=sys.stderr)
            return 2
        candidate_path = require_existing_file(
            Path(args.candidates_file).resolve(),
            "score mode",
            "run `--mode generate` first so generated_candidates.jsonl exists.",
        )
        candidate_rows = load_jsonl(candidate_path)
        rows = score_candidates(candidate_rows, output_dir, reward_config)
        write_json(output_dir / "score_summary.json", summarize_score_rows(rows))
        print(json.dumps({"scored_candidates": len(rows), "file": str((output_dir / 'candidate_scores.jsonl').resolve())}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "export-preference":
        if not args.scores_file:
            print("--scores-file is required for export-preference mode.", file=sys.stderr)
            return 2
        score_path = require_existing_file(
            Path(args.scores_file).resolve(),
            "export-preference mode",
            "run `--mode score` first so candidate_scores.jsonl exists.",
        )
        score_rows = load_jsonl(score_path)
        print(json.dumps(export_preference_dataset(score_rows, output_dir, args.min_preference_gap), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "pipeline":
        dataset_summary = export_sft_dataset(examples, output_dir, args.split_ratio, args.seed)
        iteration_total = max(1, args.iterations)
        history_rows: list[dict[str, Any]] = []
        iteration_summaries: list[dict[str, Any]] = []
        for iteration in range(1, iteration_total + 1):
            iteration_output_dir = output_dir if iteration_total == 1 else output_dir / f"iteration_{iteration:03d}"
            try:
                candidate_rows = generate_candidates(examples=examples, output_dir=iteration_output_dir, endpoint=args.ollama_url, model_name=args.model, num_candidates=args.num_candidates, temperature=args.temperature, max_tokens=args.max_tokens, generation_backend=args.generation_backend, hf_device_map=args.hf_device_map, hf_torch_dtype=args.hf_torch_dtype, hf_trust_remote_code=args.hf_trust_remote_code, hf_adapter_path=args.hf_adapter_path)
            except Exception as exc:
                if args.stop_on_generate_error:
                    raise
                artifact_files = write_iteration_history(output_dir, history_rows)
                pipeline_summary = {"dataset_summary": dataset_summary, "completed_iterations": len(history_rows), "generation_error": str(exc), "model": args.model, "reward_config": reward_config, "artifact_files": artifact_files}
                write_json(output_dir / "pipeline_summary.json", pipeline_summary)
                print(json.dumps(pipeline_summary, ensure_ascii=False, indent=2))
                return 1
            score_rows = score_candidates(candidate_rows, iteration_output_dir, reward_config)
            preference_summary = export_preference_dataset(score_rows, iteration_output_dir, args.min_preference_gap)
            score_summary = summarize_score_rows(score_rows)
            write_json(iteration_output_dir / "score_summary.json", score_summary)
            iteration_metrics = summarize_iteration_metrics(score_rows, iteration)
            write_json(iteration_output_dir / "iteration_metrics.json", iteration_metrics)
            history_rows.append(iteration_metrics)
            iteration_summaries.append({"iteration": iteration, "output_dir": str(iteration_output_dir.resolve()), "generated_candidates_file": str((iteration_output_dir / "generated_candidates.jsonl").resolve()), "candidate_scores_file": str((iteration_output_dir / "candidate_scores.jsonl").resolve()), "score_summary": score_summary, "preference_summary": preference_summary, "iteration_metrics": iteration_metrics})
            print(format_iteration_metrics(iteration_metrics, iteration_total))
        artifact_files = write_iteration_history(output_dir, history_rows)
        pipeline_summary = {"dataset_summary": dataset_summary, "iterations": iteration_total, "iteration_summaries": iteration_summaries, "reward_config": reward_config, "artifact_files": artifact_files}
        write_json(output_dir / "pipeline_summary.json", pipeline_summary)
        print(json.dumps(pipeline_summary, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "rolling-dpo":
        print(json.dumps(run_rolling_dpo(args, examples, output_dir, reward_config), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "plan":
        plan = render_training_plan(examples, output_dir, args.model)
        write_text(output_dir / "training_plan.md", plan)
        print(plan)
        return 0
    print(f"Unsupported mode: {args.mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
