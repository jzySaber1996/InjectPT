#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import optimize_skill_graph_model as opt
import train_skill_graph_reinforce as reinforce_lib
import validate_skill_graphs as graph_validator

CODE_FENCE_RE = re.compile(r"^\s*```(?:mermaid)?\s*$", re.IGNORECASE)
MERMAID_HEADER_RE = re.compile(r"^\s*graph\s+TD\b", re.IGNORECASE)
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class PromptExample:
    skill_id: str
    prompt: str
    reference_graph: str
    skill_dir: str = ""
    original_prompt: str = ""
    full_reference_graph: str = ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # JSONL should be split only on literal line feeds. Using splitlines() is too broad
    # and will incorrectly split records that contain Unicode line separators inside JSON strings.
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_prompt_examples(dataset_file: Path, limit: int = 0) -> list[PromptExample]:
    rows = read_jsonl(dataset_file)
    examples = [
        PromptExample(
            skill_id=row.get("skill_id", ""),
            prompt=row.get("instruction", ""),
            reference_graph=row.get("output") or row.get("reference_graph", ""),
            skill_dir=row.get("skill_dir", ""),
            original_prompt=row.get("instruction", ""),
            full_reference_graph=row.get("output") or row.get("reference_graph", ""),
        )
        for row in rows
    ]
    if limit > 0:
        return examples[:limit]
    return examples


def normalize_graph_text(raw_text: str) -> str:
    cleaned_lines = []
    for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if CODE_FENCE_RE.match(line):
            continue
        cleaned_lines.append(line.rstrip())
    if not cleaned_lines:
        return ""

    header_index = None
    for index, line in enumerate(cleaned_lines):
        if MERMAID_HEADER_RE.match(line):
            header_index = index
            break

    if header_index is None:
        return "\n".join(cleaned_lines).strip()

    normalized = ["graph TD"]
    for line in cleaned_lines[header_index + 1 :]:
        if MERMAID_HEADER_RE.match(line):
            break
        normalized.append(line)
    return "\n".join(normalized).strip()


def format_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _sanitize_adapter_config(adapter_path: Path) -> Path:
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return adapter_path

    from peft import LoraConfig

    config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    signature = inspect.signature(LoraConfig.__init__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return adapter_path

    allowed = {name for name in signature.parameters if name != "self"}
    sanitized = {key: value for key, value in config.items() if key in allowed or key == "peft_type"}
    if sanitized == config:
        return adapter_path

    temp_dir = Path(tempfile.mkdtemp(prefix="skill-graph-peft-compat-"))
    for source in adapter_path.iterdir():
        target = temp_dir / source.name
        if source.name == "adapter_config.json":
            continue
        if source.is_file():
            target.write_bytes(source.read_bytes())
    (temp_dir / "adapter_config.json").write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
    return temp_dir


def _enable_lora_training(model: Any) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def load_policy_model(
    *,
    model_name_or_path: str,
    adapter_path: str = "",
    torch_dtype: str = "bfloat16",
    device: str = "cuda",
    trust_remote_code: bool = False,
    trainable: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    dtype_value = opt.resolve_hf_torch_dtype(torch_dtype)
    resolved_model_path = str(Path(model_name_or_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        torch_dtype=dtype_value,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)

    if adapter_path:
        compatible_path = _sanitize_adapter_config(Path(adapter_path).expanduser().resolve())
        try:
            model = PeftModel.from_pretrained(model, str(compatible_path), is_trainable=trainable)
        except TypeError:
            model = PeftModel.from_pretrained(model, str(compatible_path))
            if trainable:
                _enable_lora_training(model)
    elif trainable:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules or list(DEFAULT_TARGET_MODULES),
        )
        model = get_peft_model(model, config)

    if trainable:
        if hasattr(model, 'config'):
            model.config.use_cache = False
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        if hasattr(model, 'gradient_checkpointing_enable'):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            except TypeError:
                model.gradient_checkpointing_enable()
    model.train(mode=trainable)
    if not trainable:
        for param in model.parameters():
            param.requires_grad = False
    return tokenizer, model


def generate_text(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
    cutoff_len: int = 4096,
) -> tuple[str, torch.Tensor, int]:
    formatted_prompt = format_prompt(tokenizer, prompt)
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cutoff_len,
    )
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    do_sample = temperature > 0
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    was_training = bool(getattr(model, 'training', False))
    if was_training:
        model.eval()
    try:
        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
    finally:
        if was_training:
            model.train()
    generated = output_ids[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, output_ids[0].detach(), int(input_ids.shape[-1])


def score_graph_text(skill_id: str, graph_text: str, reward_config: dict[str, float], temp_dir: Path) -> dict[str, Any]:
    normalized = normalize_graph_text(graph_text)
    graph_label = str((temp_dir / f"{skill_id}.md").resolve())
    try:
        report = graph_validator.validate_graph_data(
            skill_id=skill_id,
            graph_text=normalized,
            graph_label=graph_label,
        )
        report_dict = graph_validator.report_to_dict(report)
    except Exception as exc:
        report_dict = {
            "skill_id": skill_id,
            "graph_file": graph_label,
            "status": "fail",
            "checks": {},
            "issues": [{"severity": "error", "code": "exception", "message": str(exc)}],
            "witness_path": None,
        }
    reward, breakdown = opt.compute_reward(report_dict, reward_config)
    return {
        "skill_id": skill_id,
        "graph_text": normalized,
        "report": report_dict,
        "reward": float(reward),
        "reward_breakdown": breakdown,
    }


def summarize_scored_rows(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored_rows:
        return {
            "candidate_count": 0,
            "avg_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
            "status_counts": {},
            "z3_pass_rate": 0.0,
            "path_exists_rate": 0.0,
            "pass_rate": 0.0,
            "pass_with_warnings_rate": 0.0,
            "exact_match_rate": 0.0,
            "graph_td_prefix_rate": 0.0,
            "edge_marker_rate": 0.0,
        }

    rewards = [float(row["reward"]) for row in scored_rows]
    candidate_count = len(scored_rows)
    status_counts: dict[str, int] = {}
    z3_pass = 0
    path_exists = 0
    pass_count = 0
    pass_with_warnings = 0
    exact_match = 0
    graph_td_prefix = 0
    edge_marker = 0
    for row in scored_rows:
        status = row["report"].get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        graph_text = str(row.get("graph_text", "") or "")
        if reinforce_lib.has_graph_td_prefix(graph_text):
            graph_td_prefix += 1
        if reinforce_lib.has_edge_marker(graph_text):
            edge_marker += 1
        if row["report"].get("witness_path"):
            z3_pass += 1
        if row["report"].get("checks", {}).get("path_exists"):
            path_exists += 1
        if status == "pass":
            pass_count += 1
        elif status == "pass_with_warnings":
            pass_with_warnings += 1
        if normalize_graph_text(row.get("reference_graph", "")).strip() == row.get("graph_text", "").strip():
            exact_match += 1

    return {
        "candidate_count": candidate_count,
        "avg_reward": round(sum(rewards) / candidate_count, 4),
        "max_reward": round(max(rewards), 4),
        "min_reward": round(min(rewards), 4),
        "status_counts": status_counts,
        "z3_pass_rate": round(z3_pass / candidate_count, 4),
        "path_exists_rate": round(path_exists / candidate_count, 4),
        "pass_rate": round(pass_count / candidate_count, 4),
        "pass_with_warnings_rate": round(pass_with_warnings / candidate_count, 4),
        "exact_match_rate": round(exact_match / candidate_count, 4),
        "graph_td_prefix_rate": round(graph_td_prefix / candidate_count, 4),
        "edge_marker_rate": round(edge_marker / candidate_count, 4),
    }
