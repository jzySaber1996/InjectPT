#!/usr/bin/env python3
"""Analyze indirect tool-instruction-injection propagation with DeepSeek.

The script pairs each normalized skill's `graph_wo_check.md` with its
`SKILL.md`, asks DeepSeek for a defensive threat analysis, then asks DeepSeek
again to simulate threat propagation chains through the workflow graph.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from framework_utils import (
    FRAMEWORK_SLOT_METRIC,
    STRATEGY_STEP_METRIC,
    evaluate_result_coverage,
    normalize_propagation_schema,
    validate_framework_definition,
    validate_parsing_strategy,
)
from z3_reward_evaluator import evaluate_z3_reward
from openclaw_runtime_evaluator import (
    add_openclaw_runtime_args,
    evaluate_openclaw_runtime_reward,
    openclaw_config_from_args,
)


DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_API_URL = os.environ.get(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/v1/chat/completions",
)
SKILL_MARKDOWN_CANDIDATES = ("SKILL.md", "SKILLS.md")
DEFAULT_GRAPH_NAME = "graph_wo_check.md"
EVOLUTION_MAX_ADDITIONS_PER_FIELD = 1
EVOLUTION_MAX_TOTAL_ADDITIONS = 2
EVOLUTION_MAX_PRUNES_PER_FIELD = 1
EVOLUTION_MAX_TOTAL_PRUNES = 1
EVOLUTION_MAX_TEXT_UPDATES_PER_FIELD = 1
EVOLUTION_MAX_TOTAL_TEXT_UPDATES = 1
PROTECTED_REQUIRED_STAGE_IDS = {
    "attacked_tool",
    "tainted_output",
    "trust_boundary_crossing",
    "impact_or_containment",
}
PROTECTED_GATE_IDS = {
    "untrusted_input_gate",
    "sanitization_gate",
    "trust_promotion_gate",
    "permission_gate",
    "state_persistence_gate",
    "external_effect_gate",
    "containment_gate",
    "no_external_source_gate",
}
PROTECTED_IMPACT_DIMENSIONS = {
    "confidentiality",
    "integrity",
    "availability",
    "authority",
    "business",
    "no_impact",
}
UNDERSTANDABILITY_METRIC_KEYS = (
    "deepseek_understandability_score",
    "semantic_alignment_score",
    "information_coverage_score",
    "redundancy_score",
    "compression_need_score",
)
NODE_DEF_RE = re.compile(r"\b([A-Za-z0-9_]+)\[([^\]]*)\]")
EDGE_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*-->\s*(?:\|([^|]*)\|\s*)?"
    r"([A-Za-z0-9_]+)(?:\[[^\]]*\])?\s*$"
)
STRUCTURED_FIELD_RE = re.compile(
    r"\b(name|task|input|output|constraint)\s*:\s*(.*?)(?=(?:;\s*"
    r"(?:name|task|input|output|constraint)\s*:)|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ThreatRecord:
    skill_id: str
    skill_dir: Path
    skill_path: Path
    graph_path: Path
    manifest_path: Path | None


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        api_url: str = DEFAULT_API_URL,
        *,
        retries: int = 3,
        retry_seconds: float = 8.0,
    ):
        if not api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY.")
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.retries = max(0, retries)
        self.retry_seconds = max(0.0, retry_seconds)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        request_label: str = "chat",
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        prompt_chars = len(system_prompt) + len(user_prompt)
        transient_statuses = {429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                delay = self.retry_seconds * (2**attempt)
                print(
                    f"[deepseek-retry] label={request_label} error={exc} "
                    f"prompt_chars={prompt_chars} attempt={attempt + 1}/{self.retries + 1}; "
                    f"sleep={delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue

            status = response.status_code
            response_body = sanitize_label(response.text)[:500]
            if status in transient_statuses and attempt < self.retries:
                retry_after = response.headers.get("Retry-After", "")
                delay = self.retry_seconds * (2**attempt)
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                print(
                    f"[deepseek-retry] label={request_label} status={status} "
                    f"prompt_chars={prompt_chars} attempt={attempt + 1}/{self.retries + 1}; "
                    f"sleep={delay:.1f}s body={response_body!r}",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise requests.HTTPError(
                    f"{exc}; label={request_label}; prompt_chars={prompt_chars}; "
                    f"response_body={response_body!r}",
                    response=response,
                ) from exc

            data = response.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        if last_error is not None:
            raise last_error
        raise RuntimeError("DeepSeek request failed without response")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(repo_root / "data" / "inner_representation_v2"),
        help="Normalized skill dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(repo_root / "artifacts" / "threat_propagation_analysis"),
        help="Directory for JSON, Mermaid, and run summaries.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=str(repo_root / "prompts" / "threat_security_system_prompt.md"),
        help="System prompt Markdown path.",
    )
    parser.add_argument(
        "--analysis-prompt-file",
        default=str(repo_root / "prompts" / "threat_analysis_prompt.md"),
        help="Threat analysis prompt template path.",
    )
    parser.add_argument(
        "--propagation-prompt-file",
        default=str(repo_root / "prompts" / "threat_propagation_prompt.md"),
        help="Threat propagation prompt template path.",
    )
    parser.add_argument(
        "--framework-file",
        default=str(repo_root / "template" / "threat_propagation_framework.json"),
        help="Simplified threat propagation framework JSON path.",
    )
    parser.add_argument(
        "--parsing-strategy-file",
        "--strategy-file",
        dest="parsing_strategy_file",
        default=str(repo_root / "template" / "threat_propagation_parsing_strategy.json"),
        help="Threat propagation parsing strategy JSON path.",
    )
    parser.add_argument(
        "--max-framework-chars",
        type=int,
        default=12000,
        help="Maximum framework-definition characters sent to DeepSeek.",
    )
    parser.add_argument(
        "--max-strategy-chars",
        type=int,
        default=18000,
        help="Maximum parsing-strategy characters sent to DeepSeek.",
    )
    parser.add_argument(
        "--chain-template-file",
        default=str(repo_root / "template" / "threat_propagation_chain_template.native.json"),
        help="Seed formal threat propagation chain template JSON path. The file is read-only unless --update-chain-template-file is set.",
    )
    parser.add_argument(
        "--chain-template-archive-root",
        default="",
        help=(
            "Directory used to archive evolved chain templates. "
            "Defaults to the directory containing --chain-template-file."
        ),
    )
    parser.add_argument(
        "--template-evolution-prompt-file",
        default=str(repo_root / "prompts" / "threat_template_evolution_prompt.md"),
        help="Prompt template used to evolve the chain template.",
    )
    parser.add_argument(
        "--template-understandability-prompt-file",
        default=str(repo_root / "prompts" / "threat_template_understandability_prompt.md"),
        help="Prompt template used to evaluate DeepSeek understandability of the chain template.",
    )
    parser.add_argument(
        "--evolve-chain-template",
        action="store_true",
        help="Evolve the in-memory chain template after each successful skill analysis.",
    )
    parser.add_argument(
        "--update-chain-template-file",
        action="store_true",
        help="Write evolved templates back to --chain-template-file. By default, the seed template is not modified.",
    )
    parser.add_argument(
        "--skip-template-understandability",
        action="store_true",
        help="Skip the extra DeepSeek call that evaluates template understandability before evolution.",
    )
    parser.add_argument(
        "--max-chain-template-chars",
        type=int,
        default=18000,
        help="Maximum chain-template characters sent to DeepSeek.",
    )
    parser.add_argument(
        "--max-template-evolution-tokens",
        type=int,
        default=2600,
        help="Max completion tokens for template evolution calls.",
    )
    parser.add_argument(
        "--max-template-understandability-tokens",
        type=int,
        default=2600,
        help="Max completion tokens for template understandability evaluation calls.",
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Specific canonical skill id(s) to process. Can be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N matched skills. 0 means no limit.",
    )
    parser.add_argument(
        "--graph-name",
        default=DEFAULT_GRAPH_NAME,
        help="Workflow graph filename inside each skill directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="DeepSeek model name.",
    )
    parser.add_argument(
        "--max-skill-chars",
        type=int,
        default=32000,
        help="Maximum SKILL.md characters sent to DeepSeek per skill.",
    )
    parser.add_argument(
        "--max-graph-chars",
        type=int,
        default=18000,
        help="Maximum graph_wo_check.md characters sent to DeepSeek per skill.",
    )
    parser.add_argument(
        "--max-analysis-tokens",
        type=int,
        default=3200,
        help="Max completion tokens for the threat analysis call.",
    )
    parser.add_argument(
        "--max-chain-tokens",
        type=int,
        default=3200,
        help="Max completion tokens for the propagation-chain call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for both DeepSeek calls.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between skills.",
    )
    parser.add_argument(
        "--deepseek-retries",
        type=int,
        default=3,
        help="Retry count for transient DeepSeek 429/5xx/network errors.",
    )
    parser.add_argument(
        "--deepseek-retry-seconds",
        type=float,
        default=8.0,
        help="Base sleep seconds for exponential DeepSeek retry backoff.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-skill threat analysis outputs.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Store raw DeepSeek responses in the JSON output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover matching graph/SKILL pairs; do not call DeepSeek.",
    )
    add_openclaw_runtime_args(parser)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def resolve_skill_markdown_path(skill_dir: Path) -> Path | None:
    for filename in SKILL_MARKDOWN_CANDIDATES:
        candidate = skill_dir / filename
        if candidate.exists():
            return candidate
    return None


def load_manifest(skill_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    manifest_path = skill_dir / "manifest.json"
    if not manifest_path.exists():
        return None, {}
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return manifest_path, {}
    if not isinstance(manifest, dict):
        return manifest_path, {}
    return manifest_path, manifest


def resolve_graph_path(skill_dir: Path, graph_name: str, manifest: dict[str, Any]) -> Path | None:
    candidates = [graph_name]
    manifest_graph = manifest.get("graphFile")
    if isinstance(manifest_graph, str) and manifest_graph not in candidates:
        candidates.append(manifest_graph)
    for filename in candidates:
        candidate = skill_dir / filename
        if candidate.exists():
            return candidate
    return None


def iter_skill_ids(data_root: Path) -> list[str]:
    index_path = data_root / "index.json"
    if index_path.exists():
        data = load_json(index_path)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {index_path}")
        ids = []
        for item in data:
            if not isinstance(item, dict):
                continue
            skill_id = str(item.get("canonicalId") or "").strip()
            if skill_id:
                ids.append(skill_id)
        return sorted(set(ids))
    return sorted(child.name for child in data_root.iterdir() if child.is_dir())


def collect_records(
    data_root: Path,
    *,
    selected_ids: set[str] | None,
    graph_name: str,
) -> list[ThreatRecord]:
    records: list[ThreatRecord] = []
    for skill_id in iter_skill_ids(data_root):
        if selected_ids and skill_id not in selected_ids:
            continue
        skill_dir = data_root / skill_id
        if not skill_dir.is_dir():
            continue
        manifest_path, manifest = load_manifest(skill_dir)
        skill_path = resolve_skill_markdown_path(skill_dir)
        graph_path = resolve_graph_path(skill_dir, graph_name, manifest)
        if skill_path is None or graph_path is None:
            continue
        records.append(
            ThreatRecord(
                skill_id=skill_id,
                skill_dir=skill_dir,
                skill_path=skill_path,
                graph_path=graph_path,
                manifest_path=manifest_path,
            )
        )
    return sorted(records, key=lambda record: record.skill_id)


def sanitize_label(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_structured_fields(label: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in STRUCTURED_FIELD_RE.finditer(label):
        fields[match.group(1).lower()] = sanitize_label(match.group(2))
    return fields


def parse_mermaid_graph(text: str) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for line in text.splitlines():
        for match in NODE_DEF_RE.finditer(line):
            node_id = match.group(1)
            label = sanitize_label(match.group(2))
            fields = extract_structured_fields(label)
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "name": fields.get("name", ""),
                "task": fields.get("task", ""),
                "input": fields.get("input", ""),
                "output": fields.get("output", ""),
                "constraint": fields.get("constraint", ""),
            }
        edge = EDGE_RE.match(line.strip())
        if edge:
            src, condition, dst = edge.groups()
            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "condition": sanitize_label(condition or ""),
                }
            )
            nodes.setdefault(src, {"id": src, "label": "", "name": ""})
            nodes.setdefault(dst, {"id": dst, "label": "", "name": ""})
    return {
        "nodes": [nodes[node_id] for node_id in sorted(nodes)],
        "edges": edges,
    }


def truncate_text(text: str, max_chars: int, label: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[TRUNCATED {label}: omitted {omitted} chars]"


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_json_fence(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(stripped[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if not isinstance(data, dict):
        raise ValueError("DeepSeek response must be a JSON object")
    return data


def compact_json(data: Any, *, limit: int = 24000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return truncate_text(text, limit, "json")


def render_prompt_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered.strip()


def build_threat_analysis_prompt(
    template: str,
    record: ThreatRecord,
    manifest: dict[str, Any],
    graph_markdown: str,
    skill_markdown: str,
    parsed_graph: dict[str, Any],
    *,
    max_graph_chars: int,
    max_skill_chars: int,
) -> str:
    manifest_summary = {
        "canonicalId": manifest.get("canonicalId", record.skill_id),
        "name": manifest.get("name", record.skill_id),
        "description": manifest.get("description", ""),
        "sourceRelativePath": manifest.get("sourceRelativePath"),
        "copiedDocs": manifest.get("copiedDocs", []),
        "files": manifest.get("files", {}),
    }
    return render_prompt_template(
        template,
        {
            "skill_id": record.skill_id,
            "manifest_summary": compact_json(manifest_summary, limit=6000),
            "parsed_graph": compact_json(parsed_graph, limit=22000),
            "graph_markdown": truncate_text(graph_markdown, max_graph_chars, "graph_wo_check.md"),
            "skill_markdown": truncate_text(skill_markdown, max_skill_chars, "SKILL.md"),
        },
    )


def build_propagation_prompt(
    template: str,
    record: ThreatRecord,
    parsed_graph: dict[str, Any],
    threat_analysis: dict[str, Any],
    framework: dict[str, Any],
    parsing_strategy: dict[str, Any],
    chain_template: str,
    *,
    max_framework_chars: int,
    max_strategy_chars: int,
    max_chain_template_chars: int,
    graph_markdown: str = "",
    max_graph_chars: int = 18000,
) -> str:
    return render_prompt_template(
        template,
        {
            "skill_id": record.skill_id,
            "framework": compact_json(framework, limit=max_framework_chars),
            "strategy": compact_json(parsing_strategy, limit=max_strategy_chars),
            "chain_template": truncate_text(
                chain_template,
                max_chain_template_chars,
                "threat_propagation_chain_template.json",
            ),
            "parsed_graph": compact_json(parsed_graph, limit=22000),
            "graph_markdown": truncate_text(graph_markdown, max_graph_chars, "graph_wo_check.md") if graph_markdown else "",
            "threat_analysis": compact_json(threat_analysis, limit=26000),
        },
    )


def load_chain_template(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Chain template must be formal JSON: {path}")
    template = load_json(path)
    validate_chain_template(template)
    return template


def load_threat_framework(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Threat propagation framework must be JSON: {path}")
    return validate_framework_definition(load_json(path))


def load_threat_parsing_strategy(path: Path, framework: dict[str, Any]) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Threat propagation parsing strategy must be JSON: {path}")
    return validate_parsing_strategy(load_json(path), framework)


def validate_logic_expression(expr: Any, allowed_refs: set[str]) -> None:
    if isinstance(expr, str):
        if expr not in allowed_refs:
            raise ValueError(f"Unknown gate or logic reference in template expression: {expr}")
        return
    if not isinstance(expr, dict):
        raise ValueError(f"Logic expression must be a string ref or object: {expr!r}")
    op = expr.get("op")
    args = expr.get("args")
    if op not in {"AND", "OR", "NOT"}:
        raise ValueError(f"Unsupported logic operator: {op}")
    if not isinstance(args, list):
        raise ValueError(f"Logic expression args must be a list: {expr!r}")
    if op == "NOT" and len(args) != 1:
        raise ValueError(f"NOT gate requires exactly one arg: {expr!r}")
    if op in {"AND", "OR"} and len(args) < 2:
        raise ValueError(f"{op} gate requires at least two args: {expr!r}")
    for arg in args:
        validate_logic_expression(arg, allowed_refs)


def validate_chain_template(template: Any) -> None:
    if not isinstance(template, dict):
        raise ValueError("Chain template must be a JSON object")
    anchors = template.get("anchors")
    if not isinstance(anchors, dict):
        raise ValueError("Chain template missing anchors object")
    for anchor in ("propagation_chain", "gate_units"):
        if not isinstance(anchors.get(anchor), dict):
            raise ValueError(f"Chain template missing anchors.{anchor}")

    propagation_chain = anchors["propagation_chain"]
    required_stages = propagation_chain.get("required_stages")
    if not isinstance(required_stages, list) or not required_stages:
        raise ValueError("anchors.propagation_chain.required_stages must be a non-empty list")
    stage_ids = set()
    for stage in required_stages:
        if not isinstance(stage, dict) or not stage.get("id"):
            raise ValueError("Each propagation stage must be an object with id")
        stage_ids.add(str(stage["id"]))
    for required_stage in ("attacked_tool", "impact_or_containment"):
        if required_stage not in stage_ids:
            raise ValueError(f"Chain template missing required stage: {required_stage}")

    gate_units = anchors["gate_units"]
    operators = gate_units.get("operators")
    if not isinstance(operators, dict):
        raise ValueError("anchors.gate_units.operators must be an object")
    for operator in ("AND", "OR", "NOT"):
        if operator not in operators:
            raise ValueError(f"Chain template missing gate operator: {operator}")

    gate_catalog = gate_units.get("gate_catalog")
    if not isinstance(gate_catalog, list) or not gate_catalog:
        raise ValueError("anchors.gate_units.gate_catalog must be a non-empty list")
    gate_ids = set()
    for gate in gate_catalog:
        if not isinstance(gate, dict) or not gate.get("id"):
            raise ValueError("Each gate_catalog item must be an object with id")
        gate_ids.add(str(gate["id"]))

    logic_templates = gate_units.get("logic_templates")
    if not isinstance(logic_templates, list) or not logic_templates:
        raise ValueError("anchors.gate_units.logic_templates must be a non-empty list")
    logic_ids = {str(item.get("id")) for item in logic_templates if isinstance(item, dict) and item.get("id")}
    allowed_refs = gate_ids | logic_ids
    for logic in logic_templates:
        if not isinstance(logic, dict) or not logic.get("id"):
            raise ValueError("Each logic_templates item must be an object with id")
        validate_logic_expression(logic.get("expression"), allowed_refs)


def render_logic_expression(expr: Any) -> str:
    if isinstance(expr, str):
        return expr
    if isinstance(expr, dict):
        op = str(expr.get("op"))
        args = expr.get("args") if isinstance(expr.get("args"), list) else []
        rendered_args = ", ".join(render_logic_expression(arg) for arg in args)
        return f"{op}({rendered_args})"
    return str(expr)


def render_formal_chain_template(template: dict[str, Any]) -> str:
    validate_chain_template(template)
    anchors = template["anchors"]
    propagation_chain = anchors["propagation_chain"]
    gate_units = anchors["gate_units"]

    lines = [
        "# Formal Threat Propagation Chain Template",
        f"template_id: {template.get('template_id', '')}",
        f"schema_version: {template.get('schema_version', '')}",
        "",
        "## Anchor: propagation_chain",
        f"purpose: {propagation_chain.get('purpose', '')}",
        f"start_rule: {propagation_chain.get('start_rule', '')}",
        f"terminal_rule: {propagation_chain.get('terminal_rule', '')}",
        "required_stages:",
    ]
    for stage in propagation_chain.get("required_stages", []):
        lines.append(
            "- "
            f"id={stage.get('id')}; required={stage.get('required')}; "
            f"role={stage.get('role')}; description={stage.get('description', '')}"
        )

    lines.extend(["", "edge_requirements:"])
    for item in propagation_chain.get("edge_requirements", []):
        lines.append(f"- {item}")

    lines.extend(["", "impact_dimensions:"])
    for item in propagation_chain.get("impact_dimensions", []):
        lines.append(f"- {item}")

    lines.extend(["", "mermaid_requirements:"])
    for item in propagation_chain.get("mermaid_requirements", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Anchor: gate_units", f"purpose: {gate_units.get('purpose', '')}", "operators:"])
    for operator, details in gate_units.get("operators", {}).items():
        meaning = details.get("meaning", "") if isinstance(details, dict) else ""
        lines.append(f"- {operator}: {meaning}")

    lines.extend(["", "gate_catalog:"])
    for gate in gate_units.get("gate_catalog", []):
        lines.append(
            "- "
            f"id={gate.get('id')}; question={gate.get('question', '')}; "
            f"true_effect={gate.get('true_effect', '')}"
        )

    lines.extend(["", "logic_templates:"])
    for logic in gate_units.get("logic_templates", []):
        lines.append(
            "- "
            f"id={logic.get('id')}; description={logic.get('description', '')}; "
            f"expression={render_logic_expression(logic.get('expression'))}"
        )

    lines.extend([
        "",
        "## Output Contract",
        compact_json(template.get("output_contract", {}), limit=8000),
        "",
        "## Formal JSON",
        "```json",
        json.dumps(template, ensure_ascii=False, indent=2),
        "```",
    ])
    return "\n".join(lines).strip()


def build_template_evolution_prompt(
    template: str,
    *,
    skill_id: str,
    chain_template: dict[str, Any],
    threat_analysis: dict[str, Any],
    propagation: dict[str, Any],
    mermaid: str,
    template_understandability: dict[str, Any] | None = None,
) -> str:
    return render_prompt_template(
        template,
        {
            "skill_id": skill_id,
            "chain_template": compact_json(chain_template, limit=30000),
            "threat_analysis": compact_json(threat_analysis, limit=24000),
            "propagation": compact_json(propagation, limit=24000),
            "mermaid": mermaid,
            "template_understandability": compact_json(
                template_understandability or {},
                limit=18000,
            ),
        },
    )


def build_template_understandability_prompt(
    template: str,
    *,
    skill_id: str,
    chain_template: dict[str, Any],
    threat_analysis: dict[str, Any],
    propagation: dict[str, Any],
    mermaid: str,
) -> str:
    return render_prompt_template(
        template,
        {
            "skill_id": skill_id,
            "chain_template": compact_json(chain_template, limit=30000),
            "threat_analysis": compact_json(threat_analysis, limit=24000),
            "propagation": compact_json(propagation, limit=24000),
            "mermaid": mermaid,
        },
    )

def safe_template_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if re.fullmatch(r"[a-z][a-z0-9_]{2,80}", value):
        return value
    return ""


def safe_template_text(value: Any, *, max_chars: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    text = sanitize_label(value)
    if not text or "```" in text:
        return ""
    return text[:max_chars]


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(1.0, max(0.0, score)), 4)


def list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_template_understandability_report(
    report: dict[str, Any],
    *,
    skill_id: str,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        report = {}
    normalized = dict(report)
    normalized["skill_id"] = str(report.get("skill_id") or skill_id)
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    normalized["metrics"] = {
        key: clamp_score(metrics.get(key, report.get(key)))
        for key in UNDERSTANDABILITY_METRIC_KEYS
    }
    for key in (
        "understood_information",
        "misunderstood_information",
        "not_understood_information",
        "not_applicable_information",
        "redundant_information",
        "semantic_alignment",
    ):
        normalized[key] = list_or_empty(report.get(key))
    suggestions = report.get("optimization_suggestions")
    normalized["optimization_suggestions"] = suggestions if isinstance(suggestions, dict) else {}
    normalized["summary"] = safe_template_text(report.get("summary"), max_chars=1000)
    return normalized


def summarize_template_understandability(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    return {
        "template_understandability_evaluated": True,
        "deepseek_understandability_score": metrics.get("deepseek_understandability_score", 0.0),
        "semantic_alignment_score": metrics.get("semantic_alignment_score", 0.0),
        "information_coverage_score": metrics.get("information_coverage_score", 0.0),
        "redundancy_score": metrics.get("redundancy_score", 0.0),
        "compression_need_score": metrics.get("compression_need_score", 0.0),
        "understood_information_count": len(list_or_empty(report.get("understood_information"))),
        "misunderstood_information_count": len(list_or_empty(report.get("misunderstood_information"))),
        "not_understood_information_count": len(list_or_empty(report.get("not_understood_information"))),
        "redundant_information_count": len(list_or_empty(report.get("redundant_information"))),
    }


def limited_evolution_items(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:EVOLUTION_MAX_ADDITIONS_PER_FIELD]


def limited_pruning_items(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:EVOLUTION_MAX_PRUNES_PER_FIELD]


def limited_text_update_items(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:EVOLUTION_MAX_TEXT_UPDATES_PER_FIELD]


def existing_item_ids(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()
    return {str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")}


def evolution_additions(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("has_updates") is False:
        return {}
    additions = response.get("additions")
    if isinstance(additions, dict):
        return additions
    return response


def evolution_pruning(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("has_updates") is False:
        return {}
    pruning = response.get("pruning")
    return pruning if isinstance(pruning, dict) else {}


def evolution_text_updates(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("has_updates") is False:
        return {}
    updates = response.get("text_updates")
    return updates if isinstance(updates, dict) else {}


def select_limited_evolution_additions(additions: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    total = 0
    for key in (
        "required_stages",
        "gate_catalog",
        "logic_templates",
        "impact_dimensions",
        "mermaid_requirements",
    ):
        if total >= EVOLUTION_MAX_TOTAL_ADDITIONS:
            break
        items = limited_evolution_items(additions.get(key))
        if not items:
            continue
        remaining = EVOLUTION_MAX_TOTAL_ADDITIONS - total
        selected[key] = items[:remaining]
        total += len(selected[key])
    if total < EVOLUTION_MAX_TOTAL_ADDITIONS:
        output_contract = additions.get("output_contract")
        if isinstance(output_contract, dict) and output_contract:
            selected["output_contract"] = dict(
                list(output_contract.items())[: EVOLUTION_MAX_TOTAL_ADDITIONS - total]
            )
    return selected


def select_limited_evolution_pruning(pruning: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    total = 0
    for key in (
        "logic_template_ids",
        "gate_ids",
        "required_stage_ids",
        "impact_dimensions",
        "mermaid_requirements",
        "output_contract_keys",
    ):
        if total >= EVOLUTION_MAX_TOTAL_PRUNES:
            break
        items = limited_pruning_items(pruning.get(key))
        if not items:
            continue
        remaining = EVOLUTION_MAX_TOTAL_PRUNES - total
        selected[key] = items[:remaining]
        total += len(selected[key])
    return selected


def select_limited_evolution_text_updates(updates: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    total = 0
    for key in ("required_stages", "gate_catalog", "logic_templates", "mermaid_requirements"):
        if total >= EVOLUTION_MAX_TOTAL_TEXT_UPDATES:
            break
        items = limited_text_update_items(updates.get(key))
        if not items:
            continue
        remaining = EVOLUTION_MAX_TOTAL_TEXT_UPDATES - total
        selected[key] = items[:remaining]
        total += len(selected[key])
    if total < EVOLUTION_MAX_TOTAL_TEXT_UPDATES:
        output_contract = updates.get("output_contract")
        if isinstance(output_contract, dict) and output_contract:
            selected["output_contract"] = dict(
                list(output_contract.items())[: EVOLUTION_MAX_TOTAL_TEXT_UPDATES - total]
            )
    return selected

def append_required_stage(template: dict[str, Any], additions: dict[str, Any]) -> None:
    stages = template["anchors"]["propagation_chain"].setdefault("required_stages", [])
    known_ids = existing_item_ids(stages)
    for item in limited_evolution_items(additions.get("required_stages")):
        if not isinstance(item, dict):
            continue
        stage_id = safe_template_id(item.get("id"))
        description = safe_template_text(item.get("description"))
        if not stage_id or stage_id in known_ids or not description:
            continue
        role = safe_template_text(item.get("role"), max_chars=80) or "propagation"
        stages.append(
            {
                "id": stage_id,
                "required": bool(item.get("required", False)),
                "role": role,
                "description": description,
            }
        )
        known_ids.add(stage_id)


def append_gate_unit(template: dict[str, Any], additions: dict[str, Any]) -> None:
    gate_catalog = template["anchors"]["gate_units"].setdefault("gate_catalog", [])
    known_ids = existing_item_ids(gate_catalog)
    for item in limited_evolution_items(additions.get("gate_catalog")):
        if not isinstance(item, dict):
            continue
        gate_id = safe_template_id(item.get("id"))
        question = safe_template_text(item.get("question"))
        true_effect = safe_template_text(item.get("true_effect"))
        if not gate_id or gate_id in known_ids or not question or not true_effect:
            continue
        gate_catalog.append({"id": gate_id, "question": question, "true_effect": true_effect})
        known_ids.add(gate_id)


def append_logic_template(template: dict[str, Any], additions: dict[str, Any]) -> None:
    gate_units = template["anchors"]["gate_units"]
    gate_catalog = gate_units.setdefault("gate_catalog", [])
    logic_templates = gate_units.setdefault("logic_templates", [])
    gate_ids = existing_item_ids(gate_catalog)
    logic_ids = existing_item_ids(logic_templates)
    for item in limited_evolution_items(additions.get("logic_templates")):
        if not isinstance(item, dict):
            continue
        logic_id = safe_template_id(item.get("id"))
        description = safe_template_text(item.get("description"))
        expression = item.get("expression")
        if not logic_id or logic_id in logic_ids or not description:
            continue
        try:
            validate_logic_expression(expression, gate_ids | logic_ids | {logic_id})
        except ValueError:
            continue
        logic_templates.append(
            {"id": logic_id, "description": description, "expression": expression}
        )
        logic_ids.add(logic_id)


def append_impact_dimension(template: dict[str, Any], additions: dict[str, Any]) -> None:
    dimensions = template["anchors"]["propagation_chain"].setdefault("impact_dimensions", [])
    known = {str(item) for item in dimensions}
    for item in limited_evolution_items(additions.get("impact_dimensions")):
        dimension = safe_template_id(item)
        if dimension and dimension not in known:
            dimensions.append(dimension)
            known.add(dimension)


def append_mermaid_requirement(template: dict[str, Any], additions: dict[str, Any]) -> None:
    requirements = template["anchors"]["propagation_chain"].setdefault("mermaid_requirements", [])
    known = {str(item) for item in requirements}
    for item in limited_evolution_items(additions.get("mermaid_requirements")):
        requirement = safe_template_text(item, max_chars=300)
        if requirement and requirement not in known:
            requirements.append(requirement)
            known.add(requirement)


def append_output_contract_rule(template: dict[str, Any], additions: dict[str, Any]) -> None:
    updates = additions.get("output_contract")
    if not isinstance(updates, dict):
        return
    contract = template.setdefault("output_contract", {})
    if not isinstance(contract, dict):
        return
    added = 0
    for key, value in updates.items():
        if added >= EVOLUTION_MAX_ADDITIONS_PER_FIELD:
            break
        safe_key = safe_template_id(key)
        safe_value = safe_template_text(value)
        if not safe_key or safe_key in contract or not safe_value:
            continue
        contract[safe_key] = safe_value
        added += 1


def logic_expression_refs(expr: Any) -> set[str]:
    if isinstance(expr, str):
        return {expr}
    if not isinstance(expr, dict):
        return set()
    args = expr.get("args")
    if not isinstance(args, list):
        return set()
    refs: set[str] = set()
    for arg in args:
        refs.update(logic_expression_refs(arg))
    return refs


def prune_required_stages(template: dict[str, Any], pruning: dict[str, Any]) -> None:
    target_ids = {safe_template_id(item) for item in pruning.get("required_stage_ids", [])}
    target_ids.discard("")
    if not target_ids:
        return
    stages = template["anchors"]["propagation_chain"].setdefault("required_stages", [])
    kept = []
    for stage in stages:
        if not isinstance(stage, dict):
            kept.append(stage)
            continue
        stage_id = str(stage.get("id") or "")
        must_keep = stage.get("required") is True or stage_id in PROTECTED_REQUIRED_STAGE_IDS
        if stage_id in target_ids and not must_keep:
            continue
        kept.append(stage)
    template["anchors"]["propagation_chain"]["required_stages"] = kept


def prune_gate_units(template: dict[str, Any], pruning: dict[str, Any]) -> None:
    gate_units = template["anchors"]["gate_units"]
    logic_templates = gate_units.setdefault("logic_templates", [])
    target_logic_ids = {safe_template_id(item) for item in pruning.get("logic_template_ids", [])}
    target_logic_ids.discard("")
    if target_logic_ids:
        referenced_by_kept: set[str] = set()
        for logic in logic_templates:
            if not isinstance(logic, dict):
                continue
            logic_id = str(logic.get("id") or "")
            if logic_id in target_logic_ids:
                continue
            referenced_by_kept.update(logic_expression_refs(logic.get("expression")))
        gate_units["logic_templates"] = [
            logic
            for logic in logic_templates
            if not (
                isinstance(logic, dict)
                and str(logic.get("id") or "") in target_logic_ids
                and str(logic.get("id") or "") not in referenced_by_kept
            )
        ]

    target_gate_ids = {safe_template_id(item) for item in pruning.get("gate_ids", [])}
    target_gate_ids.discard("")
    if not target_gate_ids:
        return
    referenced_refs: set[str] = set()
    for logic in gate_units.get("logic_templates", []):
        if isinstance(logic, dict):
            referenced_refs.update(logic_expression_refs(logic.get("expression")))
    gate_catalog = gate_units.setdefault("gate_catalog", [])
    gate_units["gate_catalog"] = [
        gate
        for gate in gate_catalog
        if not (
            isinstance(gate, dict)
            and str(gate.get("id") or "") in target_gate_ids
            and str(gate.get("id") or "") not in PROTECTED_GATE_IDS
            and str(gate.get("id") or "") not in referenced_refs
        )
    ]


def prune_impact_dimensions(template: dict[str, Any], pruning: dict[str, Any]) -> None:
    target_ids = {safe_template_id(item) for item in pruning.get("impact_dimensions", [])}
    target_ids.discard("")
    if not target_ids:
        return
    dimensions = template["anchors"]["propagation_chain"].setdefault("impact_dimensions", [])
    template["anchors"]["propagation_chain"]["impact_dimensions"] = [
        item
        for item in dimensions
        if not (str(item) in target_ids and str(item) not in PROTECTED_IMPACT_DIMENSIONS)
    ]


def prune_mermaid_requirements(template: dict[str, Any], pruning: dict[str, Any]) -> None:
    targets = {safe_template_text(item, max_chars=300) for item in pruning.get("mermaid_requirements", [])}
    targets.discard("")
    if not targets:
        return
    requirements = template["anchors"]["propagation_chain"].setdefault("mermaid_requirements", [])
    template["anchors"]["propagation_chain"]["mermaid_requirements"] = [
        item for item in requirements if sanitize_label(str(item)) not in targets
    ]


def prune_output_contract_rules(template: dict[str, Any], pruning: dict[str, Any]) -> None:
    target_keys = {safe_template_id(item) for item in pruning.get("output_contract_keys", [])}
    target_keys.discard("")
    if not target_keys:
        return
    contract = template.setdefault("output_contract", {})
    if not isinstance(contract, dict):
        return
    for key in list(contract):
        if key in target_keys:
            contract.pop(key, None)


def apply_evolution_pruning(template: dict[str, Any], pruning: dict[str, Any]) -> None:
    prune_gate_units(template, pruning)
    prune_required_stages(template, pruning)
    prune_impact_dimensions(template, pruning)
    prune_mermaid_requirements(template, pruning)
    prune_output_contract_rules(template, pruning)


def update_required_stage_text(template: dict[str, Any], updates: dict[str, Any]) -> None:
    stages = template["anchors"]["propagation_chain"].setdefault("required_stages", [])
    for item in limited_text_update_items(updates.get("required_stages")):
        if not isinstance(item, dict):
            continue
        stage_id = safe_template_id(item.get("id"))
        description = safe_template_text(item.get("description"))
        role = safe_template_text(item.get("role"), max_chars=80)
        if not stage_id:
            continue
        for stage in stages:
            if not isinstance(stage, dict) or str(stage.get("id") or "") != stage_id:
                continue
            if description:
                stage["description"] = description
            if role:
                stage["role"] = role
            break


def update_gate_text(template: dict[str, Any], updates: dict[str, Any]) -> None:
    gates = template["anchors"]["gate_units"].setdefault("gate_catalog", [])
    for item in limited_text_update_items(updates.get("gate_catalog")):
        if not isinstance(item, dict):
            continue
        gate_id = safe_template_id(item.get("id"))
        question = safe_template_text(item.get("question"))
        true_effect = safe_template_text(item.get("true_effect"))
        if not gate_id:
            continue
        for gate in gates:
            if not isinstance(gate, dict) or str(gate.get("id") or "") != gate_id:
                continue
            if question:
                gate["question"] = question
            if true_effect:
                gate["true_effect"] = true_effect
            break


def update_logic_template_text(template: dict[str, Any], updates: dict[str, Any]) -> None:
    logic_templates = template["anchors"]["gate_units"].setdefault("logic_templates", [])
    for item in limited_text_update_items(updates.get("logic_templates")):
        if not isinstance(item, dict):
            continue
        logic_id = safe_template_id(item.get("id"))
        description = safe_template_text(item.get("description"))
        if not logic_id or not description:
            continue
        for logic in logic_templates:
            if isinstance(logic, dict) and str(logic.get("id") or "") == logic_id:
                logic["description"] = description
                break


def update_mermaid_requirement_text(template: dict[str, Any], updates: dict[str, Any]) -> None:
    requirements = template["anchors"]["propagation_chain"].setdefault("mermaid_requirements", [])
    for item in limited_text_update_items(updates.get("mermaid_requirements")):
        if not isinstance(item, dict):
            continue
        current = safe_template_text(item.get("current"), max_chars=300)
        replacement = safe_template_text(item.get("replacement"), max_chars=300)
        if not current or not replacement:
            continue
        for index, requirement in enumerate(requirements):
            if sanitize_label(str(requirement)) == current:
                requirements[index] = replacement
                break


def update_output_contract_text(template: dict[str, Any], updates: dict[str, Any]) -> None:
    contract_updates = updates.get("output_contract")
    if not isinstance(contract_updates, dict):
        return
    contract = template.setdefault("output_contract", {})
    if not isinstance(contract, dict):
        return
    for key, value in list(contract_updates.items())[:EVOLUTION_MAX_TEXT_UPDATES_PER_FIELD]:
        safe_key = safe_template_id(key)
        safe_value = safe_template_text(value)
        if safe_key in contract and safe_value:
            contract[safe_key] = safe_value


def apply_evolution_text_updates(template: dict[str, Any], updates: dict[str, Any]) -> None:
    update_required_stage_text(template, updates)
    update_gate_text(template, updates)
    update_logic_template_text(template, updates)
    update_mermaid_requirement_text(template, updates)
    update_output_contract_text(template, updates)


def apply_evolution_delta(
    current_chain_template: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    evolved = deepcopy(current_chain_template)
    validate_chain_template(evolved)

    additions = evolution_additions(response)
    if isinstance(additions, dict):
        additions = select_limited_evolution_additions(additions)
        append_required_stage(evolved, additions)
        append_gate_unit(evolved, additions)
        append_logic_template(evolved, additions)
        append_impact_dimension(evolved, additions)
        append_mermaid_requirement(evolved, additions)
        append_output_contract_rule(evolved, additions)

    text_updates = evolution_text_updates(response)
    if isinstance(text_updates, dict):
        text_updates = select_limited_evolution_text_updates(text_updates)
        apply_evolution_text_updates(evolved, text_updates)

    pruning = evolution_pruning(response)
    if isinstance(pruning, dict):
        pruning = select_limited_evolution_pruning(pruning)
        apply_evolution_pruning(evolved, pruning)

    validate_chain_template(evolved)
    return evolved

def normalize_evolved_template(
    current_chain_template: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(response.get("anchors"), dict):
        validate_chain_template(response)
        return response
    return apply_evolution_delta(current_chain_template, response)


def evaluate_template_understandability(
    client: DeepSeekClient,
    *,
    system_prompt_template: str,
    understandability_prompt_template: str,
    current_chain_template: dict[str, Any],
    result: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    skill_id = str(result.get("skill_id") or "")
    prompt = build_template_understandability_prompt(
        understandability_prompt_template,
        skill_id=skill_id,
        chain_template=current_chain_template,
        threat_analysis=result.get("threat_analysis", {}),
        propagation=result.get("propagation", {}),
        mermaid=str(result.get("mermaid") or ""),
    )
    raw_output = client.complete(
        system_prompt_template,
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        request_label=f"{skill_id or 'unknown'}:template-understandability",
    )
    response = parse_json_object(raw_output)
    return normalize_template_understandability_report(response, skill_id=skill_id), raw_output


def evolve_chain_template(
    client: DeepSeekClient,
    *,
    system_prompt_template: str,
    evolution_prompt_template: str,
    current_chain_template: dict[str, Any],
    result: dict[str, Any],
    template_understandability: dict[str, Any] | None,
    max_tokens: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    prompt = build_template_evolution_prompt(
        evolution_prompt_template,
        skill_id=str(result.get("skill_id") or ""),
        chain_template=current_chain_template,
        threat_analysis=result.get("threat_analysis", {}),
        propagation=result.get("propagation", {}),
        mermaid=str(result.get("mermaid") or ""),
        template_understandability=template_understandability,
    )
    raw_output = client.complete(
        system_prompt_template,
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        request_label=f"{result.get('skill_id') or 'unknown'}:template-evolution",
    )
    response = parse_json_object(raw_output)
    evolved = normalize_evolved_template(current_chain_template, response)
    return evolved, raw_output, response

def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip(".-") or "template"


def archive_chain_template(
    chain_template_file: Path,
    template: dict[str, Any],
    *,
    archive_dir: Path,
    skill_id: str,
) -> Path:
    safe_skill_id = sanitize_filename(skill_id)
    archive_name = f"{safe_skill_id}_{chain_template_file.name}"
    archive_path = archive_dir / archive_name
    suffix = 2
    while archive_path.exists():
        archive_path = archive_dir / f"{safe_skill_id}_{suffix}_{chain_template_file.name}"
        suffix += 1
    write_json(archive_path, template)
    return archive_path

def extract_mermaid(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:mermaid)?\s*(graph\s+TD[\s\S]*?)```", stripped, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    graph_match = re.search(r"(graph\s+TD[\s\S]*)", stripped, re.IGNORECASE)
    if graph_match:
        return graph_match.group(1).strip()
    return stripped


def valid_mermaid(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return bool(re.match(r"^graph\s+TD\b", lines[0], re.IGNORECASE))


def mermaid_label(text: str, *, max_len: int = 120) -> str:
    text = sanitize_label(text)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return json.dumps(text, ensure_ascii=False)


def sanitize_edge_label(text: str, *, max_len: int = 90) -> str:
    text = sanitize_label(text).replace("|", "/")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text or "propagates"


def build_start_step(chain: dict[str, Any]) -> dict[str, Any]:
    starting_point = chain.get("starting_attack_point")
    if not isinstance(starting_point, dict):
        starting_point = {}
    return {
        "node_id": str(starting_point.get("node_id") or "attack_entry"),
        "node_name": str(starting_point.get("node_name") or starting_point.get("node_id") or "unknown_tool"),
        "role": "entry",
        "threat_state": "被攻击工具接收或生成带有间接指令注入风险的内容",
        "reason": str(starting_point.get("reason") or "该节点是传播链的起始攻击点"),
    }


def normalize_chain_path(chain: dict[str, Any]) -> list[dict[str, Any]]:
    start_step = build_start_step(chain)
    path = chain.get("path")
    if not isinstance(path, list):
        path = []
    normalized = [step for step in path if isinstance(step, dict)]
    if not normalized:
        return [start_step]

    first = normalized[0]
    first_id = str(first.get("node_id") or "")
    first_name = str(first.get("node_name") or "")
    start_id = start_step["node_id"]
    start_name = start_step["node_name"]
    if first_id != start_id and first_name != start_name:
        return [start_step, *normalized]

    merged = {**start_step, **first, "role": first.get("role") or "entry"}
    if merged.get("role") != "entry":
        merged["role"] = "entry"
    return [merged, *normalized[1:]]


def build_attack_process_mermaid(propagation: dict[str, Any], analysis: dict[str, Any]) -> str:
    lines = ["graph TD"]
    chains = propagation.get("chains")
    if isinstance(chains, list) and chains:
        for chain_index, chain in enumerate(chains, start=1):
            if not isinstance(chain, dict):
                continue
            path = normalize_chain_path(chain)
            edge_items = chain.get("edges") if isinstance(chain.get("edges"), list) else []
            previous_node_key = ""
            for step_index, step in enumerate(path):
                node_key = f"C{chain_index}_N{step_index}"
                node_name = str(step.get("node_name") or step.get("node_id") or "unknown")
                role = str(step.get("role") or "propagation")
                state = str(step.get("threat_state") or "")
                prefix = "被攻击工具" if step_index == 0 else role
                label = f"{prefix}: {node_name}"
                if state:
                    label = f"{label} | {state}"
                lines.append(f"    {node_key}[{mermaid_label(label)}]")
                if previous_node_key:
                    edge_label = "propagates"
                    edge_index = step_index - 1
                    if edge_index < len(edge_items) and isinstance(edge_items[edge_index], dict):
                        edge_label = str(
                            edge_items[edge_index].get("propagation_mechanism")
                            or edge_items[edge_index].get("condition")
                            or edge_label
                        )
                    lines.append(f"    {previous_node_key} -->|{sanitize_edge_label(edge_label)}| {node_key}")
                previous_node_key = node_key
    else:
        threats = analysis.get("threats")
        if isinstance(threats, list) and threats:
            for index, threat in enumerate(threats, start=1):
                if not isinstance(threat, dict):
                    continue
                location = threat.get("attack_location")
                if not isinstance(location, dict):
                    location = {}
                node_name = str(location.get("node_name") or location.get("node_id") or "unknown_tool")
                impact = threat.get("impact")
                if not isinstance(impact, dict):
                    impact = {}
                entry_key = f"T{index}_entry"
                impact_key = f"T{index}_impact"
                lines.append(f"    {entry_key}[{mermaid_label('被攻击工具: ' + node_name)}]")
                lines.append(
                    f"    {entry_key} -->|潜在传播| {impact_key}"
                    f"[{mermaid_label(str(impact.get('summary') or '潜在安全影响'))}]"
                )
        else:
            lines.append('    no_chain["No propagation chain identified"]')
    return "\n".join(lines) + "\n"


def fallback_mermaid(propagation: dict[str, Any], analysis: dict[str, Any]) -> str:
    return build_attack_process_mermaid(propagation, analysis)

def markdown_report(record: ThreatRecord, result: dict[str, Any], mermaid: str) -> str:
    analysis = result.get("threat_analysis", {})
    propagation = result.get("propagation", {})
    threats = analysis.get("threats") if isinstance(analysis, dict) else []
    chains = propagation.get("chains") if isinstance(propagation, dict) else []
    coverage = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
    coverage_section = ""
    if coverage:
        framework_score = float(coverage.get(FRAMEWORK_SLOT_METRIC, 0.0) or 0.0)
        strategy_score = float(coverage.get(STRATEGY_STEP_METRIC, 0.0) or 0.0)
        uncovered_units = coverage.get("uncovered_units") if isinstance(coverage.get("uncovered_units"), list) else []
        z3_reward = result.get("z3_reward") if isinstance(result.get("z3_reward"), dict) else {}
        runtime_reward = result.get("openclaw_runtime_reward") if isinstance(result.get("openclaw_runtime_reward"), dict) else {}
        z3_line = ""
        if z3_reward:
            z3_line = (
                f"- Z3 reward: {float(z3_reward.get('reward', 0.0) or 0.0):.4f}\n"
                f"- Framework compliance: {float(z3_reward.get('framework_compliance_score', 0.0) or 0.0):.4f}\n"
                f"- Propagation compliance: {float(z3_reward.get('propagation_compliance_score', 0.0) or 0.0):.4f}\n"
            )
        runtime_line = ""
        if runtime_reward:
            runtime_line = (
                f"- OpenClaw runtime reward: {float(runtime_reward.get('reward', 0.0) or 0.0):.4f}\n"
                f"- Attack mode: {runtime_reward.get('attack_mode', '')}\n"
                f"- Attack success rate: {float(runtime_reward.get('attack_success_rate', 0.0) or 0.0):.4f}\n"
                f"- Runtime path coverage: {float(runtime_reward.get('path_coverage_score', 0.0) or 0.0):.4f}\n"
                f"- Skill injection applied rate: {float(runtime_reward.get('skill_injection_applied_rate', 0.0) or 0.0):.4f}\n"
            )
        coverage_section = f"""
## Coverage

{z3_line}{runtime_line}- Framework path coverage: {framework_score:.4f}
- Strategy parse coverage: {strategy_score:.4f}
- Semantic units: {coverage.get("unit_count", 0)}
- Uncovered units: {len(uncovered_units)}

"""
    return f"""# Threat Propagation Analysis: {record.skill_id}

## Inputs

- Skill file: `{record.skill_path}`
- Graph file: `{record.graph_path}`

## Summary

- Threat findings: {len(threats) if isinstance(threats, list) else 0}
- Propagation chains: {len(chains) if isinstance(chains, list) else 0}

{coverage_section}## Mermaid

```mermaid
{mermaid.strip()}
```
"""



def analyze_record(
    client: DeepSeekClient,
    record: ThreatRecord,
    *,
    output_root: Path,
    system_prompt_template: str,
    analysis_prompt_template: str,
    propagation_prompt_template: str,
    framework: dict[str, Any],
    parsing_strategy: dict[str, Any],
    framework_file: Path,
    parsing_strategy_file: Path,
    chain_template: str,
    chain_template_file: Path,
    max_framework_chars: int,
    max_strategy_chars: int,
    max_chain_template_chars: int,
    max_skill_chars: int,
    max_graph_chars: int,
    max_analysis_tokens: int,
    max_chain_tokens: int,
    temperature: float,
    include_raw: bool,
    openclaw_runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    skill_markdown = read_text(record.skill_path)
    graph_markdown = read_text(record.graph_path)
    manifest = load_json(record.manifest_path) if record.manifest_path else {}
    if not isinstance(manifest, dict):
        manifest = {}
    parsed_graph = parse_mermaid_graph(graph_markdown)

    analysis_prompt = build_threat_analysis_prompt(
        analysis_prompt_template,
        record,
        manifest,
        graph_markdown,
        skill_markdown,
        parsed_graph,
        max_graph_chars=max_graph_chars,
        max_skill_chars=max_skill_chars,
    )
    raw_analysis = client.complete(
        system_prompt_template,
        analysis_prompt,
        max_tokens=max_analysis_tokens,
        temperature=temperature,
        request_label=f"{record.skill_id}:threat-analysis",
    )
    threat_analysis = parse_json_object(raw_analysis)

    propagation_prompt = build_propagation_prompt(
        propagation_prompt_template,
        record,
        parsed_graph,
        threat_analysis,
        framework,
        parsing_strategy,
        chain_template,
        max_framework_chars=max_framework_chars,
        max_strategy_chars=max_strategy_chars,
        max_chain_template_chars=max_chain_template_chars,
        graph_markdown=graph_markdown,
        max_graph_chars=max_graph_chars,
    )
    raw_propagation = client.complete(
        system_prompt_template,
        propagation_prompt,
        max_tokens=max_chain_tokens,
        temperature=temperature,
        request_label=f"{record.skill_id}:propagation",
    )
    propagation = normalize_propagation_schema(
        parse_json_object(raw_propagation),
        skill_id=record.skill_id,
    )

    model_mermaid = extract_mermaid(str(propagation.get("mermaid") or ""))
    mermaid = build_attack_process_mermaid(propagation, threat_analysis)

    generated_at = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "skill_id": record.skill_id,
        "generated_at": generated_at,
        "model": client.model,
        "input_files": {
            "skill": str(record.skill_path),
            "graph": str(record.graph_path),
            "manifest": str(record.manifest_path) if record.manifest_path else None,
        },
        "prompt_files": {
            "system": "threat_security_system_prompt.md",
            "analysis": "threat_analysis_prompt.md",
            "propagation": "threat_propagation_prompt.md",
            "framework": str(framework_file),
            "parsing_strategy": str(parsing_strategy_file),
            "chain_template": str(chain_template_file),
        },
        "parsed_graph": parsed_graph,
        "threat_analysis": threat_analysis,
        "propagation": propagation,
        "model_mermaid": model_mermaid if valid_mermaid(model_mermaid) else "",
        "mermaid": mermaid,
    }
    result["coverage"] = evaluate_result_coverage(result, framework, parsing_strategy)
    reward_summary = evaluate_z3_reward([result], framework, parsing_strategy)
    skill_rewards = reward_summary.get("skill_rewards") if isinstance(reward_summary.get("skill_rewards"), list) else []
    if skill_rewards:
        result["z3_reward"] = skill_rewards[0]
        result["coverage"]["z3_reward"] = skill_rewards[0].get("reward", 0.0)
        result["coverage"]["z3_propagation_compliance_score"] = skill_rewards[0].get("propagation_compliance_score", 0.0)
        result["coverage"]["z3_failed_constraints"] = skill_rewards[0].get("propagation_failed_constraints", [])
    result["z3_reward_summary"] = reward_summary

    skill_output_root = output_root / record.skill_id
    runtime_config = dict(openclaw_runtime_config or {})
    if runtime_config.get("enabled"):
        runtime_reward = evaluate_openclaw_runtime_reward(
            result,
            output_root=skill_output_root / "openclaw_runtime",
            config=runtime_config,
        )
        result["openclaw_runtime_reward"] = runtime_reward
        result["coverage"]["openclaw_runtime_reward"] = runtime_reward.get("reward", 0.0)
        result["coverage"]["attack_success_rate"] = runtime_reward.get("attack_success_rate", runtime_reward.get("strict_attack_success_rate", 0.0))
        result["coverage"]["strict_attack_success_rate"] = runtime_reward.get("strict_attack_success_rate", runtime_reward.get("attack_success_rate", 0.0))
        result["coverage"]["instruction_follow_success_rate"] = runtime_reward.get("instruction_follow_success_rate", 0.0)
        result["coverage"]["canary_exposure_rate"] = runtime_reward.get("canary_exposure_rate", 0.0)
        result["coverage"]["runtime_path_coverage_score"] = runtime_reward.get("path_coverage_score", 0.0)
        result["coverage"]["skill_injection_applied_rate"] = runtime_reward.get("skill_injection_applied_rate", 0.0)
        result["coverage"]["attack_mode"] = runtime_reward.get("attack_mode", "")

    if include_raw:
        result["raw_model_outputs"] = {
            "analysis": raw_analysis,
            "propagation": raw_propagation,
        }

    write_json(skill_output_root / "threat_analysis.json", result)
    write_json(
        skill_output_root / "propagation.json",
        {
            "skill_id": record.skill_id,
            "generated_at": generated_at,
            "model": client.model,
            "propagation": propagation,
            "mermaid": mermaid,
            "coverage": result["coverage"],
            "z3_reward": result.get("z3_reward", {}),
            "openclaw_runtime_reward": result.get("openclaw_runtime_reward", {}),
        },
    )
    write_json(skill_output_root / "coverage.json", result["coverage"])
    write_text(skill_output_root / "propagation.mmd", mermaid)
    write_text(skill_output_root / "propagation.md", markdown_report(record, result, mermaid))
    return result

def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    analysis = result.get("threat_analysis", {})
    propagation = result.get("propagation", {})
    threats = analysis.get("threats") if isinstance(analysis, dict) else []
    chains = propagation.get("chains") if isinstance(propagation, dict) else []
    coverage = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
    summary = {
        "skill_id": result.get("skill_id"),
        "generated_at": result.get("generated_at"),
        "model": result.get("model"),
        "threat_count": len(threats) if isinstance(threats, list) else 0,
        "chain_count": len(chains) if isinstance(chains, list) else 0,
        "status": "ok",
    }
    if coverage:
        uncovered_units = coverage.get("uncovered_units") if isinstance(coverage.get("uncovered_units"), list) else []
        summary.update(
            {
                FRAMEWORK_SLOT_METRIC: coverage.get(FRAMEWORK_SLOT_METRIC, 0.0),
                STRATEGY_STEP_METRIC: coverage.get(STRATEGY_STEP_METRIC, 0.0),
                "uncovered_unit_count": len(uncovered_units),
            }
        )
    z3_reward = result.get("z3_reward") if isinstance(result.get("z3_reward"), dict) else {}
    if z3_reward:
        summary.update(
            {
                "z3_reward": z3_reward.get("reward", 0.0),
                "framework_compliance_score": z3_reward.get("framework_compliance_score", 0.0),
                "propagation_compliance_score": z3_reward.get("propagation_compliance_score", 0.0),
            }
        )
    runtime_reward = result.get("openclaw_runtime_reward") if isinstance(result.get("openclaw_runtime_reward"), dict) else {}
    if runtime_reward:
        summary.update(
            {
                "openclaw_runtime_reward": runtime_reward.get("reward", 0.0),
                "attack_success_rate": runtime_reward.get("attack_success_rate", runtime_reward.get("strict_attack_success_rate", 0.0)),
                "strict_attack_success_rate": runtime_reward.get("strict_attack_success_rate", runtime_reward.get("attack_success_rate", 0.0)),
                "instruction_follow_success_rate": runtime_reward.get("instruction_follow_success_rate", 0.0),
                "canary_exposure_rate": runtime_reward.get("canary_exposure_rate", 0.0),
                "runtime_path_coverage_score": runtime_reward.get("path_coverage_score", 0.0),
                "skill_injection_applied_rate": runtime_reward.get("skill_injection_applied_rate", 0.0),
                "attack_mode": runtime_reward.get("attack_mode", ""),
                "openclaw_runtime_status": runtime_reward.get("status", ""),
            }
        )
    return summary

def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    selected_ids = set(args.skill) if args.skill else None

    records = collect_records(data_root, selected_ids=selected_ids, graph_name=args.graph_name)
    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        print("No matching graph_wo_check.md and SKILL.md pairs found.", file=sys.stderr)
        return 1

    print(f"Matched {len(records)} skill(s) with graph/SKILL pairs.")
    for record in records:
        print(f"- {record.skill_id}: {record.graph_path.name} + {record.skill_path.name}")

    if args.dry_run:
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    client = DeepSeekClient(
        api_key=api_key,
        model=args.model,
        retries=args.deepseek_retries,
        retry_seconds=args.deepseek_retry_seconds,
    )
    system_prompt_template = read_text(Path(args.system_prompt_file).resolve()).strip()
    analysis_prompt_template = read_text(Path(args.analysis_prompt_file).resolve()).strip()
    propagation_prompt_template = read_text(Path(args.propagation_prompt_file).resolve()).strip()
    framework_file = Path(args.framework_file).resolve()
    parsing_strategy_file = Path(args.parsing_strategy_file).resolve()
    framework_definition = load_threat_framework(framework_file)
    parsing_strategy = load_threat_parsing_strategy(parsing_strategy_file, framework_definition)
    template_evolution_prompt_template = read_text(Path(args.template_evolution_prompt_file).resolve()).strip()
    template_understandability_prompt_template = read_text(
        Path(args.template_understandability_prompt_file).resolve()
    ).strip()
    chain_template_file = Path(args.chain_template_file).resolve()
    chain_template_archive_root = (
        Path(args.chain_template_archive_root).resolve()
        if args.chain_template_archive_root
        else chain_template_file.parent
    )
    run_started_at = datetime.now(timezone.utc)
    chain_template_archive_dir = chain_template_archive_root / run_started_at.strftime("%Y-%m-%dT%H-%M-%SZ")
    chain_template_data = load_chain_template(chain_template_file)
    if args.evolve_chain_template:
        write_json(chain_template_archive_dir / "run_seed_chain_template.json", chain_template_data)
    chain_template = render_formal_chain_template(chain_template_data)
    summary_path = output_root / "summary.jsonl"
    failures: list[tuple[str, str]] = []

    for index, record in enumerate(records, start=1):
        output_json = output_root / record.skill_id / "threat_analysis.json"
        if output_json.exists() and not args.force:
            print(f"[skip] {record.skill_id}: threat_analysis.json exists")
            append_jsonl(
                summary_path,
                {
                    "skill_id": record.skill_id,
                    "status": "skipped",
                    "reason": "output exists",
                    "output": str(output_json),
                },
            )
            continue
        try:
            result = analyze_record(
                client,
                record,
                output_root=output_root,
                system_prompt_template=system_prompt_template,
                analysis_prompt_template=analysis_prompt_template,
                propagation_prompt_template=propagation_prompt_template,
                framework=framework_definition,
                parsing_strategy=parsing_strategy,
                framework_file=framework_file,
                parsing_strategy_file=parsing_strategy_file,
                chain_template=chain_template,
                chain_template_file=chain_template_file,
                max_framework_chars=args.max_framework_chars,
                max_strategy_chars=args.max_strategy_chars,
                max_chain_template_chars=args.max_chain_template_chars,
                max_skill_chars=args.max_skill_chars,
                max_graph_chars=args.max_graph_chars,
                max_analysis_tokens=args.max_analysis_tokens,
                max_chain_tokens=args.max_chain_tokens,
                temperature=args.temperature,
                include_raw=args.include_raw,
                openclaw_runtime_config=openclaw_config_from_args(args),
            )
            summary = summarize_result(result)
            if args.evolve_chain_template:
                summary["chain_template_file"] = str(chain_template_file)
                summary["chain_template_archive_root"] = str(chain_template_archive_root)
                summary["chain_template_archive_dir"] = str(chain_template_archive_dir)
                previous_template_data = chain_template_data
                template_snapshot_data = chain_template_data
                template_snapshot_status = "unchanged"
                skill_output_root = output_root / record.skill_id
                template_understandability: dict[str, Any] = {}
                summary["template_understandability_evaluated"] = False
                if args.skip_template_understandability:
                    summary["template_understandability_skipped"] = True
                else:
                    try:
                        template_understandability, raw_understandability = evaluate_template_understandability(
                            client,
                            system_prompt_template=system_prompt_template,
                            understandability_prompt_template=template_understandability_prompt_template,
                            current_chain_template=chain_template_data,
                            result=result,
                            max_tokens=args.max_template_understandability_tokens,
                        )
                        result["template_understandability"] = template_understandability
                        write_json(
                            skill_output_root / "template_understandability.json",
                            {
                                "skill_id": record.skill_id,
                                "generated_at": datetime.now(timezone.utc).isoformat(),
                                "model": client.model,
                                "template_understandability": template_understandability,
                            },
                        )
                        write_json(skill_output_root / "threat_analysis.json", result)
                        summary.update(summarize_template_understandability(template_understandability))
                        if args.include_raw:
                            write_text(
                                skill_output_root / "template_understandability.raw.json",
                                raw_understandability.rstrip() + "\n",
                            )
                    except Exception as exc:
                        summary["template_understandability_error"] = str(exc)
                        print(
                            f"[understandability-skip] {record.skill_id}: continue without template understandability; {exc}",
                            file=sys.stderr,
                        )
                try:
                    evolved_template_data, raw_evolution, evolution_response = evolve_chain_template(
                        client,
                        system_prompt_template=system_prompt_template,
                        evolution_prompt_template=template_evolution_prompt_template,
                        current_chain_template=chain_template_data,
                        result=result,
                        template_understandability=template_understandability,
                        max_tokens=args.max_template_evolution_tokens,
                    )
                    template_changed = evolved_template_data != previous_template_data
                    summary["template_evolved"] = template_changed
                    write_json(skill_output_root / "chain_template_evolution.json", evolution_response)
                    if args.include_raw:
                        write_text(skill_output_root / "chain_template_evolution.raw.json", raw_evolution.rstrip() + "\n")
                    if template_changed:
                        write_json(skill_output_root / "chain_template.before.json", previous_template_data)
                        write_json(skill_output_root / "chain_template.after.json", evolved_template_data)
                        if args.update_chain_template_file:
                            write_json(chain_template_file, evolved_template_data)
                            summary["chain_template_file_updated"] = True
                        else:
                            summary["chain_template_file_updated"] = False
                        chain_template_data = evolved_template_data
                        chain_template = render_formal_chain_template(chain_template_data)
                        template_snapshot_data = evolved_template_data
                        template_snapshot_status = "evolved"
                    else:
                        template_snapshot_status = "unchanged"
                except Exception as exc:
                    summary["template_evolved"] = False
                    summary["template_evolution_error"] = str(exc)
                    template_snapshot_status = "evolution_failed"
                    print(
                        f"[template-skip] {record.skill_id}: keep previous template; {exc}",
                        file=sys.stderr,
                    )
                summary["chain_template_snapshot_status"] = template_snapshot_status
                try:
                    archive_path = archive_chain_template(
                        chain_template_file,
                        template_snapshot_data,
                        archive_dir=chain_template_archive_dir,
                        skill_id=record.skill_id,
                    )
                    summary["chain_template_archive"] = str(archive_path)
                    summary["chain_template_snapshot"] = str(archive_path)
                    if template_snapshot_status == "evolved":
                        if args.update_chain_template_file:
                            print(f"[template] evolved after {record.skill_id} -> {chain_template_file}; snapshot {archive_path}")
                        else:
                            print(f"[template] evolved after {record.skill_id}; snapshot {archive_path}")
                    elif template_snapshot_status == "unchanged":
                        print(f"[template] unchanged after {record.skill_id}; snapshot {archive_path}")
                    else:
                        print(f"[template] evolution failed after {record.skill_id}; current snapshot {archive_path}")
                except Exception as exc:
                    summary["chain_template_archive_error"] = str(exc)
                    print(
                        f"[template-archive-fail] {record.skill_id}: {exc}",
                        file=sys.stderr,
                    )
            append_jsonl(summary_path, summary)
            print(f"[ok] {record.skill_id} -> {output_json}")
        except Exception as exc:
            failures.append((record.skill_id, str(exc)))
            append_jsonl(
                summary_path,
                {
                    "skill_id": record.skill_id,
                    "status": "failed",
                    "error": str(exc),
                },
            )
            print(f"[fail] {record.skill_id}: {exc}", file=sys.stderr)
        if args.sleep_seconds > 0 and index < len(records):
            time.sleep(args.sleep_seconds)

    if args.evolve_chain_template:
        final_template_path = chain_template_archive_dir / "run_final_chain_template.json"
        write_json(final_template_path, chain_template_data)
        print(f"[template] final evolved template snapshot {final_template_path}")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for skill_id, error in failures:
            print(f"- {skill_id}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
