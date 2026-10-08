#!/usr/bin/env python3
"""Generate Mermaid workflow graphs for normalized skills via DeepSeek.

The script reads skills under `data/inner_representation/<skill>/SKILL.md`,
prompts DeepSeek to extract a query-oriented workflow, and writes the Mermaid
result to `graph_wo_check.md` in each skill directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_API_URL = os.environ.get(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/v1/chat/completions",
)
SQUARE_NODE_DEF_RE = re.compile(r"\b([A-Za-z0-9_]+)\[([^\]]*)\]")
ANY_NODE_DEF_RE = re.compile(
    r"\b([A-Za-z0-9_]+)(\[[^\]]*\]|\{[^}]*\}|\(\([^)]*\)\)|\([^)]*\))"
)
NODE_EDGE_SUFFIX_RE = r"(?:\[[^\]]*\]|\{[^}]*\}|\(\([^)]*\)\)|\([^)]*\))?"
EDGE_RE = re.compile(
    rf"^\s*([A-Za-z0-9_]+){NODE_EDGE_SUFFIX_RE}\s*-->\s*(?:\|[^|]*\|\s*)?([A-Za-z0-9_]+){NODE_EDGE_SUFFIX_RE}\s*$"
)
STRUCTURED_FIELD_NAMES = ("name", "task", "input", "output", "constraint")
STRUCTURED_FIELD_RE = re.compile(
    r"\b(name|task|input|output|constraint)\s*:\s*(.*?)(?=(?:;\s*(?:name|task|input|output|constraint)\s*:)|$)",
    re.IGNORECASE,
)
SNAKE_CASE_RE = re.compile(r"^[a-z0-9_]+$")
LEGACY_EXECUTION_RE = re.compile(r"(script\s*[-_ ]?\s*execution|脚本执行)", re.IGNORECASE)
START_LABEL_RE = re.compile(r"(任务开始|开始|start|entry|入口|初始化|接收查询|接收输入)", re.IGNORECASE)
FINISH_LABEL_RE = re.compile(r"(任务完成|完成|finish|done|返回结果|结束|输出结果)", re.IGNORECASE)
SCRIPT_FILE_RE = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|sh|js|ts|tsx|jsx|mjs|cjs))")
COMMAND_CODE_RE = re.compile(r"`([^`]+)`")
SKILL_MARKDOWN_CANDIDATES = ("SKILL.md", "SKILLS.md")


@dataclass
class SkillRecord:
    skill_id: str
    skill_dir: Path
    manifest: dict[str, Any]
    skill_markdown: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(repo_root / "data" / "inner_representation"),
        help="Normalized skill dataset root.",
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
        "--output-name",
        default="graph_wo_check.md",
        help="Output Mermaid filename inside each skill directory.",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(repo_root / "prompts" / "skill_workflow_mermaid_prompt.md"),
        help="Prompt template path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="DeepSeek model name.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2400,
        help="Max completion tokens per skill.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between API calls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call DeepSeek or write files; print selected skills only.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def load_prompt_template(path: Path) -> str:
    return read_text(path)


def load_index(data_root: Path) -> list[dict[str, Any]]:
    index_path = data_root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing dataset index: {index_path}")
    data = load_json(index_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {index_path}")
    return data


def resolve_skill_markdown_path(skill_dir: Path) -> Path | None:
    for filename in SKILL_MARKDOWN_CANDIDATES:
        candidate = skill_dir / filename
        if candidate.exists():
            return candidate
    return None


def build_fallback_manifest(skill_id: str, skill_dir: Path) -> dict[str, Any]:
    meta_path = skill_dir / "_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        loaded = load_json(meta_path)
        if isinstance(loaded, dict):
            meta = loaded

    slug = str(meta.get("slug") or skill_id).strip() or skill_id
    return {
        "canonicalId": skill_id,
        "rawSlug": slug,
        "name": slug,
        "description": "",
        "sourceRelativePath": str(skill_dir),
        "sourceDatasetPath": str(skill_dir),
        "sourceParents": [],
        "primaryJson": None,
        "copiedDocs": [],
        "syntheticMeta": True,
    }


def iter_skill_entries(data_root: Path) -> list[dict[str, Any]]:
    index_path = data_root / "index.json"
    if index_path.exists():
        return load_index(data_root)

    entries: list[dict[str, Any]] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        skill_path = resolve_skill_markdown_path(child)
        if skill_path is None:
            continue
        entries.append(
            {
                "canonicalId": child.name,
                "rawSlug": child.name,
                "name": child.name,
                "description": "",
                "syntheticMeta": True,
            }
        )
    return entries


def collect_skills(data_root: Path, selected_ids: set[str] | None) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    for entry in iter_skill_entries(data_root):
        skill_id = str(entry.get("canonicalId") or "").strip()
        if not skill_id:
            continue
        if selected_ids and skill_id not in selected_ids:
            continue
        skill_dir = data_root / skill_id
        manifest_path = skill_dir / "manifest.json"
        skill_path = resolve_skill_markdown_path(skill_dir)
        if skill_path is None:
            continue
        manifest = build_fallback_manifest(skill_id, skill_dir)
        if manifest_path.exists():
            try:
                loaded_manifest = load_json(manifest_path)
            except (OSError, ValueError, json.JSONDecodeError):
                loaded_manifest = None
            if isinstance(loaded_manifest, dict):
                manifest = loaded_manifest
        skill_markdown = read_text(skill_path)
        records.append(
            SkillRecord(
                skill_id=skill_id,
                skill_dir=skill_dir,
                manifest=manifest,
                skill_markdown=skill_markdown,
            )
        )
    records.sort(key=lambda item: item.skill_id)
    return records


def summarize_manifest(manifest: dict[str, Any]) -> str:
    payload = {
        "sourceRelativePath": manifest.get("sourceRelativePath"),
        "sourceParents": manifest.get("sourceParents"),
        "primaryJson": manifest.get("primaryJson"),
        "copiedDocs": manifest.get("copiedDocs"),
        "syntheticMeta": manifest.get("syntheticMeta"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_prompt(template: str, record: SkillRecord) -> str:
    return template.format(
        skill_id=record.skill_id,
        skill_name=record.manifest.get("name", record.skill_id),
        skill_description=record.manifest.get("description", ""),
        manifest_summary=summarize_manifest(record.manifest),
        skill_markdown=record.skill_markdown,
    )


def extract_mermaid(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:mermaid)?\s*(graph\s+TD[\s\S]*?)```", stripped, re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    graph_match = re.search(r"(graph\s+TD[\s\S]*)", stripped, re.IGNORECASE)
    if graph_match:
        stripped = graph_match.group(1).strip()
    return stripped


def sanitize_label(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace('"', "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_mermaid(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty Mermaid output")
    if not re.match(r"^graph\s+TD\b", lines[0], re.IGNORECASE):
        raise ValueError("Mermaid output must start with 'graph TD'")
    normalized = [re.sub(r"\s+", " ", lines[0].strip())]
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        if "-->" not in line and not ANY_NODE_DEF_RE.search(line):
            continue
        normalized.append(f"    {line}")
    return "\n".join(normalized).strip() + "\n"


def parse_edge_line(line: str) -> tuple[str, str] | None:
    match = EDGE_RE.match(line)
    if not match:
        return None
    return match.group(1), match.group(2)


def extract_shape_label(shape_token: str) -> str:
    if shape_token.startswith("[[") and shape_token.endswith("]]"):
        return shape_token[2:-2]
    if shape_token.startswith("[") and shape_token.endswith("]"):
        return shape_token[1:-1]
    if shape_token.startswith("{") and shape_token.endswith("}"):
        return shape_token[1:-1]
    if shape_token.startswith("((") and shape_token.endswith("))"):
        return shape_token[2:-2]
    if shape_token.startswith("(") and shape_token.endswith(")"):
        return shape_token[1:-1]
    return shape_token


def parse_graph_structure(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    node_labels: dict[str, str] = {}
    node_ids: set[str] = set()
    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)
    edges: list[tuple[str, str]] = []

    for line in lines[1:]:
        for match in ANY_NODE_DEF_RE.finditer(line):
            node_id = match.group(1)
            node_ids.add(node_id)
            node_labels.setdefault(node_id, sanitize_label(extract_shape_label(match.group(2))))
        edge = parse_edge_line(line.strip())
        if edge is None:
            continue
        src, dst = edge
        node_ids.add(src)
        node_ids.add(dst)
        outgoing[src] += 1
        incoming[dst] += 1
        edges.append((src, dst))

    start_nodes = sorted(node for node in node_ids if incoming.get(node, 0) == 0)
    finish_nodes = sorted(node for node in node_ids if outgoing.get(node, 0) == 0)
    return {
        "lines": lines,
        "node_ids": node_ids,
        "node_labels": node_labels,
        "incoming": dict(incoming),
        "outgoing": dict(outgoing),
        "edges": edges,
        "start_nodes": start_nodes,
        "finish_nodes": finish_nodes,
    }


def extract_structured_fields(label: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in STRUCTURED_FIELD_RE.finditer(label):
        fields[match.group(1).lower()] = sanitize_label(match.group(2))
    return fields


def to_snake_case(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("::", "_")
    value = re.sub(r"[\s./:-]+", "_", value)
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def unique_node_id(preferred: str, existing: set[str]) -> str:
    candidate = preferred
    suffix = 2
    while candidate in existing:
        candidate = f"{preferred}_{suffix}"
        suffix += 1
    return candidate


def split_task_and_constraint(label: str, role: str) -> tuple[str, str]:
    fields = extract_structured_fields(label)
    if fields:
        return fields.get("task", ""), fields.get("constraint", "")

    text = sanitize_label(label)
    if not text:
        return "", ""

    task = text
    constraint = ""

    constraint_match = re.search(r"(?:约束|constraint)\s*[:：]\s*(.*)$", task, re.IGNORECASE)
    if constraint_match:
        constraint = sanitize_label(constraint_match.group(1))
        task = sanitize_label(task[: constraint_match.start()])

    task = re.sub(r"^(?:script\s*[-_ ]?\s*execution|脚本执行)\s*[:：]\s*", "", task, flags=re.IGNORECASE)
    task = re.sub(r"^(?:任务开始|开始|start)\s*[:：]\s*", "", task, flags=re.IGNORECASE)
    task = re.sub(r"^(?:任务完成|完成|finish|done)\s*[:：]\s*", "", task, flags=re.IGNORECASE)
    task = sanitize_label(task)

    if not task and role == "start":
        task = "接收查询请求并初始化执行上下文"
    if not task and role == "finish":
        task = "返回最终结果"

    return task, constraint


def infer_input_output(task: str, constraint: str, role: str) -> tuple[str, str]:
    merged = f"{task} {constraint}"
    lower = merged.lower()

    if role == "start":
        return "用户查询或任务目标", "初始执行上下文"
    if role == "finish":
        return "最终处理结果", "返回给用户的结果"

    if any(token in merged for token in ["校验", "检查", "鉴权", "认证", "环境", "凭据", "参数合法"]):
        return "查询请求、环境配置或待校验参数", "校验结果或可执行上下文"
    if any(token in merged for token in ["构建", "生成请求", "组装", "准备"]) or any(token in lower for token in ["build", "prepare"]):
        return "已校验的查询条件和上下文", "标准化请求载荷"
    if any(token in merged for token in ["发送", "请求", "调用", "查询", "抓取", "检索", "读取", "获取", "打开", "快照"]) or any(token in lower for token in ["request", "query", "fetch", "read", "open", "snapshot"]):
        return "查询参数、凭据或上一步输出", "原始响应或查询结果"
    if any(token in merged for token in ["解析", "提取", "分析", "转换", "结构化", "汇总"]) or any(token in lower for token in ["parse", "extract", "analy", "transform", "structure"]):
        return "原始响应或中间数据", "结构化结果或提取字段"
    if any(token in merged for token in ["重试", "回滚", "等待", "限流", "恢复"]) or any(token in lower for token in ["retry", "rollback", "wait", "rate limit"]):
        return "失败上下文、速率状态或待恢复流程", "重试结果或恢复后的上下文"
    if any(token in merged for token in ["提示", "返回前置", "补充", "修正", "纠正", "授权"]) or any(token in lower for token in ["prompt", "fix", "correct", "authorize"]):
        return "错误信息、缺失条件或当前上下文", "修正建议或返回前置步骤"
    return "上一步输出和当前上下文", "当前步骤结果"


def infer_default_constraint(task: str, role: str) -> str:
    if role == "start":
        return "全图唯一入口"
    if role == "finish":
        return "全图唯一出口"
    if any(token in task for token in ["重试", "回滚", "等待"]):
        return "保留原查询上下文并遵守重试条件"
    if any(token in task for token in ["校验", "检查", "鉴权", "认证"]):
        return "必须满足前置检查后才能继续"
    if any(token in task for token in ["解析", "结构化", "提取"]):
        return "输出必须可被下一步稳定消费"
    return "遵循SKILL.md中的前置条件和参数约束"


def extract_name_candidate(label: str, task: str, role: str) -> str:
    fields = extract_structured_fields(label)
    existing_name = fields.get("name", "")
    if existing_name:
        return existing_name

    text = label
    file_match = SCRIPT_FILE_RE.search(text)
    if file_match:
        return file_match.group(1)

    for token in COMMAND_CODE_RE.findall(text):
        token = token.strip()
        if not token:
            continue
        file_match = SCRIPT_FILE_RE.search(token)
        if file_match:
            return file_match.group(1)
        if any(ch.isalpha() for ch in token):
            return token

    if role == "start":
        return "start_node"
    if role == "finish":
        return "finish_node"
    return task or "workflow_step"


def make_unique_name(name: str, used_names: set[str], fallback: str) -> str:
    candidate = to_snake_case(name) or to_snake_case(fallback) or "workflow_step"
    unique = candidate
    suffix = 2
    while unique in used_names:
        unique = f"{candidate}_{suffix}"
        suffix += 1
    used_names.add(unique)
    return unique


def build_label_fields(node_id: str, label: str, role: str, used_names: set[str]) -> dict[str, str]:
    existing_fields = extract_structured_fields(label)
    task, constraint = split_task_and_constraint(label, role)

    task = existing_fields.get("task") or task
    if not task:
        if role == "start":
            task = "接收查询请求并初始化执行上下文"
        elif role == "finish":
            task = "返回最终结果"
        else:
            task = "执行当前查询步骤"

    constraint = existing_fields.get("constraint") or constraint or infer_default_constraint(task, role)
    inferred_input, inferred_output = infer_input_output(task, constraint, role)
    input_text = existing_fields.get("input") or inferred_input
    output_text = existing_fields.get("output") or inferred_output

    name_candidate = existing_fields.get("name") or extract_name_candidate(label, task, role)
    fallback_name = "start_node" if role == "start" else "finish_node" if role == "finish" else node_id.lower()
    name = make_unique_name(name_candidate, used_names, fallback_name)

    return {
        "name": name,
        "task": sanitize_label(task),
        "input": sanitize_label(input_text),
        "output": sanitize_label(output_text),
        "constraint": sanitize_label(constraint),
    }


def format_structured_label(fields: dict[str, str]) -> str:
    ordered = [f"{field}: {fields[field]}" for field in STRUCTURED_FIELD_NAMES]
    return sanitize_label("; ".join(ordered))


def choose_canonical_start(graph: dict[str, Any]) -> str:
    candidates = list(graph["start_nodes"])
    if not candidates:
        candidates = [src for src, _ in graph["edges"]]
    if not candidates:
        candidates = sorted(graph["node_ids"])

    seen: set[str] = set()
    ranked: list[tuple[float, str]] = []
    for index, node_id in enumerate(candidates):
        if node_id in seen:
            continue
        seen.add(node_id)
        label = graph["node_labels"].get(node_id, "")
        score = 0.0
        if START_LABEL_RE.search(label):
            score += 20
        if re.search(r"(?:^|_)(start|entry)(?:_|$)", node_id, re.IGNORECASE):
            score += 6
        if any(token in label for token in ["接收", "输入", "初始化", "校验", "检查"]):
            score += 4
        if FINISH_LABEL_RE.search(label):
            score -= 10
        score -= index * 0.01
        ranked.append((score, node_id))
    ranked.sort()
    return ranked[-1][1]


def choose_canonical_finish(finish_nodes: list[str], node_labels: dict[str, str]) -> str:
    ranked = []
    for index, node_id in enumerate(finish_nodes):
        label = node_labels.get(node_id, "")
        score = 0.0
        if FINISH_LABEL_RE.search(label):
            score += 10
        if "任务完成" in label or "返回" in label:
            score += 6
        if "失败" in label or "终止" in label:
            score -= 2
        score -= index * 0.01
        ranked.append((score, node_id))
    ranked.sort()
    return ranked[-1][1]


def choose_canonical_finish_source(graph: dict[str, Any]) -> str:
    candidates = [dst for _, dst in graph["edges"]]
    if not candidates:
        candidates = sorted(graph["node_ids"])
    seen: set[str] = set()
    ranked: list[tuple[float, str]] = []
    for index, node_id in enumerate(candidates):
        if node_id in seen:
            continue
        seen.add(node_id)
        label = graph["node_labels"].get(node_id, "")
        score = 0.0
        if FINISH_LABEL_RE.search(label):
            score += 12
        if any(token in label for token in ["解析", "结构化", "返回", "输出"]):
            score += 4
        score -= index * 0.01
        ranked.append((score, node_id))
    ranked.sort()
    return ranked[-1][1]


def ensure_single_start_node(text: str) -> str:
    graph = parse_graph_structure(text)
    start_nodes = graph["start_nodes"]
    if len(start_nodes) == 1:
        return text

    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    synthetic_id = unique_node_id("start_node", set(graph["node_ids"]))
    synthetic_label = (
        "name: start_node; task: 接收查询请求并初始化执行上下文; "
        "input: 用户查询或任务目标; output: 初始执行上下文; constraint: 全图唯一入口"
    )

    if len(start_nodes) == 0:
        target = choose_canonical_start(graph)
        lines.append(f"    {synthetic_id}[{synthetic_label}] --> {target}")
        return "\n".join(lines).strip() + "\n"

    first_target = start_nodes[0]
    lines.append(f"    {synthetic_id}[{synthetic_label}] --> {first_target}")
    for target in start_nodes[1:]:
        lines.append(f"    {synthetic_id} --> {target}")
    return "\n".join(lines).strip() + "\n"


def ensure_single_finish_node(text: str) -> str:
    graph = parse_graph_structure(text)
    finish_nodes = graph["finish_nodes"]
    if len(finish_nodes) == 1:
        return text

    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]

    if len(finish_nodes) == 0:
        synthetic_id = unique_node_id("finish_node", set(graph["node_ids"]))
        source = choose_canonical_finish_source(graph)
        synthetic_label = (
            "name: finish_node; task: 返回最终结果; input: 最终处理结果; "
            "output: 返回给用户的结果; constraint: 全图唯一出口"
        )
        lines.append(f"    {source} --> {synthetic_id}[{synthetic_label}]")
        return "\n".join(lines).strip() + "\n"

    canonical_finish = choose_canonical_finish(finish_nodes, graph["node_labels"])
    for node_id in finish_nodes:
        if node_id == canonical_finish:
            continue
        lines.append(f"    {node_id} --> {canonical_finish}")
    return "\n".join(lines).strip() + "\n"


def ensure_structured_node_labels(text: str) -> str:
    graph = parse_graph_structure(text)
    if not graph["node_ids"]:
        return text

    start_nodes = set(graph["start_nodes"])
    finish_nodes = set(graph["finish_nodes"])
    used_names: set[str] = set()
    final_labels: dict[str, str] = {}

    for node_id in sorted(graph["node_ids"]):
        role = "step"
        if node_id in start_nodes:
            role = "start"
        elif node_id in finish_nodes:
            role = "finish"
        source_label = graph["node_labels"].get(node_id, "")
        fields = build_label_fields(node_id, source_label, role, used_names)
        final_labels[node_id] = format_structured_label(fields)

    updated_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue

        def replace_node(match: re.Match[str]) -> str:
            node_id = match.group(1)
            return f"{node_id}[{final_labels[node_id]}]"

        updated_lines.append(ANY_NODE_DEF_RE.sub(replace_node, line).rstrip())

    unlabeled_nodes = sorted(node_id for node_id in graph["node_ids"] if node_id not in graph["node_labels"])
    for node_id in unlabeled_nodes:
        updated_lines.append(f"    {node_id}[{final_labels[node_id]}]")

    return "\n".join(updated_lines).strip() + "\n"


def repair_common_mermaid_issues(text: str) -> str:
    repaired = ensure_single_start_node(text)
    repaired = ensure_single_finish_node(repaired)
    repaired = ensure_structured_node_labels(repaired)
    return repaired


def validate_mermaid(text: str) -> list[str]:
    errors: list[str] = []
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not re.match(r"^graph\s+TD\b", lines[0], re.IGNORECASE):
        errors.append("missing graph TD header")
        return errors

    graph = parse_graph_structure(text)
    all_nodes = graph["node_ids"]
    outgoing = len(graph["edges"])

    if outgoing == 0:
        errors.append("no edges found")

    start_nodes = graph["start_nodes"]
    finish_nodes = graph["finish_nodes"]

    if len(start_nodes) != 1:
        errors.append(f"expected exactly 1 start node, got {len(start_nodes)}: {start_nodes}")
    if len(finish_nodes) != 1:
        errors.append(f"expected exactly 1 finish node, got {len(finish_nodes)}: {finish_nodes}")

    conditional_edges = sum(1 for line in lines[1:] if "-->|" in line)
    if conditional_edges == 0:
        errors.append("missing conditional branches")

    if not all_nodes:
        errors.append("no nodes found")

    for node_id in sorted(all_nodes):
        label = graph["node_labels"].get(node_id, "")
        if not label:
            errors.append(f"node {node_id} missing structured label")
            continue
        fields = extract_structured_fields(label)
        missing = [field for field in STRUCTURED_FIELD_NAMES if not fields.get(field)]
        if missing:
            errors.append(f"node {node_id} missing fields: {missing}")
            continue
        if not SNAKE_CASE_RE.fullmatch(fields["name"]):
            errors.append(f"node {node_id} has non-snake_case name: {fields['name']}")

    return errors


class DeepSeekClient:
    def __init__(self, api_key: str, model: str, api_url: str = DEFAULT_API_URL):
        if not api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY.")
        self.api_key = api_key
        self.model = model
        self.api_url = api_url

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )


def build_repair_prompt(record: SkillRecord, invalid_mermaid: str, errors: list[str]) -> str:
    error_text = "\n".join(f"- {item}" for item in errors)
    return f"""你需要修复一段 Mermaid 状态机，使其符合要求并且语法稳定。

要求：
- 只输出 Mermaid 内容
- 必须以 `graph TD` 开头
- 只能有一个开始节点和一个完成节点
- 所有节点都必须使用方括号节点：`node_id[文本]`
- 不要使用 `{{}}`、`()`、`(())` 等其他节点形状
- 每个节点标签都必须严格包含以下五个字段，且按这个顺序输出：
  `name: ...; task: ...; input: ...; output: ...; constraint: ...`
- `name` 必须是小写英文 snake_case
- 如果 `SKILL.md` 里出现明确的脚本名、命令名、工具名，优先把它规范化后用于 `name`
- `task` 简要描述这一步做什么
- `input` 说明该步输入
- `output` 说明该步输出
- `constraint` 说明约束条件
- 需要有条件分支
- 节点文案简短
- 允许回滚/重试分支
- 所有失败分支不能形成多个终点；必须回退到前序步骤，或最终汇聚到唯一完成节点
- 不要把失败节点、终止节点、提示节点保留成独立终点

skill: {record.skill_id}

当前 Mermaid:
{invalid_mermaid}

发现的问题:
{error_text}
"""


def generate_mermaid_for_skill(
    client: DeepSeekClient,
    prompt_template: str,
    record: SkillRecord,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    prompt = render_prompt(prompt_template, record)
    raw_output = client.complete(prompt, max_tokens=max_tokens, temperature=temperature)
    mermaid = repair_common_mermaid_issues(normalize_mermaid(extract_mermaid(raw_output)))
    errors = validate_mermaid(mermaid)
    if not errors:
        return mermaid

    repair_prompt = build_repair_prompt(record, mermaid, errors)
    repaired_output = client.complete(repair_prompt, max_tokens=max_tokens, temperature=0.0)
    repaired = repair_common_mermaid_issues(normalize_mermaid(extract_mermaid(repaired_output)))
    repaired_errors = validate_mermaid(repaired)
    if repaired_errors:
        raise ValueError(
            f"Mermaid validation failed for {record.skill_id}: {repaired_errors}"
        )
    return repaired


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    prompt_file = Path(args.prompt_file).resolve()
    selected_ids = set(args.skill) if args.skill else None
    records = collect_skills(data_root, selected_ids)
    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        print("No skills matched.", file=sys.stderr)
        return 1

    print(f"Matched {len(records)} skill(s).")
    for record in records:
        print(f"- {record.skill_id}")

    if args.dry_run:
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    client = DeepSeekClient(api_key=api_key, model=args.model)
    prompt_template = load_prompt_template(prompt_file)

    failures: list[tuple[str, str]] = []
    output_name = args.output_name
    for index, record in enumerate(records, start=1):
        output_path = record.skill_dir / output_name
        if output_path.exists() and not args.force:
            print(f"[skip] {record.skill_id}: {output_name} exists")
            continue
        try:
            mermaid = generate_mermaid_for_skill(
                client,
                prompt_template,
                record,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            write_text(output_path, mermaid)
            print(f"[ok] {record.skill_id} -> {output_path}")
        except Exception as exc:
            failures.append((record.skill_id, str(exc)))
            print(f"[fail] {record.skill_id}: {exc}", file=sys.stderr)
        if args.sleep_seconds > 0 and index < len(records):
            time.sleep(args.sleep_seconds)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for skill_id, error in failures:
            print(f"- {skill_id}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
