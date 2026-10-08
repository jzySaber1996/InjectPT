#!/usr/bin/env python3
"""Validate Mermaid workflow graphs with structural checks, Z3 path solving,
and LLM-ready semantic review prompts.

This script targets normalized workflow graphs under
`data/inner_representation/<skill>/graph_wo_check.md`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from z3 import And, Bool, If, Implies, Int, IntVal, Not, Optimize, Or, Solver, Sum, sat

ANY_NODE_DEF_RE = re.compile(
    r"\b([A-Za-z0-9_]+)(\[[^\]]*\]|\{[^}]*\}|\(\([^)]*\)\)|\([^)]*\))"
)
EDGE_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)(?:\[[^\]]*\]|\{[^}]*\}|\(\([^)]*\)\)|\([^)]*\))?\s*-->\s*(?:\|([^|]*)\|\s*)?([A-Za-z0-9_]+)(?:\[[^\]]*\]|\{[^}]*\}|\(\([^)]*\)\)|\([^)]*\))?\s*$"
)
STRUCTURED_FIELD_RE = re.compile(
    r"\b(name|task|input|output|constraint)\s*:\s*(.*?)(?=(?:;\s*(?:name|task|input|output|constraint)\s*:)|$)",
    re.IGNORECASE,
)
START_LABEL_RE = re.compile(r"(任务开始|开始|start|entry|入口|初始化|接收)", re.IGNORECASE)
FINISH_LABEL_RE = re.compile(r"(任务完成|完成|finish|done|返回结果|结束|输出结果)", re.IGNORECASE)
NEGATIVE_CONDITION_TOKENS = (
    "失败", "错误", "异常", "不完整", "缺失", "技术问题", "限流", "重试", "回滚", "等待",
    "无效", "不足", "未通过", "fail", "error", "invalid", "retry", "rollback", "wait",
    "missing", "denied", "blocked",
)
POSITIVE_CONDITION_TOKENS = (
    "通过", "成功", "完成", "有效", "已验证", "无errors", "ready", "ok", "pass", "success",
    "complete", "valid",
)

SEQUENCE_RULES = (
    ("hypothesis", "sample_size"),
    ("sample_size", "run_test"),
    ("metrics", "run_test"),
    ("variant", "run_test"),
    ("pre_launch", "run_test"),
    ("run_test", "analy"),
    ("analy", "document"),
    ("document", "recommend"),
)
TOKEN_RULES = (
    ("constraint_requires_sample_size", ("样本量", "sample size"), ("sample_size", "样本量")),
    ("constraint_requires_metrics", ("主要指标", "次要指标", "护栏指标", "metric"), ("metric", "指标")),
    ("constraint_requires_tracking", ("追踪", "tracking", "埋点"), ("track", "tracking", "telemetry", "instrument", "埋点")),
    ("constraint_requires_qa", ("qa", "quality assurance"), ("qa", "quality assurance", "validate")),
    ("constraint_requires_auth", ("oauth", "api key", "凭据", "认证", "鉴权"), ("auth", "credential", "token", "key", "oauth", "api key", "凭据")),
)


@dataclass
class Node:
    node_id: str
    name: str
    task: str
    input_text: str
    output_text: str
    constraint: str
    raw_label: str


@dataclass
class Edge:
    src: str
    dst: str
    condition: str
    cost: int


@dataclass
class Issue:
    severity: str
    code: str
    message: str


@dataclass
class WitnessPath:
    nodes: list[str]
    edges: list[dict[str, Any]]
    total_cost: int


@dataclass
class ValidationReport:
    skill_id: str
    graph_file: str
    status: str
    start_node: str | None
    finish_node: str | None
    checks: dict[str, Any]
    issues: list[Issue]
    witness_path: WitnessPath | None
    llm_review_prompt: str


@dataclass
class BatchSummary:
    total: int
    pass_count: int
    pass_with_warnings_count: int
    fail_count: int
    issue_code_counts: dict[str, int]
    status_by_skill: dict[str, str]
    top_issue_examples: dict[str, list[str]]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(repo_root / "data" / "inner_representation"),
        help="Root directory containing normalized skill folders.",
    )
    parser.add_argument(
        "--graph-name",
        default="graph_wo_check.md",
        help="Workflow graph filename inside each skill directory.",
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Skill id(s) to validate. Can be repeated.",
    )
    parser.add_argument(
        "--graph-file",
        action="append",
        default=[],
        help="Specific graph file(s) to validate. Can be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Validate only the first N matched targets. 0 means no limit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a human-readable report.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write `<graph>.validation.json` next to each graph.",
    )
    parser.add_argument(
        "--write-llm-prompt",
        action="store_true",
        help="Write `<graph>.llm_review.md` next to each graph.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def sanitize_label(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace('"', "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def extract_structured_fields(label: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in STRUCTURED_FIELD_RE.finditer(label):
        fields[match.group(1).lower()] = sanitize_label(match.group(2))
    return fields


def classify_edge_cost(condition: str) -> int:
    condition = sanitize_label(condition)
    if not condition:
        return 1
    lowered = condition.lower()
    if any(token in condition for token in NEGATIVE_CONDITION_TOKENS) or any(token in lowered for token in NEGATIVE_CONDITION_TOKENS):
        return 5
    if any(token in condition for token in POSITIVE_CONDITION_TOKENS) or any(token in lowered for token in POSITIVE_CONDITION_TOKENS):
        return 0
    return 1


def parse_edge_line(line: str) -> Edge | None:
    match = EDGE_RE.match(line.strip())
    if not match:
        return None
    condition = sanitize_label(match.group(2) or "")
    return Edge(
        src=match.group(1),
        dst=match.group(3),
        condition=condition,
        cost=classify_edge_cost(condition),
    )


def parse_graph_lines(lines: list[str], graph_label: str) -> tuple[dict[str, Node], list[Edge], list[str]]:
    normalized_lines = [line.rstrip() for line in lines if line.strip()]
    if not normalized_lines:
        raise ValueError(f"Empty graph file: {graph_label}")
    if not re.match(r"^graph\s+TD\b", normalized_lines[0], re.IGNORECASE):
        raise ValueError(f"Graph must start with 'graph TD': {graph_label}")

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []

    for line in normalized_lines[1:]:
        for match in ANY_NODE_DEF_RE.finditer(line):
            node_id = match.group(1)
            raw_label = sanitize_label(extract_shape_label(match.group(2)))
            fields = extract_structured_fields(raw_label)
            existing = nodes.get(node_id)
            if existing and existing.raw_label:
                continue
            nodes[node_id] = Node(
                node_id=node_id,
                name=fields.get("name", ""),
                task=fields.get("task", ""),
                input_text=fields.get("input", ""),
                output_text=fields.get("output", ""),
                constraint=fields.get("constraint", ""),
                raw_label=raw_label,
            )
        edge = parse_edge_line(line)
        if edge is not None:
            edges.append(edge)

    for edge in edges:
        if edge.src not in nodes:
            nodes[edge.src] = Node(edge.src, "", "", "", "", "", "")
        if edge.dst not in nodes:
            nodes[edge.dst] = Node(edge.dst, "", "", "", "", "", "")

    return nodes, edges, normalized_lines


def parse_graph(graph_path: Path) -> tuple[dict[str, Node], list[Edge], list[str]]:
    text = read_text(graph_path)
    return parse_graph_lines(text.splitlines(), str(graph_path))


def build_degree_maps(nodes: dict[str, Node], edges: list[Edge]) -> tuple[dict[str, int], dict[str, int], dict[str, list[Edge]], dict[str, list[Edge]]]:
    indegree = {node_id: 0 for node_id in nodes}
    outdegree = {node_id: 0 for node_id in nodes}
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    incoming: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges:
        outdegree[edge.src] += 1
        indegree[edge.dst] += 1
        outgoing[edge.src].append(edge)
        incoming[edge.dst].append(edge)
    return indegree, outdegree, outgoing, incoming


def identify_start_finish(nodes: dict[str, Node], indegree: dict[str, int], outdegree: dict[str, int]) -> tuple[list[str], list[str], str | None, str | None]:
    name_starts = sorted(node_id for node_id, node in nodes.items() if node.name == "start_node")
    name_finishes = sorted(node_id for node_id, node in nodes.items() if node.name == "finish_node")
    degree_starts = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    degree_finishes = sorted(node_id for node_id, degree in outdegree.items() if degree == 0)

    canonical_start = None
    canonical_finish = None
    if len(name_starts) == 1:
        canonical_start = name_starts[0]
    elif len(degree_starts) == 1:
        canonical_start = degree_starts[0]

    if len(name_finishes) == 1:
        canonical_finish = name_finishes[0]
    elif len(degree_finishes) == 1:
        canonical_finish = degree_finishes[0]

    return degree_starts, degree_finishes, canonical_start, canonical_finish


def bfs_reachable(start: str, outgoing: dict[str, list[Edge]]) -> set[str]:
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        for edge in outgoing.get(node_id, []):
            if edge.dst not in visited:
                queue.append(edge.dst)
    return visited


def reverse_bfs_reachable(finish: str, incoming: dict[str, list[Edge]]) -> set[str]:
    visited: set[str] = set()
    queue: deque[str] = deque([finish])
    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        for edge in incoming.get(node_id, []):
            if edge.src not in visited:
                queue.append(edge.src)
    return visited


def build_node_order_index(witness_path: WitnessPath | None) -> dict[str, int]:
    if not witness_path:
        return {}
    return {node_id: index for index, node_id in enumerate(witness_path.nodes)}


def find_nodes_by_keyword(nodes: dict[str, Node], keywords: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for node_id, node in nodes.items():
        blob = " ".join([node.name, node.task, node.input_text, node.output_text, node.constraint]).lower()
        if any(keyword.lower() in blob for keyword in keywords):
            matches.append(node_id)
    return sorted(set(matches))


def evaluate_sequence_rules(nodes: dict[str, Node], witness_path: WitnessPath | None) -> tuple[list[Issue], dict[str, Any]]:
    order_index = build_node_order_index(witness_path)
    checks: dict[str, Any] = {"sequence_rules": []}
    issues: list[Issue] = []
    for before_keyword, after_keyword in SEQUENCE_RULES:
        before_nodes = find_nodes_by_keyword(nodes, (before_keyword,))
        after_nodes = find_nodes_by_keyword(nodes, (after_keyword,))
        satisfied = True
        detail = {
            "before_keyword": before_keyword,
            "after_keyword": after_keyword,
            "before_nodes": before_nodes,
            "after_nodes": after_nodes,
            "satisfied": True,
        }
        if order_index and before_nodes and after_nodes:
            before_pos = min(order_index[node_id] for node_id in before_nodes if node_id in order_index)
            after_pos = min(order_index[node_id] for node_id in after_nodes if node_id in order_index)
            satisfied = before_pos < after_pos
            detail["before_pos"] = before_pos
            detail["after_pos"] = after_pos
            detail["satisfied"] = satisfied
        checks["sequence_rules"].append(detail)
        if before_nodes and after_nodes and not satisfied:
            issues.append(Issue(
                "warning",
                "sequence_rule",
                f"expected `{before_keyword}` before `{after_keyword}` on witness path, got {before_nodes} after/beside {after_nodes}",
            ))
    return issues, checks


def solve_constraint_feasibility(nodes: dict[str, Node], witness_path: WitnessPath | None) -> tuple[list[Issue], dict[str, Any]]:
    order_index = build_node_order_index(witness_path)
    solver = Solver()
    checks: dict[str, Any] = {"constraint_rules": []}
    issues: list[Issue] = []
    symbols: dict[str, Any] = {}

    for node_id in nodes:
        symbols[node_id] = Bool(f"active_{node_id}")
        if witness_path and node_id in order_index:
            solver.add(symbols[node_id])
        else:
            solver.add(symbols[node_id] == False)

    for rule_code, trigger_keywords, provider_keywords in TOKEN_RULES:
        triggered_nodes = []
        provider_nodes = find_nodes_by_keyword(nodes, provider_keywords)
        for node_id, node in nodes.items():
            blob = " ".join([node.task, node.constraint, node.input_text]).lower()
            if any(keyword.lower() in blob for keyword in trigger_keywords):
                triggered_nodes.append(node_id)
        triggered_nodes = sorted(set(triggered_nodes))
        rule_payload = {
            "code": rule_code,
            "triggered_nodes": triggered_nodes,
            "provider_nodes": provider_nodes,
            "sat": True,
        }
        for node_id in triggered_nodes:
            if not provider_nodes:
                rule_payload["sat"] = False
                issues.append(Issue(
                    "warning",
                    rule_code,
                    f"node {node_id} references {trigger_keywords} but no provider node matches {provider_keywords}",
                ))
                continue
            if node_id in provider_nodes:
                rule_payload.setdefault("per_node", []).append({
                    "node": node_id,
                    "candidate_providers": provider_nodes,
                    "sat": True,
                    "reason": "self_provider",
                })
                continue
            candidate_constraints = []
            for provider_id in provider_nodes:
                if not witness_path or provider_id not in order_index or node_id not in order_index:
                    continue
                provider_before = Bool(f"{rule_code}_{provider_id}_before_{node_id}")
                solver.add(provider_before == (order_index[provider_id] < order_index[node_id]))
                candidate_constraints.append(provider_before)
            if candidate_constraints:
                solver.add(Implies(symbols[node_id], Or(candidate_constraints)))
                sat_result = solver.check() == sat
                rule_payload.setdefault("per_node", []).append({
                    "node": node_id,
                    "candidate_providers": provider_nodes,
                    "sat": sat_result,
                })
                if not sat_result:
                    rule_payload["sat"] = False
                    issues.append(Issue(
                        "warning",
                        rule_code,
                        f"node {node_id} has no earlier provider for keywords {provider_keywords} on witness path",
                    ))
            else:
                rule_payload["sat"] = False
                issues.append(Issue(
                    "warning",
                    rule_code,
                    f"node {node_id} references {trigger_keywords} but witness path has no earlier provider node for {provider_keywords}",
                ))
        checks["constraint_rules"].append(rule_payload)
    checks["constraint_solver_sat"] = solver.check() == sat
    return issues, checks


def build_batch_summary(reports: list[ValidationReport]) -> BatchSummary:
    issue_code_counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)
    status_by_skill: dict[str, str] = {}
    pass_count = 0
    pass_with_warnings_count = 0
    fail_count = 0
    for report in reports:
        status_by_skill[report.skill_id] = report.status
        if report.status == "pass":
            pass_count += 1
        elif report.status == "pass_with_warnings":
            pass_with_warnings_count += 1
        else:
            fail_count += 1
        for issue in report.issues:
            issue_code_counts[issue.code] += 1
            if len(examples[issue.code]) < 5:
                examples[issue.code].append(f"{report.skill_id}: {issue.message}")
    return BatchSummary(
        total=len(reports),
        pass_count=pass_count,
        pass_with_warnings_count=pass_with_warnings_count,
        fail_count=fail_count,
        issue_code_counts=dict(sorted(issue_code_counts.items())),
        status_by_skill=status_by_skill,
        top_issue_examples=dict(sorted(examples.items())),
    )


def batch_summary_to_dict(summary: BatchSummary) -> dict[str, Any]:
    return asdict(summary)


def strongly_connected_components(nodes: dict[str, Node], outgoing: dict[str, list[Edge]]) -> list[list[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node_id: str) -> None:
        nonlocal index
        indices[node_id] = index
        lowlink[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)

        for edge in outgoing.get(node_id, []):
            neighbor = edge.dst
            if neighbor not in indices:
                visit(neighbor)
                lowlink[node_id] = min(lowlink[node_id], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node_id] = min(lowlink[node_id], indices[neighbor])

        if lowlink[node_id] == indices[node_id]:
            component: list[str] = []
            while stack:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node_id:
                    break
            components.append(sorted(component))

    for node_id in sorted(nodes):
        if node_id not in indices:
            visit(node_id)
    return components


def bool_sum(items: list[Any]) -> Any:
    if not items:
        return IntVal(0)
    return Sum(items)


def solve_witness_path(nodes: dict[str, Node], edges: list[Edge], start: str, finish: str) -> WitnessPath | None:
    optimizer = Optimize()
    node_ids = sorted(nodes)
    edge_keys = [f"{edge.src}->{edge.dst}#{index}" for index, edge in enumerate(edges)]
    on_path = {node_id: Bool(f"on_path_{node_id}") for node_id in node_ids}
    order = {node_id: Int(f"order_{node_id}") for node_id in node_ids}
    use_edge = {edge_key: Bool(f"use_{index}") for index, edge_key in enumerate(edge_keys)}

    incoming_keys: dict[str, list[str]] = defaultdict(list)
    outgoing_keys: dict[str, list[str]] = defaultdict(list)
    for edge_key, edge in zip(edge_keys, edges):
        incoming_keys[edge.dst].append(edge_key)
        outgoing_keys[edge.src].append(edge_key)

    for node_id in node_ids:
        optimizer.add(order[node_id] >= 0)
        optimizer.add(order[node_id] < len(node_ids))

    for edge_key, edge in zip(edge_keys, edges):
        optimizer.add(Implies(use_edge[edge_key], on_path[edge.src]))
        optimizer.add(Implies(use_edge[edge_key], on_path[edge.dst]))
        optimizer.add(Implies(use_edge[edge_key], order[edge.src] < order[edge.dst]))

    optimizer.add(on_path[start])
    optimizer.add(on_path[finish])
    optimizer.add(order[start] == 0)

    for node_id in node_ids:
        incoming_count = bool_sum([If(use_edge[key], 1, 0) for key in incoming_keys.get(node_id, [])])
        outgoing_count = bool_sum([If(use_edge[key], 1, 0) for key in outgoing_keys.get(node_id, [])])
        if node_id == start:
            optimizer.add(incoming_count == 0)
            optimizer.add(outgoing_count == 1)
        elif node_id == finish:
            optimizer.add(incoming_count == 1)
            optimizer.add(outgoing_count == 0)
        else:
            optimizer.add(Implies(on_path[node_id], And(incoming_count == 1, outgoing_count == 1)))
            optimizer.add(Implies(Not(on_path[node_id]), And(incoming_count == 0, outgoing_count == 0)))

    for index, left in enumerate(node_ids):
        for right in node_ids[index + 1 :]:
            optimizer.add(Implies(And(on_path[left], on_path[right]), order[left] != order[right]))

    total_cost = bool_sum([If(use_edge[edge_key], edge.cost, 0) for edge_key, edge in zip(edge_keys, edges)])
    total_length = bool_sum([If(use_edge[edge_key], 1, 0) for edge_key in edge_keys])
    optimizer.minimize(total_cost)
    optimizer.minimize(total_length)

    if optimizer.check() != sat:
        return None

    model = optimizer.model()
    chosen_edges: list[Edge] = []
    for edge_key, edge in zip(edge_keys, edges):
        if model.eval(use_edge[edge_key], model_completion=True):
            chosen_edges.append(edge)

    if not chosen_edges:
        return None

    successor: dict[str, Edge] = {edge.src: edge for edge in chosen_edges}
    ordered_nodes = [start]
    ordered_edges: list[dict[str, Any]] = []
    current = start
    seen = {start}
    while current != finish:
        edge = successor.get(current)
        if edge is None:
            return None
        ordered_edges.append({"src": edge.src, "dst": edge.dst, "condition": edge.condition, "cost": edge.cost})
        current = edge.dst
        if current in seen and current != finish:
            return None
        seen.add(current)
        ordered_nodes.append(current)

    return WitnessPath(
        nodes=ordered_nodes,
        edges=ordered_edges,
        total_cost=sum(edge["cost"] for edge in ordered_edges),
    )


def render_llm_review_prompt(skill_id: str, graph_file: Path, nodes: dict[str, Node], edges: list[Edge], witness_path: WitnessPath | None, checks: dict[str, Any]) -> str:
    node_payload = [
        {
            "id": node.node_id,
            "name": node.name,
            "task": node.task,
            "input": node.input_text,
            "output": node.output_text,
            "constraint": node.constraint,
        }
        for node in sorted(nodes.values(), key=lambda item: item.node_id)
    ]
    edge_payload = [
        {"src": edge.src, "dst": edge.dst, "condition": edge.condition, "cost": edge.cost}
        for edge in edges
    ]
    witness_payload = asdict(witness_path) if witness_path else None
    summary_payload = {
        "path_exists": checks.get("path_exists"),
        "all_nodes_reachable_from_start": checks.get("all_nodes_reachable_from_start"),
        "all_nodes_can_reach_finish": checks.get("all_nodes_can_reach_finish"),
        "unreachable_from_start": checks.get("unreachable_from_start"),
        "cannot_reach_finish": checks.get("cannot_reach_finish"),
    }
    return (
        f"# LLM Semantic Review for {skill_id}\n\n"
        f"Graph file: `{graph_file}`\n\n"
        "You are reviewing whether this workflow graph is semantically valid and can achieve its final goal.\n\n"
        "Tasks:\n"
        "1. Check every node: whether task/input/output/constraint are internally consistent.\n"
        "2. Check every edge: whether the source output can realistically satisfy the destination input.\n"
        "3. Check the witness path: whether it represents a logically correct happy path from start to finish.\n"
        "4. Identify missing prerequisite steps, impossible transitions, or steps that should be reordered.\n"
        "5. Output JSON with `node_checks`, `edge_checks`, `path_check`, `goal_check`, and `overall_verdict`.\n\n"
        f"## Structural Summary\n```json\n{json.dumps(summary_payload, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## Nodes\n```json\n{json.dumps(node_payload, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## Edges\n```json\n{json.dumps(edge_payload, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## Witness Path\n```json\n{json.dumps(witness_payload, ensure_ascii=False, indent=2)}\n```\n"
    )


def validate_graph_data(*, skill_id: str, graph_text: str, graph_label: str) -> ValidationReport:
    nodes, edges, lines = parse_graph_lines(graph_text.splitlines(), graph_label)
    indegree, outdegree, outgoing, incoming = build_degree_maps(nodes, edges)
    degree_starts, degree_finishes, start_node, finish_node = identify_start_finish(nodes, indegree, outdegree)

    issues: list[Issue] = []
    checks: dict[str, Any] = {
        "line_count": len(lines),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "degree_start_candidates": degree_starts,
        "degree_finish_candidates": degree_finishes,
        "name_start_candidates": sorted(node_id for node_id, node in nodes.items() if node.name == "start_node"),
        "name_finish_candidates": sorted(node_id for node_id, node in nodes.items() if node.name == "finish_node"),
    }

    if len(checks["name_start_candidates"]) != 1:
        issues.append(Issue("error", "unique_start_name", f"expected exactly one `name: start_node`, got {checks['name_start_candidates']}"))
    if len(checks["name_finish_candidates"]) != 1:
        issues.append(Issue("error", "unique_finish_name", f"expected exactly one `name: finish_node`, got {checks['name_finish_candidates']}"))
    if len(degree_starts) != 1:
        issues.append(Issue("error", "unique_start_degree", f"expected exactly one indegree-0 start node, got {degree_starts}"))
    if len(degree_finishes) != 1:
        issues.append(Issue("error", "unique_finish_degree", f"expected exactly one outdegree-0 finish node, got {degree_finishes}"))

    for node in sorted(nodes.values(), key=lambda item: item.node_id):
        missing = [
            field_name
            for field_name, value in (
                ("name", node.name),
                ("task", node.task),
                ("input", node.input_text),
                ("output", node.output_text),
                ("constraint", node.constraint),
            )
            if not value
        ]
        if missing:
            issues.append(Issue("error", "missing_fields", f"node {node.node_id} missing fields: {missing}"))

    reachable_from_start: set[str] = set()
    can_reach_finish: set[str] = set()
    witness_path: WitnessPath | None = None

    if start_node is not None:
        reachable_from_start = bfs_reachable(start_node, outgoing)
    if finish_node is not None:
        can_reach_finish = reverse_bfs_reachable(finish_node, incoming)

    if start_node and finish_node:
        checks["path_exists"] = finish_node in reachable_from_start
        if not checks["path_exists"]:
            issues.append(Issue("error", "path_exists", f"finish node {finish_node} is not reachable from {start_node}"))
        witness_path = solve_witness_path(nodes, edges, start_node, finish_node)
        if witness_path is None:
            issues.append(Issue("error", "z3_witness_path", "Z3 could not synthesize a valid start-to-finish witness path"))
    else:
        checks["path_exists"] = False
        issues.append(Issue("error", "missing_terminal_nodes", "could not identify canonical start and finish nodes"))

    checks["reachable_from_start"] = sorted(reachable_from_start)
    checks["can_reach_finish"] = sorted(can_reach_finish)
    checks["unreachable_from_start"] = sorted(node_id for node_id in nodes if node_id not in reachable_from_start) if start_node else sorted(nodes)
    checks["cannot_reach_finish"] = sorted(node_id for node_id in nodes if node_id not in can_reach_finish) if finish_node else sorted(nodes)
    checks["all_nodes_reachable_from_start"] = not checks["unreachable_from_start"]
    checks["all_nodes_can_reach_finish"] = not checks["cannot_reach_finish"]

    if checks["unreachable_from_start"]:
        issues.append(Issue("warning", "unreachable_nodes", f"nodes unreachable from start: {checks['unreachable_from_start']}"))
    if checks["cannot_reach_finish"]:
        issues.append(Issue("warning", "dead_end_nodes", f"nodes that cannot reach finish: {checks['cannot_reach_finish']}"))

    components = strongly_connected_components(nodes, outgoing)
    cyclical_components: list[list[str]] = []
    for component in components:
        if len(component) > 1:
            cyclical_components.append(component)
        elif len(component) == 1:
            node_id = component[0]
            if any(edge.src == node_id and edge.dst == node_id for edge in outgoing.get(node_id, [])):
                cyclical_components.append(component)
    checks["cyclical_components"] = cyclical_components
    if cyclical_components:
        issues.append(Issue("warning", "cycles", f"graph contains cycles/SCCs: {cyclical_components}"))

    sequence_issues, sequence_checks = evaluate_sequence_rules(nodes, witness_path)
    issues.extend(sequence_issues)
    checks.update(sequence_checks)

    constraint_issues, constraint_checks = solve_constraint_feasibility(nodes, witness_path)
    issues.extend(constraint_issues)
    checks.update(constraint_checks)

    llm_review_prompt = render_llm_review_prompt(
        skill_id=skill_id,
        graph_file=Path(graph_label),
        nodes=nodes,
        edges=edges,
        witness_path=witness_path,
        checks=checks,
    )

    has_error = any(issue.severity == "error" for issue in issues)
    has_warning = any(issue.severity == "warning" for issue in issues)
    status = "fail" if has_error else "pass_with_warnings" if has_warning else "pass"

    return ValidationReport(
        skill_id=skill_id,
        graph_file=graph_label,
        status=status,
        start_node=start_node,
        finish_node=finish_node,
        checks=checks,
        issues=issues,
        witness_path=witness_path,
        llm_review_prompt=llm_review_prompt,
    )


def validate_graph(graph_path: Path) -> ValidationReport:
    return validate_graph_data(
        skill_id=graph_path.parent.name,
        graph_text=read_text(graph_path),
        graph_label=str(graph_path),
    )


def iter_targets(data_root: Path, graph_name: str, skills: list[str], graph_files: list[str], limit: int) -> list[Path]:
    targets: list[Path] = []
    if graph_files:
        targets.extend(Path(item).resolve() for item in graph_files)
    elif skills:
        targets.extend((data_root / skill / graph_name).resolve() for skill in skills)
    else:
        for skill_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
            graph_path = skill_dir / graph_name
            if graph_path.exists():
                targets.append(graph_path.resolve())
    if limit > 0:
        targets = targets[:limit]
    return targets


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    return {
        "skill_id": report.skill_id,
        "graph_file": report.graph_file,
        "status": report.status,
        "start_node": report.start_node,
        "finish_node": report.finish_node,
        "checks": report.checks,
        "issues": [asdict(issue) for issue in report.issues],
        "witness_path": asdict(report.witness_path) if report.witness_path else None,
        "llm_review_prompt": report.llm_review_prompt,
    }


def render_text_report(report: ValidationReport) -> str:
    lines = [
        f"[{report.status}] {report.skill_id}",
        f"graph: {report.graph_file}",
        f"start: {report.start_node}",
        f"finish: {report.finish_node}",
        f"nodes: {report.checks['node_count']}, edges: {report.checks['edge_count']}",
        f"path_exists: {report.checks['path_exists']}",
        f"all_nodes_reachable_from_start: {report.checks['all_nodes_reachable_from_start']}",
        f"all_nodes_can_reach_finish: {report.checks['all_nodes_can_reach_finish']}",
    ]
    if report.witness_path:
        lines.append("witness_path: " + " -> ".join(report.witness_path.nodes))
    if report.issues:
        lines.append("issues:")
        for issue in report.issues:
            lines.append(f"- [{issue.severity}] {issue.code}: {issue.message}")
    else:
        lines.append("issues: none")
    return "\n".join(lines)


def write_sidecar_files(report: ValidationReport, write_json: bool, write_llm_prompt: bool) -> None:
    graph_path = Path(report.graph_file)
    stem = graph_path.with_suffix("")
    if write_json:
        json_path = stem.parent / f"{stem.name}.validation.json"
        json_path.write_text(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    if write_llm_prompt:
        prompt_path = stem.parent / f"{stem.name}.llm_review.md"
        prompt_path.write_text(report.llm_review_prompt, encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    targets = iter_targets(data_root, args.graph_name, args.skill, args.graph_file, args.limit)
    if not targets:
        print("No graph targets matched.", file=sys.stderr)
        return 1

    reports: list[ValidationReport] = []
    exit_code = 0
    for graph_path in targets:
        try:
            report = validate_graph(graph_path)
            reports.append(report)
            write_sidecar_files(report, args.write_json, args.write_llm_prompt)
        except Exception as exc:
            exit_code = 2
            fallback = {
                "skill_id": graph_path.parent.name,
                "graph_file": str(graph_path),
                "status": "fail",
                "issues": [{"severity": "error", "code": "exception", "message": str(exc)}],
            }
            if args.json:
                print(json.dumps(fallback, ensure_ascii=False, indent=2))
            else:
                print(f"[fail] {graph_path.parent.name}\n- [error] exception: {exc}")
    for report in reports:
        if report.status == "fail":
            exit_code = max(exit_code, 2)
        elif report.status == "pass_with_warnings":
            exit_code = max(exit_code, 1)
        if args.json:
            print(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2))
        else:
            print(render_text_report(report))
            print()

    summary = build_batch_summary(reports)
    if args.json:
        print(json.dumps({"batch_summary": batch_summary_to_dict(summary)}, ensure_ascii=False, indent=2))
    else:
        print("Batch Summary")
        print(f"total: {summary.total}")
        print(f"pass: {summary.pass_count}")
        print(f"pass_with_warnings: {summary.pass_with_warnings_count}")
        print(f"fail: {summary.fail_count}")
        if summary.issue_code_counts:
            print("issue_counts:")
            for code, count in summary.issue_code_counts.items():
                print(f"- {code}: {count}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
