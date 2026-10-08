#!/usr/bin/env python3
"""Shared helpers for threat propagation framework, strategy, and metrics."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any


FRAMEWORK_PATH_METRIC = "framework_path_coverage_score"
STRATEGY_PARSE_METRIC = "strategy_parse_coverage_score"

# Keep the old variable names as import-compatible aliases while changing the metric semantics.
FRAMEWORK_SLOT_METRIC = FRAMEWORK_PATH_METRIC
STRATEGY_STEP_METRIC = STRATEGY_PARSE_METRIC

PROTECTED_COMPONENT_IDS = {"entry", "carrier", "trust_shift", "terminal"}
PROTECTED_GATE_IDS = {"untrusted_source", "trust_promoted", "external_effect"}
PROTECTED_STRATEGY_STEP_IDS = {"find_entry", "trace_carrier", "mark_trust_shift", "close_with_outcome"}


def _is_chain_like(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    has_path = isinstance(value.get("path"), list)
    has_start = isinstance(value.get("starting_attack_point"), dict)
    has_edges = isinstance(value.get("edges"), list)
    return has_path and (has_start or has_edges or bool(value.get("id")) or bool(value.get("title")))


def _single_node_chain(value: Any) -> dict[str, Any] | None:
    """Convert a model-emitted node hint into the smallest evaluable chain."""
    node_id = str(value.get("node_id") or value.get("id") or "").strip()
    node_name = str(value.get("node_name") or value.get("name") or node_id).strip()
    if not node_id and not node_name:
        return None
    reason = str(value.get("reason") or value.get("evidence") or value.get("threat_state") or "single-node propagation hint").strip()
    threat_state = str(value.get("threat_state") or reason or "untrusted content enters this node").strip()
    impact = value.get("impact") if isinstance(value.get("impact"), dict) else {}
    impact_summary = str(value.get("summary") or value.get("impact_summary") or impact.get("summary") or reason or "possible propagation into downstream workflow output").strip()
    return {
        "id": str(value.get("chain_id") or "C1"),
        "title": str(value.get("title") or "single-node fallback chain"),
        "source_threat_ids": [str(value.get("source_threat_id") or value.get("threat_id") or "T1")],
        "starting_attack_point": {"node_id": node_id, "node_name": node_name, "reason": reason},
        "path": [{
            "node_id": node_id,
            "node_name": node_name,
            "role": "entry",
            "threat_state": threat_state,
            "reason": reason,
        }],
        "edges": [],
        "impact": {
            "summary": impact_summary,
            "stride": impact.get("stride", []),
            "owasp_agent_threats": impact.get("owasp_agent_threats", []),
        },
        "risk_level": str(value.get("risk_level") or value.get("severity") or "medium"),
        "confidence": str(value.get("confidence") or "low"),
        "synthesized_from_single_node": True,
    }
def normalize_propagation_schema(propagation: Any, *, skill_id: str = "") -> dict[str, Any]:
    """Normalize model output to the canonical {skill_id, chains, mermaid, notes} schema.

    DeepSeek sometimes follows the singular chain template and emits one chain
    object directly. Downstream coverage and runtime evaluators require the
    top-level `chains` array, so keep this compatibility layer centralized.
    """
    if not isinstance(propagation, dict):
        return {
            "skill_id": skill_id,
            "chains": [],
            "mermaid": "",
            "notes": ["invalid propagation output: expected JSON object"],
            "schema_normalized": True,
            "normalization_reason": "non_object_output",
        }

    normalized = deepcopy(propagation)
    output_skill_id = str(normalized.get("skill_id") or skill_id)
    notes = normalized.get("notes") if isinstance(normalized.get("notes"), list) else []

    chains = normalized.get("chains")
    if isinstance(chains, list):
        normalized["chains"] = [chain for chain in chains if isinstance(chain, dict)]
        normalized["skill_id"] = output_skill_id
        normalized.setdefault("mermaid", "")
        normalized.setdefault("notes", notes)
        if normalized["chains"]:
            return normalized

    nested = normalized.get("propagation")
    if isinstance(nested, dict):
        nested_normalized = normalize_propagation_schema(nested, skill_id=output_skill_id)
        nested_normalized.setdefault("notes", [])
        if notes:
            nested_normalized["notes"] = [*nested_normalized.get("notes", []), *notes]
        return nested_normalized

    chain_obj = normalized.get("chain") if isinstance(normalized.get("chain"), dict) else None
    if chain_obj is None and _is_chain_like(normalized):
        chain_obj = {
            key: value
            for key, value in normalized.items()
            if key not in {"skill_id", "mermaid", "notes", "schema_normalized", "normalization_reason"}
        }

    if isinstance(chain_obj, dict):
        return {
            "skill_id": output_skill_id,
            "chains": [chain_obj],
            "mermaid": str(normalized.get("mermaid") or ""),
            "notes": [*notes, "schema normalized: wrapped single chain object into chains[0]"],
            "schema_normalized": True,
            "normalization_reason": "single_chain_object",
        }
    single_node_chain = _single_node_chain(normalized)
    if single_node_chain is not None:
        return {
            "skill_id": output_skill_id,
            "chains": [single_node_chain],
            "mermaid": str(normalized.get("mermaid") or ""),
            "notes": [*notes, "schema normalized: synthesized chains[0] from single node hint"],
            "schema_normalized": True,
            "normalization_reason": "single_node_hint",
        }

    normalized["skill_id"] = output_skill_id
    normalized["chains"] = []
    normalized.setdefault("mermaid", "")
    normalized["notes"] = [*notes, "schema normalized: no chains found"]
    normalized["schema_normalized"] = True
    normalized["normalization_reason"] = "missing_chains"
    return normalized


BASE_FRAMEWORK: dict[str, Any] = {
    "schema_version": "1.0",
    "framework_id": "threat_propagation_framework",
    "language": "zh-CN",
    "summary": "用少量稳定组件描述间接工具指令注入在 agent workflow 中的传播过程，独立于具体 skill。",
    "core_components": [
        {
            "id": "entry",
            "label": "攻击入口",
            "required": True,
            "definition": "不可信内容第一次进入工作流并可能影响后续决策的位置。",
            "role_hints": ["entry"],
            "match_signals": ["被攻击工具", "攻击入口", "不可信输入", "用户请求", "外部内容", "注入"],
        },
        {
            "id": "carrier",
            "label": "传播载体",
            "required": True,
            "definition": "污染内容被打包成输出、参数、上下文、缓存或状态并继续传递的阶段。",
            "role_hints": ["propagation"],
            "match_signals": ["传播", "输出", "参数", "上下文", "转发", "缓存", "记忆", "状态"],
        },
        {
            "id": "trust_shift",
            "label": "信任跨越",
            "required": True,
            "definition": "污染内容被后续节点当作更可信的指令、约束、计划或工具输入。",
            "role_hints": ["propagation"],
            "match_signals": ["信任边界", "高信任", "可信上下文", "执行计划", "工具参数", "被当作指令", "提升为可信"],
        },
        {
            "id": "effect",
            "label": "影响作用点",
            "required": False,
            "definition": "传播开始显化为权限扩大、状态改变、资源消耗或外部副作用的阶段。",
            "role_hints": ["propagation", "impact"],
            "match_signals": ["权限", "写入", "调用", "发送", "修改", "状态改变", "资源消耗", "外部影响"],
        },
        {
            "id": "containment",
            "label": "阻断机制",
            "required": False,
            "definition": "清洗、人工确认、allowlist、隔离、沙箱或回滚等使传播被阻断或降级的阶段。",
            "role_hints": ["containment"],
            "match_signals": ["阻断", "清洗", "确认", "allowlist", "schema 校验", "沙箱", "回滚", "拒绝"],
        },
        {
            "id": "terminal",
            "label": "终局",
            "required": True,
            "definition": "链路最终落到安全影响、传播阻断或无可行路径的结论。",
            "role_hints": ["impact", "containment"],
            "match_signals": ["最终影响", "影响达成", "传播中断", "无可行攻击路径", "返回给用户", "终点"],
        },
    ],
    "decision_gates": [
        {
            "id": "untrusted_source",
            "question": "是否存在来自用户、网页、文件、API、邮件、搜索结果、日志、数据库或记忆的不可信来源？",
            "true_effect": "传播链具备攻击入口。",
            "match_signals": ["用户请求", "外部输入", "网页", "文件", "API", "日志", "数据库", "记忆"],
        },
        {
            "id": "sanitized",
            "question": "污染内容是否经过清洗、转义、结构化解析、引用隔离或来源标记？",
            "true_effect": "传播被削弱或阻断。",
            "match_signals": ["清洗", "转义", "结构化解析", "引用", "来源标记", "过滤"],
        },
        {
            "id": "trust_promoted",
            "question": "污染内容是否被提升为可信上下文、工具参数、执行计划或后续 prompt？",
            "true_effect": "链路完成一次信任跨越。",
            "match_signals": ["可信上下文", "工具参数", "执行计划", "后续 prompt", "高信任", "信任边界"],
        },
        {
            "id": "state_or_privilege",
            "question": "污染内容是否进入更高权限、更长生命周期或更敏感的状态对象？",
            "true_effect": "影响范围被放大。",
            "match_signals": ["权限", "allowed=True", "记忆", "缓存", "文件写入", "数据库写入", "配置修改"],
        },
        {
            "id": "external_effect",
            "question": "链路是否产生外部可见效果，例如返回结果、调用 API、发消息、提交任务或写文件？",
            "true_effect": "安全影响开始显化。",
            "match_signals": ["返回给用户", "调用 API", "发消息", "写文件", "提交", "对外输出"],
        },
        {
            "id": "contained",
            "question": "链路是否被人工确认、allowlist、schema 校验、隔离、沙箱或回滚阻断？",
            "true_effect": "传播链收敛到 containment 终局。",
            "match_signals": ["人工确认", "allowlist", "schema 校验", "隔离", "沙箱", "回滚", "拒绝"],
        },
    ],
    "evidence_axes": [
        {
            "id": "content_flow",
            "label": "内容流",
            "definition": "污染数据或文本怎样从一个节点移动到下一个节点。",
        },
        {
            "id": "instruction_flow",
            "label": "指令流",
            "definition": "不可信内容怎样被解释为指令、约束或计划。",
        },
        {
            "id": "authority_flow",
            "label": "权限流",
            "definition": "权限、信任级别或可访问范围怎样被放大或误用。",
        },
        {
            "id": "state_flow",
            "label": "状态流",
            "definition": "污染内容怎样进入记忆、缓存、文件、数据库或其他持久状态。",
        },
        {
            "id": "effect_flow",
            "label": "效果流",
            "definition": "传播怎样落成外部副作用、资源消耗或安全影响。",
        },
    ],
    "output_contract": {
        "chain_start": "每条传播链必须从真实被攻击工具或节点开始，而不是正常 workflow 的 start node。",
        "path_order": "path 必须按攻击传播顺序组织，而不是按完整业务路径平铺。",
        "edge_reasoning": "edges.propagation_mechanism 必须解释污染内容、指令、权限或状态怎样移动。",
        "terminal_summary": "impact.summary 必须说明最终影响、阻断结果或无可行路径。",
        "defensive_scope": "输出保持防御性和高层描述，不包含攻击 payload 或操作细节。",
    },
}

BASE_STRATEGY: dict[str, Any] = {
    "strategy_id": "threat_propagation_parsing_strategy",
    "framework_id": "threat_propagation_framework",
    "version": "1.0",
    "summary": "定义人类与 DeepSeek 应如何使用 threat_propagation_framework 解析真实 skill 的威胁传播链。",
    "reading_order": [
        "先读第一阶段 threat_analysis，定位已知攻击入口和 propagation_candidates。",
        "再读 framework 的 core_components 与 decision_gates，明确本轮链路需要解释哪些结构。",
        "然后读 parsed graph，沿真实节点和边选择最短、最可解释的攻击传播路径。",
        "最后用 output_contract 检查 path、edges 和 impact 是否完整。",
    ],
    "steps": [
        {
            "id": "find_entry",
            "title": "定位入口",
            "instruction": "优先从 threat_analysis.attack_location 选择最早接触不可信内容的工具或节点作为起点。",
            "target_components": ["entry"],
            "required_fields": ["starting_attack_point.reason", "path[*].reason"],
        },
        {
            "id": "trace_carrier",
            "title": "跟踪载体",
            "instruction": "沿 graph 中真实后继边，说明污染内容怎样变成输出、参数、上下文或状态并继续传播。",
            "target_components": ["carrier"],
            "required_fields": ["path[*].threat_state", "edges[*].propagation_mechanism"],
        },
        {
            "id": "mark_trust_shift",
            "title": "标注信任跨越",
            "instruction": "只在污染内容被后续节点当作可信输入、指令、约束或执行计划时，标记 trust_shift。",
            "target_components": ["trust_shift"],
            "required_fields": ["path[*].reason", "edges[*].propagation_mechanism"],
        },
        {
            "id": "locate_effect_or_containment",
            "title": "定位影响或阻断",
            "instruction": "识别链路是在权限/状态/外部副作用处显化，还是被阻断、回滚、人工确认或无路径结束。",
            "target_components": ["effect", "containment", "terminal"],
            "required_fields": ["path[*].reason", "impact.summary"],
        },
        {
            "id": "close_with_outcome",
            "title": "收束终局",
            "instruction": "用 impact.summary 给出防御性终局结论，并确保 path 的最后一个节点与终局一致。",
            "target_components": ["terminal"],
            "required_fields": ["impact.summary"],
        },
    ],
    "slot_mapping_rules": [
        {
            "slot_id": "entry",
            "fields": ["starting_attack_point.reason", "path[*].reason", "path[*].threat_state"],
            "role_hints": ["entry"],
            "match_signals": ["攻击入口", "被攻击工具", "不可信输入", "用户请求"],
        },
        {
            "slot_id": "carrier",
            "fields": ["path[*].threat_state", "edges[*].propagation_mechanism"],
            "role_hints": ["propagation"],
            "match_signals": ["传播", "输出", "参数", "上下文", "状态"],
        },
        {
            "slot_id": "trust_shift",
            "fields": ["path[*].reason", "edges[*].propagation_mechanism"],
            "role_hints": ["propagation"],
            "match_signals": ["信任边界", "高信任", "可信上下文", "被当作指令", "提升为可信"],
        },
        {
            "slot_id": "effect",
            "fields": ["path[*].reason", "edges[*].propagation_mechanism", "impact.summary"],
            "role_hints": ["impact", "propagation"],
            "match_signals": ["权限", "写入", "调用", "发送", "外部影响", "资源消耗"],
        },
        {
            "slot_id": "containment",
            "fields": ["path[*].reason", "impact.summary"],
            "role_hints": ["containment"],
            "match_signals": ["阻断", "回滚", "人工确认", "allowlist", "拒绝"],
        },
        {
            "slot_id": "terminal",
            "fields": ["impact.summary", "path[*].reason"],
            "role_hints": ["impact", "containment"],
            "match_signals": ["最终影响", "影响达成", "传播中断", "无可行攻击路径", "返回给用户"],
        },
    ],
    "gate_mapping_rules": [
        {
            "gate_id": "untrusted_source",
            "fields": ["starting_attack_point.reason", "path[*].reason"],
            "match_signals": ["用户请求", "外部输入", "网页", "文件", "API", "日志", "数据库", "记忆"],
        },
        {
            "gate_id": "trust_promoted",
            "fields": ["path[*].reason", "edges[*].propagation_mechanism"],
            "match_signals": ["可信上下文", "工具参数", "执行计划", "后续 prompt", "信任边界"],
        },
        {
            "gate_id": "external_effect",
            "fields": ["edges[*].propagation_mechanism", "impact.summary"],
            "match_signals": ["返回给用户", "调用 API", "发消息", "写文件", "提交"],
        },
        {
            "gate_id": "contained",
            "fields": ["path[*].reason", "impact.summary"],
            "match_signals": ["人工确认", "allowlist", "schema 校验", "隔离", "沙箱", "回滚", "拒绝"],
        },
    ],
    "quality_checks": [
        "starting_attack_point 与 path[0] 必须指向同一个真实入口节点。",
        "每条 edges.propagation_mechanism 都要解释传播的资产、指令、权限或状态。",
        "如果链路被阻断，path.reason 或 impact.summary 必须点明阻断机制。",
        "impact.summary 必须与 path 的最后一个节点语义一致。",
    ],
    "metric_guidance": [
        {
            "metric_id": FRAMEWORK_SLOT_METRIC,
            "improvement_hint": "优先补齐真实链路中经常出现但当前 core_components 无法稳定解释的阶段。",
        },
        {
            "metric_id": STRATEGY_STEP_METRIC,
            "improvement_hint": "优先简化或补全 mapping_rules 与 quality_checks，让更多真实链路可以被一致解析。",
        },
    ],
}


def deep_copy_base_framework() -> dict[str, Any]:
    return deepcopy(BASE_FRAMEWORK)


def deep_copy_base_strategy() -> dict[str, Any]:
    return deepcopy(BASE_STRATEGY)


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def contains_signal(texts: list[str], signals: list[str]) -> bool:
    normalized_texts = [normalize_text(text) for text in texts if normalize_text(text)]
    if not normalized_texts:
        return False
    for signal in signals:
        normalized_signal = normalize_text(signal)
        if not normalized_signal:
            continue
        for text in normalized_texts:
            if normalized_signal in text:
                return True
    return False


def unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _safe_id(value: Any, *, prefix: str) -> str:
    text = normalize_text(value).replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = text.strip("_")
    if not text:
        text = prefix
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,79}", text):
        text = f"{prefix}_{text[:32] or 'item'}"
        text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return text[:80]


def _safe_signal_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return unique_strings([str(value or "").strip() for value in values if str(value or "").strip()])[:12]


def _safe_component(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    component_id = _safe_id(value.get("id"), prefix="component")
    label = str(value.get("label") or component_id).strip()[:40]
    definition = str(value.get("definition") or "").strip()[:400]
    if not label or not definition:
        return None
    return {
        "id": component_id,
        "label": label,
        "required": bool(value.get("required", False)),
        "definition": definition,
        "role_hints": unique_strings(value.get("role_hints") if isinstance(value.get("role_hints"), list) else []),
        "match_signals": _safe_signal_list(value.get("match_signals")),
    }


def _safe_gate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    gate_id = _safe_id(value.get("id"), prefix="gate")
    question = str(value.get("question") or "").strip()[:280]
    true_effect = str(value.get("true_effect") or "").strip()[:220]
    if not question or not true_effect:
        return None
    return {
        "id": gate_id,
        "question": question,
        "true_effect": true_effect,
        "match_signals": _safe_signal_list(value.get("match_signals")),
    }


def _safe_axis(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    axis_id = _safe_id(value.get("id"), prefix="axis")
    label = str(value.get("label") or axis_id).strip()[:40]
    definition = str(value.get("definition") or "").strip()[:240]
    if not label or not definition:
        return None
    return {"id": axis_id, "label": label, "definition": definition}


def _safe_strategy_step(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    step_id = _safe_id(value.get("id"), prefix="step")
    title = str(value.get("title") or step_id).strip()[:60]
    instruction = str(value.get("instruction") or "").strip()[:500]
    target_components = unique_strings(value.get("target_components") if isinstance(value.get("target_components"), list) else [])
    required_fields = unique_strings(value.get("required_fields") if isinstance(value.get("required_fields"), list) else [])
    if not title or not instruction or not target_components:
        return None
    return {
        "id": step_id,
        "title": title,
        "instruction": instruction,
        "target_components": target_components,
        "required_fields": required_fields,
    }


def _safe_mapping_rule(value: Any, *, id_key: str, prefix: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    rule_id = _safe_id(value.get(id_key), prefix=prefix)
    fields = unique_strings(value.get("fields") if isinstance(value.get("fields"), list) else [])
    role_hints = unique_strings(value.get("role_hints") if isinstance(value.get("role_hints"), list) else [])
    signals = _safe_signal_list(value.get("match_signals"))
    if not fields and not signals and not role_hints:
        return None
    result = {id_key: rule_id, "fields": fields, "match_signals": signals}
    if role_hints:
        result["role_hints"] = role_hints
    return result


def derive_framework_definition(chain_template: dict[str, Any] | None = None) -> dict[str, Any]:
    framework = deep_copy_base_framework()
    if not isinstance(chain_template, dict):
        return framework

    stage_map = {
        "attacked_tool": "entry",
        "tainted_output": "carrier",
        "trust_boundary_crossing": "trust_shift",
        "instruction_interpretation": "trust_shift",
        "privileged_action": "effect",
        "impact_or_containment": "terminal",
    }
    anchors = chain_template.get("anchors") if isinstance(chain_template.get("anchors"), dict) else {}
    propagation = anchors.get("propagation_chain") if isinstance(anchors.get("propagation_chain"), dict) else {}
    gate_units = anchors.get("gate_units") if isinstance(anchors.get("gate_units"), dict) else {}

    components_by_id = {item["id"]: item for item in framework["core_components"]}
    for stage in propagation.get("required_stages", []):
        if not isinstance(stage, dict):
            continue
        target_id = stage_map.get(str(stage.get("id") or ""))
        if not target_id or target_id not in components_by_id:
            continue
        component = components_by_id[target_id]
        description = str(stage.get("description") or "").strip()
        if description:
            component["definition"] = description
        if stage.get("required") is True:
            component["required"] = True
        role = str(stage.get("role") or "").strip()
        if role and role not in component["role_hints"]:
            component["role_hints"].append(role)

    gate_map = {
        "untrusted_input_gate": "untrusted_source",
        "sanitization_gate": "sanitized",
        "trust_promotion_gate": "trust_promoted",
        "permission_gate": "state_or_privilege",
        "state_persistence_gate": "state_or_privilege",
        "external_effect_gate": "external_effect",
        "containment_gate": "contained",
    }
    gates_by_id = {item["id"]: item for item in framework["decision_gates"]}
    for gate in gate_units.get("gate_catalog", []):
        if not isinstance(gate, dict):
            continue
        target_id = gate_map.get(str(gate.get("id") or ""))
        if not target_id or target_id not in gates_by_id:
            continue
        target = gates_by_id[target_id]
        question = str(gate.get("question") or "").strip()
        true_effect = str(gate.get("true_effect") or "").strip()
        if question:
            target["question"] = question
        if true_effect:
            target["true_effect"] = true_effect

    output_contract = chain_template.get("output_contract")
    if isinstance(output_contract, dict):
        contract = framework["output_contract"]
        for target_key, source_key in (
            ("chain_start", "starting_attack_point_rule"),
            ("path_order", "path_rule"),
            ("edge_reasoning", "edge_rule"),
            ("terminal_summary", "impact_rule"),
            ("defensive_scope", "gate_reasoning_rule"),
        ):
            value = str(output_contract.get(source_key) or "").strip()
            if value:
                contract[target_key] = value
    return framework


def derive_parsing_strategy(framework: dict[str, Any], chain_template: dict[str, Any] | None = None) -> dict[str, Any]:
    strategy = deep_copy_base_strategy()
    strategy["framework_id"] = str(framework.get("framework_id") or BASE_STRATEGY["framework_id"])
    if not isinstance(chain_template, dict):
        return strategy
    output_contract = chain_template.get("output_contract")
    if isinstance(output_contract, dict):
        quality_checks = strategy.setdefault("quality_checks", [])
        for key in ("starting_attack_point_rule", "path_rule", "edge_rule", "impact_rule"):
            text = str(output_contract.get(key) or "").strip()
            if text and text not in quality_checks:
                quality_checks.append(text)
    return strategy


def validate_framework_definition(framework: Any) -> dict[str, Any]:
    if not isinstance(framework, dict):
        raise ValueError("Framework must be a JSON object")
    if not str(framework.get("framework_id") or "").strip():
        raise ValueError("Framework missing framework_id")
    components = framework.get("core_components")
    if not isinstance(components, list) or not components:
        raise ValueError("Framework missing core_components")
    component_ids: set[str] = set()
    for component in components:
        safe_component = _safe_component(component)
        if safe_component is None:
            raise ValueError(f"Invalid component: {component!r}")
        if safe_component["id"] in component_ids:
            raise ValueError(f"Duplicate component id: {safe_component['id']}")
        component_ids.add(safe_component["id"])
    gates = framework.get("decision_gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("Framework missing decision_gates")
    gate_ids: set[str] = set()
    for gate in gates:
        safe_gate = _safe_gate(gate)
        if safe_gate is None:
            raise ValueError(f"Invalid gate: {gate!r}")
        if safe_gate["id"] in gate_ids:
            raise ValueError(f"Duplicate gate id: {safe_gate['id']}")
        gate_ids.add(safe_gate["id"])
    axes = framework.get("evidence_axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("Framework missing evidence_axes")
    for axis in axes:
        if _safe_axis(axis) is None:
            raise ValueError(f"Invalid evidence axis: {axis!r}")
    contract = framework.get("output_contract")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("Framework missing output_contract")
    return framework


def validate_parsing_strategy(strategy: Any, framework: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(strategy, dict):
        raise ValueError("Parsing strategy must be a JSON object")
    if str(strategy.get("framework_id") or "") != str(framework.get("framework_id") or ""):
        raise ValueError("Parsing strategy framework_id mismatch")
    if not isinstance(strategy.get("steps"), list) or not strategy["steps"]:
        raise ValueError("Parsing strategy missing steps")
    component_ids = {item["id"] for item in framework["core_components"]}
    for step in strategy["steps"]:
        safe_step = _safe_strategy_step(step)
        if safe_step is None:
            raise ValueError(f"Invalid strategy step: {step!r}")
        unknown_components = set(safe_step["target_components"]) - component_ids
        if unknown_components:
            raise ValueError(f"Strategy step references unknown components: {sorted(unknown_components)}")
    slot_rules = strategy.get("slot_mapping_rules")
    if not isinstance(slot_rules, list) or not slot_rules:
        raise ValueError("Parsing strategy missing slot_mapping_rules")
    for rule in slot_rules:
        safe_rule = _safe_mapping_rule(rule, id_key="slot_id", prefix="slot")
        if safe_rule is None:
            raise ValueError(f"Invalid slot mapping rule: {rule!r}")
        if safe_rule["slot_id"] not in component_ids:
            raise ValueError(f"Slot mapping references unknown slot: {safe_rule['slot_id']}")
    gate_ids = {item["id"] for item in framework["decision_gates"]}
    gate_rules = strategy.get("gate_mapping_rules")
    if not isinstance(gate_rules, list) or not gate_rules:
        raise ValueError("Parsing strategy missing gate_mapping_rules")
    for rule in gate_rules:
        safe_rule = _safe_mapping_rule(rule, id_key="gate_id", prefix="gate")
        if safe_rule is None:
            raise ValueError(f"Invalid gate mapping rule: {rule!r}")
        if safe_rule["gate_id"] not in gate_ids:
            raise ValueError(f"Gate mapping references unknown gate: {safe_rule['gate_id']}")
    if not isinstance(strategy.get("quality_checks"), list) or not strategy["quality_checks"]:
        raise ValueError("Parsing strategy missing quality_checks")
    return strategy


def load_framework_definition(path: str, read_text_fn) -> dict[str, Any]:
    framework = json.loads(read_text_fn(path))
    return validate_framework_definition(framework)


def load_parsing_strategy(path: str, read_text_fn, framework: dict[str, Any]) -> dict[str, Any]:
    strategy = json.loads(read_text_fn(path))
    return validate_parsing_strategy(strategy, framework)


TOKEN_RE = re.compile(r"([^.[\]]+)(\[\*\])?")


def extract_field_values(data: Any, field_path: str) -> list[str]:
    if not field_path:
        return []
    current: list[Any] = [data]
    for part in field_path.split("."):
        match = TOKEN_RE.fullmatch(part)
        if not match:
            return []
        key, iterate = match.groups()
        next_values: list[Any] = []
        for item in current:
            if not isinstance(item, dict):
                continue
            value = item.get(key)
            if value is None:
                continue
            if iterate:
                if isinstance(value, list):
                    next_values.extend(value)
            else:
                next_values.append(value)
        current = next_values
        if not current:
            return []
    result: list[str] = []
    for item in current:
        if isinstance(item, list):
            result.extend(str(value) for value in item if str(value or "").strip())
        elif isinstance(item, dict):
            for value in item.values():
                if str(value or "").strip():
                    result.append(str(value))
        elif str(item or "").strip():
            result.append(str(item))
    return result


def chain_text_bundle(chain: dict[str, Any]) -> dict[str, list[str]]:
    bundle = {
        "starting_attack_point": extract_field_values(chain, "starting_attack_point.reason"),
        "path_reason": extract_field_values(chain, "path[*].reason"),
        "path_state": extract_field_values(chain, "path[*].threat_state"),
        "edge_reason": extract_field_values(chain, "edges[*].propagation_mechanism"),
        "impact": extract_field_values(chain, "impact.summary"),
    }
    bundle["all"] = []
    for values in bundle.values():
        bundle["all"].extend(values)
    return bundle


def _component_rule_map(strategy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for rule in strategy.get("slot_mapping_rules", []):
        if isinstance(rule, dict) and rule.get("slot_id"):
            mapping[str(rule["slot_id"])] = rule
    return mapping


UNIT_AXIS_SIGNAL_HINTS: dict[str, list[str]] = {
    "content_flow": ["污染", "内容", "数据", "输出", "参数", "上下文", "传递", "结果", "载体"],
    "instruction_flow": ["指令", "命令", "prompt", "计划", "约束", "执行", "当作指令", "解析"],
    "authority_flow": ["权限", "授权", "allowed", "信任", "可信", "访问", "越权", "高信任"],
    "state_flow": ["状态", "会话", "缓存", "记忆", "文件", "数据库", "持久", "配置"],
    "effect_flow": ["返回", "调用", "发送", "写入", "提交", "泄露", "影响", "副作用", "执行"],
}

UNIT_COMPONENT_BY_TYPE: dict[str, list[str]] = {
    "entry": ["entry"],
    "path_step": ["carrier"],
    "edge_propagation": ["carrier"],
    "terminal": ["terminal"],
}

UNIT_COMPONENT_BY_ROLE: dict[str, list[str]] = {
    "entry": ["entry"],
    "propagation": ["carrier"],
    "impact": ["effect", "terminal"],
    "containment": ["containment", "terminal"],
}

UNIT_STEP_BY_TYPE: dict[str, list[str]] = {
    "entry": ["find_entry"],
    "path_step": ["trace_carrier", "mark_trust_shift", "locate_effect_or_containment"],
    "edge_propagation": ["trace_carrier", "mark_trust_shift"],
    "terminal": ["locate_effect_or_containment", "close_with_outcome"],
}

TRUST_SHIFT_HINTS = ["信任", "可信", "高信任", "当作指令", "提升", "工具参数", "执行计划", "prompt", "allowed"]
CONTAINMENT_HINTS = ["阻断", "清洗", "确认", "allowlist", "schema", "沙箱", "回滚", "拒绝", "无可行"]


def _field_matches(rule_field: str, unit_fields: list[str]) -> bool:
    return any(rule_field == field for field in unit_fields)


def _unit_text(unit: dict[str, Any]) -> str:
    return " ".join(str(value or "") for value in unit.get("texts", []) if str(value or "").strip())


def _unit_weight(unit_type: str, role: str, text: str) -> float:
    if unit_type == "edge_propagation":
        return 1.5
    if role == "propagation" and contains_signal([text], TRUST_SHIFT_HINTS):
        return 1.5
    return 1.0


def _add_unit(
    units: list[dict[str, Any]],
    *,
    unit_id: str,
    unit_type: str,
    role: str = "",
    fields: list[str] | None = None,
    texts: list[Any] | None = None,
    node_id: str = "",
    node_name: str = "",
) -> None:
    clean_texts = unique_strings([str(value or "") for value in (texts or []) if str(value or "").strip()])
    if not clean_texts:
        return
    text = " ".join(clean_texts)
    units.append(
        {
            "unit_id": unit_id,
            "unit_type": unit_type,
            "role": role,
            "fields": fields or [],
            "texts": clean_texts,
            "text": text,
            "node_id": node_id,
            "node_name": node_name,
            "weight": _unit_weight(unit_type, role, text),
        }
    )


def extract_threat_units(chain: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    start = chain.get("starting_attack_point") if isinstance(chain.get("starting_attack_point"), dict) else {}
    if start:
        _add_unit(
            units,
            unit_id="entry:starting_attack_point",
            unit_type="entry",
            role="entry",
            fields=["starting_attack_point.node_id", "starting_attack_point.node_name", "starting_attack_point.reason"],
            texts=[start.get("node_name"), start.get("reason")],
            node_id=str(start.get("node_id") or ""),
            node_name=str(start.get("node_name") or ""),
        )

    path = chain.get("path") if isinstance(chain.get("path"), list) else []
    for index, step in enumerate(path):
        if not isinstance(step, dict):
            continue
        role = str(step.get("role") or "").strip()
        fields = ["path[*].node_id", "path[*].node_name", "path[*].role"]
        if str(step.get("threat_state") or "").strip():
            fields.append("path[*].threat_state")
        if str(step.get("reason") or "").strip():
            fields.append("path[*].reason")
        _add_unit(
            units,
            unit_id=f"path:{index}:{step.get('node_id') or ''}",
            unit_type="path_step",
            role=role,
            fields=fields,
            texts=[step.get("node_name"), role, step.get("threat_state"), step.get("reason")],
            node_id=str(step.get("node_id") or ""),
            node_name=str(step.get("node_name") or ""),
        )

    edges = chain.get("edges") if isinstance(chain.get("edges"), list) else []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        fields = ["edges[*].src_node_id", "edges[*].dst_node_id"]
        if str(edge.get("condition") or "").strip():
            fields.append("edges[*].condition")
        if str(edge.get("propagation_mechanism") or "").strip():
            fields.append("edges[*].propagation_mechanism")
        _add_unit(
            units,
            unit_id=f"edge:{index}:{edge.get('src_node_id') or ''}->{edge.get('dst_node_id') or ''}",
            unit_type="edge_propagation",
            role="propagation",
            fields=fields,
            texts=[edge.get("condition"), edge.get("propagation_mechanism")],
            node_id=str(edge.get("src_node_id") or ""),
            node_name=f"{edge.get('src_node_id') or ''}->{edge.get('dst_node_id') or ''}",
        )

    impact = chain.get("impact") if isinstance(chain.get("impact"), dict) else {}
    terminal_texts = [impact.get("summary"), chain.get("risk_level"), chain.get("confidence")]
    if impact:
        _add_unit(
            units,
            unit_id="terminal:impact",
            unit_type="terminal",
            role="containment" if contains_signal([str(impact.get("summary") or "")], CONTAINMENT_HINTS) else "impact",
            fields=["impact.summary", "risk_level", "confidence"],
            texts=terminal_texts,
        )
    return units


def match_framework_unit(unit: dict[str, Any], framework: dict[str, Any]) -> dict[str, Any]:
    text = _unit_text(unit)
    role = str(unit.get("role") or "")
    unit_type = str(unit.get("unit_type") or "")
    default_components = set(UNIT_COMPONENT_BY_TYPE.get(unit_type, [])) | set(UNIT_COMPONENT_BY_ROLE.get(role, []))
    if contains_signal([text], TRUST_SHIFT_HINTS):
        default_components.add("trust_shift")
    if contains_signal([text], CONTAINMENT_HINTS):
        default_components.add("containment")

    component_matches: list[str] = []
    for component in framework.get("core_components", []):
        if not isinstance(component, dict):
            continue
        component_id = str(component.get("id") or "")
        role_hints = {str(value) for value in component.get("role_hints", [])}
        signals = unique_strings(component.get("match_signals", []))
        if component_id in default_components or role in role_hints or contains_signal([text], signals):
            component_matches.append(component_id)

    gate_defaults: set[str] = set()
    if unit_type == "entry":
        gate_defaults.add("untrusted_source")
    if unit_type == "terminal" or role == "impact":
        gate_defaults.add("external_effect")
    if role == "containment" or contains_signal([text], CONTAINMENT_HINTS):
        gate_defaults.add("contained")
    if contains_signal([text], TRUST_SHIFT_HINTS):
        gate_defaults.add("trust_promoted")

    gate_matches: list[str] = []
    for gate in framework.get("decision_gates", []):
        if not isinstance(gate, dict):
            continue
        gate_id = str(gate.get("id") or "")
        if gate_id in gate_defaults or contains_signal([text], unique_strings(gate.get("match_signals", []))):
            gate_matches.append(gate_id)

    axis_matches: list[str] = []
    for axis in framework.get("evidence_axes", []):
        if not isinstance(axis, dict):
            continue
        axis_id = str(axis.get("id") or "")
        signals = UNIT_AXIS_SIGNAL_HINTS.get(axis_id, []) + [axis.get("label", ""), axis.get("definition", "")]
        if contains_signal([text], unique_strings(signals)):
            axis_matches.append(axis_id)

    return {
        "components": sorted(set(component_matches)),
        "decision_gates": sorted(set(gate_matches)),
        "evidence_axes": sorted(set(axis_matches)),
        "covered": bool(component_matches or gate_matches or axis_matches),
    }


def _strategy_step_covers_unit(step: dict[str, Any], unit: dict[str, Any], framework_matches: dict[str, Any]) -> bool:
    step_id = str(step.get("id") or "")
    unit_type = str(unit.get("unit_type") or "")
    unit_fields = [str(field) for field in unit.get("fields", [])]
    required_fields = [str(field) for field in step.get("required_fields", [])]
    target_components = {str(value) for value in step.get("target_components", [])}
    component_overlap = bool(target_components & set(framework_matches.get("components", [])))
    default_step = step_id in UNIT_STEP_BY_TYPE.get(unit_type, [])
    field_overlap = any(_field_matches(field, unit_fields) for field in required_fields)
    return (component_overlap or default_step) and field_overlap


def _mapping_rule_covers_unit(rule: dict[str, Any], unit: dict[str, Any], id_key: str, framework_matches: dict[str, Any]) -> bool:
    unit_fields = [str(field) for field in unit.get("fields", [])]
    fields = [str(field) for field in rule.get("fields", [])]
    field_overlap = any(_field_matches(field, unit_fields) for field in fields)
    if not field_overlap:
        return False
    rule_id = str(rule.get(id_key) or "")
    if id_key == "slot_id" and rule_id in set(framework_matches.get("components", [])):
        return True
    if id_key == "gate_id" and rule_id in set(framework_matches.get("decision_gates", [])):
        return True
    return contains_signal([_unit_text(unit)], unique_strings(rule.get("match_signals", [])))


def match_strategy_unit(unit: dict[str, Any], framework_matches: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    step_matches = [
        str(step.get("id"))
        for step in strategy.get("steps", [])
        if isinstance(step, dict) and _strategy_step_covers_unit(step, unit, framework_matches)
    ]
    slot_matches = [
        str(rule.get("slot_id"))
        for rule in strategy.get("slot_mapping_rules", [])
        if isinstance(rule, dict) and _mapping_rule_covers_unit(rule, unit, "slot_id", framework_matches)
    ]
    gate_matches = [
        str(rule.get("gate_id"))
        for rule in strategy.get("gate_mapping_rules", [])
        if isinstance(rule, dict) and _mapping_rule_covers_unit(rule, unit, "gate_id", framework_matches)
    ]
    return {
        "steps": sorted(set(step_matches)),
        "slot_mapping_rules": sorted(set(slot_matches)),
        "gate_mapping_rules": sorted(set(gate_matches)),
        "covered": bool(step_matches or slot_matches or gate_matches),
    }


def _weighted_score(units: list[dict[str, Any]], key: str) -> float:
    total_weight = sum(float(unit.get("weight", 1.0)) for unit in units)
    if total_weight <= 0:
        return 0.0
    covered_weight = sum(float(unit.get("weight", 1.0)) for unit in units if unit.get(key))
    return round(covered_weight / total_weight, 4)


def _unit_type_coverage(units: list[dict[str, Any]], key: str) -> dict[str, float]:
    by_type: dict[str, dict[str, float]] = {}
    for unit in units:
        unit_type = str(unit.get("unit_type") or "unknown")
        entry = by_type.setdefault(unit_type, {"covered": 0.0, "total": 0.0})
        weight = float(unit.get("weight", 1.0))
        entry["total"] += weight
        if unit.get(key):
            entry["covered"] += weight
    return {
        unit_type: round(values["covered"] / values["total"], 4) if values["total"] else 0.0
        for unit_type, values in sorted(by_type.items())
    }


def _summarize_unit(unit: dict[str, Any], *, skill_id: Any = None, chain_id: Any = None, max_text: int = 240) -> dict[str, Any]:
    return {
        "skill_id": skill_id,
        "chain_id": chain_id,
        "unit_id": unit.get("unit_id"),
        "unit_type": unit.get("unit_type"),
        "role": unit.get("role"),
        "fields": unit.get("fields", []),
        "weight": unit.get("weight", 1.0),
        "text": str(unit.get("text") or "")[:max_text],
        "covered_by_framework": bool(unit.get("covered_by_framework")),
        "covered_by_strategy": bool(unit.get("covered_by_strategy")),
        "framework_matches": unit.get("framework_matches", {}),
        "strategy_matches": unit.get("strategy_matches", {}),
    }


def evaluate_chain_against_framework(
    chain: dict[str, Any],
    framework: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    units = extract_threat_units(chain)
    for unit in units:
        framework_matches = match_framework_unit(unit, framework)
        strategy_matches = match_strategy_unit(unit, framework_matches, strategy)
        unit["framework_matches"] = framework_matches
        unit["strategy_matches"] = strategy_matches
        unit["covered_by_framework"] = framework_matches["covered"]
        unit["covered_by_strategy"] = strategy_matches["covered"]

    component_ids = [str(item.get("id")) for item in framework.get("core_components", []) if isinstance(item, dict)]
    step_ids = [str(item.get("id")) for item in strategy.get("steps", []) if isinstance(item, dict)]
    component_hits = {
        component_id: any(component_id in unit.get("framework_matches", {}).get("components", []) for unit in units)
        for component_id in component_ids
    }
    step_hits = {
        step_id: any(step_id in unit.get("strategy_matches", {}).get("steps", []) for unit in units)
        for step_id in step_ids
    }
    missing_components = [
        str(component.get("id"))
        for component in framework.get("core_components", [])
        if isinstance(component, dict) and component.get("required") is True and not component_hits.get(str(component.get("id")))
    ]
    missing_steps = [step_id for step_id, hit in step_hits.items() if not hit]
    uncovered_units = [
        _summarize_unit(unit, chain_id=chain.get("id"))
        for unit in units
        if not unit.get("covered_by_framework") or not unit.get("covered_by_strategy")
    ]

    return {
        "unit_count": len(units),
        "unit_weight": round(sum(float(unit.get("weight", 1.0)) for unit in units), 4),
        "semantic_units": [_summarize_unit(unit, chain_id=chain.get("id"), max_text=180) for unit in units],
        "component_hits": component_hits,
        "component_details": {
            component_id: {"hit": hit, "unit_count": sum(1 for unit in units if component_id in unit.get("framework_matches", {}).get("components", []))}
            for component_id, hit in component_hits.items()
        },
        "step_hits": step_hits,
        "step_details": {
            step_id: {"hit": hit, "unit_count": sum(1 for unit in units if step_id in unit.get("strategy_matches", {}).get("steps", []))}
            for step_id, hit in step_hits.items()
        },
        FRAMEWORK_SLOT_METRIC: _weighted_score(units, "covered_by_framework"),
        STRATEGY_STEP_METRIC: _weighted_score(units, "covered_by_strategy"),
        "framework_unit_type_coverage": _unit_type_coverage(units, "covered_by_framework"),
        "strategy_unit_type_coverage": _unit_type_coverage(units, "covered_by_strategy"),
        "missing_components": missing_components,
        "missing_steps": missing_steps,
        "uncovered_units": uncovered_units,
    }


def _weighted_average(items: list[dict[str, Any]], metric: str) -> float:
    total_weight = sum(float(item.get("unit_weight", 0.0)) for item in items)
    if total_weight <= 0:
        return 0.0
    return round(sum(float(item.get(metric, 0.0)) * float(item.get("unit_weight", 0.0)) for item in items) / total_weight, 4)


def _simple_average(items: list[dict[str, Any]], metric: str) -> float:
    if not items:
        return 0.0
    return round(sum(float(item.get(metric, 0.0)) for item in items) / len(items), 4)


def _merge_unit_type_coverage(items: list[dict[str, Any]], key: str) -> dict[str, float]:
    totals: dict[str, dict[str, float]] = {}
    for item in items:
        for unit in item.get("semantic_units", []):
            unit_type = str(unit.get("unit_type") or "unknown")
            entry = totals.setdefault(unit_type, {"covered": 0.0, "total": 0.0})
            weight = float(unit.get("weight", 1.0))
            entry["total"] += weight
            if unit.get(key):
                entry["covered"] += weight
    return {
        unit_type: round(values["covered"] / values["total"], 4) if values["total"] else 0.0
        for unit_type, values in sorted(totals.items())
    }


def evaluate_result_coverage(
    result: dict[str, Any],
    framework: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    propagation = normalize_propagation_schema(result.get("propagation"), skill_id=str(result.get("skill_id") or ""))
    chains = propagation.get("chains") if isinstance(propagation.get("chains"), list) else []
    chain_evaluations: list[dict[str, Any]] = []
    for chain in chains:
        if not isinstance(chain, dict):
            continue
        evaluation = evaluate_chain_against_framework(chain, framework, strategy)
        chain_evaluations.append(
            {
                "chain_id": str(chain.get("id") or ""),
                "title": str(chain.get("title") or ""),
                **evaluation,
            }
        )

    if not chain_evaluations:
        return {
            "skill_id": result.get("skill_id"),
            "chain_count": 0,
            "unit_count": 0,
            "unit_weight": 0.0,
            FRAMEWORK_SLOT_METRIC: 0.0,
            STRATEGY_STEP_METRIC: 0.0,
            "component_hit_rate": {},
            "step_hit_rate": {},
            "framework_unit_type_coverage": {},
            "strategy_unit_type_coverage": {},
            "uncovered_units": [],
            "chains": [],
        }

    component_ids = [item["id"] for item in framework.get("core_components", []) if isinstance(item, dict)]
    step_ids = [item["id"] for item in strategy.get("steps", []) if isinstance(item, dict)]
    component_hit_rate = {
        component_id: round(
            sum(1 for evaluation in chain_evaluations if evaluation["component_hits"].get(component_id)) / len(chain_evaluations),
            4,
        )
        for component_id in component_ids
    }
    step_hit_rate = {
        step_id: round(
            sum(1 for evaluation in chain_evaluations if evaluation["step_hits"].get(step_id)) / len(chain_evaluations),
            4,
        )
        for step_id in step_ids
    }
    uncovered_units = []
    for evaluation in chain_evaluations:
        for unit in evaluation["uncovered_units"]:
            enriched = dict(unit)
            enriched["skill_id"] = result.get("skill_id")
            enriched["chain_id"] = evaluation["chain_id"]
            uncovered_units.append(enriched)
    return {
        "skill_id": result.get("skill_id"),
        "chain_count": len(chain_evaluations),
        "unit_count": sum(int(evaluation.get("unit_count", 0)) for evaluation in chain_evaluations),
        "unit_weight": round(sum(float(evaluation.get("unit_weight", 0.0)) for evaluation in chain_evaluations), 4),
        FRAMEWORK_SLOT_METRIC: _weighted_average(chain_evaluations, FRAMEWORK_SLOT_METRIC),
        STRATEGY_STEP_METRIC: _weighted_average(chain_evaluations, STRATEGY_STEP_METRIC),
        "component_hit_rate": component_hit_rate,
        "step_hit_rate": step_hit_rate,
        "framework_unit_type_coverage": _merge_unit_type_coverage(chain_evaluations, "covered_by_framework"),
        "strategy_unit_type_coverage": _merge_unit_type_coverage(chain_evaluations, "covered_by_strategy"),
        "uncovered_units": uncovered_units,
        "chains": chain_evaluations,
    }


def aggregate_corpus_coverage(
    results: list[dict[str, Any]],
    framework: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    skill_evaluations = [evaluate_result_coverage(result, framework, strategy) for result in results]
    skill_evaluations = [evaluation for evaluation in skill_evaluations if evaluation["chain_count"] > 0]
    if not skill_evaluations:
        return {
            "skill_count": 0,
            "chain_count": 0,
            "unit_count": 0,
            "unit_weight": 0.0,
            "metrics": {
                FRAMEWORK_SLOT_METRIC: 0.0,
                STRATEGY_STEP_METRIC: 0.0,
            },
            "mean_metrics": {
                FRAMEWORK_SLOT_METRIC: 0.0,
                STRATEGY_STEP_METRIC: 0.0,
            },
            "component_hit_rate": {},
            "step_hit_rate": {},
            "framework_unit_type_coverage": {},
            "strategy_unit_type_coverage": {},
            "uncovered_units": [],
            "skill_scores": [],
        }

    chain_count = sum(evaluation["chain_count"] for evaluation in skill_evaluations)
    component_ids = [item["id"] for item in framework.get("core_components", []) if isinstance(item, dict)]
    step_ids = [item["id"] for item in strategy.get("steps", []) if isinstance(item, dict)]
    component_hit_rate = {
        component_id: round(
            sum(evaluation["component_hit_rate"].get(component_id, 0.0) * evaluation["chain_count"] for evaluation in skill_evaluations)
            / chain_count,
            4,
        )
        for component_id in component_ids
    }
    step_hit_rate = {
        step_id: round(
            sum(evaluation["step_hit_rate"].get(step_id, 0.0) * evaluation["chain_count"] for evaluation in skill_evaluations)
            / chain_count,
            4,
        )
        for step_id in step_ids
    }
    all_chain_evaluations = [chain for evaluation in skill_evaluations for chain in evaluation["chains"]]
    uncovered_units = [unit for evaluation in skill_evaluations for unit in evaluation.get("uncovered_units", [])]
    skill_scores = []
    for evaluation in skill_evaluations:
        missing_components = sorted({component_id for chain in evaluation["chains"] for component_id in chain["missing_components"]})
        missing_steps = sorted({step_id for chain in evaluation["chains"] for step_id in chain["missing_steps"]})
        skill_scores.append(
            {
                "skill_id": evaluation["skill_id"],
                "chain_count": evaluation["chain_count"],
                "unit_count": evaluation["unit_count"],
                "unit_weight": evaluation["unit_weight"],
                FRAMEWORK_SLOT_METRIC: evaluation[FRAMEWORK_SLOT_METRIC],
                STRATEGY_STEP_METRIC: evaluation[STRATEGY_STEP_METRIC],
                "missing_components": missing_components,
                "missing_steps": missing_steps,
                "uncovered_unit_count": len(evaluation.get("uncovered_units", [])),
                "uncovered_units": evaluation.get("uncovered_units", [])[:5],
            }
        )
    skill_scores.sort(
        key=lambda item: (
            item[FRAMEWORK_SLOT_METRIC] + item[STRATEGY_STEP_METRIC],
            -int(item.get("uncovered_unit_count", 0)),
            -float(item.get("unit_weight", 0.0)),
            item["skill_id"],
        )
    )
    return {
        "skill_count": len(skill_evaluations),
        "chain_count": chain_count,
        "unit_count": sum(evaluation["unit_count"] for evaluation in skill_evaluations),
        "unit_weight": round(sum(float(evaluation["unit_weight"]) for evaluation in skill_evaluations), 4),
        "metrics": {
            FRAMEWORK_SLOT_METRIC: _weighted_average(skill_evaluations, FRAMEWORK_SLOT_METRIC),
            STRATEGY_STEP_METRIC: _weighted_average(skill_evaluations, STRATEGY_STEP_METRIC),
        },
        "mean_metrics": {
            FRAMEWORK_SLOT_METRIC: _simple_average(skill_evaluations, FRAMEWORK_SLOT_METRIC),
            STRATEGY_STEP_METRIC: _simple_average(skill_evaluations, STRATEGY_STEP_METRIC),
        },
        "component_hit_rate": component_hit_rate,
        "step_hit_rate": step_hit_rate,
        "framework_unit_type_coverage": _merge_unit_type_coverage(all_chain_evaluations, "covered_by_framework"),
        "strategy_unit_type_coverage": _merge_unit_type_coverage(all_chain_evaluations, "covered_by_strategy"),
        "uncovered_units": uncovered_units[:100],
        "skill_scores": skill_scores,
    }


def build_metric_history_entry(
    iteration: int,
    coverage: dict[str, Any],
    *,
    reason: str = "",
    updated: bool = False,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "updated": updated,
        "reason": reason,
        "skill_count": coverage.get("skill_count", 0),
        "chain_count": coverage.get("chain_count", 0),
        "metrics": coverage.get("metrics", {}),
        "mean_metrics": coverage.get("mean_metrics", {}),
        "component_hit_rate": coverage.get("component_hit_rate", {}),
        "step_hit_rate": coverage.get("step_hit_rate", {}),
    }


def render_metric_brief(coverage: dict[str, Any], *, max_skills: int = 8) -> dict[str, Any]:
    framework_unit_type_coverage = (
        coverage.get("framework_unit_type_coverage")
        if isinstance(coverage.get("framework_unit_type_coverage"), dict)
        else {}
    )
    strategy_unit_type_coverage = (
        coverage.get("strategy_unit_type_coverage")
        if isinstance(coverage.get("strategy_unit_type_coverage"), dict)
        else {}
    )
    skill_scores = coverage.get("skill_scores") if isinstance(coverage.get("skill_scores"), list) else []
    uncovered_units = coverage.get("uncovered_units") if isinstance(coverage.get("uncovered_units"), list) else []
    return {
        "skill_count": coverage.get("skill_count", 0),
        "chain_count": coverage.get("chain_count", 0),
        "unit_count": coverage.get("unit_count", 0),
        "unit_weight": coverage.get("unit_weight", 0.0),
        "metrics": coverage.get("metrics", {}),
        "mean_metrics": coverage.get("mean_metrics", {}),
        "sample_info": coverage.get("sample_info", {}),
        "primary_reward": coverage.get("primary_reward", 0.0),
        "primary_reward_source": coverage.get("primary_reward_source", "z3"),
        "z3_reward_summary": coverage.get("z3_reward_summary", {}),
        "openclaw_runtime_reward_summary": coverage.get("openclaw_runtime_reward_summary", {}),
        "lowest_framework_unit_type_coverage": sorted(framework_unit_type_coverage.items(), key=lambda item: item[1])[:5],
        "lowest_strategy_unit_type_coverage": sorted(strategy_unit_type_coverage.items(), key=lambda item: item[1])[:5],
        "top_uncovered_units": uncovered_units[:max_skills],
        "lowest_scoring_skills": skill_scores[:max_skills],
    }


def plot_metric_history(history: list[dict[str, Any]], output_path: str) -> None:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None

    iterations = [int(item.get("iteration", 0)) for item in history]
    def metric_source(item: dict[str, Any]) -> dict[str, Any]:
        mean_metrics = item.get("mean_metrics")
        if isinstance(mean_metrics, dict) and mean_metrics:
            return mean_metrics
        metrics = item.get("metrics")
        return metrics if isinstance(metrics, dict) else {}

    framework_values = [float(metric_source(item).get(FRAMEWORK_SLOT_METRIC, 0.0)) for item in history]
    strategy_values = [float(metric_source(item).get(STRATEGY_STEP_METRIC, 0.0)) for item in history]

    if sns is not None:
        sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 5))
    if sns is not None:
        sns.lineplot(x=iterations, y=framework_values, marker="o", label="Framework Path Coverage", ax=axis)
        sns.lineplot(x=iterations, y=strategy_values, marker="o", label="Strategy Parse Coverage", ax=axis)
    else:
        axis.plot(iterations, framework_values, marker="o", label="Framework Path Coverage")
        axis.plot(iterations, strategy_values, marker="o", label="Strategy Parse Coverage")
        axis.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.8)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Score")
    axis.set_title("Threat Propagation Framework Coverage (Mean When Available)")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _index_by_id(items: list[dict[str, Any]], item_id: str = "id") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and item.get(item_id):
            result[str(item[item_id])] = item
    return result


def apply_framework_updates(
    framework: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    evolved = deepcopy(framework)
    framework_updates = updates.get("framework") if isinstance(updates.get("framework"), dict) else {}
    for section_key, protector, safe_fn in (
        ("core_components", PROTECTED_COMPONENT_IDS, _safe_component),
        ("decision_gates", PROTECTED_GATE_IDS, _safe_gate),
        ("evidence_axes", set(), _safe_axis),
    ):
        current_items = evolved.get(section_key)
        if not isinstance(current_items, list):
            continue
        section_updates = framework_updates.get(section_key) if isinstance(framework_updates.get(section_key), dict) else {}
        index = _index_by_id(current_items)

        for addition in section_updates.get("add", [])[:2]:
            safe_item = safe_fn(addition)
            if safe_item is None or safe_item["id"] in index:
                continue
            current_items.append(safe_item)
            index[safe_item["id"]] = safe_item

        for text_update in section_updates.get("text_updates", [])[:2]:
            safe_item = safe_fn(text_update)
            if safe_item is None:
                continue
            current = index.get(safe_item["id"])
            if current is None:
                continue
            for key, value in safe_item.items():
                if key == "id":
                    continue
                current[key] = value

        prune_ids = [str(value) for value in section_updates.get("prune_ids", [])[:1] if str(value)]
        if prune_ids:
            evolved[section_key] = [
                item
                for item in current_items
                if not (
                    isinstance(item, dict)
                    and str(item.get("id") or "") in prune_ids
                    and str(item.get("id") or "") not in protector
                )
            ]

    output_updates = framework_updates.get("output_contract")
    if isinstance(output_updates, dict):
        contract = evolved.get("output_contract")
        if isinstance(contract, dict):
            additions = output_updates.get("add") if isinstance(output_updates.get("add"), dict) else {}
            text_updates = output_updates.get("text_updates") if isinstance(output_updates.get("text_updates"), dict) else {}
            prune_keys = [str(value) for value in output_updates.get("prune_keys", [])[:1] if str(value)]
            for key, value in list(additions.items())[:2]:
                safe_key = _safe_id(key, prefix="contract")
                safe_value = str(value or "").strip()[:280]
                if safe_key not in contract and safe_value:
                    contract[safe_key] = safe_value
            for key, value in list(text_updates.items())[:2]:
                safe_key = str(key or "").strip()
                safe_value = str(value or "").strip()[:280]
                if safe_key in contract and safe_value:
                    contract[safe_key] = safe_value
            for key in prune_keys:
                if key in contract:
                    contract.pop(key, None)

    validate_framework_definition(evolved)
    return evolved


def apply_strategy_updates(
    strategy: dict[str, Any],
    updates: dict[str, Any],
    framework: dict[str, Any],
) -> dict[str, Any]:
    evolved = deepcopy(strategy)
    strategy_updates = updates.get("strategy") if isinstance(updates.get("strategy"), dict) else {}

    for section_key, protector, safe_fn, id_key in (
        ("steps", PROTECTED_STRATEGY_STEP_IDS, _safe_strategy_step, "id"),
        ("slot_mapping_rules", set(), lambda item: _safe_mapping_rule(item, id_key="slot_id", prefix="slot"), "slot_id"),
        ("gate_mapping_rules", set(), lambda item: _safe_mapping_rule(item, id_key="gate_id", prefix="gate"), "gate_id"),
    ):
        current_items = evolved.get(section_key)
        if not isinstance(current_items, list):
            continue
        section_updates = strategy_updates.get(section_key) if isinstance(strategy_updates.get(section_key), dict) else {}
        index = _index_by_id(current_items, item_id=id_key)

        for addition in section_updates.get("add", [])[:2]:
            safe_item = safe_fn(addition)
            if safe_item is None or safe_item[id_key] in index:
                continue
            current_items.append(safe_item)
            index[safe_item[id_key]] = safe_item

        for text_update in section_updates.get("text_updates", [])[:2]:
            safe_item = safe_fn(text_update)
            if safe_item is None:
                continue
            current = index.get(safe_item[id_key])
            if current is None:
                continue
            for key, value in safe_item.items():
                if key == id_key:
                    continue
                current[key] = value

        prune_ids = [str(value) for value in section_updates.get("prune_ids", [])[:1] if str(value)]
        if prune_ids:
            evolved[section_key] = [
                item
                for item in current_items
                if not (
                    isinstance(item, dict)
                    and str(item.get(id_key) or "") in prune_ids
                    and str(item.get(id_key) or "") not in protector
                )
            ]

    quality_updates = strategy_updates.get("quality_checks")
    if isinstance(quality_updates, dict):
        checks = evolved.get("quality_checks")
        if isinstance(checks, list):
            for addition in quality_updates.get("add", [])[:2]:
                text = str(addition or "").strip()[:220]
                if text and text not in checks:
                    checks.append(text)
            prune_items = {str(item or "").strip() for item in quality_updates.get("prune", [])[:1] if str(item or "").strip()}
            if prune_items:
                evolved["quality_checks"] = [item for item in checks if str(item) not in prune_items]

    validate_parsing_strategy(evolved, framework)
    return evolved
