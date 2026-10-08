#!/usr/bin/env python3
"""Z3-based reward evaluator for threat propagation framework training."""

from __future__ import annotations

from typing import Any

from z3 import BoolVal, Implies, Solver, sat

from framework_utils import normalize_propagation_schema

FRAMEWORK_REWARD_WEIGHT = 0.4
PROPAGATION_REWARD_WEIGHT = 0.6
ALLOWED_PATH_ROLES = {"entry", "propagation", "impact", "containment"}
VALID_RISK_LEVELS = {"low", "medium", "high", "critical"}
VALID_CONFIDENCE = {"low", "medium", "high"}


def _sat(expr: Any) -> bool:
    solver = Solver()
    solver.add(expr)
    return solver.check() == sat


def _score_constraints(checks: list[tuple[str, Any]]) -> dict[str, Any]:
    if not checks:
        return {
            "score": 0.0,
            "satisfied": 0,
            "total": 0,
            "z3_satisfiable": False,
            "failed_constraints": ["no_constraints"],
        }
    failed = [name for name, expr in checks if not _sat(expr)]
    satisfied = len(checks) - len(failed)
    return {
        "score": round(satisfied / len(checks), 4),
        "satisfied": satisfied,
        "total": len(checks),
        "z3_satisfiable": not failed,
        "failed_constraints": failed,
    }


def _component_ids(framework: dict[str, Any]) -> set[str]:
    return {
        str(item.get("id") or "")
        for item in framework.get("core_components", [])
        if isinstance(item, dict) and str(item.get("id") or "")
    }


def _gate_ids(framework: dict[str, Any]) -> set[str]:
    return {
        str(item.get("id") or "")
        for item in framework.get("decision_gates", [])
        if isinstance(item, dict) and str(item.get("id") or "")
    }


def _strategy_step_ids(strategy: dict[str, Any]) -> set[str]:
    return {
        str(item.get("id") or "")
        for item in strategy.get("steps", [])
        if isinstance(item, dict) and str(item.get("id") or "")
    }


def _observed_roles(results: list[dict[str, Any]]) -> set[str]:
    roles: set[str] = set()
    for result in results:
        propagation = normalize_propagation_schema(result.get("propagation"), skill_id=str(result.get("skill_id") or ""))
        chains = propagation.get("chains") if isinstance(propagation.get("chains"), list) else []
        for chain in chains:
            if not isinstance(chain, dict):
                continue
            path = chain.get("path") if isinstance(chain.get("path"), list) else []
            for step in path:
                if isinstance(step, dict) and str(step.get("role") or ""):
                    roles.add(str(step.get("role") or ""))
    return roles


def evaluate_framework_compliance(
    framework: dict[str, Any],
    strategy: dict[str, Any],
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    component_ids = _component_ids(framework)
    gate_ids = _gate_ids(framework)
    step_ids = _strategy_step_ids(strategy)
    observed = _observed_roles(results or [])

    slot_rule_ids = {
        str(item.get("slot_id") or "")
        for item in strategy.get("slot_mapping_rules", [])
        if isinstance(item, dict) and str(item.get("slot_id") or "")
    }
    gate_rule_ids = {
        str(item.get("gate_id") or "")
        for item in strategy.get("gate_mapping_rules", [])
        if isinstance(item, dict) and str(item.get("gate_id") or "")
    }
    strategy_target_ids = {
        str(component_id)
        for step in strategy.get("steps", [])
        if isinstance(step, dict)
        for component_id in (step.get("target_components") if isinstance(step.get("target_components"), list) else [])
    }

    has_entry_model = "entry" in component_ids and "untrusted_source" in gate_ids
    has_propagation_model = bool({"carrier", "trust_shift"} & component_ids) and bool(
        {"trust_promoted", "state_or_privilege"} & gate_ids
    )
    has_terminal_model = "terminal" in component_ids and bool({"external_effect", "contained"} & gate_ids)
    has_containment_model = "containment" in component_ids and "contained" in gate_ids
    has_effect_model = bool({"effect", "terminal"} & component_ids) and "external_effect" in gate_ids

    checks = [
        ("framework_has_entry_component", BoolVal("entry" in component_ids)),
        ("framework_has_carrier_or_trust_shift_component", BoolVal(bool({"carrier", "trust_shift"} & component_ids))),
        ("framework_has_terminal_component", BoolVal("terminal" in component_ids)),
        ("framework_has_untrusted_source_gate", BoolVal("untrusted_source" in gate_ids)),
        ("framework_has_trust_or_state_gate", BoolVal(bool({"trust_promoted", "state_or_privilege"} & gate_ids))),
        ("framework_has_external_or_containment_gate", BoolVal(bool({"external_effect", "contained"} & gate_ids))),
        ("strategy_has_entry_step", BoolVal(bool({"find_entry"} & step_ids))),
        ("strategy_has_trace_step", BoolVal(bool({"trace_carrier", "mark_trust_shift"} & step_ids))),
        ("strategy_has_terminal_step", BoolVal(bool({"locate_effect_or_containment", "close_with_outcome"} & step_ids))),
        ("strategy_step_targets_are_known_components", BoolVal(strategy_target_ids <= component_ids)),
        ("slot_rules_reference_known_components", BoolVal(slot_rule_ids <= component_ids)),
        ("gate_rules_reference_known_gates", BoolVal(gate_rule_ids <= gate_ids)),
        ("observed_entry_can_be_modeled", Implies(BoolVal("entry" in observed), BoolVal(has_entry_model))),
        ("observed_propagation_can_be_modeled", Implies(BoolVal("propagation" in observed), BoolVal(has_propagation_model))),
        ("observed_impact_can_be_modeled", Implies(BoolVal("impact" in observed), BoolVal(has_effect_model or has_terminal_model))),
        ("observed_containment_can_be_modelled", Implies(BoolVal("containment" in observed), BoolVal(has_containment_model or has_terminal_model))),
    ]
    result = _score_constraints(checks)
    result.update(
        {
            "observed_roles": sorted(observed),
            "component_ids": sorted(component_ids),
            "gate_ids": sorted(gate_ids),
            "strategy_step_ids": sorted(step_ids),
        }
    )
    return result


def _nonempty(value: Any) -> bool:
    return bool(str(value or "").strip())


def _graph_node_ids(result: dict[str, Any]) -> set[str]:
    parsed_graph = result.get("parsed_graph") if isinstance(result.get("parsed_graph"), dict) else {}
    nodes = parsed_graph.get("nodes") if isinstance(parsed_graph.get("nodes"), list) else []
    return {
        str(node.get("id") or "")
        for node in nodes
        if isinstance(node, dict) and str(node.get("id") or "")
    }


def _graph_edge_pairs(result: dict[str, Any]) -> set[tuple[str, str]]:
    parsed_graph = result.get("parsed_graph") if isinstance(result.get("parsed_graph"), dict) else {}
    edges = parsed_graph.get("edges") if isinstance(parsed_graph.get("edges"), list) else []
    return {
        (str(edge.get("src") or ""), str(edge.get("dst") or ""))
        for edge in edges
        if isinstance(edge, dict) and str(edge.get("src") or "") and str(edge.get("dst") or "")
    }


def _chain_constraints(chain: dict[str, Any], result: dict[str, Any]) -> list[tuple[str, Any]]:
    chain_id = str(chain.get("id") or "chain")
    start = chain.get("starting_attack_point") if isinstance(chain.get("starting_attack_point"), dict) else {}
    path = chain.get("path") if isinstance(chain.get("path"), list) else []
    edges = chain.get("edges") if isinstance(chain.get("edges"), list) else []
    impact = chain.get("impact") if isinstance(chain.get("impact"), dict) else {}
    graph_node_ids = _graph_node_ids(result)
    graph_edge_pairs = _graph_edge_pairs(result)
    first = path[0] if path and isinstance(path[0], dict) else {}
    last = path[-1] if path and isinstance(path[-1], dict) else {}
    expected_edge_count = max(0, len(path) - 1)
    adjacency_ok = len(edges) == expected_edge_count
    path_node_ids = [
        str(step.get("node_id") or "")
        for step in path
        if isinstance(step, dict) and str(step.get("node_id") or "")
    ]
    if adjacency_ok:
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict) or index + 1 >= len(path):
                adjacency_ok = False
                break
            src = str(edge.get("src_node_id") or "")
            dst = str(edge.get("dst_node_id") or "")
            expected_src = str(path[index].get("node_id") or "") if isinstance(path[index], dict) else ""
            expected_dst = str(path[index + 1].get("node_id") or "") if isinstance(path[index + 1], dict) else ""
            if src != expected_src or dst != expected_dst:
                adjacency_ok = False
                break

    graph_nodes_ok = bool(graph_node_ids) and bool(path_node_ids) and all(node_id in graph_node_ids for node_id in path_node_ids)
    graph_start_ok = bool(graph_node_ids) and _nonempty(start.get("node_id")) and str(start.get("node_id") or "") in graph_node_ids
    graph_edges_ok = bool(graph_edge_pairs) and len(path_node_ids) >= 2 and all(
        (path_node_ids[index], path_node_ids[index + 1]) in graph_edge_pairs
        for index in range(len(path_node_ids) - 1)
    )

    roles = [str(step.get("role") or "") for step in path if isinstance(step, dict)]
    reasons_ok = all(_nonempty(step.get("reason")) for step in path if isinstance(step, dict)) if path else False
    states_ok = all(_nonempty(step.get("threat_state")) for step in path if isinstance(step, dict)) if path else False
    nodes_ok = all(_nonempty(step.get("node_id")) and _nonempty(step.get("node_name")) for step in path if isinstance(step, dict)) if path else False
    edge_mechanisms_ok = all(
        isinstance(edge, dict) and _nonempty(edge.get("propagation_mechanism"))
        for edge in edges
    ) if edges else False

    return [
        (f"{chain_id}:chain_has_id", BoolVal(_nonempty(chain.get("id")))),
        (f"{chain_id}:chain_has_title", BoolVal(_nonempty(chain.get("title")))),
        (f"{chain_id}:starting_attack_point_has_node", BoolVal(_nonempty(start.get("node_id")) and _nonempty(start.get("node_name")))),
        (f"{chain_id}:path_has_at_least_two_nodes", BoolVal(len(path) >= 2)),
        (f"{chain_id}:path0_matches_starting_attack_point", BoolVal(_nonempty(start.get("node_id")) and str(first.get("node_id") or "") == str(start.get("node_id") or ""))),
        (f"{chain_id}:path0_role_is_entry", BoolVal(str(first.get("role") or "") == "entry")),
        (f"{chain_id}:path_roles_are_allowed", BoolVal(bool(roles) and all(role in ALLOWED_PATH_ROLES for role in roles))),
        (f"{chain_id}:path_nodes_have_ids_and_names", BoolVal(nodes_ok)),
        (f"{chain_id}:path_steps_have_threat_state", BoolVal(states_ok)),
        (f"{chain_id}:path_steps_have_reason", BoolVal(reasons_ok)),
        (f"{chain_id}:edges_match_path_adjacency", BoolVal(adjacency_ok)),
        (f"{chain_id}:path_nodes_exist_in_workflow_graph", BoolVal(graph_nodes_ok)),
        (f"{chain_id}:starting_attack_point_exists_in_workflow_graph", BoolVal(graph_start_ok)),
        (f"{chain_id}:path_edges_exist_in_workflow_graph", BoolVal(graph_edges_ok)),
        (f"{chain_id}:edges_explain_propagation_mechanism", BoolVal(edge_mechanisms_ok)),
        (f"{chain_id}:terminal_role_or_summary_present", BoolVal(str(last.get("role") or "") in {"impact", "containment"} or _nonempty(impact.get("summary")))),
        (f"{chain_id}:impact_summary_present", BoolVal(_nonempty(impact.get("summary")))),
        (f"{chain_id}:risk_level_valid", BoolVal(str(chain.get("risk_level") or "") in VALID_RISK_LEVELS)),
        (f"{chain_id}:confidence_valid", BoolVal(str(chain.get("confidence") or "") in VALID_CONFIDENCE)),
    ]


def evaluate_propagation_compliance(result: dict[str, Any]) -> dict[str, Any]:
    propagation = normalize_propagation_schema(result.get("propagation"), skill_id=str(result.get("skill_id") or ""))
    chains = propagation.get("chains") if isinstance(propagation.get("chains"), list) else []
    checks: list[tuple[str, Any]] = [("propagation_has_chains", BoolVal(bool(chains)))]
    for chain in chains:
        if isinstance(chain, dict):
            checks.extend(_chain_constraints(chain, result))
        else:
            checks.append(("chain_is_object", BoolVal(False)))
    result_payload = _score_constraints(checks)
    result_payload.update(
        {
            "skill_id": result.get("skill_id"),
            "chain_count": len(chains),
        }
    )
    return result_payload


def evaluate_z3_reward(
    results: list[dict[str, Any]],
    framework: dict[str, Any],
    strategy: dict[str, Any],
    *,
    framework_weight: float = FRAMEWORK_REWARD_WEIGHT,
    propagation_weight: float = PROPAGATION_REWARD_WEIGHT,
) -> dict[str, Any]:
    framework_eval = evaluate_framework_compliance(framework, strategy, results)
    skill_rewards: list[dict[str, Any]] = []
    for result in results:
        propagation_eval = evaluate_propagation_compliance(result)
        reward = framework_weight * float(framework_eval["score"]) + propagation_weight * float(propagation_eval["score"])
        skill_rewards.append(
            {
                "skill_id": result.get("skill_id"),
                "reward": round(reward, 4),
                "framework_compliance_score": framework_eval["score"],
                "propagation_compliance_score": propagation_eval["score"],
                "propagation_failed_constraints": propagation_eval.get("failed_constraints", []),
                "propagation_evaluation": propagation_eval,
            }
        )
    if skill_rewards:
        avg_reward = sum(float(item["reward"]) for item in skill_rewards) / len(skill_rewards)
        avg_propagation = sum(float(item["propagation_compliance_score"]) for item in skill_rewards) / len(skill_rewards)
    else:
        avg_reward = 0.0
        avg_propagation = 0.0
    return {
        "reward": round(avg_reward, 4),
        "framework_compliance_score": framework_eval["score"],
        "propagation_compliance_score": round(avg_propagation, 4),
        "weights": {
            "framework_compliance": framework_weight,
            "propagation_compliance": propagation_weight,
        },
        "framework_evaluation": framework_eval,
        "skill_rewards": skill_rewards,
    }
