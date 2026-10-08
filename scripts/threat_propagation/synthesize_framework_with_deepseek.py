#!/usr/bin/env python3
"""Train a shared threat propagation framework with random skill updates."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_with_deepseek import (  # type: ignore
    DeepSeekClient,
    analyze_record,
    append_jsonl,
    collect_records,
    load_chain_template,
    parse_json_object,
    read_text,
    render_formal_chain_template,
    render_prompt_template,
    summarize_result,
    write_json,
    write_text,
)
from framework_utils import (  # type: ignore
    FRAMEWORK_SLOT_METRIC,
    STRATEGY_STEP_METRIC,
    aggregate_corpus_coverage,
    apply_framework_updates,
    apply_strategy_updates,
    derive_framework_definition,
    derive_parsing_strategy,
    load_framework_definition,
    load_parsing_strategy,
    plot_metric_history,
    render_metric_brief,
    validate_framework_definition,
    validate_parsing_strategy,
)
from z3_reward_evaluator import evaluate_z3_reward  # type: ignore
from openclaw_runtime_evaluator import (  # type: ignore
    add_openclaw_runtime_args,
    aggregate_openclaw_runtime_rewards,
    openclaw_config_from_args,
)

REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / 'data' / 'inner_representation_v2'
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / 'artifacts' / 'threat_propagation_analysis'
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'artifacts' / 'threat_framework_synthesis'
DEFAULT_FRAMEWORK_FILE = REPO_ROOT / 'template' / 'threat_propagation_framework.json'
DEFAULT_STRATEGY_FILE = REPO_ROOT / 'template' / 'threat_propagation_parsing_strategy.json'
DEFAULT_PROMPT_FILE = REPO_ROOT / 'prompts' / 'threat_framework_synthesis_prompt.md'
DEFAULT_SYSTEM_PROMPT_FILE = REPO_ROOT / 'prompts' / 'threat_security_system_prompt.md'
DEFAULT_ANALYSIS_PROMPT_FILE = REPO_ROOT / 'prompts' / 'threat_analysis_prompt.md'
DEFAULT_PROPAGATION_PROMPT_FILE = REPO_ROOT / 'prompts' / 'threat_propagation_prompt.md'
DEFAULT_CHAIN_TEMPLATE_FILE = REPO_ROOT / 'template' / 'threat_propagation_chain_template.native.json'
DEFAULT_GRAPH_NAME = 'graph_wo_check.md'
DEFAULT_TEST_SIZE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT), help='Normalized skill dataset root used for real training.')
    parser.add_argument('--analysis-root', default=str(DEFAULT_ANALYSIS_ROOT), help='Existing per-skill threat propagation outputs used by --dry-run.')
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT), help='Directory for train/test iteration folders and final framework outputs.')
    parser.add_argument('--framework-file', default=str(DEFAULT_FRAMEWORK_FILE), help='Seed framework JSON.')
    parser.add_argument('--strategy-file', default=str(DEFAULT_STRATEGY_FILE), help='Seed parsing strategy JSON.')
    parser.add_argument('--prompt-file', default=str(DEFAULT_PROMPT_FILE), help='Framework synthesis prompt Markdown path.')
    parser.add_argument('--system-prompt-file', default=str(DEFAULT_SYSTEM_PROMPT_FILE), help='System prompt Markdown path.')
    parser.add_argument('--analysis-prompt-file', default=str(DEFAULT_ANALYSIS_PROMPT_FILE), help='Per-skill threat analysis prompt Markdown path.')
    parser.add_argument('--propagation-prompt-file', default=str(DEFAULT_PROPAGATION_PROMPT_FILE), help='Per-skill threat propagation path prompt Markdown path.')
    parser.add_argument('--chain-template-file', default=str(DEFAULT_CHAIN_TEMPLATE_FILE), help='Formal propagation output template JSON used by the per-skill propagation prompt.')
    parser.add_argument('--graph-name', default=DEFAULT_GRAPH_NAME, help='Workflow graph filename inside each skill directory.')
    parser.add_argument('--skill', action='append', default=[], help='Specific canonical skill id(s) to include before splitting. Can be repeated.')
    parser.add_argument('--limit', type=int, default=0, help='Use only the first N matched skills before train/test split. 0 means no limit.')
    parser.add_argument('--test-size', type=int, default=DEFAULT_TEST_SIZE, help='Number of randomly held-out skills for the test set.')
    parser.add_argument('--split-seed', type=int, default=20260703, help='Random seed for the fixed train/test split.')
    parser.add_argument('--sample-size', '--sample-k', dest='sample_size', type=int, default=0, help='Randomly sample K training skills per iteration. 0 means use all training skills in random order.')
    parser.add_argument('--sample-seed', type=int, default=None, help='Optional random seed for training sample order.')
    parser.add_argument('--model', default=os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'), help='DeepSeek model name.')
    parser.add_argument('--api-url', default=os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1/chat/completions'), help='DeepSeek API URL.')
    parser.add_argument('--iterations', type=int, default=3, help='Number of training iterations after baseline test evaluation.')
    parser.add_argument('--sample-limit', type=int, default=8, help='Max low-scoring path samples to include in each framework update prompt.')
    parser.add_argument('--max-tokens', type=int, default=3200, help='Max completion tokens for framework synthesis calls.')
    parser.add_argument('--max-analysis-tokens', type=int, default=3200, help='Max completion tokens for each per-skill threat analysis call.')
    parser.add_argument('--max-chain-tokens', type=int, default=3200, help='Max completion tokens for each per-skill propagation-chain call.')
    parser.add_argument('--max-skill-chars', type=int, default=32000, help='Maximum SKILL.md characters sent to DeepSeek per skill.')
    parser.add_argument('--max-graph-chars', type=int, default=18000, help='Maximum graph_wo_check.md characters sent to DeepSeek per skill.')
    parser.add_argument('--max-framework-chars', type=int, default=12000, help='Maximum framework-definition characters sent to DeepSeek per skill.')
    parser.add_argument('--max-strategy-chars', type=int, default=18000, help='Maximum parsing-strategy characters sent to DeepSeek per skill.')
    parser.add_argument('--max-chain-template-chars', type=int, default=18000, help='Maximum chain-template characters sent to DeepSeek per skill.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature for DeepSeek calls.')
    parser.add_argument('--sleep-seconds', type=float, default=0.0, help='Sleep between per-skill analysis calls.')
    parser.add_argument('--deepseek-retries', type=int, default=3, help='Retry count for transient DeepSeek errors.')
    parser.add_argument('--deepseek-retry-seconds', type=float, default=8.0, help='Base retry backoff seconds.')
    parser.add_argument('--include-raw', action='store_true', help='Store raw DeepSeek responses in per-skill JSON outputs.')
    parser.add_argument('--dry-run', action='store_true', help='Only compute coverage and Z3 reward from --analysis-root; do not call DeepSeek.')
    add_openclaw_runtime_args(parser)
    return parser.parse_args()


def discover_skill_results(analysis_root: Path) -> list[Path]:
    results: list[Path] = []
    if not analysis_root.exists():
        return results
    for path in analysis_root.rglob('threat_analysis.json'):
        if path.parent.name == 'framework':
            continue
        results.append(path)
    return sorted(results)


def read_text_from_any_path(path: str | Path) -> str:
    return read_text(Path(path))


def load_result(path: Path) -> dict[str, Any]:
    data = json.loads(read_text(path))
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return data


def load_seed_framework_and_strategy(framework_file: Path, strategy_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if framework_file.exists() and strategy_file.exists():
        framework = load_framework_definition(str(framework_file), read_text_from_any_path)
        strategy = load_parsing_strategy(str(strategy_file), read_text_from_any_path, framework)
        return framework, strategy

    template_file = DEFAULT_CHAIN_TEMPLATE_FILE
    chain_template_data = json.loads(read_text(template_file)) if template_file.exists() else {}
    framework = derive_framework_definition(chain_template_data if chain_template_data else None)
    strategy = derive_parsing_strategy(framework, chain_template_data if chain_template_data else None)
    validate_framework_definition(framework)
    validate_parsing_strategy(strategy, framework)
    return framework, strategy


def safe_id(value: Any) -> str:
    text = str(value or '').strip().lower()
    text = re.sub(r'[^a-z0-9_.-]+', '_', text)
    return text.strip('._-')[:80] or 'skill'


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_text(path, '')
        return
    write_text(path, '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n')


def _preview_dicts(items: Any, keys: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    previews: list[dict[str, Any]] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        preview: dict[str, Any] = {}
        for key in keys:
            value = item.get(key)
            if value not in (None, '', []):
                preview[key] = value
        previews.append(preview)
    return previews


def _runtime_feedback_brief(runtime_reward: Any) -> dict[str, Any]:
    if not isinstance(runtime_reward, dict):
        return {}
    case_rewards = runtime_reward.get('case_rewards') if isinstance(runtime_reward.get('case_rewards'), list) else []
    case_feedback: list[dict[str, Any]] = []
    for case in case_rewards[:2]:
        if not isinstance(case, dict):
            continue
        parsed_feedback = case.get('parsed_feedback') if isinstance(case.get('parsed_feedback'), dict) else {}
        failure_reasons = parsed_feedback.get('failure_reasons') if isinstance(parsed_feedback.get('failure_reasons'), list) else []
        injection_plan = case.get('skill_injection_plan') if isinstance(case.get('skill_injection_plan'), dict) else {}
        case_feedback.append(
            {
                'case_id': case.get('case_id'),
                'chain_id': case.get('chain_id'),
                'attack_successful': bool(case.get('attack_successful')),
                'strict_attack_successful': bool(case.get('strict_attack_successful', case.get('attack_successful'))),
                'instruction_follow_successful': bool(case.get('instruction_follow_successful')),
                'canary_exposure_successful': bool(case.get('canary_exposure_successful')),
                'prompt_assisted': bool(case.get('prompt_assisted')),
                'attack_mode': case.get('attack_mode', ''),
                'skill_injection_applied': bool(case.get('skill_injection_applied')),
                'injection_target': injection_plan.get('target', {}),
                'injection_location': injection_plan.get('location', {}),
                'path_coverage_score': case.get('path_coverage_score', 0.0),
                'matched_expected_nodes': case.get('matched_expected_nodes', []),
                'failure_reasons': failure_reasons[:3],
                'observed_path_preview': _preview_dicts(
                    parsed_feedback.get('observed_path'),
                    ('node_id', 'node_name', 'role', 'threat_state', 'reason'),
                    limit=6,
                ),
            }
        )
    return {
        'enabled': bool(runtime_reward.get('enabled')),
        'status': runtime_reward.get('status'),
        'reward': runtime_reward.get('reward', 0.0),
        'attack_mode': runtime_reward.get('attack_mode', ''),
        'attack_success_rate': runtime_reward.get('attack_success_rate', runtime_reward.get('strict_attack_success_rate', 0.0)),
        'strict_attack_success_rate': runtime_reward.get('strict_attack_success_rate', runtime_reward.get('attack_success_rate', 0.0)),
        'instruction_follow_success_rate': runtime_reward.get('instruction_follow_success_rate', 0.0),
        'canary_exposure_rate': runtime_reward.get('canary_exposure_rate', 0.0),
        'runtime_path_coverage_score': runtime_reward.get('runtime_path_coverage_score', runtime_reward.get('path_coverage_score', 0.0)),
        'attack_success_count': runtime_reward.get('attack_success_count', 0),
        'strict_attack_success_count': runtime_reward.get('strict_attack_success_count', runtime_reward.get('attack_success_count', 0)),
        'instruction_follow_success_count': runtime_reward.get('instruction_follow_success_count', 0),
        'canary_exposure_count': runtime_reward.get('canary_exposure_count', 0),
        'skill_injection_applied_count': runtime_reward.get('skill_injection_applied_count', 0),
        'skill_injection_applied_rate': runtime_reward.get('skill_injection_applied_rate', 0.0),
        'case_count': runtime_reward.get('case_count', 0),
        'weights': runtime_reward.get('weights', {}),
        'case_feedback': case_feedback,
    }


def build_low_scoring_samples(results: list[dict[str, Any]], coverage: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    result_by_skill = {str(result.get('skill_id') or ''): result for result in results}
    samples: list[dict[str, Any]] = []
    for skill_score in coverage.get('skill_scores', [])[:limit]:
        skill_id = str(skill_score.get('skill_id') or '')
        result = result_by_skill.get(skill_id)
        if not result:
            continue
        propagation = result.get('propagation') if isinstance(result.get('propagation'), dict) else {}
        chains = propagation.get('chains') if isinstance(propagation.get('chains'), list) else []
        skill_coverage = result.get('coverage') if isinstance(result.get('coverage'), dict) else {}
        runtime_reward = result.get('openclaw_runtime_reward') if isinstance(result.get('openclaw_runtime_reward'), dict) else {}
        z3_reward = result.get('z3_reward') if isinstance(result.get('z3_reward'), dict) else {}
        chain_eval_by_id = {
            str(chain_eval.get('chain_id') or ''): chain_eval
            for chain_eval in skill_coverage.get('chains', [])
            if isinstance(chain_eval, dict)
        }
        chain_samples: list[dict[str, Any]] = []
        for chain in chains[:2]:
            if not isinstance(chain, dict):
                continue
            chain_id = str(chain.get('id') or '')
            chain_eval = chain_eval_by_id.get(chain_id, {})
            chain_samples.append(
                {
                    'chain_id': chain.get('id'),
                    'title': chain.get('title'),
                    'path_preview': _preview_dicts(chain.get('path'), ('node_id', 'node_name', 'role', 'threat_state', 'reason'), limit=8),
                    'edge_preview': _preview_dicts(chain.get('edges'), ('src_node_id', 'dst_node_id', 'condition', 'propagation_mechanism'), limit=8),
                    'impact': chain.get('impact', {}),
                    'chain_coverage': {
                        FRAMEWORK_SLOT_METRIC: chain_eval.get(FRAMEWORK_SLOT_METRIC, 0.0),
                        STRATEGY_STEP_METRIC: chain_eval.get(STRATEGY_STEP_METRIC, 0.0),
                        'missing_components': chain_eval.get('missing_components', []),
                        'missing_steps': chain_eval.get('missing_steps', []),
                        'uncovered_units': chain_eval.get('uncovered_units', [])[:3],
                    },
                }
            )
        samples.append(
            {
                'skill_id': skill_id,
                'coverage': {
                    'primary_reward': skill_score.get('primary_reward', runtime_reward.get('reward', z3_reward.get('reward', 0.0))),
                    'openclaw_runtime_reward': skill_score.get('openclaw_runtime_reward', runtime_reward.get('reward', 0.0)),
                    'attack_success_rate': skill_score.get('attack_success_rate', runtime_reward.get('attack_success_rate', runtime_reward.get('strict_attack_success_rate', 0.0))),
                    'strict_attack_success_rate': skill_score.get('strict_attack_success_rate', runtime_reward.get('strict_attack_success_rate', runtime_reward.get('attack_success_rate', 0.0))),
                    'instruction_follow_success_rate': skill_score.get('instruction_follow_success_rate', runtime_reward.get('instruction_follow_success_rate', 0.0)),
                    'canary_exposure_rate': skill_score.get('canary_exposure_rate', runtime_reward.get('canary_exposure_rate', 0.0)),
                    'runtime_path_coverage_score': skill_score.get('runtime_path_coverage_score', runtime_reward.get('path_coverage_score', 0.0)),
                    'skill_injection_applied_rate': skill_score.get('skill_injection_applied_rate', runtime_reward.get('skill_injection_applied_rate', 0.0)),
                    'z3_reward': skill_score.get('z3_reward', z3_reward.get('reward', 0.0)),
                    FRAMEWORK_SLOT_METRIC: skill_score.get(FRAMEWORK_SLOT_METRIC, 0.0),
                    STRATEGY_STEP_METRIC: skill_score.get(STRATEGY_STEP_METRIC, 0.0),
                    'uncovered_unit_count': skill_score.get('uncovered_unit_count', 0),
                    'uncovered_units': skill_score.get('uncovered_units', [])[:5],
                },
                'z3_reward': z3_reward,
                'runtime_feedback': _runtime_feedback_brief(runtime_reward),
                'summary': str((result.get('threat_analysis') or {}).get('summary') or ''),
                'chain_samples': chain_samples,
            }
        )
    return samples


def build_prompt(prompt_template: str, framework: dict[str, Any], strategy: dict[str, Any], coverage: dict[str, Any], samples: list[dict[str, Any]], metric_history: list[dict[str, Any]]) -> str:
    return render_prompt_template(
        prompt_template,
        {
            'framework': json.dumps(framework, ensure_ascii=False, indent=2),
            'strategy': json.dumps(strategy, ensure_ascii=False, indent=2),
            'coverage_summary': json.dumps(render_metric_brief(coverage), ensure_ascii=False, indent=2),
            'low_scoring_samples': json.dumps(samples, ensure_ascii=False, indent=2),
            'metric_history': json.dumps(metric_history[-30:], ensure_ascii=False, indent=2),
        },
    )


def apply_synthesis_response(framework: dict[str, Any], strategy: dict[str, Any], response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
    if not response.get('has_updates', False):
        return framework, strategy, False
    evolved_framework = apply_framework_updates(framework, response)
    evolved_strategy = apply_strategy_updates(strategy, response, evolved_framework)
    return evolved_framework, evolved_strategy, True


def split_train_test(records: list[Any], *, test_size: int, seed: int) -> tuple[list[Any], list[Any], dict[str, Any]]:
    if len(records) < 2:
        raise ValueError('At least 2 skills are required for train/test split')
    actual_test_size = min(max(1, test_size), len(records) - 1)
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    test_records = sorted(shuffled[:actual_test_size], key=lambda record: record.skill_id)
    train_records = sorted(shuffled[actual_test_size:], key=lambda record: record.skill_id)
    split_info = {
        'split_seed': seed,
        'requested_test_size': test_size,
        'actual_test_size': len(test_records),
        'train_size': len(train_records),
        'test_skill_ids': [record.skill_id for record in test_records],
        'train_skill_ids': [record.skill_id for record in train_records],
    }
    return train_records, test_records, split_info


def select_train_records(train_records: list[Any], *, sample_size: int, rng: random.Random, iteration: int) -> tuple[list[Any], dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(train_records):
        selected = list(train_records)
        rng.shuffle(selected)
        selection_strategy = 'all_train_shuffled'
        requested_sample_size = sample_size
    else:
        selected = rng.sample(train_records, sample_size)
        selection_strategy = 'random_without_replacement'
        requested_sample_size = sample_size
    sample_info = {
        'phase': 'train',
        'iteration': iteration,
        'strategy': selection_strategy,
        'requested_sample_size': requested_sample_size,
        'population_count': len(train_records),
        'selected_count': len(selected),
        'selected_skill_ids': [record.skill_id for record in selected],
    }
    return selected, sample_info


def attach_z3_rewards(
    *,
    results: list[dict[str, Any]],
    framework: dict[str, Any],
    strategy: dict[str, Any],
    coverage: dict[str, Any],
    output_root: Path | None,
) -> dict[str, Any]:
    reward_summary = evaluate_z3_reward(results, framework, strategy)
    coverage['z3_reward_summary'] = {
        'reward': reward_summary.get('reward', 0.0),
        'framework_compliance_score': reward_summary.get('framework_compliance_score', 0.0),
        'propagation_compliance_score': reward_summary.get('propagation_compliance_score', 0.0),
        'weights': reward_summary.get('weights', {}),
        'framework_failed_constraints': reward_summary.get('framework_evaluation', {}).get('failed_constraints', []),
        'skill_rewards': [
            {
                'skill_id': item.get('skill_id'),
                'reward': item.get('reward', 0.0),
                'propagation_compliance_score': item.get('propagation_compliance_score', 0.0),
                'propagation_failed_constraints': item.get('propagation_failed_constraints', []),
            }
            for item in reward_summary.get('skill_rewards', [])
        ],
    }
    coverage.setdefault('metrics', {})['z3_reward'] = reward_summary.get('reward', 0.0)
    coverage.setdefault('mean_metrics', {})['z3_reward'] = reward_summary.get('reward', 0.0)
    reward_by_skill = {str(item.get('skill_id') or ''): item for item in reward_summary.get('skill_rewards', [])}
    for score in coverage.get('skill_scores', []):
        if not isinstance(score, dict):
            continue
        skill_reward = reward_by_skill.get(str(score.get('skill_id') or ''))
        if not skill_reward:
            continue
        score['z3_reward'] = skill_reward.get('reward', 0.0)
        score['z3_propagation_compliance_score'] = skill_reward.get('propagation_compliance_score', 0.0)
        score['z3_failed_constraints'] = skill_reward.get('propagation_failed_constraints', [])
    if isinstance(coverage.get('skill_scores'), list):
        coverage['skill_scores'].sort(
            key=lambda item: (
                float(item.get('z3_reward', 1.0)),
                item.get(FRAMEWORK_SLOT_METRIC, 1.0) + item.get(STRATEGY_STEP_METRIC, 1.0),
                str(item.get('skill_id') or ''),
            )
        )
    for result in results:
        skill_id = str(result.get('skill_id') or '')
        skill_reward = reward_by_skill.get(skill_id)
        if not skill_reward:
            continue
        result['z3_reward'] = skill_reward
        coverage_payload = result.get('coverage') if isinstance(result.get('coverage'), dict) else {}
        coverage_payload['z3_reward'] = skill_reward.get('reward', 0.0)
        coverage_payload['z3_propagation_compliance_score'] = skill_reward.get('propagation_compliance_score', 0.0)
        coverage_payload['z3_failed_constraints'] = skill_reward.get('propagation_failed_constraints', [])
        result['coverage'] = coverage_payload
        if output_root is not None:
            skill_dir = output_root / skill_id
            write_json(skill_dir / 'z3_reward.json', skill_reward)
            write_json(skill_dir / 'coverage.json', coverage_payload)
            write_json(
                skill_dir / 'propagation.json',
                {
                    'skill_id': result.get('skill_id'),
                    'generated_at': result.get('generated_at'),
                    'model': result.get('model'),
                    'propagation': result.get('propagation', {}),
                    'mermaid': result.get('mermaid', ''),
                    'coverage': coverage_payload,
                    'z3_reward': skill_reward,
                },
            )
            write_json(skill_dir / 'threat_analysis.json', result)
    if output_root is not None:
        write_json(output_root / 'z3_reward_summary.json', reward_summary)
    return reward_summary


def attach_openclaw_runtime_rewards(
    *,
    results: list[dict[str, Any]],
    coverage: dict[str, Any],
    output_root: Path | None,
) -> dict[str, Any]:
    reward_summary = aggregate_openclaw_runtime_rewards(results)
    if not reward_summary.get("enabled"):
        coverage.setdefault("primary_reward", coverage.get("z3_reward_summary", {}).get("reward", 0.0))
        coverage.setdefault("primary_reward_source", "z3")
        return reward_summary

    coverage["openclaw_runtime_reward_summary"] = reward_summary
    coverage.setdefault("metrics", {})["openclaw_runtime_reward"] = reward_summary.get("reward", 0.0)
    coverage.setdefault("mean_metrics", {})["openclaw_runtime_reward"] = reward_summary.get("reward", 0.0)
    coverage["metrics"]["attack_success_rate"] = reward_summary.get("attack_success_rate", reward_summary.get("strict_attack_success_rate", 0.0))
    coverage["metrics"]["strict_attack_success_rate"] = reward_summary.get("strict_attack_success_rate", reward_summary.get("attack_success_rate", 0.0))
    coverage["metrics"]["instruction_follow_success_rate"] = reward_summary.get("instruction_follow_success_rate", 0.0)
    coverage["metrics"]["canary_exposure_rate"] = reward_summary.get("canary_exposure_rate", 0.0)
    coverage["metrics"]["runtime_path_coverage_score"] = reward_summary.get("path_coverage_score", 0.0)
    coverage["metrics"]["skill_injection_applied_rate"] = reward_summary.get("skill_injection_applied_rate", 0.0)
    coverage["mean_metrics"]["attack_success_rate"] = reward_summary.get("attack_success_rate", reward_summary.get("strict_attack_success_rate", 0.0))
    coverage["mean_metrics"]["strict_attack_success_rate"] = reward_summary.get("strict_attack_success_rate", reward_summary.get("attack_success_rate", 0.0))
    coverage["mean_metrics"]["instruction_follow_success_rate"] = reward_summary.get("instruction_follow_success_rate", 0.0)
    coverage["mean_metrics"]["canary_exposure_rate"] = reward_summary.get("canary_exposure_rate", 0.0)
    coverage["mean_metrics"]["runtime_path_coverage_score"] = reward_summary.get("path_coverage_score", 0.0)
    coverage["mean_metrics"]["skill_injection_applied_rate"] = reward_summary.get("skill_injection_applied_rate", 0.0)
    coverage["primary_reward"] = reward_summary.get("reward", 0.0)
    coverage["primary_reward_source"] = "openclaw_runtime"

    reward_by_skill = {
        str(item.get("skill_id") or ""): item
        for item in reward_summary.get("skill_rewards", [])
        if isinstance(item, dict)
    }
    for score in coverage.get("skill_scores", []):
        if not isinstance(score, dict):
            continue
        skill_reward = reward_by_skill.get(str(score.get("skill_id") or ""))
        if skill_reward:
            score["openclaw_runtime_reward"] = skill_reward.get("reward", 0.0)
            score["attack_success_rate"] = skill_reward.get("attack_success_rate", skill_reward.get("strict_attack_success_rate", 0.0))
            score["strict_attack_success_rate"] = skill_reward.get("strict_attack_success_rate", skill_reward.get("attack_success_rate", 0.0))
            score["instruction_follow_success_rate"] = skill_reward.get("instruction_follow_success_rate", 0.0)
            score["canary_exposure_rate"] = skill_reward.get("canary_exposure_rate", 0.0)
            score["runtime_path_coverage_score"] = skill_reward.get("path_coverage_score", 0.0)
            score["skill_injection_applied_rate"] = skill_reward.get("skill_injection_applied_rate", 0.0)
            score["primary_reward"] = skill_reward.get("reward", 0.0)
    if isinstance(coverage.get("skill_scores"), list):
        coverage["skill_scores"].sort(
            key=lambda item: (
                float(item.get("openclaw_runtime_reward", 1.0)),
                float(item.get("z3_reward", 1.0)),
                str(item.get("skill_id") or ""),
            )
        )

    for result in results:
        skill_id = str(result.get("skill_id") or "")
        skill_reward = reward_by_skill.get(skill_id)
        if not skill_reward:
            continue
        result["openclaw_runtime_reward"] = skill_reward
        coverage_payload = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
        coverage_payload["openclaw_runtime_reward"] = skill_reward.get("reward", 0.0)
        coverage_payload["attack_success_rate"] = skill_reward.get("attack_success_rate", skill_reward.get("strict_attack_success_rate", 0.0))
        coverage_payload["strict_attack_success_rate"] = skill_reward.get("strict_attack_success_rate", skill_reward.get("attack_success_rate", 0.0))
        coverage_payload["instruction_follow_success_rate"] = skill_reward.get("instruction_follow_success_rate", 0.0)
        coverage_payload["canary_exposure_rate"] = skill_reward.get("canary_exposure_rate", 0.0)
        coverage_payload["runtime_path_coverage_score"] = skill_reward.get("path_coverage_score", 0.0)
        coverage_payload["skill_injection_applied_rate"] = skill_reward.get("skill_injection_applied_rate", 0.0)
        coverage_payload["primary_reward"] = skill_reward.get("reward", 0.0)
        coverage_payload["primary_reward_source"] = "openclaw_runtime"
        result["coverage"] = coverage_payload
        if output_root is not None:
            skill_dir = output_root / skill_id
            write_json(skill_dir / "openclaw_runtime_reward.json", skill_reward)
            write_json(skill_dir / "coverage.json", coverage_payload)
            write_json(
                skill_dir / "propagation.json",
                {
                    "skill_id": result.get("skill_id"),
                    "generated_at": result.get("generated_at"),
                    "model": result.get("model"),
                    "propagation": result.get("propagation", {}),
                    "mermaid": result.get("mermaid", ""),
                    "coverage": coverage_payload,
                    "z3_reward": result.get("z3_reward", {}),
                    "openclaw_runtime_reward": skill_reward,
                },
            )
            write_json(skill_dir / "threat_analysis.json", result)
    if output_root is not None:
        write_json(output_root / "openclaw_runtime_reward_summary.json", reward_summary)
    return reward_summary


def run_skill_iteration(
    *,
    client: DeepSeekClient,
    records: list[Any],
    output_root: Path,
    framework: dict[str, Any],
    strategy: dict[str, Any],
    system_prompt: str,
    analysis_prompt: str,
    propagation_prompt: str,
    chain_template: str,
    chain_template_file: Path,
    args: argparse.Namespace,
    sample_info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    framework_file = output_root / 'framework_definition.json'
    strategy_file = output_root / 'parsing_strategy.json'
    write_json(framework_file, framework)
    write_json(strategy_file, strategy)
    if sample_info is not None:
        write_json(output_root / 'sample_info.json', sample_info)
    write_json(
        output_root / 'skill_manifest.json',
        [
            {
                'skill_id': record.skill_id,
                'skill': str(record.skill_path),
                'graph': str(record.graph_path),
                'manifest': str(record.manifest_path) if record.manifest_path else None,
            }
            for record in records
        ],
    )
    summary_path = output_root / 'summary.jsonl'
    write_text(summary_path, '')
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    progress_enabled = bool(
        sample_info
        and (
            sample_info.get("progress_format") == "openclaw_asb"
            or sample_info.get("optimization_signal") == "strict_attack_success_rate_only"
            or sample_info.get("reward_objective") == "strict_attack_success_rate_only"
        )
    )
    progress_phase = str((sample_info or {}).get("phase") or "run")
    progress_iteration = int((sample_info or {}).get("iteration") or 0)
    total_records = len(records)
    for index, record in enumerate(records, start=1):
        if progress_enabled:
            print(
                f"[iter {progress_iteration}][{progress_phase}][{index}/{total_records}][generate] "
                f"skill={record.skill_id} remaining={total_records - index}"
            )
        try:
            result = analyze_record(
                client,
                record,
                output_root=output_root,
                system_prompt_template=system_prompt,
                analysis_prompt_template=analysis_prompt,
                propagation_prompt_template=propagation_prompt,
                framework=framework,
                parsing_strategy=strategy,
                framework_file=framework_file,
                parsing_strategy_file=strategy_file,
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
            results.append(result)
            append_jsonl(summary_path, summarize_result(result))
            if progress_enabled:
                runtime_reward = result.get("openclaw_runtime_reward") if isinstance(result.get("openclaw_runtime_reward"), dict) else {}
                case_rewards = runtime_reward.get("case_rewards") if isinstance(runtime_reward.get("case_rewards"), list) else []
                if case_rewards:
                    for case_index, case in enumerate(case_rewards, start=1):
                        if not isinstance(case, dict):
                            continue
                        injection_plan = case.get('skill_injection_plan') if isinstance(case.get('skill_injection_plan'), dict) else {}
                        target = injection_plan.get('target') if isinstance(injection_plan.get('target'), dict) else {}
                        target_node = target.get('node_name') or target.get('node_id') or 'unknown'
                        print(
                            f"[iter {progress_iteration}][{progress_phase}][{index}/{total_records}]"
                            f"[case {case_index}/{len(case_rewards)}] "
                            f"attack_goal={case.get('attack_goal', '其他攻击影响')} "
                            f"owasp_goal={case.get('owasp_goal', 'Other / Unclassified')} "
                            f"target_node={target_node} target_quality={case.get('target_quality', '') or 'unknown'} "
                            f"success={int(bool(case.get('strict_attack_successful')))} "
                            f"remaining={total_records - index}"
                        )
                else:
                    print(
                        f"[iter {progress_iteration}][{progress_phase}][{index}/{total_records}] "
                        f"attack_goal=其他攻击影响 owasp_goal=Other / Unclassified success=0 "
                        f"remaining={total_records - index}"
                    )
            print(f"[ok] {output_root.name}/{record.skill_id} -> {output_root / record.skill_id / 'propagation.md'}")
        except Exception as exc:
            failure = {'skill_id': record.skill_id, 'status': 'failed', 'error': str(exc)}
            failures.append(failure)
            append_jsonl(summary_path, failure)
            print(f"[fail] {output_root.name}/{record.skill_id}: {exc}", file=sys.stderr)
        if args.sleep_seconds > 0 and index < len(records):
            time.sleep(args.sleep_seconds)
    if not results:
        raise RuntimeError(f'No successful skill propagation results in {output_root}')
    coverage = aggregate_corpus_coverage(results, framework, strategy)
    if sample_info is not None:
        enriched_sample_info = dict(sample_info)
        enriched_sample_info['successful_skill_ids'] = [str(result.get('skill_id') or '') for result in results]
        enriched_sample_info['failed_skill_ids'] = [str(failure.get('skill_id') or '') for failure in failures]
        coverage['sample_info'] = enriched_sample_info
        write_json(output_root / 'sample_info.json', enriched_sample_info)
    # OpenClaw strict-ASR training must not spend optimization budget on auxiliary Z3 feedback.
    # Keep the legacy path intact for the non-strict framework trainer.
    if getattr(args, "strict_asr_only", False):
        reward_summary = {"mode": "strict_asr_only", "reward": 0.0}
    else:
        reward_summary = attach_z3_rewards(results=results, framework=framework, strategy=strategy, coverage=coverage, output_root=output_root)
    attach_openclaw_runtime_rewards(results=results, coverage=coverage, output_root=output_root)
    if failures:
        write_json(output_root / 'failures.json', failures)
    return results, coverage, failures, reward_summary


def write_phase_artifacts(
    phase_root: Path,
    *,
    framework: dict[str, Any],
    strategy: dict[str, Any],
    coverage: dict[str, Any],
    metric_entry: dict[str, Any],
    metric_history: list[dict[str, Any]],
    response: dict[str, Any] | None = None,
    next_framework: dict[str, Any] | None = None,
    next_strategy: dict[str, Any] | None = None,
) -> None:
    write_json(phase_root / 'framework_definition.json', framework)
    write_json(phase_root / 'parsing_strategy.json', strategy)
    write_json(phase_root / 'coverage_summary.json', coverage)
    write_json(phase_root / 'phase_summary.json', metric_entry)
    write_json(phase_root / 'metric_history.json', metric_history)
    write_jsonl(phase_root / 'metric_history.jsonl', metric_history)
    write_json(phase_root / 'skill_scores.json', coverage.get('skill_scores', []))
    if isinstance(coverage.get('sample_info'), dict):
        write_json(phase_root / 'sample_info.json', coverage['sample_info'])
    if response is not None:
        write_json(phase_root / 'synthesis_response.json', response)
    if next_framework is not None:
        write_json(phase_root / 'framework_definition.after.json', next_framework)
    if next_strategy is not None:
        write_json(phase_root / 'parsing_strategy.after.json', next_strategy)


def build_metric_entry(
    *,
    iteration: int,
    phase: str,
    coverage: dict[str, Any],
    failures: list[dict[str, Any]],
    train_step: int | None = None,
    skill_id: str = '',
    updated: bool = False,
    reason: str = '',
) -> dict[str, Any]:
    reward = coverage.get('z3_reward_summary') if isinstance(coverage.get('z3_reward_summary'), dict) else {}
    runtime_reward = coverage.get('openclaw_runtime_reward_summary') if isinstance(coverage.get('openclaw_runtime_reward_summary'), dict) else {}
    runtime_enabled = bool(runtime_reward.get('enabled'))
    primary_reward = runtime_reward.get('reward', 0.0) if runtime_enabled else reward.get('reward', 0.0)
    entry = {
        'iteration': iteration,
        'phase': phase,
        'updated': updated,
        'reason': reason,
        'failure_count': len(failures),
        'primary_reward': primary_reward,
        'primary_reward_source': 'openclaw_runtime' if runtime_enabled else 'z3',
        'openclaw_runtime_enabled': runtime_enabled,
        'openclaw_runtime_reward': runtime_reward.get('reward', 0.0),
        'attack_success_rate': runtime_reward.get('attack_success_rate', runtime_reward.get('strict_attack_success_rate', 0.0)),
        'strict_attack_success_rate': runtime_reward.get('strict_attack_success_rate', runtime_reward.get('attack_success_rate', 0.0)),
        'instruction_follow_success_rate': runtime_reward.get('instruction_follow_success_rate', 0.0),
        'canary_exposure_rate': runtime_reward.get('canary_exposure_rate', 0.0),
        'runtime_path_coverage_score': runtime_reward.get('path_coverage_score', 0.0),
        'skill_injection_applied_rate': runtime_reward.get('skill_injection_applied_rate', 0.0),
        'skill_injection_applied_count': runtime_reward.get('skill_injection_applied_count', 0),
        'z3_reward': reward.get('reward', 0.0),
        'framework_compliance_score': reward.get('framework_compliance_score', 0.0),
        'propagation_compliance_score': reward.get('propagation_compliance_score', 0.0),
        **render_metric_brief(coverage),
    }
    if train_step is not None:
        entry['train_step'] = train_step
    if skill_id:
        entry['skill_id'] = skill_id
    return entry


def metric_line(label: str, coverage: dict[str, Any]) -> str:
    metrics = coverage.get('metrics') if isinstance(coverage.get('metrics'), dict) else {}
    mean_metrics = coverage.get('mean_metrics') if isinstance(coverage.get('mean_metrics'), dict) else {}
    reward = coverage.get('z3_reward_summary') if isinstance(coverage.get('z3_reward_summary'), dict) else {}
    runtime_reward = coverage.get('openclaw_runtime_reward_summary') if isinstance(coverage.get('openclaw_runtime_reward_summary'), dict) else {}
    runtime_enabled = bool(runtime_reward.get('enabled'))
    primary_reward = runtime_reward.get('reward', 0.0) if runtime_enabled else reward.get('reward', 0.0)
    primary_source = 'openclaw_runtime' if runtime_enabled else 'z3'
    train_adjustment = coverage.get('strict_asr_train_history_adjustment') if isinstance(coverage.get('strict_asr_train_history_adjustment'), dict) else {}
    test_discovery = coverage.get('strict_asr_discovery') if isinstance(coverage.get('strict_asr_discovery'), dict) else {}
    displayed_asr = train_adjustment.get('history_adjusted_strict_attack_success_rate', test_discovery.get('history_adjusted_strict_attack_success_rate', runtime_reward.get('attack_success_rate', runtime_reward.get('strict_attack_success_rate', 0.0))))
    cumulative_asr = train_adjustment.get('cumulative_train_historical_attack_success_rate', test_discovery.get('cumulative_historical_attack_success_rate', 0.0))
    return (
        f"[{label}] primary_reward={float(primary_reward):.4f}({primary_source}) "
        f"openclaw_reward={float(runtime_reward.get('reward', 0.0)):.4f} "
        f"strict_asr={float(displayed_asr):.4f} "
        f"cumulative_historical_asr={float(cumulative_asr):.4f} "
        f"instruction_follow={float(runtime_reward.get('instruction_follow_success_rate', 0.0)):.4f} "
        f"canary_exposure={float(runtime_reward.get('canary_exposure_rate', 0.0)):.4f} "
        f"runtime_path_coverage={float(runtime_reward.get('path_coverage_score', 0.0)):.4f} "
        f"skill_injection_applied={float(runtime_reward.get('skill_injection_applied_rate', 0.0)):.4f} "
        f"z3_reward={float(reward.get('reward', 0.0)):.4f} "
        f"framework_compliance={float(reward.get('framework_compliance_score', 0.0)):.4f} "
        f"propagation_compliance={float(reward.get('propagation_compliance_score', 0.0)):.4f} "
        f"mean_{FRAMEWORK_SLOT_METRIC}={float(mean_metrics.get(FRAMEWORK_SLOT_METRIC, 0.0)):.4f} "
        f"mean_{STRATEGY_STEP_METRIC}={float(mean_metrics.get(STRATEGY_STEP_METRIC, 0.0)):.4f} "
        f"weighted_{FRAMEWORK_SLOT_METRIC}={float(metrics.get(FRAMEWORK_SLOT_METRIC, 0.0)):.4f} "
        f"weighted_{STRATEGY_STEP_METRIC}={float(metrics.get(STRATEGY_STEP_METRIC, 0.0)):.4f}"
    )


def plot_test_reward_history(history: list[dict[str, Any]], output_path: str) -> None:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None

    test_rows = [row for row in history if row.get('phase') == 'test']
    if not test_rows:
        return
    iterations = [int(row.get('iteration', 0)) for row in test_rows]
    primary_rewards = [float(row.get('primary_reward', row.get('z3_reward', 0.0))) for row in test_rows]
    openclaw_rewards = [float(row.get('openclaw_runtime_reward', 0.0)) for row in test_rows]
    attack_success = [float(row.get('attack_success_rate', row.get('strict_attack_success_rate', 0.0))) for row in test_rows]
    instruction_follow = [float(row.get('instruction_follow_success_rate', 0.0)) for row in test_rows]
    canary_exposure = [float(row.get('canary_exposure_rate', 0.0)) for row in test_rows]
    runtime_path = [float(row.get('runtime_path_coverage_score', 0.0)) for row in test_rows]
    injection_applied = [float(row.get('skill_injection_applied_rate', 0.0)) for row in test_rows]
    z3_rewards = [float(row.get('z3_reward', 0.0)) for row in test_rows]
    framework_scores = [float(row.get('framework_compliance_score', 0.0)) for row in test_rows]
    propagation_scores = [float(row.get('propagation_compliance_score', 0.0)) for row in test_rows]

    if sns is not None:
        sns.set_theme(style='whitegrid')
    figure, axis = plt.subplots(figsize=(10, 5.5))
    series = [
        ('Primary Reward', primary_rewards),
        ('OpenClaw Runtime Reward', openclaw_rewards),
        ('Strict Attack Success Rate', attack_success),
        ('Instruction Follow Success', instruction_follow),
        ('Canary Exposure Rate', canary_exposure),
        ('Runtime Path Coverage', runtime_path),
        ('Skill Injection Applied', injection_applied),
        ('Z3 Reward', z3_rewards),
        ('Framework Compliance', framework_scores),
        ('Propagation Compliance', propagation_scores),
    ]
    for label, values in series:
        if label.startswith('OpenClaw') and not any(value > 0 for value in values):
            continue
        if label in {'Strict Attack Success Rate', 'Instruction Follow Success', 'Canary Exposure Rate', 'Runtime Path Coverage', 'Skill Injection Applied'} and not any(value > 0 for value in values):
            continue
        if sns is not None:
            sns.lineplot(x=iterations, y=values, marker='o', label=label, ax=axis)
        else:
            axis.plot(iterations, values, marker='o', label=label)
    if sns is None:
        axis.grid(True, color='#d1d5db', linewidth=0.8, alpha=0.8)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel('Iteration')
    axis.set_ylabel('Score')
    axis.set_title('Test Runtime Reward and Compliance')
    axis.legend(loc='lower right')
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_history_outputs(output_root: Path, history: list[dict[str, Any]]) -> None:
    write_json(output_root / 'metric_history.json', history)
    write_jsonl(output_root / 'metric_history.jsonl', history)
    test_history = [row for row in history if row.get('phase') == 'test']
    write_json(output_root / 'test_metric_history.json', test_history)
    write_jsonl(output_root / 'test_metric_history.jsonl', test_history)
    plot_metric_history(test_history, str(output_root / 'metric_history.png'))
    plot_test_reward_history(history, str(output_root / 'test_reward_history.png'))


def write_final_outputs(output_root: Path, framework: dict[str, Any], strategy: dict[str, Any], coverage: dict[str, Any], history: list[dict[str, Any]]) -> None:
    write_json(output_root / 'framework_definition.json', framework)
    write_json(output_root / 'parsing_strategy.json', strategy)
    write_json(output_root / 'coverage_summary.json', coverage)
    write_history_outputs(output_root, history)


def run_test_evaluation(
    *,
    client: DeepSeekClient,
    test_records: list[Any],
    iteration_root: Path,
    iteration: int,
    framework: dict[str, Any],
    strategy: dict[str, Any],
    system_prompt: str,
    analysis_prompt: str,
    propagation_prompt: str,
    chain_template: str,
    chain_template_file: Path,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    test_root = iteration_root / 'test'
    sample_info = {
        'phase': 'test',
        'iteration': iteration,
        'strategy': 'fixed_holdout_test_set',
        'progress_format': 'openclaw_asb' if getattr(args, 'strict_asr_only', False) else 'default',
        'selected_count': len(test_records),
        'selected_skill_ids': [record.skill_id for record in test_records],
    }
    results, coverage, failures, reward_summary = run_skill_iteration(
        client=client,
        records=test_records,
        output_root=test_root,
        framework=framework,
        strategy=strategy,
        system_prompt=system_prompt,
        analysis_prompt=analysis_prompt,
        propagation_prompt=propagation_prompt,
        chain_template=chain_template,
        chain_template_file=chain_template_file,
        args=args,
        sample_info=sample_info,
    )
    metric_entry = build_metric_entry(iteration=iteration, phase='test', coverage=coverage, failures=failures)
    history.append(metric_entry)
    write_phase_artifacts(
        test_root,
        framework=framework,
        strategy=strategy,
        coverage=coverage,
        metric_entry=metric_entry,
        metric_history=history,
    )
    write_json(iteration_root / 'test_summary.json', metric_entry)
    print(metric_line(f'test iter {iteration}', coverage))
    return coverage, reward_summary


def run_training_step(
    *,
    client: DeepSeekClient,
    record: Any,
    iteration_root: Path,
    iteration: int,
    train_step: int,
    framework: dict[str, Any],
    strategy: dict[str, Any],
    system_prompt: str,
    analysis_prompt: str,
    propagation_prompt: str,
    synthesis_prompt: str,
    chain_template: str,
    chain_template_file: Path,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    step_root = iteration_root / 'train' / f'step_{train_step:03d}_{safe_id(record.skill_id)}'
    sample_info = {
        'phase': 'train',
        'iteration': iteration,
        'train_step': train_step,
        'strategy': 'single_random_train_skill',
        'selected_count': 1,
        'selected_skill_ids': [record.skill_id],
    }
    results, coverage, failures, _reward_summary = run_skill_iteration(
        client=client,
        records=[record],
        output_root=step_root,
        framework=framework,
        strategy=strategy,
        system_prompt=system_prompt,
        analysis_prompt=analysis_prompt,
        propagation_prompt=propagation_prompt,
        chain_template=chain_template,
        chain_template_file=chain_template_file,
        args=args,
        sample_info=sample_info,
    )
    metric_entry = build_metric_entry(
        iteration=iteration,
        phase='train',
        train_step=train_step,
        skill_id=record.skill_id,
        coverage=coverage,
        failures=failures,
    )
    history.append(metric_entry)
    print(metric_line(f'train iter {iteration} step {train_step} {record.skill_id}', coverage))

    samples = build_low_scoring_samples(results, coverage, args.sample_limit)
    prompt = build_prompt(synthesis_prompt, framework, strategy, coverage, samples, history)
    raw_output = client.complete(
        system_prompt,
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_label=f'framework-synthesis-iter{iteration:02d}-step{train_step:03d}-{record.skill_id}',
    )
    response = parse_json_object(raw_output)
    next_framework, next_strategy, requested_update = apply_synthesis_response(framework, strategy, response)
    changed = requested_update and (next_framework != framework or next_strategy != strategy)
    metric_entry['updated'] = bool(changed)
    metric_entry['reason'] = str(response.get('reason') or '')
    write_phase_artifacts(
        step_root,
        framework=framework,
        strategy=strategy,
        coverage=coverage,
        metric_entry=metric_entry,
        metric_history=history,
        response=response,
        next_framework=next_framework,
        next_strategy=next_strategy,
    )
    print(f"[train-update] iter={iteration} step={train_step} skill={record.skill_id} updated={changed}")
    return next_framework, next_strategy, metric_entry


def run_dry_run(args: argparse.Namespace, output_root: Path, framework: dict[str, Any], strategy: dict[str, Any]) -> int:
    analysis_root = Path(args.analysis_root).resolve()
    result_paths = discover_skill_results(analysis_root)
    if not result_paths:
        print(f'No threat_analysis.json files found under {analysis_root}', file=sys.stderr)
        return 1
    results = [load_result(path) for path in result_paths]
    coverage = aggregate_corpus_coverage(results, framework, strategy)
    attach_z3_rewards(results=results, framework=framework, strategy=strategy, coverage=coverage, output_root=None)
    history = [build_metric_entry(iteration=0, phase='dry_run', coverage=coverage, failures=[], reason='dry-run existing analysis root')]
    write_final_outputs(output_root, framework, strategy, coverage, history)
    print(metric_line('dry-run', coverage))
    return 0


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    framework_file = Path(args.framework_file).resolve()
    strategy_file = Path(args.strategy_file).resolve()
    framework, strategy = load_seed_framework_and_strategy(framework_file, strategy_file)

    if args.dry_run:
        return run_dry_run(args, output_root, framework, strategy)

    if args.iterations < 0:
        print('--iterations must be >= 0', file=sys.stderr)
        return 1
    if args.sample_size < 0:
        print('--sample-size must be >= 0', file=sys.stderr)
        return 1
    if args.test_size <= 0:
        print('--test-size must be > 0', file=sys.stderr)
        return 1

    data_root = Path(args.data_root).resolve()
    selected_ids = set(args.skill) if args.skill else None
    records = collect_records(data_root, selected_ids=selected_ids, graph_name=args.graph_name)
    if args.limit > 0:
        records = records[: args.limit]
    if len(records) < 2:
        print(f'At least 2 matching graph_wo_check.md and SKILL.md pairs are required under {data_root}', file=sys.stderr)
        return 1

    train_records, test_records, split_info = split_train_test(records, test_size=args.test_size, seed=args.split_seed)
    if not train_records or not test_records:
        print('Train/test split produced an empty split.', file=sys.stderr)
        return 1
    write_json(output_root / 'dataset_split.json', split_info)

    api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not api_key:
        print('DEEPSEEK_API_KEY is required for train/test propagation generation.', file=sys.stderr)
        return 1

    client = DeepSeekClient(
        api_key=api_key,
        model=args.model,
        api_url=args.api_url,
        retries=args.deepseek_retries,
        retry_seconds=args.deepseek_retry_seconds,
    )
    system_prompt = read_text(Path(args.system_prompt_file).resolve()).strip()
    analysis_prompt = read_text(Path(args.analysis_prompt_file).resolve()).strip()
    propagation_prompt = read_text(Path(args.propagation_prompt_file).resolve()).strip()
    synthesis_prompt = read_text(Path(args.prompt_file).resolve()).strip()
    chain_template_file = Path(args.chain_template_file).resolve()
    chain_template = render_formal_chain_template(load_chain_template(chain_template_file))

    train_rng = random.Random(args.sample_seed)
    current_framework = framework
    current_strategy = strategy
    history: list[dict[str, Any]] = []
    final_coverage: dict[str, Any] = {}

    print(
        f"Dataset split: train={len(train_records)} test={len(test_records)} "
        f"split_seed={args.split_seed} output={output_root}"
    )

    iteration_zero_root = output_root / 'iteration_00'
    iteration_zero_root.mkdir(parents=True, exist_ok=True)
    final_coverage, _ = run_test_evaluation(
        client=client,
        test_records=test_records,
        iteration_root=iteration_zero_root,
        iteration=0,
        framework=current_framework,
        strategy=current_strategy,
        system_prompt=system_prompt,
        analysis_prompt=analysis_prompt,
        propagation_prompt=propagation_prompt,
        chain_template=chain_template,
        chain_template_file=chain_template_file,
        args=args,
        history=history,
    )
    write_history_outputs(output_root, history)

    for iteration in range(1, args.iterations + 1):
        iteration_root = output_root / f'iteration_{iteration:02d}'
        iteration_root.mkdir(parents=True, exist_ok=True)
        selected_train_records, train_sample_info = select_train_records(
            train_records,
            sample_size=args.sample_size,
            rng=train_rng,
            iteration=iteration,
        )
        write_json(iteration_root / 'train_sample_info.json', train_sample_info)
        print(
            f"[iteration {iteration}] train updates={len(selected_train_records)} "
            f"from train population={len(train_records)}"
        )
        train_summaries: list[dict[str, Any]] = []
        for train_step, record in enumerate(selected_train_records, start=1):
            current_framework, current_strategy, train_summary = run_training_step(
                client=client,
                record=record,
                iteration_root=iteration_root,
                iteration=iteration,
                train_step=train_step,
                framework=current_framework,
                strategy=current_strategy,
                system_prompt=system_prompt,
                analysis_prompt=analysis_prompt,
                propagation_prompt=propagation_prompt,
                synthesis_prompt=synthesis_prompt,
                chain_template=chain_template,
                chain_template_file=chain_template_file,
                args=args,
                history=history,
            )
            train_summaries.append(train_summary)
        write_json(iteration_root / 'train_summary.json', train_summaries)

        final_coverage, _ = run_test_evaluation(
            client=client,
            test_records=test_records,
            iteration_root=iteration_root,
            iteration=iteration,
            framework=current_framework,
            strategy=current_strategy,
            system_prompt=system_prompt,
            analysis_prompt=analysis_prompt,
            propagation_prompt=propagation_prompt,
            chain_template=chain_template,
            chain_template_file=chain_template_file,
            args=args,
            history=history,
        )
        write_json(iteration_root / 'framework_definition.json', current_framework)
        write_json(iteration_root / 'parsing_strategy.json', current_strategy)
        write_json(iteration_root / 'iteration_summary.json', history[-1])
        write_history_outputs(output_root, history)

    write_final_outputs(output_root, current_framework, current_strategy, final_coverage, history)
    print(metric_line('final test', final_coverage))
    print(
        f"[final] framework={output_root / 'framework_definition.json'} "
        f"strategy={output_root / 'parsing_strategy.json'} "
        f"reward_chart={output_root / 'test_reward_history.png'}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
