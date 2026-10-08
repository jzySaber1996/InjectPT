#!/usr/bin/env python3
"""Analyze threat propagation with a local Ollama/Qwen3 chat model.

This routes every model call through local Ollama /api/chat and does
not require any remote model API key.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_with_deepseek import (  # type: ignore
    DEFAULT_GRAPH_NAME,
    analyze_record,
    append_jsonl,
    archive_chain_template,
    collect_records,
    evaluate_template_understandability,
    evolve_chain_template,
    load_chain_template,
    load_threat_framework,
    load_threat_parsing_strategy,
    read_text,
    render_formal_chain_template,
    summarize_result,
    summarize_template_understandability,
    write_json,
    write_text,
)
from ollama_chat_client import (  # type: ignore
    DEFAULT_OLLAMA_API_URL,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_THINK,
    OllamaClient,
    parse_ollama_options,
    parse_ollama_think,
    rewrite_object_for_ollama,
    rewrite_prompt_for_ollama,
)
from openclaw_runtime_evaluator import add_openclaw_runtime_args, openclaw_config_from_args  # type: ignore


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
        default=str(repo_root / "artifacts" / "threat_propagation_analysis_ollama_qwen3"),
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
    parser.add_argument("--max-framework-chars", type=int, default=12000, help="Maximum framework-definition characters sent to Ollama.")
    parser.add_argument("--max-strategy-chars", type=int, default=18000, help="Maximum parsing-strategy characters sent to Ollama.")
    parser.add_argument(
        "--chain-template-file",
        default=str(repo_root / "template" / "threat_propagation_chain_template.native.json"),
        help="Seed formal threat propagation chain template JSON path. The file is read-only unless --update-chain-template-file is set.",
    )
    parser.add_argument(
        "--chain-template-archive-root",
        default="",
        help="Directory used to archive evolved chain templates. Defaults to the directory containing --chain-template-file.",
    )
    parser.add_argument(
        "--template-evolution-prompt-file",
        default=str(repo_root / "prompts" / "threat_template_evolution_prompt.md"),
        help="Prompt template used to evolve the chain template.",
    )
    parser.add_argument(
        "--template-understandability-prompt-file",
        default=str(repo_root / "prompts" / "threat_template_understandability_prompt.md"),
        help="Prompt template used to evaluate Ollama/Qwen3 understandability of the chain template.",
    )
    parser.add_argument("--evolve-chain-template", action="store_true", help="Evolve the in-memory chain template after each successful skill analysis.")
    parser.add_argument("--update-chain-template-file", action="store_true", help="Write evolved templates back to --chain-template-file.")
    parser.add_argument("--skip-template-understandability", action="store_true", help="Skip the extra local model call that evaluates template understandability before evolution.")
    parser.add_argument("--max-chain-template-chars", type=int, default=18000, help="Maximum chain-template characters sent to Ollama.")
    parser.add_argument("--max-template-evolution-tokens", type=int, default=2600, help="Max completion tokens for template evolution calls.")
    parser.add_argument("--max-template-understandability-tokens", type=int, default=2600, help="Max completion tokens for template understandability evaluation calls.")
    parser.add_argument("--skill", action="append", default=[], help="Specific canonical skill id(s) to process. Can be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N matched skills. 0 means no limit.")
    parser.add_argument("--graph-name", default=DEFAULT_GRAPH_NAME, help="Workflow graph filename inside each skill directory.")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name, for example qwen3, qwen3:8b, or qwen3:14b.")
    parser.add_argument("--api-url", "--ollama-url", dest="api_url", default=DEFAULT_OLLAMA_API_URL, help="Ollama chat API URL.")
    parser.add_argument("--max-skill-chars", type=int, default=32000, help="Maximum SKILL.md characters sent to Ollama per skill.")
    parser.add_argument("--max-graph-chars", type=int, default=18000, help="Maximum graph_wo_check.md characters sent to Ollama per skill.")
    parser.add_argument("--max-analysis-tokens", type=int, default=3200, help="Max completion tokens for the threat analysis call.")
    parser.add_argument("--max-chain-tokens", type=int, default=3200, help="Max completion tokens for the propagation-chain call.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for all Ollama calls.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between skills.")
    parser.add_argument("--ollama-retries", type=int, default=2, help="Retry count for transient local Ollama errors.")
    parser.add_argument("--ollama-retry-seconds", type=float, default=5.0, help="Base retry backoff seconds.")
    parser.add_argument("--ollama-timeout-seconds", type=float, default=600.0, help="Timeout seconds for each Ollama request.")
    parser.add_argument("--ollama-keep-alive", default=DEFAULT_OLLAMA_KEEP_ALIVE, help="Ollama keep_alive value. Empty disables it.")
    parser.add_argument(
        "--ollama-think",
        choices=("auto", "true", "false"),
        default=DEFAULT_OLLAMA_THINK if DEFAULT_OLLAMA_THINK in {"auto", "true", "false"} else "false",
        help="Whether to request Ollama thinking mode. false is recommended for machine-readable JSON.",
    )
    parser.add_argument("--ollama-format-json", dest="ollama_format_json", action="store_true", default=True, help="Request Ollama JSON mode. This is enabled by default.")
    parser.add_argument("--no-ollama-format-json", dest="ollama_format_json", action="store_false", help="Do not pass format=json to Ollama.")
    parser.add_argument("--ollama-option", action="append", default=[], help="Extra Ollama option as key=value, for example --ollama-option num_ctx=65536.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-skill threat analysis outputs.")
    parser.add_argument("--include-raw", action="store_true", help="Store raw Ollama responses in the JSON output.")
    parser.add_argument("--dry-run", action="store_true", help="Only discover matching graph/SKILL pairs; do not call Ollama.")
    add_openclaw_runtime_args(parser)
    return parser.parse_args()


def load_prompt(path: str) -> str:
    return rewrite_prompt_for_ollama(read_text(Path(path).resolve()).strip())


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

    client = OllamaClient(
        model=args.model,
        api_url=args.api_url,
        retries=args.ollama_retries,
        retry_seconds=args.ollama_retry_seconds,
        timeout_seconds=args.ollama_timeout_seconds,
        keep_alive=args.ollama_keep_alive,
        think=parse_ollama_think(args.ollama_think),
        use_json_format=bool(args.ollama_format_json),
        options=parse_ollama_options(args.ollama_option),
    )
    system_prompt_template = load_prompt(args.system_prompt_file)
    analysis_prompt_template = load_prompt(args.analysis_prompt_file)
    propagation_prompt_template = load_prompt(args.propagation_prompt_file)
    framework_file = Path(args.framework_file).resolve()
    parsing_strategy_file = Path(args.parsing_strategy_file).resolve()
    framework_definition = rewrite_object_for_ollama(load_threat_framework(framework_file))
    parsing_strategy = rewrite_object_for_ollama(load_threat_parsing_strategy(parsing_strategy_file, framework_definition))
    template_evolution_prompt_template = load_prompt(args.template_evolution_prompt_file)
    template_understandability_prompt_template = load_prompt(args.template_understandability_prompt_file)
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

    print(f"Using local Ollama model: {client.model} via {client.api_url}")

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
                    print(f"[template-skip] {record.skill_id}: keep previous template; {exc}", file=sys.stderr)
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
                    print(f"[template-archive-fail] {record.skill_id}: {exc}", file=sys.stderr)
            append_jsonl(summary_path, summary)
            print(f"[ok] {record.skill_id} -> {output_json}")
        except Exception as exc:
            failures.append((record.skill_id, str(exc)))
            append_jsonl(summary_path, {"skill_id": record.skill_id, "status": "failed", "error": str(exc)})
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
