#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TRAIN_KEYS = [
    'ce_loss_normalized',
    'graph_semantic_loss',
    'structure_raw_loss',
    'total_loss',
    'graph_semantic_score',
    'graph_completeness_score',
    'avg_z3_reachability_score',
    'full_connectivity_rate',
    'strict_mermaid_syntax_score',
    'elapsed_seconds',
]

VALIDATION_KEYS = [
    'graph_completeness_rate',
    'truth_graph_semantic_score',
    'truth_node_set_similarity',
    'truth_edge_set_similarity',
    'truth_path_similarity',
    'full_connectivity_rate',
    'strict_mermaid_syntax_rate',
    'z3_pass_rate',
    'path_exists_rate',
    'candidate_count',
]

TRAIN_GROUPS = {
    'loss_metrics': ['ce_loss_normalized', 'graph_semantic_loss', 'structure_raw_loss', 'total_loss'],
    'structure_score_metrics': [
        'graph_semantic_score',
        'graph_completeness_score',
        'avg_z3_reachability_score',
        'full_connectivity_rate',
        'strict_mermaid_syntax_score',
    ],
    'train_runtime_metrics': ['elapsed_seconds'],
}

VALIDATION_GROUPS = {
    'validation_structure_metrics': [
        'validation/graph_completeness_rate',
        'validation/truth_graph_semantic_score',
        'validation/truth_node_set_similarity',
        'validation/truth_edge_set_similarity',
        'validation/truth_path_similarity',
    ],
    'validation_z3_metrics': [
        'validation/full_connectivity_rate',
        'validation/strict_mermaid_syntax_rate',
        'validation/z3_pass_rate',
        'validation/path_exists_rate',
    ],
}

DASHBOARD_IMAGES = [
    ('Loss Metrics', 'loss_metrics.png'),
    ('Structure Score Metrics', 'structure_score_metrics.png'),
    ('Validation Structure Metrics', 'validation_structure_metrics.png'),
    ('Validation Z3 Metrics', 'validation_z3_metrics.png'),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').split('\n'):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_train_frame(rows: list[dict[str, Any]], keys: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for item in rows:
        step = item.get('step')
        if step is None:
            continue
        note = item.get('note', '')
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            records.append({'step': step, 'metric': key, 'value': value, 'note': note})
    return pd.DataFrame(records)


def build_validation_frame(rows: list[dict[str, Any]], keys: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for item in rows:
        step = item.get('step')
        payload = item.get('validation')
        if step is None or not isinstance(payload, dict):
            continue
        note = item.get('note', '')
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            records.append({'step': step, 'metric': f'validation/{key}', 'value': value, 'note': note})
    return pd.DataFrame(records)


def plot_groups(df: pd.DataFrame, output_dir: Path, groups: dict[str, list[str]], x_key: str, quiet: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='talk')
    created: list[Path] = []
    for stem, metrics in groups.items():
        subset = df[df['metric'].isin(metrics)].copy()
        if subset.empty:
            continue
        plt.figure(figsize=(12, 7))
        ax = sns.lineplot(data=subset, x=x_key, y='value', hue='metric', marker='o')
        ax.set_title(stem.replace('_', ' ').title())
        ax.set_xlabel(x_key.title())
        ax.set_ylabel('Value')
        ax.figure.tight_layout()
        figure_path = output_dir / f'{stem}.png'
        ax.figure.savefig(figure_path, dpi=160)
        plt.close(ax.figure)
        created.append(figure_path)
        if not quiet:
            print(f'Saved figure: {figure_path}')
    return created


def write_dashboard_html(output_dir: Path, refresh_seconds: int = 5, quiet: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    sections: list[str] = []
    for title, filename in DASHBOARD_IMAGES:
        image_path = output_dir / filename
        if not image_path.exists():
            continue
        sections.append(
            f'<section><h2>{title}</h2><img src="{filename}?ts={timestamp}" alt="{title}"></section>'
        )
    if not sections:
        sections.append('<p>No plot images have been generated yet.</p>')
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{max(1, int(refresh_seconds))}">
  <title>Structured SFT Live Metrics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; color: #111; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #444; }}
    section {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin: 20px 0; padding: 16px; }}
    img {{ width: 100%; height: auto; display: block; }}
    code {{ background: #eee; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Structured SFT Live Metrics</h1>
  <p>This page refreshes every {max(1, int(refresh_seconds))} seconds.</p>
  <p>Directory: <code>{output_dir}</code></p>
  {''.join(sections)}
</body>
</html>
"""
    dashboard_path = output_dir / 'live_dashboard.html'
    dashboard_path.write_text(html, encoding='utf-8')
    if not quiet:
        print(f'Saved dashboard: {dashboard_path}')
    return dashboard_path


def render_log_file(
    log_file: Path,
    output_dir: Path | None = None,
    *,
    refresh_dashboard: bool = False,
    dashboard_refresh_seconds: int = 5,
    quiet: bool = False,
) -> dict[str, Path]:
    resolved_log_file = Path(log_file).resolve()
    if not resolved_log_file.exists():
        raise SystemExit(f'Missing log file: {resolved_log_file}')

    resolved_output_dir = Path(output_dir).resolve() if output_dir else resolved_log_file.parent / 'seaborn_plots'
    rows = load_jsonl(resolved_log_file)
    if not rows:
        raise SystemExit('No rows found in structured_sft_train_log.jsonl')

    train_df = build_train_frame(rows, TRAIN_KEYS)
    validation_df = build_validation_frame(rows, VALIDATION_KEYS)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    if not train_df.empty:
        train_csv = resolved_output_dir / 'structured_sft_train_metrics_long.csv'
        train_df.to_csv(train_csv, index=False)
        artifacts['train_csv'] = train_csv
        if not quiet:
            print(f'Saved metric table: {train_csv}')
        plot_groups(train_df, resolved_output_dir, TRAIN_GROUPS, 'step', quiet=quiet)

    if not validation_df.empty:
        validation_csv = resolved_output_dir / 'structured_sft_validation_metrics_long.csv'
        validation_df.to_csv(validation_csv, index=False)
        artifacts['validation_csv'] = validation_csv
        if not quiet:
            print(f'Saved metric table: {validation_csv}')
        plot_groups(validation_df, resolved_output_dir, VALIDATION_GROUPS, 'step', quiet=quiet)

    if train_df.empty and validation_df.empty:
        raise SystemExit('No matching metrics found in structured_sft_train_log.jsonl')

    if refresh_dashboard:
        artifacts['dashboard'] = write_dashboard_html(
            resolved_output_dir,
            refresh_seconds=dashboard_refresh_seconds,
            quiet=quiet,
        )

    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot structured SFT train/eval metrics with seaborn.')
    parser.add_argument('--log-file', required=True, help='Path to structured_sft_train_log.jsonl')
    parser.add_argument('--output-dir', default='', help='Directory to save figures. Defaults to a seaborn_plots folder beside the log file')
    parser.add_argument('--refresh-dashboard', action='store_true', help='Also write an auto-refresh HTML dashboard.')
    parser.add_argument('--dashboard-refresh-seconds', type=int, default=5, help='Auto-refresh interval for the HTML dashboard.')
    args = parser.parse_args()

    render_log_file(
        log_file=Path(args.log_file),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        refresh_dashboard=args.refresh_dashboard,
        dashboard_refresh_seconds=args.dashboard_refresh_seconds,
        quiet=False,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
