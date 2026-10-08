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
    'avg_reward',
    'avg_shaped_reward',
    'baseline_reward',
    'avg_advantage',
    'avg_scaled_advantage',
    'max_reward',
    'min_reward',
    'effective_trajectories',
    'sampled_trajectories',
    'rollout_round_count',
    'num_samples_per_prompt',
    'target_min_pg_updates_per_step',
    'skipped_optimizer_step',
    'buffer_only_update',
    'anchor_only_step',
    'success_buffer_size',
    'seeded_success_buffer_count',
    'success_buffer_add_count',
    'success_buffer_batch_count',
    'graph_prefix_pass_count',
    'graph_prefix_skip_count',
    'graph_prefix_pass_rate',
    'edge_marker_pass_count',
    'edge_marker_pass_rate',
    'finish_name_pass_count',
    'finish_name_pass_rate',
    'full_start_reach_pass_count',
    'full_start_reach_pass_rate',
    'full_finish_reach_pass_count',
    'full_finish_reach_pass_rate',
    'pg_eligible_count',
    'pg_eligible_rate',
    'pg_update_count',
    'pg_update_rate',
    'anchor_only_count',
    'negative_advantage_skip_count',
    'advantage_scale',
    'elapsed_seconds',
]

VALIDATION_KEYS = [
    'avg_reward',
    'max_reward',
    'min_reward',
    'z3_pass_rate',
    'path_exists_rate',
    'pass_rate',
    'pass_with_warnings_rate',
    'exact_match_rate',
    'candidate_count',
]

TRAIN_GROUPS = {
    'train_reward_metrics': ['avg_reward', 'baseline_reward', 'avg_scaled_advantage'],
    'train_rl_gate_metrics': [
        'finish_name_pass_rate',
        'pg_eligible_rate',
        'pg_update_rate',
    ],
    'train_sampling_metrics': [
        'sampled_trajectories',
        'rollout_round_count',
        'pg_update_count',
        'target_min_pg_updates_per_step',
        'success_buffer_size',
        'seeded_success_buffer_count',
        'success_buffer_add_count',
        'success_buffer_batch_count',
        'anchor_only_step',
        'buffer_only_update',
        'skipped_optimizer_step',
    ],
}

VALIDATION_GROUPS = {
    'validation_pass_rates': [
        'validation/z3_pass_rate',
        'validation/path_exists_rate',
        'validation/pass_rate',
    ],
}

DASHBOARD_IMAGES = [
    ('Train Reward Metrics', 'train_reward_metrics.png'),
    ('Train RL Gate Metrics', 'train_rl_gate_metrics.png'),
    ('Train Sampling Metrics', 'train_sampling_metrics.png'),
    ('Validation Pass Rates', 'validation_pass_rates.png'),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
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
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta http-equiv=\"refresh\" content=\"{max(1, int(refresh_seconds))}\">
  <title>REINFORCE Live Metrics</title>
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
  <h1>REINFORCE Live Metrics</h1>
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
        raise SystemExit('No rows found in reinforce_train_log.jsonl')

    train_df = build_train_frame(rows, TRAIN_KEYS)
    validation_df = build_validation_frame(rows, VALIDATION_KEYS)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    if not train_df.empty:
        train_csv = resolved_output_dir / 'reinforce_train_metrics_long.csv'
        train_df.to_csv(train_csv, index=False)
        artifacts['train_csv'] = train_csv
        if not quiet:
            print(f'Saved metric table: {train_csv}')
        plot_groups(train_df, resolved_output_dir, TRAIN_GROUPS, 'step', quiet=quiet)

    if not validation_df.empty:
        validation_csv = resolved_output_dir / 'reinforce_validation_metrics_long.csv'
        validation_df.to_csv(validation_csv, index=False)
        artifacts['validation_csv'] = validation_csv
        if not quiet:
            print(f'Saved metric table: {validation_csv}')
        plot_groups(validation_df, resolved_output_dir, VALIDATION_GROUPS, 'step', quiet=quiet)

    if train_df.empty and validation_df.empty:
        raise SystemExit('No matching metrics found in reinforce_train_log.jsonl')

    if refresh_dashboard:
        artifacts['dashboard'] = write_dashboard_html(
            resolved_output_dir,
            refresh_seconds=dashboard_refresh_seconds,
            quiet=quiet,
        )
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot REINFORCE reward and validation metrics with seaborn.')
    parser.add_argument('--log-file', required=True, help='Path to reinforce_train_log.jsonl')
    parser.add_argument('--output-dir', default='', help='Directory to save figures. Defaults to a seaborn_plots folder beside the log file')
    parser.add_argument('--refresh-dashboard', action='store_true', help='Generate an auto-refresh HTML dashboard beside the plots.')
    parser.add_argument('--dashboard-refresh-seconds', type=int, default=5, help='Auto-refresh interval for the HTML dashboard.')
    args = parser.parse_args()

    render_log_file(
        log_file=Path(args.log_file),
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        refresh_dashboard=args.refresh_dashboard,
        dashboard_refresh_seconds=args.dashboard_refresh_seconds,
        quiet=False,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
