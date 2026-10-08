#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_KEYS = [
    'loss',
    'eval_loss',
    'rewards/accuracies',
    'eval_rewards/accuracies',
    'rewards/chosen',
    'rewards/rejected',
    'rewards/margins',
    'eval_rewards/chosen',
    'eval_rewards/rejected',
    'eval_rewards/margins',
    'learning_rate',
]

ROLLING_KEYS = [
    'avg_reward',
    'avg_best_reward_per_skill',
    'z3_pass_rate',
    'path_exists_rate',
    'pass_rate',
    'preferences_created',
    'cumulative_preferences',
    'candidate_count',
]


def load_trainer_state(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload.get('log_history', [])


def load_rolling_history(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise SystemExit('rolling_dpo_history.json must contain a JSON list')
    return payload


def build_frame(log_history: list[dict], keys: list[str]) -> pd.DataFrame:
    rows = []
    for item in log_history:
        step = item.get('step')
        epoch = item.get('epoch')
        for key in keys:
            if key in item:
                rows.append({'step': step, 'epoch': epoch, 'metric': key, 'value': item[key]})
    return pd.DataFrame(rows)


def build_rolling_frame(history_rows: list[dict], keys: list[str]) -> pd.DataFrame:
    rows = []
    for item in history_rows:
        round_index = item.get('round', item.get('iteration'))
        sampled_skill_count = len(item.get('sampled_skill_ids', []))
        dpo_returncode = item.get('dpo', {}).get('returncode')
        dpo_elapsed_seconds = item.get('dpo', {}).get('elapsed_seconds')
        for key in keys:
            if key in item:
                rows.append(
                    {
                        'round': round_index,
                        'metric': key,
                        'value': item[key],
                        'sampled_skill_count': sampled_skill_count,
                        'dpo_returncode': dpo_returncode,
                        'dpo_elapsed_seconds': dpo_elapsed_seconds,
                    }
                )
    return pd.DataFrame(rows)


def plot_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='talk')

    groups = {
        'loss_metrics': ['loss', 'eval_loss'],
        'reward_accuracy_metrics': ['rewards/accuracies', 'eval_rewards/accuracies'],
        'reward_value_metrics': [
            'rewards/chosen', 'rewards/rejected', 'rewards/margins',
            'eval_rewards/chosen', 'eval_rewards/rejected', 'eval_rewards/margins',
        ],
        'learning_rate': ['learning_rate'],
    }

    for stem, metrics in groups.items():
        subset = df[df['metric'].isin(metrics)].copy()
        if subset.empty:
            continue
        plt.figure(figsize=(12, 7))
        ax = sns.lineplot(data=subset, x='step', y='value', hue='metric', marker='o')
        ax.set_title(stem.replace('_', ' ').title())
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Value')
        ax.figure.tight_layout()
        figure_path = output_dir / f'{stem}.png'
        ax.figure.savefig(figure_path, dpi=160)
        plt.close(ax.figure)
        print(f'Saved figure: {figure_path}')


def plot_rolling_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='talk')

    groups = {
        'rolling_reward_metrics': ['avg_reward', 'avg_best_reward_per_skill'],
        'rolling_validation_rates': ['z3_pass_rate', 'path_exists_rate', 'pass_rate'],
        'rolling_preference_growth': ['preferences_created', 'cumulative_preferences', 'candidate_count'],
    }

    for stem, metrics in groups.items():
        subset = df[df['metric'].isin(metrics)].copy()
        if subset.empty:
            continue
        plt.figure(figsize=(12, 7))
        ax = sns.lineplot(data=subset, x='round', y='value', hue='metric', marker='o')
        ax.set_title(stem.replace('_', ' ').title())
        ax.set_xlabel('Round')
        ax.set_ylabel('Value')
        ax.figure.tight_layout()
        figure_path = output_dir / f'{stem}.png'
        ax.figure.savefig(figure_path, dpi=160)
        plt.close(ax.figure)
        print(f'Saved figure: {figure_path}')

    dpo_df = (
        df[['round', 'dpo_returncode', 'dpo_elapsed_seconds']]
        .drop_duplicates()
        .sort_values('round')
    )
    if not dpo_df.empty and dpo_df['dpo_returncode'].notna().any():
        plt.figure(figsize=(12, 7))
        ax = sns.scatterplot(data=dpo_df, x='round', y='dpo_returncode', s=140)
        ax.set_title('Rolling DPO Return Codes')
        ax.set_xlabel('Round')
        ax.set_ylabel('DPO Return Code')
        ax.figure.tight_layout()
        figure_path = output_dir / 'rolling_dpo_return_codes.png'
        ax.figure.savefig(figure_path, dpi=160)
        plt.close(ax.figure)
        print(f'Saved figure: {figure_path}')

    if not dpo_df.empty and dpo_df['dpo_elapsed_seconds'].notna().any():
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=dpo_df, x='round', y='dpo_elapsed_seconds', color='#4C72B0')
        ax.set_title('Rolling DPO Elapsed Seconds')
        ax.set_xlabel('Round')
        ax.set_ylabel('Seconds')
        ax.figure.tight_layout()
        figure_path = output_dir / 'rolling_dpo_elapsed_seconds.png'
        ax.figure.savefig(figure_path, dpi=160)
        plt.close(ax.figure)
        print(f'Saved figure: {figure_path}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot DPO train/eval metrics with seaborn.')
    parser.add_argument('--trainer-state', default='', help='Path to trainer_state.json')
    parser.add_argument('--rolling-history', default='', help='Path to rolling_dpo_history.json')
    parser.add_argument('--output-dir', default='', help='Directory to save figures. Defaults to a seaborn_plots folder beside the input file')
    args = parser.parse_args()

    if bool(args.trainer_state) == bool(args.rolling_history):
        raise SystemExit('Pass exactly one of --trainer-state or --rolling-history')

    if args.trainer_state:
        trainer_state = Path(args.trainer_state).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else trainer_state.parent / 'seaborn_plots'
        log_history = load_trainer_state(trainer_state)
        df = build_frame(log_history, DEFAULT_KEYS)
        if df.empty:
            raise SystemExit('No matching metrics found in trainer_state.json')
        csv_path = output_dir / 'dpo_metrics_long.csv'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f'Saved metric table: {csv_path}')
        plot_metrics(df, output_dir)
        return 0

    rolling_history = Path(args.rolling_history).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else rolling_history.parent / 'seaborn_plots'
    history_rows = load_rolling_history(rolling_history)
    df = build_rolling_frame(history_rows, ROLLING_KEYS)
    if df.empty:
        raise SystemExit('No matching metrics found in rolling_dpo_history.json')
    csv_path = output_dir / 'rolling_dpo_metrics_long.csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f'Saved metric table: {csv_path}')
    plot_rolling_metrics(df, output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
