# DeepSeek Threat Propagation Analysis

这套脚本用于在 `data/inner_representation_v2/` 中配对搜索每个 skill 的
`graph_wo_check.md` 与 `SKILL.md`，并调用 DeepSeek 做两阶段安全分析：

1. 基于流程图节点、边和 `SKILL.md` 详细说明，识别间接工具指令注入的发生位置、影响、STRIDE/OWASP-style Agent 威胁类型和缓解建议。
2. 基于第一阶段 JSON 分析结果、简化后的理论框架和解析策略，再模拟威胁传播链路，并输出传播链 JSON 与 Mermaid 图；`propagation.md` 中的 Mermaid 面向攻击过程，起点是被攻击的工具/节点。

## Inputs

- `data/inner_representation_v2/<skill>/graph_wo_check.md`
- `data/inner_representation_v2/<skill>/SKILL.md`
- `data/inner_representation_v2/<skill>/manifest.json` when available

`data/inner_representation_v2/` 是仓库内的默认输入位置，对应归一化后的 v2 数据视图。

## Required environment

- `DEEPSEEK_API_KEY`
- Optional:
  - `DEEPSEEK_MODEL`
  - `DEEPSEEK_API_URL`

Prompt templates:

- `prompts/threat_security_system_prompt.md`
- `prompts/threat_analysis_prompt.md`
- `prompts/threat_propagation_prompt.md`
- `prompts/threat_template_evolution_prompt.md`

Framework and parsing strategy:

- `template/threat_propagation_framework.json`
- `template/threat_propagation_parsing_strategy.json`

Legacy chain template, kept for optional archive/evolution compatibility:

- `template/threat_propagation_chain_template.json`

## Usage

Preview matched graph/SKILL pairs without calling DeepSeek:

```bash
python3 scripts/threat_propagation/analyze_with_deepseek.py --skill browserctl --dry-run
```

Analyze one skill:

```bash
python3 scripts/threat_propagation/analyze_with_deepseek.py --skill browserctl --force
```

Analyze the first 10 matched skills:

```bash
python3 scripts/threat_propagation/analyze_with_deepseek.py --limit 10 --force
```

Analyze a batch and evolve the chain template after each successful skill:

```bash
python3 scripts/threat_propagation/analyze_with_deepseek.py --limit 10 --force --evolve-chain-template
```

Analyze a limited batch with explicit result and template-archive output locations:

```bash
python3 scripts/threat_propagation/analyze_with_deepseek.py \
  --limit 10 \
  --output-root artifacts/threat_propagation_analysis_limit10 \
  --chain-template-archive-root artifacts/threat_template_archive \
  --force \
  --evolve-chain-template
```

Run unified framework/strategy synthesis from existing per-skill results without calling DeepSeek:

```bash
python3 scripts/threat_propagation/synthesize_framework_with_deepseek.py --dry-run
```

Run three DeepSeek-guided synthesis iterations and write coverage metrics plus a seaborn chart:

```bash
python3 scripts/threat_propagation/synthesize_framework_with_deepseek.py --iterations 3
```

Use another normalized dataset root:

```bash
python3 scripts/threat_propagation/analyze_with_deepseek.py \
  --data-root data/inner_representation \
  --skill 1password \
  --force
```

## Outputs

Default output root:

- `artifacts/threat_propagation_analysis/`

Per skill:

- `artifacts/threat_propagation_analysis/<skill>/threat_analysis.json`
- `artifacts/threat_propagation_analysis/<skill>/propagation.json`
- `artifacts/threat_propagation_analysis/<skill>/propagation.mmd`
- `artifacts/threat_propagation_analysis/<skill>/propagation.md`
- `artifacts/threat_propagation_analysis/<skill>/chain_template.before.json` when `--evolve-chain-template` updates the template
- `artifacts/threat_propagation_analysis/<skill>/chain_template.after.json` when `--evolve-chain-template` updates the template

Run summary:

- `artifacts/threat_propagation_analysis/summary.jsonl`

Framework synthesis outputs:

- `artifacts/threat_framework_synthesis/framework_definition.json`
- `artifacts/threat_framework_synthesis/parsing_strategy.json`
- `artifacts/threat_framework_synthesis/coverage_summary.json`
- `artifacts/threat_framework_synthesis/metric_history.jsonl`
- `artifacts/threat_framework_synthesis/metric_history.png`

## Notes

- The script performs two DeepSeek calls per skill.
- Existing per-skill outputs are skipped unless `--force` is set.
- Prompts ask for defensive analysis only and avoid exploit payloads.
- The propagation chain is simulated by DeepSeek as JSON; `propagation.md` and `propagation.mmd` are generated from that JSON so the first node is always the attacked tool/node, not the normal workflow start node.

## Framework Coverage Metrics

`framework_path_coverage_score` measures the weighted share of DeepSeek-generated threat propagation units that can be explained by the framework components, decision gates, or evidence axes. `strategy_parse_coverage_score` measures the weighted share of the same units that can be parsed by the strategy steps or mapping rules. The synthesis script recomputes both metrics before and after every DeepSeek-proposed update, records uncovered units for optimization, prints the values per iteration, and renders `metric_history.png` with seaborn when available.

## Template Evolution

`template/threat_propagation_chain_template.json` defines the reusable formal propagation-chain structure. It contains two required anchors: `anchors.propagation_chain` for the attack path structure and `anchors.gate_units` for AND/OR/NOT gate logic. The script parses this JSON and renders the parsed logic into `{chain_template}` for every prediction.

By default the script only reads the JSON template. When `--evolve-chain-template` is set, the loop performs one additional DeepSeek call after each successful skill analysis and may update the formal template with reusable stages, gate units, AND/OR/NOT logic templates, impact reasoning, or Mermaid constraints. The next skill in the same run uses the updated template.

Template evolution uses a conservative delta strategy: DeepSeek is asked to return only small `additions`, not a full template. The prompt asks for one high-information reusable concept when possible, and `has_updates: false` when the current chain does not justify a reusable change. The script merges at most two total additions and at most one new item per allowed field into the current local template, preserves `anchors.propagation_chain`, `anchors.gate_units`, and `AND/OR/NOT`, then validates the merged JSON before writing anything. If the model returns no usable additions, the template is left unchanged. If validation fails, the script records `template_evolution_error`, keeps the previous template, and continues; the current skill result remains valid. When `--evolve-chain-template` is enabled, every successfully analyzed skill stores one template snapshot under the single run-level directory named by the run start time: `template/YYYY-MM-DDTHH-MM-SSZ/` by default, or `<archive-root>/YYYY-MM-DDTHH-MM-SSZ/` when `--chain-template-archive-root` is set. Changed templates are also written back to the root JSON file. The directory is written to `summary.jsonl` as `chain_template_archive_dir`, each snapshot path is written as `chain_template_snapshot`, and `chain_template_snapshot_status` is `evolved`, `unchanged`, or `evolution_failed`.

## Formal Template Structure

The default JSON template uses a compact formal structure:

- `anchors.propagation_chain`: defines the attack-chain start rule, required propagation stages, edge requirements, impact dimensions, and Mermaid requirements.
- `anchors.gate_units`: defines gate operators `AND`, `OR`, `NOT`, reusable gate units, and logic templates that describe when entry, propagation, impact, or containment is reached.
- `output_contract`: defines how generated chains should bind `starting_attack_point`, `path`, `edges`, impact, and gate reasoning.
- `evolution_policy`: constrains what template evolution may add and what it must not change.

Successful template updates are archived as JSON files under one UTC run-start timestamp directory, for example `template/2026-07-02T15-30-00Z/06-ai_threat_propagation_chain_template.json`, `template/2026-07-02T15-30-00Z/07-tool_threat_propagation_chain_template.json`, or under `artifacts/threat_template_archive/2026-07-02T15-30-00Z/` when `--chain-template-archive-root artifacts/threat_template_archive` is used.

## Qwen2.5 Reward-Ranked Training

`scripts/threat_propagation/train_framework_qwen25_reward_dpo.py` trains Qwen2.5
to propose the same incremental framework and parsing-strategy JSON actions used
by `synthesize_framework_with_deepseek.py`.

For every selected training skill, the script evaluates the current state, samples
several Qwen actions, validates every action with `apply_framework_updates` and
`apply_strategy_updates`, then re-evaluates each valid change with the existing
DeepSeek propagation environment. Its reward is the existing primary reward:

- with `--openclaw-runtime-eval`, use `openclaw_runtime_reward`, which defaults
to 90% strict canary success and 10% runtime path coverage;
- otherwise, use the existing Z3 reward.

For an identical policy prompt, the highest- and lowest-reward valid Qwen actions
form a chosen/rejected pair. The script writes the LLaMA-Factory ranking data to
`llamafactory_data/threat_framework_reward_dpo.jsonl`. When
`--run-llamafactory-dpo` is enabled, it generates a per-round LLaMA-Factory DPO
YAML, invokes `llamafactory-cli train`, and reloads the resulting LoRA adapter
for the next Qwen rollout round. This gives a continuous reward-ranked RLHF/DPO
loop without modifying the installed LLaMA-Factory source to inject an in-process
custom GRPO reward function.

The evaluated environment remains controlled: actions are limited to the existing
framework/strategy schema, while the OpenClaw evaluator retains its staged,
read-only, non-destructive canary workflow.

Validate dataset discovery and configuration without loading Qwen or calling
DeepSeek:

```bash
python3 scripts/threat_propagation/train_framework_qwen25_reward_dpo.py \
  --config llamafactory/threat_framework_reward_rl_local.yaml \
  --prepare-only
```

Run one small end-to-end training round. It requires `DEEPSEEK_API_KEY`, the local
Qwen2.5 model, an installed `llamafactory-cli`, and the existing local OpenClaw
profile. Start with a small dataset because every candidate is evaluated in the
same propagation and runtime environment:

```bash
DEEPSEEK_API_KEY=... \
bash llamafactory/run_threat_framework_reward_rl.sh \
  llamafactory/threat_framework_reward_rl_local.yaml \
  --limit 4 --test-size 1 --rounds 1 --sample-size 1 --samples-per-prompt 3
```

To inspect generated rewards and preference pairs before model training, invoke
the Python script directly and omit `--run-llamafactory-dpo`. Important artifacts:

- `artifacts/threat_framework_qwen25_reward_rl/reward_history.jsonl`
- `artifacts/threat_framework_qwen25_reward_rl/dpo_preferences.jsonl`
- `artifacts/threat_framework_qwen25_reward_rl/llamafactory_data/dataset_info.json`
- `artifacts/threat_framework_qwen25_reward_rl/round_*/llamafactory_dpo.yaml`
- `artifacts/threat_framework_qwen25_reward_rl/round_*/test/reward_summary.json`
