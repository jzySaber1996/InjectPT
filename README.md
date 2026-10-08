# AgentThreatMining

#### Description

AgentThreatMining is used to organize, normalize, and analyze Agent skill data, and to generate intermediate workflow representations from `SKILL.md` files for threat mining and process analysis.

The repository currently provides two main capabilities:

1. Export raw skill data into a normalized directory that external tools can consume reliably
2. Use DeepSeek + prompts to read skill documents and generate query-oriented Mermaid workflow graphs

#### Repository Layout

- `data/hot100-300/`: raw dataset
- `data/inner_representation/`: normalized, tool-ready dataset
- `scripts/export_tool_ready_data.py`: exports raw data into the normalized layout
- `scripts/generate_skill_graphs_with_deepseek.py`: calls DeepSeek and writes `graph_wo_check.md`
- `prompts/skill_workflow_mermaid_prompt.md`: prompt template for workflow graph generation
- `docs/deepseek_graph_generation.md`: additional documentation for DeepSeek graph generation

#### Requirements

1. Python 3.10+
2. `requests` installed
3. `DEEPSEEK_API_KEY` set if you want to use DeepSeek graph generation

Example:

```bash
pip install requests
export DEEPSEEK_API_KEY=your_api_key
```

#### Usage

1. Export the normalized dataset

```bash
python3 scripts/export_tool_ready_data.py
```

This generates:

- `data/inner_representation/index.json`
- `data/inner_representation/by_slug.json`
- `data/inner_representation/stats.json`
- `data/inner_representation/<skill>/...`

Each skill directory uses stable filenames such as:

- `SKILL.md`
- `_meta.json`
- `primary.json`
- `classified.json`
- `classified_version2.json`
- `human.json`
- `regex.json`
- `manifest.json`

2. Generate a query workflow graph for a single skill

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --skill peekaboo --force
```

3. Generate query workflow graphs in batch

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --force
```

4. Preview matched skills without calling DeepSeek

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --skill 1password --dry-run
```

#### Graph Generation Rules

The generation script reads `data/inner_representation/<skill>/SKILL.md` and `manifest.json`, then instructs the model to:

1. Collapse tool execution, command execution, data reads, and API calls into workflow nodes
2. Emit fixed node fields for every node: `name`, `task`, `input`, `output`, and `constraint`
3. Use lowercase snake_case for `name`; if the skill exposes an original script or command name, normalize and reuse it
4. Build the workflow only around query, read, retrieval, and analysis paths
5. Output a `graph TD` Mermaid state-machine-style workflow with exactly one start node and one finish node
6. Include branch conditions, failure paths, and rollback or retry logic when needed

The generated result is written to:

- `data/inner_representation/<skill>/graph_wo_check.md`

#### Notes

- External tools should consume the normalized dataset instead of depending on the raw `hot100-300/` tree
- After DeepSeek returns a graph, the script validates the Mermaid structure; if the first output is invalid, it performs one automatic repair retry
- If `graph_wo_check.md` already exists for a skill, the script skips it by default; use `--force` to overwrite

#### Contribution

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request
