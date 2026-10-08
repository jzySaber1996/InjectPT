# DeepSeek Skill Workflow Graph Generation

This workflow reads normalized skills from `data/inner_representation/` and
uses DeepSeek to generate query-oriented Mermaid workflow graphs.

## Inputs

- `data/inner_representation/<skill>/SKILL.md`
- `data/inner_representation/<skill>/manifest.json`

## Output

- `data/inner_representation/<skill>/graph_wo_check.md`

## Required environment

- `DEEPSEEK_API_KEY`
- Optional:
  - `DEEPSEEK_MODEL`
  - `DEEPSEEK_API_URL`

## Example

Generate one skill:

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --skill peekaboo --force
```

Generate the first 10 skills:

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --limit 10 --force
```

Generate all skills:

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --force
```

Preview matched skills without calling DeepSeek:

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --skill 1password --dry-run
```

## Behavior

- Focuses on query/read/retrieval/analysis paths from `SKILL.md`
- Collapses tool execution, command execution, API calls, and data reads into workflow nodes
- Requires every node label to include `name`, `task`, `input`, `output`, and `constraint`
- Uses lowercase snake_case for the `name` field, reusing original script or command names when possible
- Requires a single start node and a single finish node
- Requires branch conditions and failure handling
- Retries once with a repair prompt if the first Mermaid output fails validation
