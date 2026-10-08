# InjectPT

#### 介绍

InjectPT 用于整理、标准化和分析 Agent skill 数据，并基于 `SKILL.md` 生成可用于威胁挖掘和流程分析的中间表示。

当前仓库已经支持两类核心能力：

1. 将原始 skill 数据整理为外部工具可稳定访问的标准化目录
2. 基于 DeepSeek + Prompt 读取 skill 文档，生成查询型 Mermaid 任务流程图

#### 目录结构

- `data/hot100-300/`：原始数据
- `data/inner_representation/`：标准化后的可消费数据目录
- `scripts/export_tool_ready_data.py`：原始数据导出为标准化目录
- `scripts/generate_skill_graphs_with_deepseek.py`：调用 DeepSeek 生成 `graph_wo_check.md`
- `prompts/skill_workflow_mermaid_prompt.md`：流程图生成提示词模板
- `docs/deepseek_graph_generation.md`：DeepSeek 流程图生成补充说明

#### 环境要求

1. Python 3.10+
2. 安装 `requests`
3. 若使用 DeepSeek 生成功能，需要设置环境变量 `DEEPSEEK_API_KEY`

示例：

```bash
pip install requests
export DEEPSEEK_API_KEY=your_api_key
```

#### 使用说明

1. 导出标准化数据目录

```bash
python3 scripts/export_tool_ready_data.py
```

执行后会生成：

- `data/inner_representation/index.json`
- `data/inner_representation/by_slug.json`
- `data/inner_representation/stats.json`
- `data/inner_representation/<skill>/...`

其中每个 skill 目录包含统一文件名，例如：

- `SKILL.md`
- `_meta.json`
- `primary.json`
- `classified.json`
- `classified_version2.json`
- `human.json`
- `regex.json`
- `manifest.json`

2. 针对单个 skill 生成查询流程图

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --skill peekaboo --force
```

3. 批量生成查询流程图

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --force
```

4. 仅预览将被处理的 skill，不调用 DeepSeek

```bash
python3 scripts/generate_skill_graphs_with_deepseek.py --skill 1password --dry-run
```

#### 流程图生成规则

生成脚本会读取 `data/inner_representation/<skill>/SKILL.md` 和 `manifest.json`，并要求模型：

1. 将工具执行、命令执行、数据读取、API 调用统一抽象为工作流节点
2. 为每个节点输出固定字段：`name`、`task`、`input`、`output`、`constraint`
3. `name` 使用小写 snake_case；若 skill 中出现原始脚本/命令名，优先规范化后直接使用
4. 仅围绕查询、读取、检索、分析路径构造流程
5. 输出 `graph TD` Mermaid 状态机，并保证全图只有一个开始节点和一个完成节点
6. 包含条件判断、失败分支、必要时的回滚或重试逻辑

生成结果默认写入：

- `data/inner_representation/<skill>/graph_wo_check.md`

#### 说明

- 标准化目录用于给外部工具提供稳定路径，不建议直接依赖原始 `hot100-300/` 的目录结构
- DeepSeek 输出后脚本会做一次 Mermaid 结构校验；若第一次输出不合格，会自动进行一次修复重试
- 若某个 skill 已存在 `graph_wo_check.md`，默认跳过；加 `--force` 会覆盖

#### 参与贡献

1. Fork 本仓库
2. 新建特性分支
3. 提交修改
4. 发起 Pull Request
