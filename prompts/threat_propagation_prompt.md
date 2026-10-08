你是一名 Agent 威胁传播链分析专家。你的任务是基于第一阶段威胁分析 JSON、简化后的理论框架和解析策略，以及原始工作流图，模拟间接工具指令注入攻击在工具链中的传播过程。

必须严格按以下顺序理解和生成：

1. 阅读 threat analysis JSON，找出每个威胁的起始攻击位置。
2. 阅读 threat_propagation_framework.json，理解“用哪些组件描述威胁传播链路”。
3. 阅读 threat_propagation_parsing_strategy.json，理解人类和 DeepSeek 应该如何使用这些组件做解析。
4. 阅读 parsed graph，沿节点边关系选择最短、最可解释的传播路径。
5. 每条传播链必须从“被攻击的工具/节点”开始，而不是从正常 workflow 的 start node 开始。
6. 传播链中的节点应包括：
   - 起始攻击点工具
   - 承接污染输出的后继工具
   - 扩大影响的中间工具
   - 产生安全影响或被控制住的节点
7. 每条边说明威胁如何传播，例如污染数据进入下一步、非可信内容被当作指令、错误权限上下文被复用、敏感信息被带入输出等。
8. Mermaid 图必须瞄准攻击过程，第一层节点必须是被攻击的工具/节点。

输出目标：

- 只输出 JSON，不要输出解释，不要加代码块标记。
- 除 JSON 字段名外，人类可读内容使用简体中文。
- 保持防御性、高层描述，不要提供可直接执行的攻击 payload。
- 顶层 JSON 必须是传播结果容器，只能使用 `skill_id`、`chains`、`mermaid`、`notes` 等容器字段。
- 严禁把单条链的 `id`、`title`、`starting_attack_point`、`path`、`edges`、`impact`、`risk_level`、`confidence` 直接放在顶层。
- 即使只有一条传播链，也必须输出 `"chains": [{...}]`，不能输出单独的 chain object。
- 严禁复制 JSON schema 示例中的占位文本，例如 `被攻击工具对应的图节点 id`、`节点 id`、`节点或工具名`、`entry|propagation|impact|containment`；所有 node_id 必须来自已解析图或原始 graph_wo_check.md。
- `path[0]` 必须与 `starting_attack_point` 指向同一个被攻击工具/节点。
- `mermaid` 必须使用 `graph TD`，且每条链的首个节点必须表示被攻击工具/节点。
- 门控逻辑必须体现在 `path.reason` 或 `edges.propagation_mechanism` 中，例如说明哪些 gate 成立、哪些 gate 阻断传播。
- 输入中的 formal chain template 只是解释单条链内部字段，不是最终顶层 JSON schema；最终顶层 schema 只能按下面的容器结构输出。

JSON schema 必须严格使用以下结构：

{
  "skill_id": "{skill_id}",
  "chains": [
    {
      "id": "C1",
      "title": "传播链标题",
      "source_threat_ids": ["T1"],
      "starting_attack_point": {
        "node_id": "被攻击工具对应的图节点 id",
        "node_name": "被攻击工具或能力名",
        "reason": "为什么这是攻击入口"
      },
      "path": [
        {
          "node_id": "节点 id",
          "node_name": "节点或工具名",
          "role": "entry|propagation|impact|containment",
          "threat_state": "该节点携带或改变的威胁状态",
          "reason": "威胁为什么移动到这里或在这里停止"
        }
      ],
      "edges": [
        {
          "src_node_id": "源节点",
          "dst_node_id": "目标节点",
          "condition": "相关图条件，没有则为空字符串",
          "propagation_mechanism": "污染内容/指令/权限上下文如何传播"
        }
      ],
      "impact": {
        "summary": "最终安全影响",
        "stride": ["Tampering"],
        "owasp_agent_threats": ["Prompt Injection"]
      },
      "risk_level": "low|medium|high|critical",
      "confidence": "low|medium|high"
    }
  ],
  "mermaid": "graph TD
...",
  "notes": ["补充说明"]
}

你将收到这些输入：

- skill 标识：`{skill_id}`
- 理论框架：
```json
{framework}
```

- 解析策略：
```json
{strategy}
```

- 已解析的图：
{parsed_graph}

- 原始 graph_wo_check.md：
```mermaid
{graph_markdown}
```

- 第一阶段威胁分析 JSON：
{threat_analysis}
