# LLaMA-Factory DPO 对接说明

## 已完成的对接

仓库现在已经直接导出 LLaMA-Factory 可读取的数据集：

- SFT 训练集：`artifacts/graph_model_optimization/llamafactory_data/skill_graph_sft_train.jsonl`
- SFT 验证集：`artifacts/graph_model_optimization/llamafactory_data/skill_graph_sft_val.jsonl`
- DPO 偏好集：`artifacts/graph_model_optimization/llamafactory_data/skill_graph_dpo.jsonl`
- 数据集注册：`artifacts/graph_model_optimization/llamafactory_data/dataset_info.json`

其中 DPO 数据集已声明 `ranking: true`，可直接给 LLaMA-Factory 的 `stage: dpo` 使用。

## 环境建议

当前机器实测环境：

- Python `3.12.3`
- GPU `RTX A6000 48GB`
- CUDA Driver `12.4`

建议单卡 4bit QLoRA：

- SFT: `Qwen/Qwen2.5-7B-Instruct`
- DPO: 同底模继续 LoRA 偏好训练
- 精度：`bf16`
- 量化：`4bit`

## 安装

```bash
conda create -n lf_dpo python=3.11 -y
conda activate lf_dpo
pip install -U pip setuptools wheel
pip install -r /root/JZY/agent-threat-mining/llamafactory/requirements.txt
```

如果要直接使用官方仓库：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

## 准备数据

```bash
bash /root/JZY/agent-threat-mining/llamafactory/prepare_lf_data.sh
```

如果已经存在 `candidate_scores.jsonl`，脚本会一并导出 DPO 偏好集。

只导出 DPO 的命令：

```bash
python3 /root/JZY/agent-threat-mining/scripts/optimize_skill_graph_model.py   --mode export-preference   --scores-file /root/JZY/agent-threat-mining/artifacts/graph_model_optimization/candidate_scores.jsonl   --output-dir /root/JZY/agent-threat-mining/artifacts/graph_model_optimization   --min-preference-gap 0.5
```

## 训练

SFT：

```bash
bash /root/JZY/agent-threat-mining/llamafactory/run_sft.sh
```

DPO：

```bash
bash /root/JZY/agent-threat-mining/llamafactory/run_dpo.sh
```

配置文件：

- `llamafactory/train_qwen25_lora_sft.yaml`
- `llamafactory/train_qwen25_lora_dpo.yaml`

## 训练顺序

1. 先跑 `build-dataset` 导出 SFT 数据。
2. 先做 SFT 微调。
3. 用 SFT 后模型重新采样候选。
4. 跑 `score` 和 `export-preference`。
5. 再做 DPO。

## 注意事项

- 新增了 `--min-preference-gap`，避免 reward 太接近时生成低质量偏好对。
- 如果 `chosen` 和 `rejected` 相同，会自动跳过。
- 当前 `ollama` 进程占了约 `6.7GB` 显存，训练前建议停掉不必要的推理进程。
- 如果后续希望 DPO 接着 SFT 的 LoRA 权重训练，应把 DPO YAML 里的 `model_name_or_path` / `adapter_name_or_path` 改成你的 SFT 输出。
