# eval/ —— 评估代码（来源注明）

## 来源

- **仓库**：https://github.com/hhnqqq/Biology-Instructions
  - commit `600acaa`（"Add evaluation"，2025-04-14）
- **论文**：*Biology-Instructions: A Dataset and Benchmark for Multi-Omics Sequence Understanding Capability of Large Language Models*（arXiv:2412.19191，EMNLP 2025 Findings）
- **作者**：hhnqqq 等（上海人工智能实验室等）

## 文件清单与用途

| 文件 | 用途 | 备注 |
|---|---|---|
| `evaluate.py` | 对模型输出 jsonl 计算各任务指标（PCC/MCC/Acc/R²/Spearman/AUC/Fmax/混合分数） | 未做任何修改；输入格式：`input/label/task/model_output`（`result` 字段会自动改名为 `model_output`） |
| `register_tasks.json` | 24 个任务的类型 / 提示 hint / omics / 指标注册表 | 原样拷贝 |
| `ec_labels.json` | FunctionEC 任务的 EC 编号标签 | 原样拷贝 |

## 本项目使用方式（评估协议对齐）

本项目（三段式后训练流水线）**沿用论文官方评估协议**以保证指标口径一致：
1. 对 `data_prep/output/eval_set.jsonl` 中的每个样本，由待评估模型生成回答，产出 jsonl（`input/label/task/model_output`）；
2. 运行 `python evaluate.py --model_name <name> --OMICS all_omics --input_file_path <outputs.jsonl>`；
3. 结果写入 `metrics_result/metrics_result_<name>_all_omics.json`（按 omics 分组，指标已 ×100）。

注意：`evaluate.py` 内部会加载 `../model/twitter-roberta-base-sentiment-latest`（二分类 fallback 用情感模型），本项目通过让模型直接输出 yes/no 规避该依赖，或按需修改 `evaluate.py` 中 `MODEL` 常量路径。

## 许可说明

- 上游仓库**未附带 LICENSE 文件**（截至拷贝时，git log 最近提交 2025-04-14）。
- 上述文件仅用于**本项目的学习、复现与求职用途**，非商业用途；如需商业使用或分发，请联系原作者确认授权。
- 本项目对这三个文件**未做修改**，保留其原始内容以维持评估协议一致性。
