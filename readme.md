# BioAlign —— 基于 QLoRA 与 DPO 的生物医学大模型后训练流水线

在资源受限环境（单卡 T4，16GB）下，以 **Qwen2.5-3B** 为基座，完整实现大模型后训练（Post-training）三阶段流水线，使通用模型获得多组学（DNA/RNA/蛋白/多分子）生物序列的理解与任务回答能力，并通过偏好对齐提升回答质量。

## 技术栈

`transformers` + `peft` + `bitsandbytes`（4-bit QLoRA）+ 自实现 DPO（不依赖 trl）

## 流水线（三阶段）

| 阶段 | 做什么 | 方法 |
|---|---|---|
| **① 领域继续预训练** | 让模型"认识"生物序列（GRCh38 人类基因组 / RNAcentral ncRNA / Swiss-Prot 蛋白，23.6 万条） | bf16 LoRA+（B 学习率=A×4）+ packing，next-token 训练 |
| **② PEFT 指令微调** | 让模型"回答"21 类生物学任务（Biology-Instructions 28.98 万条净化后 QA） | 4-bit QLoRA SFT，只对 assistant 部分算 loss |
| **③ DPO 偏好对齐** | 让模型"答得更好"（自构 2-3 万对生物领域偏好数据） | 自实现 DPO loss，π_ref 为 SFT 模型冻结 |

## 成果

- 三阶段**消融验证**各环节有效：继续预训练增益、指令微调提升、DPO 不掉领域任务能力
- 单卡 T4（<12GB）跑通工业级后训练技术栈全流程，验证有限算力下复现前沿对齐技术的可行性
- 数据管线（三路防泄漏划分、模板均衡净化、偏好自构）与训练代码全部自建，可复现

## 文档导航

| 文档 | 内容 |
|---|---|
| [`TECH_NOTES.md`](TECH_NOTES.md) | **技术原理笔记**：显存换算（m/v、可训练参数、激活估算）、踩坑原理（peft adapter 训练、trl 版本）、冒烟/并行/DPO 语义等 |
| [`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md) | 方案 + Kaggle 实施计划（每个 notebook 的完整命令） |
| [`data_prep/README.md`](data_prep/README.md) | 数据管线：三路划分、净化、来源记录、面试问答 |
| [`train/README.md`](train/README.md) | 训练代码：脚本用法、冒烟记录、bug 清单 |
| [`eval/README.md`](eval/README.md) | 评估代码（论文官方协议，来源注明） |

## 数据与代码位置

- 原始数据：`dataset/`（论文数据）、`seq/`（序列三源）—— 不入库
- 处理产物：`data_prep/output/` —— 不入库（可脚本重生成）
- 训练代码：`train/*.py`；评估：`eval/`；数据脚本：`data_prep/scripts/`
