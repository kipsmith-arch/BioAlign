# Parser v2 使用说明

## 背景

原 `eval/evaluate.py` 的输出解析（`classify_by_keywords`、`extract_numeric_values`）过于脆弱：

| 现象 | 根因 |
|------|------|
| 二分类任务大量 MCC=0 | `positive_keywords = ['yes']`，绝大多数自然语言回答不含 "yes" |
| 多分类任务 Fmax ≈ 0 | 正则找不到 EC 号前缀边界（`\bEC` 在 `EC2.4.1.-` 中不匹配） |
| 数字提取误抓 | 没有范围限定，模型自由文里的"长度 200 bp"会被当预测 |
| 多分类 Modification AUC = NaN | `extract_modifications` 没处理 `none` |

## 文件

- `eval/parser_v2.py`：独立模块，提供 `parse_binary_classification`、`parse_regression_number`、`parse_ec_numbers` 等函数。
- `eval/evaluate_v2.py`：兼容 `evaluate.py` 的 CLI，可选 `--use_old_parser` 对比。
- `eval/_compare.py`：自动生成 4 模型 × 2 parser 的对比表。

## 用法

```bash
# 用 v2 parser
python eval/evaluate_v2.py --model_name stage3 --OMICS all_omics --input_file_path output/eval_stage3.jsonl

# 用旧 parser 对比
python eval/evaluate_v2.py --model_name stage3 --OMICS all_omics --input_file_path output/eval_stage3.jsonl --use_old_parser

# 输出文件名带后缀
python eval/evaluate_v2.py --model_name stage3 --OMICS all_omics --input_file_path output/eval_stage3.jsonl --out_suffix v2
```

输出到 `eval/metrics_result/metrics_result_{model}_{omics}{suffix}.json`。

## v2 设计要点

1. **结构化字段优先（v2.1）**：先识别 `<ans>...</ans>` 块（Reason+Answer 格式），再 fallback 到 `Answer: positive` / `Classification: negative` / `Result: 3.14` 等标记，最后 fallback 到关键词。
2. **关键词分级**：强 positive / 强 negative / 不确定 三类信号；不确定信号判为 None，不强行预测。
3. **范围限定**：回归任务可传入 `(min, max)`，剔除模型自由文里的无关数字（如"长度 200"）。
4. **多分类标签识别**：按长度倒序匹配（避免 `miRNA` 误匹配 `scaRNA`）。
5. **EC 号识别**：处理 `EC2.4.1.-` 这种紧贴前缀的格式（不能用 `\bEC\b`）。

## 提取优先级链

`extract_structured_field` 按以下优先级尝试：

```
1. <ans>label</ans>            ← Reason+Answer 格式（Phase 2 推荐）
2. <reason>...</reason>          ← 保留推理过程（Phase 3 用于 LLM-judge）
3. Answer: xxx                  ← 兼容老格式
4. Classification: xxx
5. Result: xxx
6. 返回 None                     ← 走关键词 fallback（v1 兼容）
```

## 当前结果（4 模型对比，自然语言训练时代）

回归任务（spearman/R2/PCC）v2 与 OLD 完全一致；多分类任务（NoncodingRNAFamily）完全一致。

差异主要在 binary classification：
- **OLD**：覆盖率虚高（靠"no evidence"等通用 negative 词匹配），但**大量误判**，所以"看起来 MCC=0 但其实是瞎猜的"。
- **V2**：覆盖率低（只有明确 yes/no/positive/negative 表达的样本被算），但**预测准确率高**。MCC 出现负值是因为很多模型**真的把分类方向学反了**——这是训练数据问题，不是 parser 问题。

## Phase 2：Reason + Answer 改造（已完成基础设施，待重训）

详见 [`docs/REASON_ANSWER_DECISION.md`](../docs/REASON_ANSWER_DECISION.md)。

### 已完成

- ✅ `parser_v2.py` 支持 `<ans>` 提取
- ✅ `SYSTEM_PROMPT` 改为 Reason + Answer 格式（`train/common.py`）
- ✅ 训练数据 output 转结构化（`data_prep/scripts/07_format_reason_answer.py`）
- ✅ 单元自测 26/26 通过（含 6 个 Reason+Answer 格式测试）

### 待执行

- [ ] 重训 Stage2 / Stage3
- [ ] 重跑 4 个 eval jsonl
- [ ] 对比 metrics

## 自测

```bash
cd eval && python parser_v2.py
```

应输出 `=== 26/26 passed ===`。
