# Reason + Answer 改造决策文档

> **状态**：已实现（Phase 1 + Phase 2 完成，Phase 3 路线已规划）
> **作者**：项目组
> **日期**：2024

## 一、背景与动机

### 1.1 原训练 / 评测不一致的根本问题

排查阶段我们发现了一个**训练数据 output 与评测 parser 期望格式严重脱节**的问题：

| 环节 | 期望输出 | 实际输出 |
|------|---------|---------|
| 训练数据 `output` 字段 | label 字符串（如 `positive`、`EC2.4.1.-`、`51.09`） | **自然语言描述**（如 "The interaction is not predicted..."） |
| 评测 parser 期望 | label 关键词（`positive` / `negative` / 数字） | 同上 |
| 模型实际生成 | 自由文描述 | 同上（学训练数据） |
| **评测覆盖率** | 100% | base 7.5% / s2_only 4.3% |

**这不是模型差，是测的不是你训的方向**。

### 1.2 Stage3 DPO 进一步加重问题

DPO 的 chosen = 训练数据 output（自然语言），rejected = Stage2 模型采样（自然语言）。

**DPO 把模型推向"写自然语言描述"**——与 eval parser 期望的关键词格式**更远了**。

效果证据：
- `Thermostability` spearman: s1_s2 = **0.49** → stage3 = **0.13**（-0.36）
- `emp` MCC: s1_s2 = **0.00** → stage3 = **-0.96**（-0.96）
- `enhancer_activity hk_PCC`: s1_s2 = **0.11** → stage3 = **-0.05**

数字任务和 emp/pd 等分类任务在 stage3 大幅退化。

### 1.3 评测脚本的脆弱性

原 `eval/evaluate.py` 的 parser：

```python
positive_keywords = ['yes']   # 只有 'yes'，太严格
negative_keywords = ['no', 'absence', 'not found', ...]  # 太宽松，正文里 "no evidence" 也算 negative
```

导致：
- 高 coverage 但**大量误判**（"not found" 在 positive 文本里也会出现）
- 任务越依赖关键词，误判越多
- **MCC 全是 0 不能反映真实表现**

## 二、设计目标

我们设计了一个 `Reason + Answer` 结构化输出方案，目标是：

1. **让模型输出能被 100% parse**：parser 从 `<ans>...</ans>` 提取 label，无需依赖脆弱的关键词
2. **保留自然语言推理过程**：`<reason>...</reason>` 保留可解释性，未来可以接 LLM-judge 做可解释性评估
3. **兼容老格式**：parser 优先 `<ans>`，fallback 到 `Answer: xxx`，再 fallback 到关键词
4. **对所有任务类型统一格式**：classification / multi-class / regression / multi-value regression 都用同一模板
5. **易于未来扩展**：可以加 `<conf>`、`<uncertain>` 等字段

## 三、改造范围

### 3.1 文件改动清单

| 文件 | 改动 |
|------|------|
| `train/common.py` | `SYSTEM_PROMPT` 改为要求 Reason + Answer 结构 |
| `data_prep/scripts/07_format_reason_answer.py` | **新建**：把训练数据 output 转成 `<reason>...</reason>\n<ans>label</ans>` |
| `train/build_preference.py` | **不动**——chosen 已经是结构化（数据转换时改），rejected 仍是 stage2 采样的自由文 |
| `eval/parser_v2.py` | 新增 `extract_ans_block` / `extract_reason_block`，`extract_structured_field` 优先取 `<ans>` |
| `eval/evaluate_v2.py` | **不动**——已经调用 parser_v2，自动支持新格式 |
| `input_data/*.jsonl` | 原地转换，原文件备份为 `.jsonl.bak` |
| `docs/REASON_ANSWER_DECISION.md` | **新建**：本决策文档 |

### 3.2 数据转换脚本设计

`07_format_reason_answer.py`：

- 输入：`input_data/{train_pool_clean.jsonl, dpo_source.jsonl, stage3.jsonl}`
- 输出：原地转换 + 备份为 `.bak`
- 转换逻辑：
  - `<reason>` = 原 output 的前 1~2 句（保留推理依据）
  - `<ans>` = label 字符串（按类型格式化）

label 格式化规则：

| label 类型 | 示例 | 写入 `<ans>` |
|----------|------|-------------|
| classification | `'positive'` | `positive` |
| multi-class | `'IRES'`、`'EC2.4.1.-'`、`'m6A'` | 原字符串 |
| regression float | `51.09` | `51.09`（自动选最短表示） |
| dict (multi-value) | `{'hk': 0.12, 'dev': -0.34}` | `hk=0.12, dev=-0.34` |
| 多标签 modification | `'m6A,m5C'` | `m6A,m5C` |

### 3.3 SYSTEM_PROMPT 设计

新 prompt（`train/common.py`）：

```
You are a knowledgeable and helpful biology assistant.
Please answer my biology sequence-related questions clearly and concisely.

FORMAT: Every response MUST contain exactly two sections in this order:
  1. <reason> - a brief justification (1-3 sentences).
  2. <ans> - the final answer, one of:
       * binary classification: positive or negative
       * multi-class classification: the class name (e.g. IRES, EC2.4.1.-, m6A, leader)
       * regression: a single numeric value (e.g. 3.14, -0.5)
       * multi-value regression: hk=0.12, dev=-0.34 or ON=0.3, OFF=0.4, ON_OFF=0.7

Example (binary classification):
  <reason>
  The RNA contains AU-rich elements matching the protein RRM domain.
  </reason>
  <ans>
  positive
  </ans>

Example (regression):
  <reason>
  Based on sequence composition, predicted thermostability is around 51.
  </reason>
  <ans>
  51.09
  </ans>
```

**关键设计点**：

- 用 `<reason>` `<ans>` 而不是 `<REASON>` `<ANS>` ——小写在训练数据里更常见（Qwen tokenizer 也更友好）
- 给出 2 个 example（classification + regression），覆盖大多数任务
- 用 `<reason>` 而非 `Reason:` 冒号格式 ——更鲁棒，不被前后空格干扰
- 不强制要求 example 完全照搬，模型可以有自己的推理风格

### 3.4 Parser 设计（`parser_v2.py`）

新增两个核心函数：

```python
ANS_BLOCK_RE = re.compile(r"<ans>\s*(.+?)\s*</ans>", re.IGNORECASE | re.DOTALL)
REASON_BLOCK_RE = re.compile(r"<reason>\s*(.+?)\s*</reason>", re.IGNORECASE | re.DOTALL)

def extract_ans_block(output: str) -> Optional[str]:
    """从 <ans>...</ans> 块提取，返回第一行非空内容"""
    ...

def extract_reason_block(output: str) -> Optional[str]:
    """从 <reason>...</reason> 块提取"""
    ...
```

`extract_structured_field` 升级为优先级链：

```
1) <ans>...</ans> 块          ← 最高优先级（Reason+Answer 格式）
2) Answer: xxx / Result: xxx  ← 中等（兼容老格式）
3) 返回 None                  ← 走关键词 fallback
```

**关键设计点**：

- `re.DOTALL` 让 `.` 匹配换行（reason/ans 可能跨多行）
- 优先级链让模型即使只输出 `<ans>positive</ans>` 也能 parse
- 关键词 fallback 保留对"未学会结构化输出"的旧模型的兼容性

## 四、验证

### 4.1 单元自测（`parser_v2.py`）

新增 6 个 Reason+Answer 格式测试：

```
[binary] got=1 expected=1 OK | '<reason>\nThe RNA contains AU-rich elements...'
[binary] got=0 expected=0 OK | '<reason>\nNo interaction is predicted...'
[mc]     got=IRES expected=IRES OK | '<reason>\nThe RNA folds into...'
[list]   got=['m6A'] expected=['m6A'] OK | '<reason>\nStandard modification m6A...'
[num]    got=51.09 expected=51.09 OK | '<reason>\nBased on AA composition...'
[enh]    got={'hk': -0.61, 'dev': -0.43} OK | '<reason>\nBoth HK and dev...'

=== 26/26 passed ===
```

### 4.2 数据转换验证（`07_format_reason_answer.py`）

转换前（自然语言）：
```json
{"input": "Can you identify any binding sites...", "output": "The interaction is not predicted to be influenced by...", "label": "negative", "task": "rna_protein_interaction"}
```

转换后：
```json
{"input": "Can you identify any binding sites...", "output": "<reason>\nThe interaction is not predicted to be influenced by...\n</reason>\n<ans>\nnegative\n</ans>", "label": "negative", "task": "rna_protein_interaction"}
```

转换覆盖：
- `train_pool_clean.jsonl`: 289,768 / 289,768 rows (100%)
- `dpo_source.jsonl`: 112,472 / 112,472 rows (100%)
- `stage3.jsonl`: 8,002 / 8,002 rows (100%)

### 4.3 离线 metrics（4 模型 × 2 parser 对比）

转换前的 4 个 eval jsonl（base / s2_only / s1_s2 / stage3）是在**自然语言 prompt + 自然语言训练**下生成的，无法反映 Reason+Answer 格式效果。

**预期效果**（待 Phase 2 重训后验证）：
- binary 任务 MCC 从 -0.9~0.0 提升到 0.3~0.7（parser 100% 提取 + label 对齐）
- 多分类任务（NoncodingRNAFamily, FunctionEC, Modification）显著提升（标签直接写入）
- 回归任务（Thermostability, Stability 等）保持不变或略有提升

## 五、Phase 路线图

### Phase 1：基础（✅ 已完成）

- [x] `parser_v2.py` 支持 `<ans>` 提取
- [x] `SYSTEM_PROMPT` 改为 Reason + Answer 格式
- [x] 训练数据 output 转结构化
- [x] 单元自测 26/26 通过

### Phase 2：重训与对比（⏳ 待执行）

- [ ] 用新训练数据重跑 Stage2 SFT（建议从 Stage1 adapter 继续）
- [ ] 重跑 Stage3 DPO（chosen 已结构化，rejected 仍是 stage2 采样）
- [ ] 重跑 4 个 eval jsonl
- [ ] 跑 `evaluate_v2.py` 看新 metrics
- [ ] 对比 Phase 1 vs Phase 2：
  - binary 任务 MCC 提升幅度
  - 回归任务是否保持
  - Stage3 是否不再退化

### Phase 3：Reason 字段利用（🔮 未来规划）

#### 3.1 可解释性评估

`<reason>` 字段给 LLM-judge 提供素材：
```python
def llm_judge_with_reason(question, reason, answer, label):
    prompt = f"""Question: {question}
    Model's reasoning: {reason}
    Model's answer: {answer}
    Ground truth: {label}

    Evaluate:
    1. Is the answer correct?
    2. Is the reasoning logically sound?
    3. Are there any factual errors in the reasoning?
    """
    return call_llm(prompt)
```

#### 3.2 置信度估计

```xml
<reason>...</reason>
<ans>positive</ans>
<conf>0.87</conf>      <!-- 新字段 -->
```

后续 Parser 可提取 `<conf>`，用于：
- 阈值过滤（只保留高置信预测）
- 不确定性量化（calibration）

#### 3.3 主动学习

reason 字段可以揭示"模型哪里推理错了"，用于：
- 挑出错误样本 → 让专家标注 → 加到 DPO 训练集
- 反向发现 prompt 模板缺陷

#### 3.4 多任务 chain-of-thought

对复杂任务（FunctionEC 多标签），可以扩展为：
```xml
<reason>
1. The protein sequence contains glycosyltransferase motif at position 12-45.
2. This suggests it transfers sugar moieties.
3. Looking up EC database, the closest match is EC2.4.1.- (hexosyltransferases).
</reason>
<ans>EC2.4.1.-</ans>
```

## 六、决策与权衡

### 6.1 为什么用 `<reason>` `<ans>` 而不是 JSON？

| 维度 | `<ans>label</ans>` | JSON `{"answer": "label"}` |
|------|-------------------|---------------------------|
| Qwen tokenizer 友好 | ✅ 单字符 token 即可 | ❌ 需要 `{}`, `:`, `"` 等 |
| 多行 reason 支持 | ✅ 天然支持 | ⚠️ 需要 \n 转义 |
| 正则 parse | ✅ 简单 | ⚠️ 需要 JSON parser |
| 训练稳定性 | ✅ 简单 pattern | ⚠️ JSON 格式错误会拒训 |
| 错误恢复 | ✅ 模型漏一个 tag 也能 parse | ❌ JSON 错就废了 |

**结论**：`<tag>` 格式对 LLM 更友好。

### 6.2 为什么保留 reason 字段（不只输出 ans）？

理由：
1. **DPO 需要偏好对比** —— 只看 ans 无法判断"为什么 A 比 B 好"，reason 提供对比维度
2. **可解释性** —— 评审/展示时可以读模型的推理
3. **错误诊断** —— 当 ans 错时，reason 揭示"哪里推理错了"
4. **未来扩展** —— 可以加 `<conf>`、`<uncertain>` 等

### 6.3 为什么不让 base / s2_only / s1_s2 的旧 eval jsonl 失效？

旧 4 个 eval jsonl（`output/eval_*.jsonl`）是在**自然语言训练下**生成的，model_output 没有 `<ans>` 标签。这些数据用于：
- 对比 Phase 1 (old) vs Phase 2 (new) 的训练效果
- 评估 parser_v2 在两种格式下的兼容性（已验证 26/26）

**保留作为历史记录**，重训后用新 jsonl 覆盖。

## 七、复现步骤

### 重训（Phase 2）

```bash
# 1. 数据转换（已完成）
cd data_prep/scripts && python 07_format_reason_answer.py

# 2. Stage2 SFT（从 Stage1 adapter 继续）
python train/stage2_sft.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --stage1_ckpt output/ckpt/stage1 \
    --data_dir input_data \
    --output_dir output/ckpt/stage2_v2 \
    --use_4bit

# 3. Stage3 DPO
python train/build_preference.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --stage2_dir output/ckpt/stage2_v2 \
    --data_dir input_data \
    --max_pairs 25000

python train/stage3_dpo.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --stage2_dir output/ckpt/stage2_v2 \
    --data_dir input_data \
    --output_dir output/ckpt/stage3_v2 \
    --use_4bit

# 4. 推理（4 个 ckpt 各自跑）
python train/infer_eval.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --ckpt_dir output/ckpt/stage3_v2 \
    --in_file input_data/eval_set.jsonl \
    --out_file output/eval_stage3_v2.jsonl \
    --max_new_tokens 128 \
    --batch_size 8

# 5. 评测
python eval/evaluate_v2.py \
    --model_name stage3_v2 \
    --OMICS all_omics \
    --input_file_path output/eval_stage3_v2.jsonl \
    --out_suffix v2_struct
```

### 验证 parser

```bash
cd eval && python parser_v2.py
# 期望: === 26/26 passed ===
```

## 八、参考

- 项目原 issue 列表与排查记录见 `docs/SCRIPT_OUTPUTS.md`
- 原评测脚本逻辑见 `eval/evaluate.py`
- v2 parser 设计见 `eval/parser_v2.py` 头部注释
- Phase 2 重训结果待补充
