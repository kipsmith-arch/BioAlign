# 脚本输出清单（文件 + 内容含义）

> 本文件按"脚本 → 输出文件 → 内容含义"整理仓库中每个脚本运行后会产出的东西，
> 包括数据管线、训练、评估、推理四个模块。供排查产物、写报告、面试回答"每条数据从哪来、长什么样"用。

---

## 0. 速查总览

| 脚本 | 运行方式 | 输出文件 |
|---|---|---|
| `data_prep/scripts/01_split_stage2.py` | `python` | `output/train_pool.jsonl`、`output/dpo_source.jsonl`、`output/eval_set.jsonl`、`output/prep_stats.json` |
| `data_prep/scripts/02_stage3_convert.py` | `python` | `output/stage3.jsonl` |
| `data_prep/scripts/03_seq_prepare.py` | `python` | `output/stage1_pretrain.jsonl`、`output/seq_stats.json` |
| `data_prep/scripts/04_smoke.py` | `python` | `output/smoke.jsonl` |
| `data_prep/scripts/05_dedup_template.py` | `python --template_mult 1.0` | `output/train_pool_clean.jsonl`、`output/clean_stats.json` |
| `data_prep/scripts/06_token_len_stats.py` | `python`（需 env `TOK_PATH`） | 仅 stdout 分位数打印（不落盘） |
| `train/stage1_pretrain.py` | `python` / `torchrun` | `--output_dir/`：adapter 权重 + tokenizer + `checkpoint-N/`（含 `trainer_state.json` 等） |
| `train/stage2_sft.py` | `python` / `torchrun` | 同上（SFT 版） |
| `train/build_preference.py` | **仅 `python`（单卡）** | `--output_dir/dpo_pairs.jsonl`（原子发布） |
| `train/stage3_dpo.py` | `python` / `torchrun` | `--output_dir/`：DPO 后 adapter + tokenizer + `checkpoint-N/` |
| `train/infer_eval.py` | `python` | `--output_dir/<out_file>`（默认 `eval_outputs.jsonl`）；`--run_eval` 时追加触发 `eval/evaluate.py` 产物 |
| `eval/evaluate.py` | `python` | `metrics_result/metrics_result_<name>_<OMICS>.json`、`processed_data/<name>/<task>_processed_data.json`、`logging/metrics_<name>_<OMICS>_<ts>.log` |
| `infer/baseline_runner.py` | `python` | `--out_file`（推理 jsonl）+ `--metrics_file`（指标 json） |
| `infer/fast_infer.py` | `python`（Linux，需 vLLM） | `--out_file`（推理 jsonl）+ `--metrics_file`（指标 json） |
| `infer/bench.py` | `python --all [--smoke]` | `bench/raw/<tag>.jsonl`、`bench/raw/<tag>_metrics.json`、`bench/bench_summary.csv` |

---

## 1. 数据准备（`data_prep/scripts/`）

所有输出默认写在 `data_prep/output/`（脚本内相对路径）。`05` 可用 `--data_dir` 覆盖。

### 1.1 `01_split_stage2.py` —— stage2 三路防泄漏划分

输入：`../dataset/stage2_train.jsonl`（330 万条）。按 task 分层、固定种子（SEED=42）确定性抽样，三路严格不相交。

| 输出文件 | 内容含义 |
|---|---|
| `output/train_pool.jsonl` | **SFT 原始采样池**（默认 30 万条）。每行一个原始训练样本：`{"input": 问题(含序列), "output": 标准答案, "label": 标签, "task": 任务名}`。→ 供 `[05]` 净化，最终供 stage2_sft |
| `output/dpo_source.jsonl` | **DPO 偏好对构造源**（默认 11 万条，每 task ≤5000）。格式同 train_pool，但 **SFT 不可见**。→ 供 `build_preference.py` |
| `output/eval_set.jsonl` | **评估集**（默认 1.89 万条，每 task 300~500）。格式同 train_pool，**SFT/DPO 均不可见**，保证评估指标干净。→ 供 `infer_eval.py` 推理 |
| `output/prep_stats.json` | 划分统计：`seed`、`source`、`overlap_exact`（三路精确重叠，应≈0）、`source_total`、`source_tasks`、三路各自的 `total`/`num_tasks`/`per_task` 分布。**校验"无泄漏"的证据** |

### 1.2 `02_stage3_convert.py` —— stage3 xlsx → jsonl

输入：`../dataset/stage3.xlsx`（8002 条 GPT-4o-mini 精修推理问答对）。

| 输出文件 | 内容含义 |
|---|---|
| `output/stage3.jsonl` | 每行 `{"input", "task", "label", "output"}`（注意是 `output` 字段，与 train_pool 的格式一致）。→ 由 `[05]` 并入 SFT 训练数据 |

### 1.3 `03_seq_prepare.py` —— Stage 1 预训练序列数据

输入：三源 FASTA（`../seq/` 下 GRCh38 人类基因组 / RNAcentral ncRNA / UniProt Swiss-Prot）。

| 输出文件 | 内容含义 |
|---|---|
| `output/stage1_pretrain.jsonl` | **Stage 1 继续预训练语料**（默认 23.6 万条）。每行 `{"text": "<dna>序列</dna>"}` 或 `<rna>`/`<protein>` 包裹（type token 帮助区分 omics 语义；`PREFIX=False` 时无包裹）。→ 供 `stage1_pretrain.py` |
| `output/seq_stats.json` | 抽样统计：`seed`、`per_source`（每源条数）、`source_scanned`（源总扫描量）、`n_skipped_fragments`（N 占比过滤掉的片段数）、`alphabet_observed`（每源字母表）、`length_stats`（每源序列长度 min/max）。**用于复查抽样质量** |

### 1.4 `04_smoke.py` —— 冒烟测试集

输入：`output/train_pool.jsonl`。

| 输出文件 | 内容含义 |
|---|---|
| `output/smoke.jsonl` | 每 task 最多抽 100 条（seed=42），格式同 train_pool。用于 10~20 分钟内验证训练/评估全流程代码正确性（与正式训练同一套代码、只改数据量） |

### 1.5 `05_dedup_template.py` —— 训练集净化（去重 + 模板均衡 + 合并 stage3）

输入：`output/train_pool.jsonl` + `output/stage3.jsonl`（可选，不存在则跳过）。

| 输出文件 | 内容含义 |
|---|---|
| `output/train_pool_clean.jsonl` | **Stage 2 SFT 正式训练数据**（默认 289,768 条）。完全去重（task+input+output 相同删除）→ 模板骨架感知均衡采样（每模板 cap=`max(50, ceil(avg×mult))`）→ 追加 stage3。→ 供 `stage2_sft.py --train_file train_pool_clean.jsonl` |
| `output/clean_stats.json` | 净化统计：`raw`、`dedup_removed`、`stage3_added`、`final`、`template_mult`、`top5_ratio_before/after`（模板同质化前后对比）、`per_task` 数组（每 task 的 `raw`/`templates`/`cap`/`kept`）。**"模板同质化被缓解"的证据** |

### 1.6 `06_token_len_stats.py` —— token 长度分布统计

输入：`output/train_pool_clean.jsonl`（默认，可改 `DATA_PATH`），tokenizer 路径取 env `TOK_PATH`。

**不产生文件**：只在 stdout 打印分位数表（SFT 输入/输出/总长、DPO chosen/rejected 总长）并给出 `max_len` 覆盖率建议（如 `max_len=1024: 保留 98.2% 样本`）。用于实测决定训练 `--max_len`。

---

## 2. 训练（`train/`）

三个 stage 脚本 + `infer_eval.py` 的输出目录由 `--output_dir` 指定（如 `ckpt/stage1`）。写出的结构完全相同：

```
<output_dir>/
├── adapter_model.safetensors      # LoRA adapter 权重（不含 base，可独立加载）
├── adapter_config.json            # LoRA 配置（r/alpha/dropout/target_modules/base 路径等，PEFT 标准）
├── README.md                      # 训练参数摘要
├── tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
├── added_tokens.json / special_tokens_map.json / chat_template.jinja
└── checkpoint-N/                  # Trainer 按 save_steps 自动生成的中间断点
    ├── adapter_model.safetensors / adapter_config.json   # 该步的 adapter
    ├── trainer_state.json         # 训练状态：global_step、log_history(每步 loss/learning_rate/grad_norm)、epoch、best_* 等
    ├── optimizer.pt / scheduler.pt / rng_state.pth       # 优化器/调度器/随机状态（可断点续训）
    ├── training_args.bin          # TrainingArguments 序列化
    └── tokenizer 全套文件
```

- 中间 checkpoint 是**可续训的完整状态**；最终 `output_dir` 根下的 adapter 是训练结束后的最终权重。
- 训练日志走 stdout（`ProgressCallback` 打印 `[进度] step ... | loss=... | 显存 alloc/reserved/peak/free`），**不写日志文件**。

### 2.1 `stage1_pretrain.py` —— Stage 1 领域继续预训练

输入：`--data_dir/stage1_pretrain.jsonl`。输出：见上目录结构（bf16/4bit LoRA+ adapter）。无其他文件。

### 2.2 `stage2_sft.py` —— Stage 2 QLoRA SFT

输入：`--data_dir/--train_file`（默认 `train_pool_clean.jsonl`）。输出：见上目录结构。`--resume_adapter` 传 Stage1 adapter 时可做 stage1+stage2 连续训练分支。

### 2.3 `build_preference.py` —— 自构 DPO 偏好对

输入：base 模型 + `--stage2_dir`（Stage2 adapter，用于生成 rejected）+ `--data_dir/--in_file`（默认 `dpo_source.jsonl`）。

| 输出文件 | 内容含义 |
|---|---|
| `--output_dir/dpo_pairs.jsonl`（默认 output_dir=data_dir） | **DPO 训练数据**（默认 ≤25000 对）。每行 trl DPOTrainer 格式：`{"prompt": [{"role":"user","content": input}], "chosen": [{"role":"assistant","content": 标准答案}], "rejected": [{"role":"assistant","content": stage2 模型高温采样输出}]}`。已过滤 rejected 为空/与 chosen 相同的无区分度样本。→ 供 `stage3_dpo.py` |
| `*.rank{rank}`（临时） | 多卡 sharding 时各 rank 的中间分片，merge 后删除；原子发布（`.tmp` → `os.replace`）保证不产出半合并文件 |

注意：多卡 DDP 下会 OOM，代码 hard assert 强制单卡运行。

### 2.4 `stage3_dpo.py` —— Stage 3 DPO

输入：`--data_dir/--dpo_data`（默认 `dpo_pairs.jsonl`）+ `--stage2_dir`（π_ref 与 π 共同初始化）。输出：见上目录结构（DPO 后 adapter）。

### 2.5 `infer_eval.py` —— 推理 + 可选评估

输入：`--data_dir/--in_file`（默认 `eval_set.jsonl`）+ `--ckpt_dir`（adapter；不传则评估基座零样本基线）。

| 输出文件 | 内容含义 |
|---|---|
| `--output_dir/<out_file>`（默认 `eval_outputs.jsonl`） | **评估输出**，`eval/evaluate.py` 兼容格式：每行 `{"input", "label", "task", "model_output"}`（model_output 为贪心解码生成的回答）。可被 `eval/evaluate.py` 直接消费 |
| （`--run_eval` 时）| 自动调用 `eval/evaluate.py --model_name <out_file 去 .jsonl> --OMICS all_omics`，产出 §3 的评估文件 |

---

## 3. 评估（`eval/evaluate.py`）

输入：推理输出 jsonl（`input/label/task/model_output`，`result` 字段会自动改名）+ `--model_name` + `--OMICS`。输出三个产物：

| 输出文件 | 内容含义 |
|---|---|
| `metrics_result/metrics_result_<model_name>_<OMICS>.json` | **最终指标**，按 omics 分组的字典：`{omics: {task: {指标名: 数值}}}`。数值已 `round(×100, 2)`（如 0.876 → 87.6）。指标类型由 `register_tasks.json` 决定：Spearman / R² / mixed_score / MCC / Acc / AUC / Fmax / PCC（含 hk/dev 或 ON/OFF/ON_OFF 多值）。**写报告用这个** |
| `processed_data/<model_name>/<task>_processed_data.json` | **逐任务处理明细**：每样本的 `input`、原始 `label`、`original_model_output`、以及正则提取后的 `processed_model_output`（回归取第一个数字 / 二分类 0/1 / RNA family / 修饰列表 / EC 列表等）。**排查"指标为什么低"（解析失败/格式问题）看这个** |
| `logging/metrics_<model_name>_<OMICS>_<时间戳>.log` | 评估全过程日志：警告（数值提取失败、inf 计入 0 分、F1 NaN 等）+ 每任务指标打印。排查被丢弃样本的入口 |

---

## 4. 推理加速（`infer/`）

三个推理脚本输出格式**字节级一致**（键：`input/label/task/model_output`），可直接喂给 `eval/evaluate.py`。

### 4.1 `baseline_runner.py` —— HF generate 对照基线

| 输出文件 | 内容含义 |
|---|---|
| `--out_file` | 推理结果 jsonl（同上格式，贪心解码） |
| `--metrics_file` | 性能指标 JSON：`tag`、`engine="hf_generate"`、`quantization`、`n_samples`、`batch_size`、`elapsed_sec`、`throughput_samples_per_sec`、`sample_latency_ms_avg/p50/p95/p99`、`peak_gpu_mem_gib`、`warmup_batches`、时间戳等。**benchmark 报告数据源** |

### 4.2 `fast_infer.py` —— vLLM 4-bit + LoRA 热加载

| 输出文件 | 内容含义 |
|---|---|
| `--out_file` | 推理结果 jsonl（与 baseline 字节级一致） |
| `--metrics_file` | 性能指标 JSON：`engine="vllm"`、`quantization="bitsandbytes_nf4_4bit"`、continuous batching 的 `throughput_samples_per_sec`、`sample_latency_ms_avg`（p50≈avg；vLLM 0.6 不公开 p95/p99，为 `null`）、`peak_gpu_mem_gib`、`max_lora_rank`、`gpu_memory_utilization` 等，附 `notes` 说明口径 |

### 4.3 `bench.py` —— 一键 5 组对照实验

| 输出文件 | 内容含义 |
|---|---|
| `bench/raw/<tag>.jsonl` | 每组实验的推理输出（`bf16`/`4bit`/`4bit_b8`/`vllm_4bit`/`vllm_4bit_b16`），可直接评估 |
| `bench/raw/<tag>_metrics.json` | 对应组的性能指标（见 4.1/4.2，失败组为 `{"tag", "error", "elapsed_sec"}`） |
| `bench/bench_summary.csv` | **5 组横向对比宽表**（每次运行追加行）：核心列 tag/engine/quantization/throughput/latency p50~p99/peak 显存/elapsed，额外字段自动并入，pandas 可直接读 |

---

## 5. 完整数据流（脚本间依赖）

```
dataset/stage2_train.jsonl
  └─[01] → train_pool.jsonl ──[05]→ train_pool_clean.jsonl ──→ stage2_sft.py ──→ ckpt/stage2 (adapter)
         → dpo_source.jsonl ── build_preference.py ──→ dpo_pairs.jsonl ──→ stage3_dpo.py ──→ ckpt/stage3
         → eval_set.jsonl ──→ infer_eval.py / baseline_runner / fast_infer ──→ eval_outputs.jsonl
dataset/stage3.xlsx ──[02]→ stage3.jsonl ──┘(并入[05])
seq/*.fasta ──[03]→ stage1_pretrain.jsonl ──→ stage1_pretrain.py ──→ ckpt/stage1 (adapter)
train_pool ──[04]→ smoke.jsonl（冒烟全流程）
[06] token 长度统计 ── 决定 max_len（仅打印）
eval_outputs.jsonl ──→ eval/evaluate.py ──→ metrics_result/*.json + processed_data/* + logging/*
```

## 6. 注意事项

- 推理/评估类输出（jsonl）默认**不入库**（`.gitignore` 忽略 `*.jsonl`/`*.json`/`ckpt`/`data_prep/output`），需可复现时重新生成。
- `evaluate.py` 的指标是 **×100 后的数值**，与论文口径一致；对比时注意单位。
- `trainer_state.json` 是判断训练是否正常收敛的第一手文件（`log_history` 里 loss 曲线、`best_model_checkpoint`）。
- `prep_stats.json` 的 `overlap_exact` 与 `clean_stats.json` 的 `top5_ratio_before/after` 是数据管线的**质量审计证据**（防泄漏 / 抗模板同质化），写报告或面试时直接引用。
