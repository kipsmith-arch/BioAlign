# 三段式后训练流水线方案（A100×4 7B / 每晚一个 stage / 全量数据）

> 目标：在 **A100 40GB ×4** 上用 **Qwen2.5-7B-Instruct** 跑通完整后训练流水线：
> **① 领域继续预训练（4-bit QLoRA）→ ② PEFT（4-bit QLoRA SFT）→ ③ RL（DPO 偏好对齐，自实现 loss）**
>
> **部署节奏**：每晚挂一个 stage（独立 commit / 独立会话，自包含、可恢复、产出可下载）。共 **9 晚**（含 2 晚评估）。
>
> **数据量**：训练阶段全量（Stage 1 23.6万 / Stage 2 28.98万 / DPO 2.5万 pairs），评估全量 eval_set 1.89万 × 4 档。
>
> 技术栈：transformers + peft + bitsandbytes（QLoRA）+ 自实现 DPO loss（不依赖 trl）。
> 数据：`data_prep/output/`；代码：`train/`。

---

## 1. 资源与环境

| 项 | 值 |
|---|---|
| GPU | A100 40GB ×4（PCIe 或 NVLink 均可） |
| 模型 | Qwen2.5-7B-Instruct |
| 框架 | transformers 4.52.x + peft 0.18.x + bitsandbytes 0.4x + accelerate 1.12.x |
| 单 stage 预算 | 单晚（≤12h commit 或前台会话） |
| 部署 | Kaggle Commit（每 stage 一 notebook）或自托管 nohup/screen |

**为什么切到 A100×4 7B**：旧方案 T4×2 3B 的所有时间估算不再适用，下面时间预算按 A100×4 7B 重算。

---

## 2. 数据

```
stage1_pretrain.jsonl (23.6万) → Stage 1 全量
train_pool_clean.jsonl (28.98万) → Stage 2 全量
dpo_source.jsonl (11.2万) → build_preference 抽样 2.5万
smoke.jsonl (4700) → 冒烟测试
eval_set.jsonl (1.89万) → 评估全量（4 档，每档 1.89万）
dpo_pairs.jsonl (build_preference 产出) → Stage 3 DPO
```

三路严格不相交（行号切分）：训练 / DPO 源 / 评估互不重叠。
train_pool_clean 净化：去重 + 模板均衡采样（每模板 cap，Top5 模板从 2.0% 降到 1.3%）。

---

## 3. 九晚节奏（全量数据）

| 晚 | Stage | 数据量 | 估算 | 产出 |
|---|---|---|---|---|
| 0 | 全链路冒烟（前端） | smoke | 1.5h | smoke logs |
| 1 | Stage 1 继续预训练 | 23.6万（全量） | 4-5h | ckpt/stage1 |
| 2 | Stage 2 分支 A (stage2-only) | 28.98万（全量） | 5-6h | ckpt/stage2_only |
| 3 | Stage 2 分支 B (stage1+stage2) | 28.98万（全量） | 5-6h | ckpt/stage2_s1 |
| 4 | build_preference（**用 stage2_s1**） | 2.5万对（dpo_source 抽样）·**单卡跑** | 7-9h | dpo_pairs.jsonl |
| 5 | Stage 3 DPO（**起始 stage2_s1**） | 2.5万对 | 1.5-2h | ckpt/stage3 |
| 6 | 评估 4 档（前 2 档） | 1.89万 × 2 + batch gen | 6-7h | eval_base / eval_s2_only |
| 7 | 评估 4 档（后 2 档） | 1.89万 × 2 + batch gen | 6-7h | eval_s1_s2 / eval_stage3 |
| **8** | **推理加速 benchmark（5 组对照）** | 1.89万 × 5（adapter 用 stage2_s1） | **2-3h** | **bench_summary.csv + 4 维对比表回填** |
| **合计** | | | **36-43h, 9 晚 + 1 晚可选** | |

每晚结束后：**下载当晚产出** → 下晚作为 input Dataset 挂载。

---

## 4. 各晚详细命令

### 路径约定

```bash
MODEL_7B = /path/to/Qwen2.5-7B-Instruct
DATA_DIR  = /path/to/data_prep/output
CODE_DIR  = /path/to/train
WORK_DIR  = /path/to/working        # 或 /kaggle/working
```

### 晚 0：全链路冒烟（前端 1.5h）

**目的**：5 步全链路无报错、显存/速度预算合理。
**原则**：只缩数据量/步数，**超参与正式一致**（冒烟 = 验证正式配置的代码路径）。

```bash
# Step 1: Stage 1 冒烟（30 步，200 条，4 卡 DDP 验证 DDP 路径）
# 【公共环境 setsid】烟测也会被 SIGHUP 杀，加 setsid + disown + < /dev/null。
mkdir -p $WORK_DIR/logs
LOGFILE=$WORK_DIR/logs/smoke_stage1_$(date +%Y%m%d_%H%M).log
setsid torchrun --nproc_per_node=4 $CODE_DIR/stage1_pretrain.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage1_smoke \
  --max_len 1024 --max_steps 30 --max_samples 200 \
  --per_device_batch 4 --grad_accum 4 \
  > $LOGFILE 2>&1 < /dev/null &
disown

# Step 2: Stage 2 冒烟（60 步，100 条，max_len 与正式一致 2048——验证长序列代码路径与显存）
mkdir -p $WORK_DIR/logs
LOGFILE=$WORK_DIR/logs/smoke_stage2_$(date +%Y%m%d_%H%M).log
setsid torchrun --nproc_per_node=4 $CODE_DIR/stage2_sft.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage2_smoke \
  --max_len 2048 --max_steps 60 --max_samples 100 \
  --per_device_batch 4 --grad_accum 4 \
  > $LOGFILE 2>&1 < /dev/null &
disown

# Step 3: build_preference 冒烟（50 pairs。**此脚本为生成式推理、DDP 多卡会在 4 进程同时加载全量
# 7B model+ref → 走 OOM。代码已加 hard assert（WORLD_SIZE>1 直接 RuntimeError），必须单卡跑。**
# 50 对 生成量不大，单卡~5min 出文件。多卡加速场景建议改用 vLLM 而不是 DDP，该方案不在本项目范围内。）
mkdir -p $WORK_DIR/logs
LOGFILE=$WORK_DIR/logs/smoke_buildpref_$(date +%Y%m%d_%H%M).log
setsid python $CODE_DIR/build_preference.py \
  --model_path $MODEL_7B --stage2_dir $WORK_DIR/ckpt/stage2_smoke \
  --data_dir $DATA_DIR --output_dir $WORK_DIR \
  --max_pairs 50 \
  > $LOGFILE 2>&1 < /dev/null &
disown

# Step 4: Stage 3 DPO 冒烟（20 步，100 pairs。烟测 peak≈26.8GiB/卡、4 卡 40G A100 余量充足，
# 仍走 DDP。与 build_preference（强制单卡）区别清楚。
mkdir -p $WORK_DIR/logs
LOGFILE=$WORK_DIR/logs/smoke_stage3_$(date +%Y%m%d_%H%M).log
setsid torchrun --nproc_per_node=4 $CODE_DIR/stage3_dpo.py \
  --model_path $MODEL_7B --stage2_dir $WORK_DIR/ckpt/stage2_smoke \
  --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage3_smoke \
  --max_len 1024 --max_samples 100 --max_steps 20 \
  --per_device_batch 4 --grad_accum 4 \
  > $LOGFILE 2>&1 < /dev/null &
disown

# Step 5: infer_eval 冒烟（20 条，单卡，验证 batch gen 路径）
python $CODE_DIR/infer_eval.py \
  --model_path $MODEL_7B --ckpt_dir $WORK_DIR/ckpt/stage2_smoke \
  --data_dir $DATA_DIR --output_dir $WORK_DIR \
  --out_file eval_smoke.jsonl --max_len 1024 --max_samples 20 --batch_size 8
```

**通过标准**：5 步无报错、loss 正常下降（最后一步 `loss=?` 是日志工件）、推理格式正确。

### 晚 1：Stage 1 继续预训练（commit 4-5h）

**目的**：domain-aware 继续预训练（生物序列）。

```bash
# 【公共环境必须 setsid】长时间训练可能被 SIGHUP 杀（SSH 断 / 父 shell 退出 / watchdog）。
# setsid 开新 session + disown 从 bash jobs 移除 + < /dev/null 断 stdin。
# 信号转发在 common.py::setup_env() 也会触发 Trainer 优雅退出、保留 checkpoint-* 供下次 resume。
mkdir -p $WORK_DIR/logs
LOGFILE=$WORK_DIR/logs/stage1_$(date +%Y%m%d_%H%M).log
setsid torchrun --nproc_per_node=4 $CODE_DIR/stage1_pretrain.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage1 \
  --max_len 1024 --epochs 1 \
  --per_device_batch 4 --grad_accum 4 \
  --lr 1e-4 --lora_plus_scaler 4 \
  > $LOGFILE 2>&1 < /dev/null &
disown
echo "[$(date)] stage1 启动, 日志=$LOGFILE"
```

**配置说明**：
- 23.6万 全量 × 1 epoch
- batch 4 + grad_accum 4 = global batch 64（与 Stage 2 smoke 同参数，可比）
- 默认 4bit + 8bit AdamW + grad checkpoint + train_norm + LoRA+ scaler 4
- 3687 步 × 4.1s ≈ 4.2h

**产出**：ckpt/stage1（4bit QLoRA adapter，约 30MB）→ **下载作为晚 3 input**

### 晚 2：Stage 2 分支 A — stage2-only（commit 5-6h）

**目的**：SFT 主力（无 Stage 1）基线。

```bash
# 【公共环境必须 setsid】长时间训练可能被 SIGHUP 杀。详见晚 1 注释、§8.4。
mkdir -p $WORK_DIR/logs
LOGFILE=$WORK_DIR/logs/stage2_only_$(date +%Y%m%d_%H%M).log
setsid torchrun --nproc_per_node=4 $CODE_DIR/stage2_sft.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage2_only \
  --max_len 2048 --epochs 1 \
  --per_device_batch 4 --grad_accum 4 \
  --lr 2e-4 \
  > $LOGFILE 2>&1 < /dev/null &
disown
echo "[$(date)] stage2_only 启动, 日志=$LOGFILE"
```

**配置说明**：
- 28.98万 全量 × 1 epoch
- batch 4 + grad_accum 4 = global batch 64
- max_len 2048：bio 长序列 tail 留 headroom
- 4528 步 × 4.1s ≈ 5.2h

**产出**：ckpt/stage2_only → **下载作为晚 6 评估 input**（只为消融①对比，**不再给 DPO 流水线使用**）

### 晚 3：Stage 2 分支 B — stage1+stage2（commit 5-6h）

**目的**：给"有 Stage 1"版本，与分支 A 对比得消融①。

```bash
# 【公共环境必须 setsid】详见晚 1 注释。
mkdir -p $WORK_DIR/logs
LOGFILE=$WORK_DIR/logs/stage2_s1_$(date +%Y%m%d_%H%M).log
setsid torchrun --nproc_per_node=4 $CODE_DIR/stage2_sft.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --resume_adapter /path/to/stage1-adapter \
  --output_dir $WORK_DIR/ckpt/stage2_s1 \
  --max_len 2048 --epochs 1 \
  --per_device_batch 4 --grad_accum 4 \
  --lr 2e-4 \
  > $LOGFILE 2>&1 < /dev/null &
disown
echo "[$(date)] stage2_s1 启动, 日志=$LOGFILE"
```

**与分支 A 的唯一区别**：多 `--resume_adapter`（Stage 1 adapter 路径）。超参完全一致 → 消融①只差"有无 Stage 1"。

**注意**：`--resume_adapter` 路径是 Kaggle Dataset 挂载点（晚 1 产出的 stage1 adapter）。

**产出**：ckpt/stage2_s1 → **下载作为晚 7 input**

### 晚 4：build_preference（commit 4-5h）

**目的**：构造 DPO 偏好对（chosen=标准答案 / rejected=stage2-on-policy 采样）。

```bash
# 【强制单卡】生成式脚本多卡会 OOM。代码内有 hard assert（WORLD_SIZE>1 直接 RuntimeError）。
# 2.5万对 7B 4bit + 高温度采样 单卡 7-9h，仍能在一晚内完成。加速请走 vLLM，该方案不在本项目范围。
# 【公共环境 setsid】单卡长任务同样会被 SIGHUP 杀，同晚 1 注释。
mkdir -p $WORK_DIR/logs
LOGFILE=$WORK_DIR/logs/buildpref_$(date +%Y%m%d_%H%M).log
setsid python $CODE_DIR/build_preference.py \
  --model_path $MODEL_7B --stage2_dir /path/to/stage2-s1-adapter \
  --data_dir $DATA_DIR \
  --output_dir $WORK_DIR \
  --max_pairs 25000 \
  --max_new_tokens 96 --temperature 0.9 \
  > $LOGFILE 2>&1 < /dev/null &
disown
```

**配置说明**：
- 2.5万对从 11.2万 dpo_source 抽样（脚本内部按顺序取前 25000）
- 单卡运行（代码内 bring硬 assert拒绝多卡）：6250/decode × 2.5s ≈ 7h。曾在 4 卡跑时
  4 进程同时加载 base+adapter (×2)，3-4 卡 OOM 退出 → 弹性 launcher 60s 超时后 SIGTERM rank 0，
  留下 "未找到 .rank* 文件" 误导错误。
- **路径 B（推荐）：rejected 用 stage2_s1**（path B——pipeline 沿 stage2_s1 起点，部署场景一致；stage2_only 仍训练仅为消融①）

**产出**：dpo_pairs.jsonl → **下载作为晚 5 input**

### 晚 5：Stage 3 DPO（commit 1.5-2h）

**目的**：自实现 DPO，偏好对齐。

```bash
# 【公共环境必须 setsid】详见晚 1 注释。
mkdir -p $WORK_DIR/logs
LOGFILE=$WORK_DIR/logs/stage3_$(date +%Y%m%d_%H%M).log
setsid torchrun --nproc_per_node=4 $CODE_DIR/stage3_dpo.py \
  --model_path $MODEL_7B --stage2_dir /path/to/stage2-s1-adapter \
  --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage3 \
  --dpo_data dpo_pairs.jsonl \
  --max_len 1024 --epochs 1 \
  --per_device_batch 4 --grad_accum 4 \
  --lr 1e-5 --beta 0.1 \
  > $LOGFILE 2>&1 < /dev/null &
disown
echo "[$(date)] stage3 启动, 日志=$LOGFILE"
```

**配置说明**：
- 2.5万对 × 1 epoch
- batch 4 + grad_accum 4 = global batch 64
- max_len 1024：DPO 双模型 × 2 序列，激活 4 路
- **烟测 peak=26.8GiB/卡**（代码已优化：logits不升为 fp32 全张；ref_model.no_grad 避免保留中间层
  autograd graph；policy+ref 双重 grad checkpoint）。余量充足，能在正式数据上安全跑动。
- 391 步 × 13.5s ≈ 1.5h

**产出**：ckpt/stage3 → **下载作为晚 7 input**

### 晚 6 / 7：评估 4 档模型（前端 6-7h/晚）

**目的**：4 档模型 × 1.89万 全量 eval_set 对比，支撑消融①和②。

**为什么拆 2 晚**：单卡 1.89万 推理 ~3s/样本 × 4 档 = 63h。已加 batch generation（batch 8 提速约 5×）→ 4 档约 13h，**2 档/晚 ≈ 6-7h**。

```bash
# === 晚 6：基座 + stage2_only（每模型单卡跑 ~3h）===

# 1) 基座
python $CODE_DIR/infer_eval.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --output_dir $WORK_DIR --out_file eval_base.jsonl \
  --max_len 1024 --max_samples 18900 --batch_size 8

# 2) stage2-only
python $CODE_DIR/infer_eval.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --ckpt_dir /path/to/stage2-only-adapter \
  --output_dir $WORK_DIR --out_file eval_s2_only.jsonl \
  --max_len 1024 --max_samples 18900 --batch_size 8

# === 晚 7：stage2_s1 + stage3（DPO）===

# 3) stage2_s1 (stage1+stage2)
python $CODE_DIR/infer_eval.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --ckpt_dir /path/to/stage2-s1-adapter \
  --output_dir $WORK_DIR --out_file eval_s1_s2.jsonl \
  --max_len 1024 --max_samples 18900 --batch_size 8

# 4) stage3 (DPO)
python $CODE_DIR/infer_eval.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --ckpt_dir /path/to/stage3-adapter \
  --output_dir $WORK_DIR --out_file eval_stage3.jsonl \
  --max_len 1024 --max_samples 18900 --batch_size 8

# === 评估指标（4 档各跑一次）===
eval/evaluate.py --model_name base     --OMICS all_omics --input_file_path $WORK_DIR/eval_base.jsonl
eval/evaluate.py --model_name s2_only  --OMICS all_omics --input_file_path $WORK_DIR/eval_s2_only.jsonl
eval/evaluate.py --model_name s1_s2    --OMICS all_omics --input_file_path $WORK_DIR/eval_s1_s2.jsonl
eval/evaluate.py --model_name stage3   --OMICS all_omics --input_file_path $WORK_DIR/eval_stage3.jsonl
```

**每档 ~3h**（单卡 7B 4bit 推理 batch 8 ≈ 0.6s/样本 × 1.89万 ≈ 3h）。4 档全量评估。

---

## 5. 时间预算（A100×4 7B / 全量）

| 晚 | Stage | 预算 | 关键参数 |
|---|---|---|---|
| 0 | 冒烟 | 1.5h | 各 stage 的缩小版 |
| 1 | Stage 1 | 4-5h | 23.6万 全量, batch 4, grad_accum 4 |
| 2 | Stage 2 A | 5-6h | 28.98万 全量, batch 4, grad_accum 4, max_len 2048 |
| 3 | Stage 2 B | 5-6h | 同 A + `--resume_adapter` |
| 4 | build_preference | 7-9h | 2.5万对（**强制单卡**·代码有 hard assert） |
| 5 | Stage 3 DPO | 1.5-2h | 2.5万对, batch 4, grad_accum 4 |
| 6 | 评估（2 档） | 6-7h | 2 档 × 1.89万 + batch gen |
| 7 | 评估（2 档） | 6-7h | 2 档 × 1.89万 + batch gen |
| **8** | **推理加速 benchmark** | **2-3h** | **5 组对照（B1/B2/B3/V1/V2）单卡顺序跑** |
| **合计** | | **36-43h, 9 晚 + 1 晚可选** | |

---

## 6. 评估与消融

| 消融 | 对比 | 必要性 |
|---|---|---|
| ① **Stage 1 必要性** | `stage2_s1` vs `stage2_only`（两分支均 28.98万 全量） | 完整本地消融 |
| ② **DPO 改善对齐** | `stage3` vs `stage2_s1`（同 2.5万 pairs，路径 B 起于 stage2_s1） | **核心消融，必做** |
| ③ **数据量**（可选） | 5万 vs 28.98万 Stage 2 对比 | 时间充裕时 |

**指标**：`eval/evaluate.py` 沿用论文官方评估协议（input/label/task/model_output）。  
**定性**：对齐前后对话示例对比（README 展示 4-6 条）。

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 晚间 commit 失败 / 超时 → working 清空 | 每晚产出**立即下载**到本地 / 推到下一晚 input Dataset |
| build_preference 多卡误调 | **强制单卡**·启动时有 hard assert（WORLD_SIZE>1 直接 RuntimeError）；需要加速请用 vLLM 代替（不在本项目范围） |
| infer_eval 4 卡浪费 | 强制单卡 `python`（脚本无 rank sharding）；加了 batch gen 提速 5× |
| Stage 1 欠拟合（loss 不降） | 加数据 / 加大 rank / 加 epoch |
| DPO 后任务指标下降 | β、lr 调小；训练中监控 eval_set |
| Stage 3 显存紧张 | 7B 4bit 1024 烟测 peak≈26.8GiB/卡，余量充足；logits 路径已优化不转 fp32全张；否则降 batch 1 或 max_len 768 |
| 评估 batch gen 显存不够 | 降 batch_size 4 或 2；最差回单样本（牺牲速度） |
| 单晚跑不完 commit | 先 `--max_samples` 跑不完的，回退到合理数据量；adapter 检查点可断点续跑 |

---

## 8. 关键代码补丁

### 8.1 build_preference.py 代码兼容多卡·**运行时强制单卡**

代码内保留 sharding 同步逻辑（各 rank 写 `.rank{i}` + barrier + rank 0 合并），以备未来 vLLM
等批量推理场景复用。**入口加了硬 assert**：`if WORLD_SIZE > 1: raise RuntimeError(...)`。

调测中发现：4 卡 DDP 会让每进程同时加载 7B base+adapter，3-4 卡 OOM 退出 ⇒ elastic launcher
60s 超时后 SIGTERM rank 0，留下 "未找到 .rank* 文件" 误导性错误（实际生成未跑起来）。

- **运行时**：必须 `python train/build_preference.py ...`（WORLD_SIZE 不设）
- **代码逻辑**：sharding merge 代码块保留备用（不依赖 torchrun）

**冒烟耗时参考**：50 对 单卡 ~5min。2.5万对 全量 单卡 ~7-9h。

### 8.2 infer_eval.py 加 batch generation（已应用）

单样本循环 → 动态切批 + 左填充生成 + 按 `input_len` 切分 generated tokens。

- 命令行 `--batch_size 8`（默认 8，1 = 原单样本）
- 提速约 5×（4 档 × 1.89万：63h → ~13h）
- 显存足够时可加大 batch_size；不足时降级到 4/2

### 8.3 use_reentrant 修复（已应用）

common.py `add_lora` 显式 `gradient_checkpointing_kwargs={"use_reentrant": False}` + `setup_env` checkpoint monkey-patch（setdefault + 替换 modeling_utils 引用）。详见 git log `feat(train): 添加transformers兼容性修复`。

### 8.4 公共环境 SIGHUP 免疫：setsid + disown + < /dev/null（已应用）

公共 GPU 节点上跑 5h+ 训练会被 SIGHUP 杀（SSH 断 / 父 shell 退出 / nvidia-smi watchodg 等）。**三件套**：

- `setsid torchrun ...`：开新 session，脱离原控制终端，免疫 SIGHUP
- `< /dev/null`：断 stdin，防止后台进程读 stdin 阻塞
- `disown`：从 bash jobs 表移除，bash exit 时不再发送信号

【代码侧补丁】`common.py::setup_env()` 将 SIGHUP / SIGINT / SIGTERM 转发为 SIGTERM，触发 transformers Trainer 的 `_train_signal_handler` 走 `on_train_end` 保存 checkpoint。Trainer 默认从最新 `checkpoint-*` 续跑。

详细原理与现场记录见 [`TECH_NOTES.md` §2.13](TECH_NOTES.md)。所有正式训练命令（晚 1/2/3/5）已使用三件套。

---

## 9. 执行清单

- [ ] 准备 environment（`pip install -q transformers peft bitsandbytes accelerate datasets sentencepiece`）
- [ ] **代码补丁已应用**：build_preference 强制单卡 assert（§8.1）+ infer_eval 加 batch gen（§8.2）+ use_reentrant 修复（§8.3）+ SIGHUP 免疫三件套 + 信号转发（§8.4）+ DPO 显存限制修了 logits .float()（见 TECH_NOTES）
- [ ] 晚 0：冒烟 5 步通过
- [ ] 晚 1：Stage 1 → 下载 ckpt/stage1
- [ ] 晚 2：Stage 2 branch A → 下载 ckpt/stage2_only
- [ ] 晚 3：Stage 2 branch B → 下载 ckpt/stage2_s1
- [ ] 晚 4：build_preference → 下载 dpo_pairs.jsonl
- [ ] 晚 5：Stage 3 DPO → 下载 ckpt/stage3
- [ ] 晚 6：评估 base + stage2_only → 下载 2 个 jsonl
- [ ] 晚 7：评估 stage2_s1 + stage3 → 下载 2 个 jsonl
- [ ] 4 档 evaluate.py 出指标表 + 消融图
- [ ] 更新 README 成果、整理项目文档
- [ ] **晚 8（可选）**：推理加速 benchmark → 5 组对照实验 + `bench/bench_summary.csv` + `bench/bench_inference.md` 4 维表回填

---

## 10. 推理加速（晚 8 · 5 组对照 benchmark）

> **本节目标**：把 1.89 万条评估集的推理从"HF generate + bf16 + batch=1"升级到
> "vLLM + NF4 4-bit + LoRA hot-swap + continuous batching"，用 **4 个维度**（吞吐 / 延迟 /
> 显存 / 任务指标）做横向对比，给出 **Go / No-Go 判定**，并支撑简历 / 面试的"训推同源"
> 选型故事。
>
> **代码位置**：`infer/`（baseline_runner / fast_infer / bench）·
> **报告骨架**：`bench/bench_inference.md`（待 A100 回填数字）·
> **选型决策**：`docs/INFER_DECISION.md` · **面试问答**：`docs/INFER_QA.md`
>
> **本节是 9 晚主线的"加分项"**：晚 1-7 已经把 4 档模型（base / s2_only / s1_s2 / stage3）
> 用 `train/infer_eval.py` 跑出 4 份 `eval_*.jsonl`，晚 8 在这之上做"换引擎不换结果"的对照。
> 不阻塞主线交付，但强烈建议做——这是产品岗汇报里"推理闭环"的关键素材。

### 10.1 为什么单独一节而不是放在"晚 6/7 评估"里

| 维度 | 晚 6/7 评估 | 晚 8 推理加速 |
|---|---|---|
| 目的 | 4 档模型 × 1.89万 出任务指标 | **同一档模型 × 5 种推理方式** 出加速比 |
| 引擎 | 单一：`train/infer_eval.py`（HF generate） | 5 种：HF bf16 / HF 4bit / HF 4bit b=8 / **vLLM 4-bit** / vLLM 4-bit reserved |
| Adapter | 4 档不同（base / s2_only / s1_s2 / stage3）| **同一档**（固定 stage2_s1）—— 隔离"模型差异" |
| 横向变量 | 模型档 | 引擎 + 量化 + batch 策略 |
| 输出 | `eval_<tag>.jsonl`（给 `evaluate.py` 吃）| `bench/raw/<tag>.jsonl` + `bench_summary.csv`（横向对比）+ 4 维报告 |
| 阻断主线？ | 否（必做）| 否（加分项）|

**关键不变量**：两组输出**字节级一致**（同 `input/label/task/model_output` schema）→
晚 6/7 的 `evaluate.py` 可直接吃晚 8 的 `bench/raw/*.jsonl`，任务指标横向对比零摩擦。

### 10.2 5 组对照实验设计

| 组 | tag | 引擎 | 量化 | batch | 目的 |
|---|---|---|---|---|---|
| **B1** | `bf16` | HF generate | bf16 | 1 | 底线基线：无量化、无引擎优化 |
| **B2** | `4bit` | HF generate | NF4 4-bit | 1 | 看仅 4-bit 加载的省显存 |
| **B3** | `4bit_b8` | HF generate | NF4 4-bit | 8 | 看朴素引擎下 batch 增益上限 |
| **V1** ⭐ | `vllm_4bit` | **vLLM 0.6+** | **NF4 4-bit** | continuous | **主推方案** |
| V2 | `vllm_4bit_b16` | vLLM 0.6+ | NF4 4-bit | continuous | V1 的 reserved 名（暂与 V1 同） |

> ⭐ V1 是产品岗汇报的"主推方案"。B1/B2/B3 用于画"加速来源归因"曲线（4-bit 加载 / batch 增益 /
> 引擎升级三段拆解）。

**控制变量**（5 组完全一致）：
- 硬件：A100 40GB ×1（单卡，避免多卡变量干扰）
- 基座 + adapter：`Qwen2.5-7B-Instruct` + `ckpt/stage2_s1/`（与晚 7 评估同档）
- 数据：`data_prep/output/eval_set.jsonl`，**18,870 条全量**
- 生成参数：`max_new_tokens=64`, `do_sample=False`（贪心解码）
- 温度：0.0（vLLM 约定）/ None（HF generate 显式置 None）

**Adapter 选择**：用 `stage2_s1`（而非 `stage2_only` 或 `stage3`）—— 是项目主推流水线（Stage 1+2）
的中点，既有 SFT 增益又无 DPO 变量干扰，最能反映"推理加速"对真实任务的纯增益。

### 10.3 加速来源归因（填表后写解读）

每组实验跑完后，从 throughput 推算三段归因：

```
B1 (bf16 baseline) ─── 加速比 1.00×  ──── 起点
   ↓  4-bit 加载省显存（动态反量化省激活 + 权重显存减半）
B2 (4bit b=1)        ─── 加速比 a₁×   ──── 4-bit 加载增益: (a₁-1) × 100%
   ↓  batch=8 朴素增益（HF generate 同步，序列左填充）
B3 (4bit b=8)        ─── 加速比 a₂×   ──── batch 增益: (a₂/a₁-1) × 100%
   ↓  引擎升级：vLLM continuous batching + PagedAttention
V1 (vLLM 4-bit)      ─── 加速比 a₃×   ──── 引擎升级增益: (a₃/a₂-1) × 100%
```

**社区 SOTA 倍数参考**（A100 7B bnb 4-bit）：
- vLLM 相对 HF bf16：**5–10×**（社区常见，报告里 ≥ 3× 即算"显著加速"）
- vLLM 相对 HF 4bit b=8：**≥ 1.5×**（continuous batching 独占增益）
- 4-bit 加载相对 bf16：**1.2–1.5×**（省的是显存，速度增益来自能塞更大 batch）

### 10.4 单组实验命令

```bash
# B1：bf16 + HF generate + batch=1（底线基线）
python infer/baseline_runner.py \
    --tag bf16 \
    --model_path $MODEL_7B --ckpt_dir $CKPT_STAGE2_S1 \
    --in_file data_prep/output/eval_set.jsonl \
    --out_file bench/raw/baseline_bf16.jsonl \
    --metrics_file bench/raw/baseline_bf16_metrics.json \
    --no_4bit --batch_size 1

# B2：4bit + HF generate + batch=1
python infer/baseline_runner.py \
    --tag 4bit \
    --model_path $MODEL_7B --ckpt_dir $CKPT_STAGE2_S1 \
    --in_file data_prep/output/eval_set.jsonl \
    --out_file bench/raw/baseline_4bit.jsonl \
    --metrics_file bench/raw/baseline_4bit_metrics.json \
    --batch_size 1

# B3：4bit + HF generate + batch=8
python infer/baseline_runner.py \
    --tag 4bit_b8 \
    --model_path $MODEL_7B --ckpt_dir $CKPT_STAGE2_S1 \
    --in_file data_prep/output/eval_set.jsonl \
    --out_file bench/raw/baseline_4bit_b8.jsonl \
    --metrics_file bench/raw/baseline_4bit_b8_metrics.json \
    --batch_size 8

# V1：vLLM 4-bit + LoRA + continuous batching（主推方案）
python infer/fast_infer.py \
    --tag vllm_4bit \
    --model_path $MODEL_7B --ckpt_dir $CKPT_STAGE2_S1 \
    --in_file data_prep/output/eval_set.jsonl \
    --out_file bench/raw/vllm_4bit.jsonl \
    --metrics_file bench/raw/vllm_4bit_metrics.json
```

**一键跑全 5 组**（推荐，避免逐条复制）：

```bash
# A100 上一次性跑完 5 组对照（顺序，不并发——并发会争抢显存）
python infer/bench.py --all

# 冒烟（每组 100 条）—— 验流程 + vLLM 装环境是否正常
python infer/bench.py --all --smoke
```

每组完成后自动落盘：
- `bench/raw/<tag>.jsonl` —— 推理输出（与 `train/infer_eval.py` **字节级一致**）
- `bench/raw/<tag>_metrics.json` —— 速度 / 显存 / latency 指标
- `bench/bench_summary.csv` —— 自动追加 5 组横向比较（宽表）

### 10.5 任务指标横向对比

5 组推理输出后，调用项目原有 `eval/evaluate.py` 算 24 项任务指标：

```bash
for tag in bf16 4bit 4bit_b8 vllm_4bit; do
    python eval/evaluate.py \
        --model_name $tag \
        --OMICS all_omics \
        --input_file_path bench/raw/${tag}.jsonl
done
```

> `eval/evaluate.py` 会在 `logging/` 目录写 `metrics_<tag>_all_omics_<timestamp>.log`，
> 把 24 项任务指标落盘。把每组的这个 log 拷到 `bench/raw/<tag>_task_metrics.txt`，
> 即可在报告里横向对比"任务指标 vs 加速比 vs 显存"的 trade-off。

**Go / No-Go 判定标准**（报告里要明确写）：
- ✅ **Go**：vLLM 4-bit 相对 bf16 baseline **加速比 ≥ 3×** 且任务指标平均变化 **< 1.5%**
- ⚠️ **保留**：加速比达标但任务指标变化 1.5–3%——报告里诚实标注 trade-off
- ❌ **No-Go**：加速比 < 3× **或** 任务指标掉 > 3%——回退到 B3（HF 4bit + b=8）作过渡方案

### 10.6 时间预算（晚 8 · 2-3h）

| 组 | tag | 单卡 1.89万 预估耗时 | 说明 |
|---|---|---|---|
| B1 | bf16 | ~3h | 单样本循环，无 batch |
| B2 | 4bit | ~3h | 加载快但仍单样本 |
| B3 | 4bit_b8 | ~40min | batch=8 提速约 5× |
| V1 | vllm_4bit | ~15min | continuous batching 期望 5-10× |
| V2 | vllm_4bit_b16 | ~15min | reserved，与 V1 同 |
| **合计** | | **~5h 顺序跑** | 实际预算 2-3h（vLLM 增益 + 模型 warm 缓存命中） |

**注意事项**：
- **顺序跑，不并发**——5 组同时跑会争抢 A100 40GB 显存（vLLM 启期 `gpu_memory_utilization=0.9`）
- **每组启动 ~30s 模型加载**——5 组 × 30s = 2.5min 启动开销，已计入预算
- **vLLM 0.6+ 不支持 Windows**——必须在 Linux A100 节点跑（`infer/README.md` §3.2 详述）

### 10.7 关键设计决策（与主线的关系）

**1. 为什么选 vLLM 不选 SGLang / TGI / TensorRT-LLM**（`docs/INFER_DECISION.md` §2 详述）：

| 引擎 | 7B 4-bit | LoRA hot-swap | 安装难度 | 综合 |
|---|---|---|---|---|
| **vLLM 0.6+** ⭐ | ✅ | ✅ `LoRARequest` API | 🟢 pip 一行 | **选** |
| SGLang | ✅ | ⚠️ 实验性 | � | 备选 |
| TGI (HF) | ✅ | ❌ | 🟡 Rust | 不选 |
| TensorRT-LLM | ✅ | ⚠️ 需 engine 重建 | � 重 | 不选（时间不划算）|

**2. 为什么选 bitsandbytes 4-bit 不选 GPTQ / AWQ**（`docs/INFER_DECISION.md` §3 详述）：
- 项目已用 bnb 4-bit 做 QLoRA 训练 → 部署也用 bnb 4-bit → **权重零转换、训推一致**
- GPTQ / AWQ 需单独 calibration + 量化脚本，**1 周时间增量收益不划算**
- 报告里**诚实标注**"为追求训推同源，放弃 GPTQ 的 ~0.5% 指标优势"

**3. 为什么 LoRA 不 merge 进 base**（`infer/README.md` §4.3 详述）：
- 项目需要同时跑 stage2 / stage3 两个 adapter 对比 → **保留 PEFT 格式 + LoRARequest**
- merge_and_unload 后量化：推理略快但失去多版本灵活性，且 merge + 重量化慢

**4. 为什么输出格式与 baseline 字节级一致**（`infer/README.md` §4.4 详述）：
- 保证 `eval/evaluate.py` 直接吃两组输出，任务指标横向对比零摩擦
- 任何格式差异都会污染指标对比（"vLLM 输出比 baseline 多了个空格？"——产品岗最怕这种坑）

### 10.8 与主线代码的边界（零侵入设计）

| 已有模块 | 本节是否修改 | 说明 |
|---|---|---|
| `train/common.py` | ❌ | 复用 `load_model_tokenizer` / `SYSTEM_PROMPT` / `read_jsonl` |
| `train/infer_eval.py` | ❌ | 保留作 0.5B 冒烟 / 小规模用；正式加速走 `infer/` |
| `eval/evaluate.py` | ❌ | 沿用官方协议 |
| `eval/register_tasks.json` | ❌ | 24 任务注册表 |
| `ckpt/stage2_s1/...` | ❌ 不读 | 训练时 adapter，PEFT 格式直接喂 vLLM |

**零侵入设计**：本节纯增量，对训练 / 数据 / 评估三个既有模块**完全无影响**。
这本身是产品岗要的"在不破坏现有系统的前提下做增量"的能力展示。

### 10.9 风险与对策（推理加速专属）

| 风险 | 对策 |
|---|---|
| Windows / macOS 跑不起来（vLLM `import resource`）| 换 Linux A100 节点（`infer/README.md` §3.2）|
| 装 vLLM 后训练脚本报"undefined symbol" | **训练 / 推理分 venv**（vLLM 改写 torch / cuda runtime）|
| vLLM 启动 OOM（`gpu_memory_utilization=0.95` 太高）| 降到 0.9 → 0.85 |
| vLLM 长序列 KV cache 爆 | `max_model_len` 从 2048 降到 1024；或 `enforce_eager=True` |
| LoRA 加载后输出"乱码" | 严格沿用 `train/common.py` 的 ChatML 模板（`fast_infer.py` 已固化）|
| bitsandbytes 版本不匹配 | 锁 **0.49.2**（与训练侧同）|
| 评估指标比 baseline 低 > 2% | 升级 vLLM 至 0.6.4+；如还低，报告里诚实标注 trade-off |
| 5 组顺序跑超时 2-3h 预算 | 优先保 V1 跑完（主推方案）；B1/B2/B3 缺一两组也能算"加速来源归因" |
| bench 跑出 NaN / "?" | 检查 vLLM 版本 / bnb 版本对齐；降 `--max_samples` 重跑单组 |

### 10.10 验证 checklist（跑完晚 8 后自查）

- [ ] 5 组实验的 `bench/raw/<tag>.jsonl` 都有产出，文件大小与样本数成正比
- [ ] 5 组 `bench_summary.csv` 5 行齐全，无 NaN / "?"
- [ ] vLLM 4-bit 相对 bf16 加速比 **≥ 3×**
- [ ] vLLM 4-bit 相对 4bit + batch=8 加速比 ≥ 1.5×
- [ ] vLLM 4-bit 任务指标相对 bf16 baseline 平均变化 **< 1.5%**
- [ ] peak 显存 vLLM 4-bit < baseline bf16 **30%+**
- [ ] `bench/bench_inference.md` 4 维表填完，Go / No-Go 判定清晰
- [ ] `docs/INFER_QA.md` 至少 10 个问答写完
- [ ] `PROJECT_FOR_RESUME.md` §10 模板 D 引用到 `bench/bench_inference.md`

### 10.11 简历 / 面试引用路径

| 用途 | 路径 |
|---|---|
| 简历条目（"推理加速闭环"那行） | `PROJECT_FOR_RESUME.md` §10 模板 D |
| 面试问答（10+ 个：选型 / 量化 / 工程权衡） | `docs/INFER_QA.md` |
| 4 维对比报告（含数字 + 归因 + Go/No-Go） | `bench/bench_inference.md` |
| 选型决策矩阵（结构化） | `docs/INFER_DECISION.md` |
| 代码入口与命令 | `infer/README.md` |
