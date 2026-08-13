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
| 4 | build_preference | 2.5万对（dpo_source 抽样） | 4-5h | dpo_pairs.jsonl |
| 5 | Stage 3 DPO | 2.5万对 | 1.5-2h | ckpt/stage3 |
| 6 | 评估 4 档（前 2 档） | 1.89万 × 2 + batch gen | 6-7h | eval_base / eval_s2_only |
| 7 | 评估 4 档（后 2 档） | 1.89万 × 2 + batch gen | 6-7h | eval_s1_s2 / eval_stage3 |
| **合计** | | | **34-40h, 9 晚** | |

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
torchrun --nproc_per_node=4 $CODE_DIR/stage1_pretrain.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage1_smoke \
  --max_len 1024 --max_steps 30 --max_samples 200 \
  --per_device_batch 4 --grad_accum 4

# Step 2: Stage 2 冒烟（60 步，100 条）
torchrun --nproc_per_node=4 $CODE_DIR/stage2_sft.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage2_smoke \
  --max_len 1024 --max_steps 60 --max_samples 100 \
  --per_device_batch 4 --grad_accum 4

# Step 3: build_preference 冒烟（50 pairs，4 卡已加 sharding）
torchrun --nproc_per_node=4 $CODE_DIR/build_preference.py \
  --model_path $MODEL_7B --stage2_dir $WORK_DIR/ckpt/stage2_smoke \
  --data_dir $DATA_DIR --output_dir $WORK_DIR \
  --max_pairs 50

# Step 4: Stage 3 DPO 冒烟（20 步，100 pairs）
torchrun --nproc_per_node=4 $CODE_DIR/stage3_dpo.py \
  --model_path $MODEL_7B --stage2_dir $WORK_DIR/ckpt/stage2_smoke \
  --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage3_smoke \
  --max_len 1024 --max_samples 100 --max_steps 20 \
  --per_device_batch 4 --grad_accum 4

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
torchrun --nproc_per_node=4 $CODE_DIR/stage1_pretrain.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage1 \
  --max_len 1024 --epochs 1 \
  --per_device_batch 4 --grad_accum 4 \
  --lr 1e-4 --lora_plus_scaler 4
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
torchrun --nproc_per_node=4 $CODE_DIR/stage2_sft.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage2_only \
  --max_len 2048 --epochs 1 \
  --per_device_batch 4 --grad_accum 4 \
  --lr 2e-4
```

**配置说明**：
- 28.98万 全量 × 1 epoch
- batch 4 + grad_accum 4 = global batch 64
- max_len 2048：bio 长序列 tail 留 headroom
- 4528 步 × 4.1s ≈ 5.2h

**产出**：ckpt/stage2_only → **下载作为晚 4 / 晚 6 input**

### 晚 3：Stage 2 分支 B — stage1+stage2（commit 5-6h）

**目的**：给"有 Stage 1"版本，与分支 A 对比得消融①。

```bash
torchrun --nproc_per_node=4 $CODE_DIR/stage2_sft.py \
  --model_path $MODEL_7B --data_dir $DATA_DIR \
  --resume_adapter /path/to/stage1-adapter \
  --output_dir $WORK_DIR/ckpt/stage2_s1 \
  --max_len 2048 --epochs 1 \
  --per_device_batch 4 --grad_accum 4 \
  --lr 2e-4
```

**与分支 A 的唯一区别**：多 `--resume_adapter`（Stage 1 adapter 路径）。超参完全一致 → 消融①只差"有无 Stage 1"。

**注意**：`--resume_adapter` 路径是 Kaggle Dataset 挂载点（晚 1 产出的 stage1 adapter）。

**产出**：ckpt/stage2_s1 → **下载作为晚 7 input**

### 晚 4：build_preference（commit 4-5h）

**目的**：构造 DPO 偏好对（chosen=标准答案 / rejected=stage2-on-policy 采样）。

```bash
torchrun --nproc_per_node=4 $CODE_DIR/build_preference.py \
  --model_path $MODEL_7B --stage2_dir /path/to/stage2-only-adapter \
  --data_dir $DATA_DIR \
  --output_dir $WORK_DIR \
  --max_pairs 25000 \
  --max_new_tokens 96 --temperature 0.9
```

**配置说明**：
- 2.5万对从 11.2万 dpo_source 抽样（脚本内部按顺序取前 25000）
- 4 卡 sharding：6250/卡 × ~2.5s ≈ 4.3h
- on-policy 语义：rejected 用 stage2-only（不是 stage2_s1，避免偏好含 Stage 1 信息污染）

**产出**：dpo_pairs.jsonl → **下载作为晚 5 input**

### 晚 5：Stage 3 DPO（commit 1.5-2h）

**目的**：自实现 DPO，偏好对齐。

```bash
torchrun --nproc_per_node=4 $CODE_DIR/stage3_dpo.py \
  --model_path $MODEL_7B --stage2_dir /path/to/stage2-only-adapter \
  --data_dir $DATA_DIR \
  --output_dir $WORK_DIR/ckpt/stage3 \
  --dpo_data dpo_pairs.jsonl \
  --max_len 1024 --epochs 1 \
  --per_device_batch 4 --grad_accum 4 \
  --lr 1e-5 --beta 0.1
```

**配置说明**：
- 2.5万对 × 1 epoch
- batch 4 + grad_accum 4 = global batch 64
- max_len 1024：DPO 双模型 × 2 序列，激活 4 路；7B 4bit 1024 实测 13.4GiB/卡
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
| 4 | build_preference | 4-5h | 2.5万对（4 卡 sharding） |
| 5 | Stage 3 DPO | 1.5-2h | 2.5万对, batch 4, grad_accum 4 |
| 6 | 评估（2 档） | 6-7h | 2 档 × 1.89万 + batch gen |
| 7 | 评估（2 档） | 6-7h | 2 档 × 1.89万 + batch gen |
| **合计** | | **34-40h, 9 晚** | |

---

## 6. 评估与消融

| 消融 | 对比 | 必要性 |
|---|---|---|
| ① **Stage 1 必要性** | `stage2_s1` vs `stage2_only`（两分支均 28.98万 全量） | 完整本地消融 |
| ② **DPO 改善对齐** | `stage3` vs `stage2_only`（同 2.5万 pairs） | **核心消融，必做** |
| ③ **数据量**（可选） | 5万 vs 28.98万 Stage 2 对比 | 时间充裕时 |

**指标**：`eval/evaluate.py` 沿用论文官方评估协议（input/label/task/model_output）。  
**定性**：对齐前后对话示例对比（README 展示 4-6 条）。

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 晚间 commit 失败 / 超时 → working 清空 | 每晚产出**立即下载**到本地 / 推到下一晚 input Dataset |
| build_preference 4 卡 sharding 失效 | §8.1 给出修复；未加前用 `python` 单卡 + max_pairs 减半 |
| infer_eval 4 卡浪费 | 强制单卡 `python`（脚本无 rank sharding）；加了 batch gen 提速 5× |
| Stage 1 欠拟合（loss 不降） | 加数据 / 加大 rank / 加 epoch |
| DPO 后任务指标下降 | β、lr 调小；训练中监控 eval_set |
| Stage 3 显存紧张 | 7B 4bit 1024 实测 13.4GiB/卡，余量充足；否则降 batch 3 或 max_len 768 |
| 评估 batch gen 显存不够 | 降 batch_size 4 或 2；最差回单样本（牺牲速度） |
| 单晚跑不完 commit | 先 `--max_samples` 跑不完的，回退到合理数据量；adapter 检查点可断点续跑 |

---

## 8. 关键代码补丁

### 8.1 build_preference.py 加 rank sharding（已应用）

切分 + 各 rank 写 `.rank{i}` + barrier + rank 0 合并 → 4 卡各生成 1/4，全量 2.5万对约 4.3h。

### 8.2 infer_eval.py 加 batch generation（已应用）

单样本循环 → 动态切批 + 左填充生成 + 按 `input_len` 切分 generated tokens。

- 命令行 `--batch_size 8`（默认 8，1 = 原单样本）
- 提速约 5×（4 档 × 1.89万：63h → ~13h）
- 显存足够时可加大 batch_size；不足时降级到 4/2

### 8.3 use_reentrant 修复（已应用）

common.py `add_lora` 显式 `gradient_checkpointing_kwargs={"use_reentrant": False}` + `setup_env` checkpoint monkey-patch（setdefault + 替换 modeling_utils 引用）。详见 git log `feat(train): 添加transformers兼容性修复`。

---

## 9. 执行清单

- [ ] 准备 environment（`pip install -q transformers peft bitsandbytes accelerate datasets sentencepiece`）
- [ ] **代码补丁已应用**：build_preference 加 rank sharding（§8.1）+ infer_eval 加 batch gen（§8.2）+ use_reentrant 修复（§8.3）
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
