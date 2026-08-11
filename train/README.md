# train/ —— 三段式后训练流水线代码

> 技术栈：transformers + peft + bitsandbytes（QLoRA）+ 自实现 DPO loss。
> **单卡/双卡共用同一份代码**：device_map 按 `LOCAL_RANK` 环境变量分配，
> 本地单卡 `python` 直跑，Kaggle 双卡 `torchrun --nproc_per_node=2`（grad_accum 减半保持 global batch 一致）。

## 文件清单

| 脚本 | 用途 | 输入 → 输出 |
|---|---|---|
| `common.py` | 公共模块：QLoRA 加载、LoRA 配置、通用参数、system prompt | - |
| `stage1_pretrain.py` | Stage 1 领域继续预训练（packing + causal LM，**bf16 LoRA+**：B 学习率=A×4 + 训 RMSNorm） | `stage1_pretrain.jsonl` → adapter |
| `stage2_sft.py` | Stage 2 PEFT（4-bit QLoRA SFT，assistant-only loss） | `train_pool.jsonl` → adapter |
| `build_preference.py` | 自构 DPO 偏好数据（chosen=标准答案，rejected=**stage2 模型采样**，on-policy） | `dpo_source.jsonl` → `dpo_pairs.jsonl` |
| `stage3_dpo.py` | Stage 3 RL（DPO，自实现 loss，不依赖 trl 版本） | `dpo_pairs.jsonl` → adapter |
| `infer_eval.py` | 推理 + 评估（输出 evaluate.py 兼容格式） | `eval_set.jsonl` → `eval_outputs.jsonl` |

## 冒烟测试记录（本地 0.5B + RTX 4060 8GB，全部通过）

| 环节 | 命令要点 | 结果 |
|---|---|---|
| Stage 1 (bf16 LoRA+) | `--max_len 512 --max_samples 2000 --max_steps 30`（默认 bf16 + LoRA+ scaler=4 + train_norm） | 通过，trainable 6.66%，loss~3.1 |
| Stage 2 | `--max_len 1024 --max_samples 1000 --max_steps 60` | loss 2.20→0.38 |
| 偏好构造 | `--max_pairs 50` | 50 对，区分度正常 |
| Stage 3 DPO | `--max_samples 100 --max_steps 20 --beta 0.1` | loss 1.90→0.27 |
| 推理 | `--max_samples 20` | 输出格式兼容 evaluate.py |

冒烟中发现并修复的问题（已体现在最终代码中）：
1. `Dataset.from_list` 不接受 generator → 转 list
2. Trainer 默认 collator 不 padding → 显式 `DataCollatorForSeq2Seq`
3. trl 0.29 与 torch 2.5 不兼容（FSDPModule）→ **自实现 DPO loss**（不依赖 trl）
4. peft 0.18 `from_pretrained` 后 LoRA 参数 requires_grad=False → 显式启用
5. 新版 transformers `compute_loss` 需 `num_items_in_batch` 参数
6. **3B Stage 1 OOM（Kaggle 实测，三次收敛）**：
   - ① max_len 2048 + fp32 AdamW：超预算 → 8bit AdamW + grad checkpoint + max_len 1024
   - ② max_len 1024 仍 OOM：根因是 loss 的 logits 峰值（batch×seq×vocab×4B）→ batch 4→1
   - ③ batch 1 仍 OOM：**根因是 bf16 权重本身**——加载后已用 11.95GB（权重 6.4GB + 预留），训练峰值 13.13GB，单卡 14.56GB 装不下 → **Stage 1 默认改用 4bit QLoRA**（权重 ~2GB，峰值 ~5-6GB）；bf16 仅适合双卡 DDP
   - 附带：3B 实测 trainable=3.74%（外推 6.66% 不可靠）；属性名 `is_gradient_checkpointing`
   - 当前配置：**4bit QLoRA + 8bit AdamW + grad checkpoint + max_len 2048 + batch 1**（4bit 省下的显存让给序列长度，几乎零截断；bf16 彻底放弃）

## 本地冒烟完整命令（0.5B）

```bash
cd BioAlign
# 1. Stage 1（bf16 LoRA+，不用 --use_4bit）
python -X utf8 train/stage1_pretrain.py \
  --model_path "D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct" \
  --data_dir data_prep/output --output_dir ckpt/smoke_stage1 \
  --max_len 512 --per_device_batch 4 --grad_accum 4 --lr 1e-4 \
  --max_steps 30 --max_samples 2000

# 2. Stage 2
python -X utf8 train/stage2_sft.py \
  --model_path "D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct" \
  --data_dir data_prep/output --output_dir ckpt/smoke_stage2 \
  --max_len 1024 --per_device_batch 4 --grad_accum 4 --lr 2e-4 \
  --max_steps 60 --max_samples 1000 --use_4bit

# 3. 偏好数据（方案 A：rejected 用 stage2 模型生成，on-policy）
python -X utf8 train/build_preference.py \
  --model_path "D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct" \
  --stage2_dir ckpt/smoke_stage2 \
  --data_dir data_prep/output --output_dir data_prep/output \
  --max_pairs 50 --max_new_tokens 96 --temperature 0.9 --use_4bit

# 4. Stage 3 DPO
python -X utf8 train/stage3_dpo.py \
  --model_path "D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct" \
  --stage2_dir ckpt/smoke_stage2 --data_dir data_prep/output \
  --output_dir ckpt/smoke_stage3 --max_len 1024 \
  --per_device_batch 2 --grad_accum 4 --lr 1e-5 --beta 0.1 \
  --max_steps 20 --max_samples 100 --use_4bit

# 5. 推理
python -X utf8 train/infer_eval.py \
  --model_path "D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct" \
  --ckpt_dir ckpt/smoke_stage2 --data_dir data_prep/output \
  --output_dir data_prep/output --in_file eval_set.jsonl \
  --out_file eval_outputs_smoke.jsonl --max_new_tokens 64 \
  --max_samples 20 --use_4bit
```

## Kaggle 正式训练要点（Qwen2.5-3B）

1. 模型路径：`/kaggle/input/models/qwen-lm/qwen2.5/transformers/3b-instruct/1`
2. 数据上传：`data_prep/output/` 全部上传为 Kaggle Dataset，`--data_dir` 指向挂载路径
3. 双卡：`torchrun --nproc_per_node=2 train/stage2_sft.py ... --grad_accum <单卡的一半>`
4. **Stage 1 用 4bit QLoRA + LoRA+**（rank 64，**max_len 2048**——覆盖论文 2000 字符序列几乎零截断）；Stage 2/3 用 4bit QLoRA；bf16 在 T4 上已实测装不下（单卡/DDP 均 OOM）
5. 消融①需要两个 Stage 2 分支：`--model_path 基座`（stage2-only）vs `--model_path <stage1 产物>`
6. DPO 偏好数据在 Kaggle 上用 **stage2 模型**生成（`--max_pairs 25000`，约 1-2h）——rejected 用 stage2 采样（on-policy，同分布质量偏好，符合 DPO 语义），不用基座
7. 每 stage 单独 notebook + checkpoint 断点续跑（12h 会话限制）
