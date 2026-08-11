# 三段式后训练流水线方案（Kaggle T4×2 + Qwen2.5-3B）—— 与实现对齐版

> 目标：在 Kaggle 免费资源（T4 16GB，每周约 30h GPU 配额）上用 **Qwen2.5-3B-Instruct** 跑通完整后训练流水线：
> **① 领域继续预训练（bf16 LoRA+，生物序列）→ ② PEFT（4-bit QLoRA SFT，Biology-Instructions）→ ③ RL（DPO 偏好对齐，自实现 loss）**
>
> 技术栈：**transformers + peft + bitsandbytes**（QLoRA）+ **自实现 DPO loss**（不依赖 trl，规避版本兼容问题）。
> 代码在 `train/`（本地冒烟已通过），数据在 `data_prep/`（01~05 脚本 + 产出）。
> 定位：算法岗简历项目。核心卖点=工业级后训练全流程 + 有限算力工程压缩 + 严谨消融。

---

## 1. 资源与环境

| 项 | 值 |
|---|---|
| GPU | Kaggle 免费：**T4 ×2**（16GB×2），每周 GPU 配额 30h（墙钟，**先实测双卡是否翻倍扣配额**）、TPU 20h（不用） |
| 运行模式 | 前台交互=调试/冒烟；**Save Version（Commit）=正式长训练**（后台运行，12h/次限制） |
| 会话/作业 | 单 notebook 12h → **每 stage 一个 notebook** + checkpoint 断点续跑 |
| 框架 | `transformers` + `peft` + `bitsandbytes`（QLoRA）+ `torch`（原生 Trainer + 自实现 DPO） |
| 模型 | `Qwen2.5-3B-Instruct`（Kaggle 路径 `/kaggle/input/models/qwen-lm/qwen2.5/transformers/3b-instruct/1`） |
| 安装 | `pip install transformers peft bitsandbytes accelerate datasets sentencepiece` |

注意：**训练代码不依赖 trl**（trl 0.29 与 torch 2.5 不兼容的 FSDPModule 问题已通过自实现 DPO 规避）。代码与权重作为 **Kaggle Dataset 挂载**（/kaggle/input/），commit 运行不依赖 working 目录。

## 2. 数据利用（与 data_prep/ 完全一致）

```
dataset/ + seq/（原始数据）
  stage2_train.jsonl (330万 QA) ─[01]三路划分→ train_pool 30万 / dpo_source 11.2万 / eval_set 1.9万
  stage3.xlsx ─[02]→ stage3.jsonl (8002 推理长答案)
  GRCh38+RNAcentral+Swiss-Prot ─[03]→ stage1_pretrain.jsonl (23.6万 序列)

  train_pool ─[05]去重+模板均衡+合并stage3→ train_pool_clean.jsonl (28.98万) → Stage 2 SFT
  dpo_source ─[build_preference]→ dpo_pairs.jsonl (2-3万对) → Stage 3 DPO
  eval_set (1.9万) → 评估（SFT/DPO 均不可见，防泄漏）
  train_pool 子集 ─[04]→ smoke.jsonl (4700) → 冒烟测试
```

**三个核心设计**：
1. **三路严格不相交**（行号切分）：训练 / DPO 源 / 评估互不重叠；
2. **DPO 用独立 dpo_source**（不用 eval_set）：chosen=标准答案，若拿 eval 构造偏好对会污染评估，无法证明"DPO 不掉领域能力"；
3. **train_pool_clean 净化**：源数据是模板化生成的（每 task 仅 50~200 个模板），去重 + 模板均衡采样（每模板 cap=max(50, ceil(avg×mult))）防模板风格过拟合；Top5 模板占比 2.0%→1.3%。

## 3. 三阶段训练（代码在 train/）

| Stage | 代码 | 数据 | 方法 | 关键参数 |
|---|---|---|---|---|
| 1 继续预训练 | `stage1_pretrain.py` | stage1_pretrain.jsonl（23.6万） | **bf16 LoRA+**（B lr=A×4）+ packing + 训 RMSNorm | r=64, lr=1e-4, max_len=1024, 8bit AdamW |
| 2 SFT | `stage2_sft.py` | train_pool_clean.jsonl（28.98万） | 4-bit QLoRA，assistant-only loss | r=16, lr=1e-4~2e-4, max_len=1024 |
| 偏好构造 | `build_preference.py` | dpo_source（取 2-3 万） | chosen=标准答案 / rejected=基座生成（temp=0.9） | max_new_tokens=96 |
| 3 DPO | `stage3_dpo.py` | dpo_pairs.jsonl | **自实现 DPO loss**（-log σ(β·Δlogratio)），π_ref=stage2 冻结 | β=0.1, lr=1e-5, 1 epoch |

**消融分支**：Stage 2 需两个分支——`--model_path 基座`（stage2-only）vs `--resume_adapter stage1的adapter路径`（stage1+stage2），验证 Stage 1 必要性。

## 4. 实施计划（跑什么、怎么跑，含完整命令）

### 路径约定（以下命令统一使用）

```
MODEL_3B = /kaggle/input/models/qwen-lm/qwen2.5/transformers/3b-instruct/1
DATA_DIR = /kaggle/input/bioalign-data/output     # 数据 Dataset 挂载点
CODE_DIR = /kaggle/input/bioalign-code/train      # 代码 Dataset 挂载点（脚本目录）
```

### 第 0 步：上传准备（一次性，本地）

| 内容 | 上传为 |
|---|---|
| `data_prep/output/`（8 个数据文件，~660MB） | Kaggle Dataset（如 `bioalign-data`） |
| `train/`（7 个脚本） | Kaggle Dataset（如 `bioalign-code`） |
| 模型权重 | ✅ 已导入（`/kaggle/input/models/qwen-lm/qwen2.5/transformers/3b-instruct/1`） |

### 第 1 个 notebook：环境验证 + 配额实测（前台，~1h）

```
① !pip install -q transformers peft bitsandbytes accelerate datasets sentencepiece
② 4bit 加载冒烟：3B 模型 load + 1 次 forward（验证 bitsandbytes/CUDA 兼容）
③ 配额实测：跑 1h 小训练 → GPU 配额扣 1h 还是 2h → 决定双卡 DDP 或单卡
④ 验证数据挂载路径（$DATA_DIR 下能看到各 jsonl）
```

### 第 2 个 notebook：全链路冒烟（前台，~1-1.5h）

用 `smoke.jsonl`（4700 条）把五步全部跑通（每步 30-60 步/少量）。
**原则：只缩数据量/步数，超参与正式完全一致**（Stage 1 冒烟 max_len 也用正式值 1024，验证显存预算）：

```
Stage1:  python $CODE_DIR/stage1_pretrain.py --model_path $MODEL_3B --data_dir $DATA_DIR \
            --output_dir /kaggle/working/ckpt/stage1_smoke --max_len 1024 --max_steps 30 --max_samples 2000
Stage2:  python $CODE_DIR/stage2_sft.py --model_path $MODEL_3B --data_dir $DATA_DIR \
            --output_dir /kaggle/working/ckpt/stage2_smoke --max_len 1024 --max_steps 60 --max_samples 1000
偏好:    python $CODE_DIR/build_preference.py --model_path $MODEL_3B --data_dir $DATA_DIR \
            --output_dir /kaggle/working --max_pairs 50
DPO:     python $CODE_DIR/stage3_dpo.py --model_path $MODEL_3B --stage2_dir /kaggle/working/ckpt/stage2_smoke \
            --data_dir $DATA_DIR --output_dir /kaggle/working/ckpt/stage3_smoke \
            --max_len 1024 --max_samples 100 --max_steps 20
推理:    python $CODE_DIR/infer_eval.py --model_path $MODEL_3B --ckpt_dir /kaggle/working/ckpt/stage2_smoke \
            --data_dir $DATA_DIR --max_len 1024 --max_samples 20
```

通过标准：五步无报错、loss 正常下降、输出格式正确 → 环境一致确认。

### 第 3 个 notebook：Stage 1 继续预训练（Commit，3-6h）

### 第 3 个 notebook：Stage 1 继续预训练（Commit，3-6h）

显存预算（T4 ~15GB）：bf16 3B 权重 6GB + 激活（grad checkpoint 后 ~2-4GB）+ 8bit 优化器 →
**max_len 用 1024**（≈论文 2000 字符/1200 token 的合理近似），batch=2，8bit AdamW（默认）：

```
python $CODE_DIR/stage1_pretrain.py \
  --model_path $MODEL_3B \
  --data_dir $DATA_DIR \
  --output_dir /kaggle/working/ckpt/stage1 \
  --max_len 1024 --per_device_batch 2 --grad_accum 8 --lr 1e-4 \
  --epochs 1 --lora_plus_scaler 4
```
（--optim 默认 adamw8bit；gradient checkpointing 已在脚本内显式开启；
双卡 DDP 时 batch 减半、显存压力更小）
✅ 产出 `ckpt/stage1`（bf16 LoRA+ adapter）→ **立即下载**

### 第 4 个 notebook：Stage 2 ×2 分支（Commit ×2，各 2-4h）

```
分支 A（stage2-only，不传 --resume_adapter = 从基座训练）：
  python $CODE_DIR/stage2_sft.py --model_path $MODEL_3B --data_dir $DATA_DIR \
    --output_dir /kaggle/working/ckpt/stage2_only \
    --max_len 1024 --per_device_batch 4 --grad_accum 4 --lr 2e-4 --epochs 1

分支 B（stage1+stage2，stage1 adapter 已下载并作为 input 挂载）：
  python $CODE_DIR/stage2_sft.py --model_path $MODEL_3B --data_dir $DATA_DIR \
    --resume_adapter /kaggle/input/stage1-adapter \
    --output_dir /kaggle/working/ckpt/stage2_s1 \
    --max_len 1024 --per_device_batch 4 --grad_accum 4 --lr 2e-4 --epochs 1
```
✅ 产出两个 adapter（两分支统一 r=64，消融①只差"有无 Stage 1"）→ 下载

### 第 5 个 notebook：偏好数据构造（Commit，1-2h）

rejected 用 **stage2 模型**采样（on-policy：同一"会答"分布下的质量偏好，符合 DPO 语义），
chosen = 标准答案；区分度由 temperature 采样保证：

```
python $CODE_DIR/build_preference.py \
  --model_path $MODEL_3B --stage2_dir /kaggle/input/stage2-s1-adapter \
  --data_dir $DATA_DIR \
  --output_dir /kaggle/working --max_pairs 25000 \
  --max_new_tokens 96 --temperature 0.9
```
✅ 产出 `dpo_pairs.jsonl`（2.5 万对）→ 下载（Stage 3 与评估的 input）

### 第 6 个 notebook：Stage 3 DPO（Commit，2-3h）

```
python $CODE_DIR/stage3_dpo.py \
  --model_path $MODEL_3B --stage2_dir /kaggle/input/stage2-s1-adapter \
  --data_dir $DATA_DIR \
  --output_dir /kaggle/working/ckpt/stage3 \
  --dpo_data dpo_pairs.jsonl --max_len 1024 --per_device_batch 2 \
  --grad_accum 4 --lr 1e-5 --beta 0.1 --epochs 1
```
（`--dpo_data dpo_pairs.jsonl` 由 nb5 生成后下载、作为 input 挂载；脚本在 `--data_dir` 下读取）
✅ 产出 `ckpt/stage3` → 下载

### 第 7 个 notebook：评估（前台，1-2h）

4 档模型 × eval_set（1.89 万条，可分批）：

```
基座:     python $CODE_DIR/infer_eval.py --model_path $MODEL_3B --data_dir $DATA_DIR \
            --output_dir /kaggle/working --out_file eval_base.jsonl --max_len 1024
stage2:   python $CODE_DIR/infer_eval.py --model_path $MODEL_3B --data_dir $DATA_DIR \
            --ckpt_dir /kaggle/input/stage2-only-adapter --output_dir /kaggle/working \
            --out_file eval_s2_only.jsonl --max_len 1024
stage2+s1:python $CODE_DIR/infer_eval.py --model_path $MODEL_3B --data_dir $DATA_DIR \
            --ckpt_dir /kaggle/input/stage2-s1-adapter --output_dir /kaggle/working \
            --out_file eval_s1_s2.jsonl --max_len 1024
stage3:   python $CODE_DIR/infer_eval.py --model_path $MODEL_3B --data_dir $DATA_DIR \
            --ckpt_dir /kaggle/input/stage3-adapter --output_dir /kaggle/working \
            --out_file eval_stage3.jsonl --max_len 1024
然后 eval/evaluate.py 各跑一遍出指标 → 消融图 + 对话对比
```

### 关键提醒

1. commit 失败/超时会清空 working → 每个 commit 完成后**立即下载 checkpoint**；下一个 notebook 把上一个 checkpoint 作为 input 挂载
2. Stage 2 两分支统一 r=64，消融①只差"有无 Stage 1"
3. 双卡命令加 `torchrun --nproc_per_node=2`，grad_accum 减半（待配额实测后定）

## 5. 时间预算（T4×2，3B）

| 环节 | 时间 | 运行方式 |
|---|---|---|
| 环境验证 + 配额实测 | 1h | 前台 |
| 全链路冒烟 | 1-1.5h | 前台 |
| Stage 1（23.6万×1 epoch，bf16 LoRA+） | 3-6h | commit |
| Stage 2（28.98万×1 epoch，QLoRA）×2 分支 | 4-8h | commit×2 |
| build_preference（2.5万对，基座生成） | 1-2h | commit |
| Stage 3 DPO（2.5万对×1 epoch） | 2-3h | commit |
| 评估 + 消融（4 档模型） | 1-2h | 前台 |
| **合计** | **约 13-23h** | **30h 配额内 ✅** |

## 6. 评估与消融

1. **消融①（Stage 1 必要性）**：`stage1+stage2` vs `stage2-only` 在 eval_set 上的任务指标
2. **消融②（对齐不掉领域能力）**：`stage2` vs `stage3` 在同一 eval_set 上的任务指标
3. **消融③（数据量，可选）**：`TRAIN_TOTAL_TARGET` 15万 vs 30万（改 01 脚本重跑，几分钟）——测数据量边际效应
4. **定性**：对齐前后对话示例对比
5. 指标：`eval/evaluate.py`（沿用论文官方评估协议，输入 input/label/task/model_output）

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| T4×2 双卡扣费规则不确定 | 第一 notebook 实测 1h 扣 1h 还是 2h |
| bitsandbytes 与 CUDA 不匹配 | 装完先 4bit 加载冒烟 |
| Stage 1 欠拟合（loss 不降） | 加数据（GRCh38/RNAcentral 池还有余量）/加大 rank |
| bf16 3B Stage 1 显存（OOM 实测） | 已修复：8bit AdamW + grad checkpoint + max_len 1024（见 train/README 问题 6）；备选 --use_4bit |
| DPO 后任务指标下降 | β、lr 调小；训练中监控 eval_set |
| commit 超时 working 被清 | 每 stage 一个 commit + 完成后立即下载 |
| 模板同质化过拟合 | 已用 05 净化（模板均衡） |
| 数据量充分性 | 无先验保证，靠 loss 曲线 + eval 指标 + 消融③验证 |

## 8. 执行清单（对应第 4 节 notebook 步骤）

- [ ] 上传 `data_prep/output/` 与 `train/` 为 Kaggle Dataset
- [ ] nb1：环境验证 + 配额实测（4bit 加载冒烟、双卡扣费规则）
- [ ] nb2：前台全链路冒烟（smoke.jsonl 五步跑通）
- [ ] nb3：Stage 1 commit → 下载 ckpt/stage1
- [ ] nb4：Stage 2 两分支 commit（stage2-only / stage1+stage2）→ 下载两个 adapter
- [ ] nb5：build_preference commit（2.5 万对）→ 下载 dpo_pairs.jsonl
- [ ] nb6：Stage 3 DPO commit → 下载 ckpt/stage3
- [ ] nb7：评估 4 档模型（基座/stage2/stage1+stage2/stage3）× eval_set → 指标表 + 消融图
- [ ] 更新 readme.md 成果数据、整理项目文档
