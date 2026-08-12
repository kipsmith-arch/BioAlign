# 技术笔记（TECH_NOTES）

> 记录本项目开发过程中积累的技术原理、换算关系与设计决策的"为什么"。
> 与 `REPRODUCTION_PLAN.md`（方案）、`data_prep/README.md`（数据决策）、`train/README.md`（代码）互补——这里是**原理层**笔记。

---

## 1. 训练显存预算与换算（OOM 排查核心）

### 1.1 显存组成与可信度分级

| 项 | 公式 | 可信度 |
|---|---|---|
| 模型权重（bf16） | 参数量 × 2B/参数 | ✅ **精确**（参数量官方公布） |
| 模型权重（4bit QLoRA） | 参数量 × ~0.5B/参数 | ✅ 精确（nf4 量化后） |
| 梯度 | 可训练参数 × 2B/参数（bf16） | ✅ 公式精确，输入是估算 |
| 优化器状态（fp32 AdamW） | 可训练参数 × 8B/参数 | ✅ 公式精确，输入是估算 |
| 优化器状态（8bit AdamW） | 可训练参数 × ~1B/参数 | ✅ bnb 量化后 |
| **激活（激活函数中间量）** | **无简洁公式** | ⚠️ **纯经验量级，以实测为准** |
| **loss logits 峰值** | **batch × seq × vocab × 4B（fp32 计算 CE）** | ⚠️ **易被忽略的峰值源**（实测：batch4×1024×151936×4B≈2.49GB） |

**关键原则：权重/优化器可精确预算；激活只能给量级；还要注意 loss 计算在 vocab 维度上的 logits 峰值张量**——最终以显存监控 / OOM 报错反推为准。

**实测教训（3B Stage 1 两次 OOM）**：max_len 1024 + 8bit 优化器 + checkpoint 后仍超预算，反推根因是 loss 的 logits 峰值（batch 4 × vocab 151936 × 4B ≈ 2.49GB）——**batch 是这类峰值张量的直接缩放因子**，显存不足时优先降 batch（1），用 grad_accum 补 global batch。**

### 1.2 优化器状态为什么是 8B/参数（m + v）

Adam/AdamW（Kingma & Ba 2015）对**每个参数**维护两个 fp32 状态：

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t      ← 一阶矩（momentum），梯度方向的指数滑动平均
v_t = β₂·v_{t-1} + (1-β₂)·g_t²     ← 二阶矩，梯度平方的指数滑动平均（自适应步长）

更新：θ ← θ − lr · m̂ / (√v̂ + ε)
```

- **m**：动量，抑制震荡（β₁=0.9）；**v**：梯度大小感知，陡峭维度步长小（β₂=0.999）
- 两者必须 **fp32**（即使权重是 bf16/4bit）——指数滑动平均对精度敏感
- 每参数 8B = m(4B) + v(4B)

**8bit AdamW 为什么省显存**：把 m/v 量化为 8bit（1B/状态），配合分块动态量化 + 误差补偿，8B → ~1B/参数，精度损失很小。

### 1.3 可训练参数怎么确定（LoRA）

**公式**：每个被替换的线性层 `W(d_out×d_in)`，LoRA 分解 `W' = W + B·A`：

```
A: r × d_in，B: d_out × r   → 新增参数 = r × (d_in + d_out)
可训练参数 = Σ_{target 模块} r×(d_in+d_out) + Σ_{RMSNorm} hidden
```

**验证**：0.5B 实测 r=16 → 8,798,208，r=64 → 35,236,736，**正好 4 倍**（∝ r）✓

**实际取值不用手算**：`model.print_trainable_parameters()`（peft 内置，内部是 `sum(p.numel() for p in model.parameters() if p.requires_grad)`）。

⚠️ **外推注意**：0.5B 实测 trainable 占比 6.66%（r=64），3B 实测 **3.74%**（120M/3.21B）——**跨模型规模外推比例不可靠**（层数/维度/FFN 比例不同），精确值必须在目标模型上跑 `print_trainable_parameters()`。

### 1.4 激活为什么难算（gradient checkpointing 原理）

激活 = 前向过程中需要**保留给反向传播**的中间张量（QKV 投影输出、注意力矩阵、FFN 中间态等），量级 ∝ **batch × seq_len × hidden × 层数**（线性关系，比绝对值可靠）。

**Gradient Checkpointing（重计算）**：前向时不保存激活，反向时**重新前向一次**算出激活——以约 1 倍的额外前向时间，换取激活显存降约一个数量级（只存每层输入）。`prepare_model_for_kbit_training` 会自动开启；bf16 模式下需显式 `model.gradient_checkpointing_enable()`。

### 1.5 用 OOM 报错反推验证预算（实战方法）

```
已用显存 − 权重 − 梯度 − 优化器 − 框架开销 ≈ 激活
```

实例（3B Stage 1 OOM，T4 ~15GB）：14.34 GiB − 6.1（权重）− 0.4（梯度）− 1.6（优化器）− ~0.5（框架）≈ **激活 5.7GB**——落在经验区间，量级吻合。

**修复的有效性来自确定性大头的压缩**（不依赖激活估算精度）：
- 8bit AdamW：优化器 1.6GB → 0.3GB（确定性）
- max_len 2048→1024、batch 4→2：激活 ∝ batch×seq，约降到 1/4（线性关系）
- grad checkpoint：激活再降一个量级

### 1.6 本项目 Stage 1 的显存预算结论（T4 单卡 14.56GB 实测）

```
4bit QLoRA：权重 ~2GB + 激活/优化器 → 峰值 ~5-6GB → 单卡可跑 ✅（Stage 1 定稿）
bf16 3B：单卡加载后 11.95GB、训练峰值 13GB+ → 单卡/DDP 双卡均装不下 ❌
（DDP 每卡完整副本、固定开销不摊薄——bf16 双卡同样 ~13.5GB/卡，仍超限）
```

---

## 2. 其他技术要点速查（详细记录见各文档）

### 2.1 冒烟测试方法论（REPRODUCTION_PLAN.md nb2）
**只缩数据量/步数，不动超参**（lr/batch/rank/max_len 与正式一致）——冒烟验证的代码路径与正式完全一致，避免"冒烟过了、正式超参没验证"。冒烟不换模型（小数据量冒烟），同一模型同一套代码。

### 2.2 数据并行 vs 模型并行
- 显存装得下 → **数据并行（DDP）**：每卡完整模型副本、各看不同 batch，吞吐 ×N，每步一次梯度 all-reduce
- 显存装不下 → **模型并行**：模型拆开（TP 张量切分 / PP 流水线 / ZeRO 分片）
- **DDP 不减每卡显存**（每卡完整副本，固定开销不摊薄）——只有模型并行/ZeRO 才减单卡显存。**实测教训**：3B bf16 单卡加载后 11.95GB，DDP 双卡每卡同样 11.95GB，训练峰值仍超 14.56GB → bf16 在 T4 上双卡也救不了，只能 4bit
- 本项目 3B+4bit 单卡 16GB 绰绰有余 → **双卡 DDP 用于提速**（Kaggle 实测双卡扣 1h/1h，白赚速度），代码零改动（LOCAL_RANK 决定 device_map），仅改启动命令 + grad_accum 减半

### 2.3 数据量充分性：没有先验量化依据（data_prep/README.md 面试问答）
- "token 数 > 参数数 所以够"的论据**不成立**（与 Chinchilla 方向相反，且 Chinchilla 仅适用于从零预训练）
- 正确验证方式：loss 曲线（欠拟合/过拟合）+ eval 指标 vs 基线 + 数据量消融

### 2.4 DPO 偏好数据语义（build_preference.py 注释）
- DPO 学的应是"同一任务分布下的质量偏好"，不是"会不会答"
- rejected 用 **stage2 模型采样**（on-policy、同分布），chosen = 标准答案
- 不用基座（胡编）——偏离 DPO 本意；区分度由 temperature 采样保证
- **为什么自实现 DPO loss（不依赖 trl）**：trl 0.29 依赖 `torch.distributed.fsdp.FSDPModule`（torch 2.6+ 才有），与 torch 2.5 不兼容——为规避"版本地狱"（trl/transformers/torch 三方强绑定），DPO loss 自己实现（~30 行：对 chosen/rejected 各算对数概率差 + `-log σ(β·Δ)`），并保留 π_ref 冻结副本。副作用是代码更透明、面试可讲公式

### 2.5 模板同质化（data_prep/README.md 3.5）
模板化数据集（每 task 50~200 个模板骨架）→ 模板感知均衡采样防"模板腔"过拟合；附带发现 stage2 标签是非标准闭合（`<rna>...<rna>` 无斜杠）。

### 2.6 从 adapter 继续训练必须显式启用 requires_grad（踩坑高频）
**现象**：`PeftModel.from_pretrained` 加载 adapter 后，所有参数 `requires_grad=False`，直接训练报 "loss does not require grad"。
**原理**：`from_pretrained` 恢复的是**推理状态**——LoRA 参数虽可训练，但加载时 `requires_grad` 默认 False；必须显式开启：
```python
model = PeftModel.from_pretrained(base, adapter_dir)
for n, p in model.named_parameters():
    if "lora" in n:
        p.requires_grad_(True)
```
**适用场景**：任何"从已有 adapter 继续训练"的流程（本项目 stage2 `--resume_adapter`、stage3 DPO 的模型与 π_ref 都用到）。

### 2.7 Trainer 训练的基础坑速查（版本相关，详见 train/README）
- **默认 collator 不 padding**：长度不一的样本必须显式传 `DataCollatorForSeq2Seq`（labels 用 `label_pad_token_id=-100`），否则 batch 拼不齐报维度错误
- **新版 transformers 属性名变化**：`is_gradient_checkpointing`（旧版 `gradient_checkpointing`）——诊断代码要兼容两者
- **训练进度可见性（commit 模式下 tqdm 不可用）**：三个训练脚本注册 `ProgressCallback`（common.py）——每 `logging_steps`（25）打印 `[进度] step/总步数 百分比 loss 已用时间 ETA 显存`，仅 rank 0 打印避免 DDP 重复；`setup_env()` 设置 `PYDEVD_DISABLE_FILE_VALIDATION=1` + `TOKENIZERS_PARALLELISM=false` 消除 Kaggle 重复警告
- `Dataset.from_list` 不接受 generator（需转 list）；新版 transformers `compute_loss` 签名多了 `num_items_in_batch`——这些属 API 版本变化，踩坑记录留在 train/README。

### 2.8 max_len 的两层语义 + DeepSpeed ZeRO（显存主题延伸）
- **max_len 语义**：Stage 1 packing 下是**块大小**（不截断，仅丢弃尾部碎片，块越小碎片比例越低）；Stage 2/DPO 下是**序列长度上限**（超长样本尾部会被截断）。生物长序列任务中"序列完整性"优先 → 4bit 省出的显存应让给 max_len（本项目 1024→2048）
- **DeepSpeed ZeRO**：分片训练状态消除冗余（ZeRO-1 优化器 / ZeRO-2 +梯度 / ZeRO-3 +参数，通信开销递增）。对 3B：优化器+梯度合计仅 ~1.3GB，ZeRO-2 双卡省 ~0.6GB/卡——救不了 bf16（需省 4-6GB）；ZeRO-3 分片参数可省 ~3GB 但通信大、与 4bit 组合复杂，对 3B 不值得

### 2.9 训练时间预算：必须基于 time_per_step 实测
**核心方法**（取代所有凭印象估算）：
1. 冒烟跑 N 步（≥ logging_steps），看 summary 的 `train_runtime` / `train_steps_per_second`
2. 算 `time_per_step = train_runtime / N`
3. 正式总时间 = `总步数 × time_per_step`
4. 总步数 = `总token / (max_len × per_device_batch × 双卡数)`

**实测教训（不同硬件严重错估）**：
- T4 16GB：3B bf16 装不下（11.94GB 加载后已满）→ 只能 4bit
- P100 / T4 4bit 0.5B 2048 单卡：~15s/步（极慢，4bit dequant 在弱 TC 上开销大）
- **A100 PCIe 40GB 3B 4bit 2048 双卡：~3.2s/步**（用户实测）
- A100 40GB 可装 bf16 3B（之前 T4 上的所有 4bit 妥协在 A100 上不再必要）

**对正式 Stage 1 的预算（A100 双卡 3.2s/步）**：
- 23.6 万条 + 1024 = ~6900 步 × 3.2s ≈ **6.1 小时**（仍超 12h commit → 抽样）
- 10 万条 + 1024 = ~2900 步 ≈ **2.6 小时** ✓
- 5000 条 + 1024 = ~1450 步 ≈ **1.3 小时**（消融够用）

**根本教训**：所有"几分钟/几小时"的预算必须从实测 time_per_step 推算，不能凭印象凭 GPU 规格拍脑袋。

### 2.10 基于数据选超参（反对拍脑袋）
**原则**：任何影响数据利用的关键超参（max_len、batch、cap 比例、per-task 配额）都应**先跑数据分布脚本再定**，而不是凭印象。

**本项目实践（`06_token_len_stats.py`）**：
- 实测 SFT 1 万条 token 长度：p50=228、p95=614、max=1544
- max_len 覆盖表：768 覆盖 99.5%、1024 覆盖 99.9%（差异 0.4%，但 1024 多花 25% 显存）
- **选型**基于数据：Stage 1/2 用 1024，Stage 3 用 768（DPO 双模型激活×4 显存压力）

**反对拍脑袋的反面案例**：
- 我曾"显存不够就改 max_len=768"，这是拍脑袋（恰好数据上 768 合理，但论证方式错误）
- 之前所有"time_per_step 估算"也是拍脑袋——A100 实测 3.2s/步才纠正了 T4 外推 3-6h 的错误

**面试可讲**：选超参的标准是"先看数据分布 → 选型 → 实测验证"，不是凭直觉或硬件规格。

---

## 3. DPO 与 gradient checkpointing 原理（Stage 3 必备知识）

### 3.1 DPO 为什么需要"两个模型"

**损失函数**（Rafailov et al. 2023）：

```
L_DPO = -E[ log σ( β · ( (log π(y_c|x) − log π_ref(y_c|x)) − (log π(y_r|x) − log π_ref(y_r|x)) ) ) ]
```

四个对数似然项：**π** = 当前训练模型，**π_ref** = 参考模型，y_c/y_r = chosen/rejected。

**为什么要 π_ref（关键）**：
- 没有 ref：模型只让 chosen 似然增大、rejected 减小 → 退化成 SFT/负样本训练，**丧失偏好的相对性**
- 有 ref：模型学的是 **chosen 相对 ref 增长** vs **rejected 相对 ref 增长** 的差值 → "相对偏好"

**"两个模型"实际是什么**：
- **同一权重的两份副本**：model 和 ref_model 初始权重相同（都从 stage2 adapter 加载）
- model 训练中更新（LoRA 梯度），ref 训练中**完全冻结**（`requires_grad_(False)`）
- ref 锁定"训练前的偏好基线"，DPO 学的是相对这个基线的差值

**DPO 比 SFT 多做 3 次前向**：
```
SFT: 1× 前向（输入 → chosen 答案）
DPO: 4× 前向（π×chosen + π×rejected + π_ref×chosen + π_ref×rejected）
```
- **时间代价** ~2x（单次前向 + logp + backward 翻倍）
- **显存代价** ~2-3x（激活存两份，chosen/rejected 各一份 × 2 模型 = 4× 激活峰值）
- **本项目实测**：7B 4bit 1024 DPO 不开 checkpoint OOM（~27-30GB/卡），开 checkpoint 后 ~18-22GB/卡 ✓

**本项目实现**（`stage3_dpo.py`）：
```python
model = PeftModel.from_pretrained(base, stage2_dir)   # 训练副本
ref_model = PeftModel.from_pretrained(ref_base, stage2_dir)  # 冻结副本
for p in ref_model.parameters():
    p.requires_grad_(False)   # ref 全冻结
```

### 3.2 gradient checkpointing 是什么、为什么有效

**问题**：反向传播需要前向时**每层保存的激活张量**。大模型 + 长序列 → 激活 O(层数 × seq × hidden) 巨大。

**解决思路**：**前向时不全存，反向时重算**：

```
普通前向:   layer1→[act1]→layer2→[act1,act2]→...→layerN→[act1..N]  → 反向用这些
checkpoint:  layer1→[act1]→丢,layer2→[act2]→丢,...→layerN→[actN]  → 反向时从 actN 重新前向算 act1..N-1
```

**代价与收益**：
| 维度 | 普通 | checkpoint |
|---|---|---|
| 显存（激活） | 存 N 层 | **只存 1 层**（降一个量级） |
| 时间 | 1× 前向 | **~1.33× 前向**（反向时多算 1 次） |

**为什么 DPO 特别需要**：
- 普通 SFT：1 序列前向，激活 ×1
- DPO：chosen + rejected 两个序列 × 2 模型 = **激活 ×4**
- 不开 checkpoint：7B 4bit 1024 DPO 双模型激活 = 单 SFT 的 4 倍，~27-30GB/卡 → OOM
- 开 checkpoint：激活降一个量级 → ~18-22GB/卡 ✓

`model.gradient_checkpointing_enable()`（Trainer 模型通用 API，DPO 需对 `model` 和 `ref_model` 都调一次）就是告诉模型"前向不全存激活、反向重算"。

### 3.3 面试可讲的一句话

- **DPO 双模型**：同一权重的两份副本（一份训练、一份冻结作参考基线），实现"相对偏好"学习；4 次前向导致激活 ×4
- **gradient checkpointing**：用时间换显存，前向不全存激活、反向时重算，激活降约一个量级、训练慢约 33%
- **两者结合**：DPO 激活 ×4 必须开 checkpoint，否则大模型 DPO 必爆显存
