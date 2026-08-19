# 技术笔记（TECH_NOTES）

> 记录本项目开发过程中积累的技术原理、换算关系与设计决策的"为什么"。
> 与 `RUNBOOK.md`（方案）、`data_prep/README.md`（数据决策）、`train/README.md`（代码）互补——这里是**原理层**笔记。

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
| **stage3 DPO logits_fp32 补充** | chosen+rejected 各一张 + bf16/fp32 mix → **堆积 4× logit 峰值** | ⚠️ DPO 特有。代码 `stage3_dpo.py::token_logprobs` 已重构：logits 默认 dtype 为主下 log_softmax + 立刻 gather 到 (B, T-1)，跳避 (B, T, V) 全张 fp32 峰值 |

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

### 1.6 本项目后期阶段显存预算（实测 A100 40GB×4 / 7B 4bit）

```
Stage 1：peak ≈ 19.0GiB/卡（bf16 LoRA+，8bit AdamW，2048 序列）
Stage 2 SFT：peak ≈ 17.9GiB/卡（4bit QLoRA，code 默认 1024 序列）
Stage 3 DPO：peak ≈ 26.8GiB/卡（4bit QLoRA + ref 同副本，1024 序列，chosen+rejected）
```

Stage 2 烟测代码刚修改后（DDP×4 / 4bit / batch=2 / 1024 序列），peak 实测 17.9GiB/卡，
free=30.6GiB。Stage 3 DPO 代码 logits .float() 优化 后，peak 实测 26.8GiB/卡，
free=26.0GiB。**全都在 A100 40GB 预算内，可放心上全量**。

**实测来源**：以上数字均从烟测输出（`[进度] step ... peak=X.XG`）取证。所有数字都会随语料、
batch、max_len 变动，仅作设计参考——重新启动新训练时仍需以当次烟测的 peak 为准。

---

## 2. 其他技术要点速查（详细记录见各文档）

### 2.1 冒烟测试方法论（RUNBOOK.md nb2）
**只缩数据量/步数，不动超参**（lr/batch/rank/max_len 与正式一致）——冒烟验证的代码路径与正式完全一致，避免"冒烟过了、正式超参没验证"。冒烟不换模型（小数据量冒烟），同一模型同一套代码。

### 2.2 张量并行 TP vs 数据并行 DDP
- **DDP（数据并行）**：每卡完整模型副本、各看不同 batch，吞吐 ×N，每步一次梯度 all-reduce
- **TP（张量并行）**：把单层权重**切到多卡**（`nn.Linear(in, out)` 切 out 维度：4 卡各拿 1/4 输出，all-gather 汇总）
- **DDP 不减每卡显存**（每卡完整副本，固定开销不摊薄）——只有 TP/PP/ZeRO 才减单卡显存。**实测教训**：3B bf16 单卡加载后 11.95GB，DDP 双卡每卡同样 11.95GB → bf16 在 T4 上双卡也救不了，只能 4bit
- 本项目 3B+4bit 单卡 16GB 绰绰有余 → **双卡 DDP 用于提速**（代码零改动，仅改启动命令 + grad_accum 减半）

**TP 规则（`tp_plan`）**：HuggingFace 新模型类声明"如果有人要 TP 这个模型，按这个规则切"——Qwen2 的 tp_plan：
```python
{
  "layers.*.self_attn.q_proj": "colwise",   # 按输出切
  "layers.*.self_attn.k_proj": "colwise",
  "layers.*.self_attn.v_proj": "colwise",
  "layers.*.self_attn.o_proj": "rowwise",   # 按输入切（与 q/k/v 输出对应）
  "layers.*.mlp.gate_proj": "colwise",
  "layers.*.mlp.up_proj": "colwise",
  "layers.*.mlp.down_proj": "rowwise",
}
```
这**只是元数据声明**，不是必须用——accelerate 检查时若"声明了但没应用"会警告。

**accelerate 警告原理**：`check_tp_plan` 检测到模型 `tp_plan` 有规则、但 `device_map` 加载方式没应用 → 报"The following TP rules were not applied"。**真修（不抑制日志）**：清空 `model._tp_plan` 和 `model.config._tp_plan` = 告诉 accelerate "我们不打算做 TP"（项目用 DDP 不 TP，3B/7B 4bit 单卡够），accelerate 找不到规则 → 不警告。清空不影响模型功能。

**为什么我们不用 TP**：3B/7B 4bit 单卡显存足够（DDP 简单 + 容错好），TP 通信密集（每层 all-gather）需要 NVLink 等高速互联才能高效，PCIe/Kaggle 上不划算。

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

### 2.11 为什么 TP 按张量切（设计原理与取舍）

**核心动机**：单层能装下、整个模型装不下。**只**按层切不够——优化器状态本身就能超：

| 模型 | 单层最大 Linear | 优化器状态（Adam fp32） | 单卡够装吗？ |
|---|---|---|---|
| LLaMA-7B | 4096×4096 × 2B = 32MB | 7B × 8B = 56GB | A100 80GB 勉强 |
| LLaMA-70B | 8192×8192 × 2B = 128MB | 70B × 8B = **560GB** | ❌ |
| GPT-3 175B | 12288×12288 × 2B = 600MB | 175B × 8B = **1.4TB** | ❌ |

**单层不是问题，整个模型+梯度+优化器+激活才是**——70B 仅优化器就 560GB（远超 80GB A100），必须拆参数本身。

**两种切法对比**：

| | PP（按层切） | TP（按张量切） |
|---|---|---|
| 切什么 | 不同层放不同卡 | 单层权重切到多卡 |
| 设备利用率 | 1/N（其他卡等下一层）| **1**（每层所有卡同时算）|
| 通信 | 层间一次（activation，BLH/N）| 每层 all-reduce（BLH）|
| 计算 | 层间**串行**（必须按顺序）| 层内**并行**（同时算）|
| Pipeline bubble | 有（首尾空闲）| 无 |
| 拓扑要求 | 慢速互联可 | **需高速互联**（NVLink 900GB/s）|
| 适用 | 极深模型 | 宽模型（单层宽）|

**TP 三大优势**：
1. **设备利用率 = 1**（vs PP 的 1/N）：每层所有卡都工作
2. **无 pipeline bubble**：每层 all-reduce 后立即进入下一层
3. **计算/通信重叠**：TP 的 all-reduce 可与下一层 matmul 用不同 CUDA stream 并行执行，通信被计算"掩藅"（实际开销更低）

**代价（担忧是对的）**：
- 每个 Linear 要写并行版本（Megatron `ColumnParallelLinear` / `RowParallelLinear` 约 500 行 + 严格处理 q/k/v colwise ↔ o proj rowwise 的对应切分，否则 all-reduce 错位）
- 通信密集，需 NVLink/NVSwitch（PCIe 30GB/s 上 70B TP 会慢 10x）
- 框架集成复杂（accelerate TP 实验性 / NeMo / TransformerEngine）
- 调试困难（bug 表现为 loss 错 / NaN / 慢，难定位）
- **这正是 HuggingFace 引入 `tp_plan` 的原因**：模型作者声明"这些层按这个规则切"，accelerate 加载时按规则切——用户不用手写

**大模型必须 TP×PP 组合**（两者优点互补）：
```
GPT-3 175B 训练 1024 张 A100 = 16 段 PP × 每段 8-way TP × 每段 1 个 layer
LLaMA-70B 训练 2048 张 A100 = 16×16 组合
```

**本项目不用 TP 的理由**（项目选型逻辑）：
- 3B/7B 4bit 单卡装得下（3.5-4GB），不需要拆
- TP 复杂度高、调试难、对框架要求高
- Kaggle T4 **无 NVLink**（只有 PCIe），TP 通信会拖垮训练
- 项目目标："流程跑通 + 消融对比"，DDP 简单 + 容错好 + 满足需求

**面试可讲**：为什么"按张量切"而不是"按层切"——单卡装不下整个模型（优化器状态就超），但单层装得下，所以切单层权重到多卡并行算 → 设备利用率 1、无气泡、通信可重叠，代价是每层要写并行版本 + 需高速互联。大模型必须 TP×PP 组合，本项目 3B/7B 单卡够 + Kaggle 无 NVLink → 选 DDP。

### 2.12 build_preference：代码兼容多卡，**运行时强制单卡**

#### 2.12.1 事实状态

`build_preference.py` 里有完整的 DDP 同步逻辑（sharding + `.rank{i}` + barrier + rank 0 合并），
但入口处加了 hard assert：**`WORLD_SIZE > 1` 直接 RuntimeError**。

```python
if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
    raise RuntimeError(
        "[Pref] build_preference 不支持 torchrun 多卡（生成式脚本，不需 DDP）。"
        "请用 `python train/build_preference.py ...` 直接跑。"
    )
```

但 sharding 代码块仍保留。以备未来使用 vLLM / 多机推理场景复用，**本身不依赖 torchrun 多卡**。

#### 2.12.2 为什么强制单卡（踩坑历史）

某次 4 卡 DDP 启动下，错误仅能看到 `[Pref] 未找到 .rank* 文件` —— 误导使用者以为“写入失败”。
根因：4 个 rank 同时加载 7B 4bit base + adapter（×2：policy + ref）实例，总显存顶到 OOM？**3-4
卡退出** —— elastic launcher 默认 60s 超时后 **SIGTERM** 主进程，rank 0 还未写到文件就死了。

**两件本质错误**：
1. **生成式推理不需要 DDP**：`model.generate()` 不反传、不跨进程梯度同步。4 卡运行只是 “4 份完整
   模型并行生成同一批样本”，既不快、还造成 4× 显存。
2. **代码 .rank* sharding 是为错误场景设计的**：原为多卡加速准备，但 7B 4bit 在 40GB A100 上多
   卡并不会快，反而是 OOM 设计。

**正确范式**：
- 生成式任务：单卡 sequential 或 vLLM batched inference。
- 训练任务（需反向、同步梯度）才用 DDP。

#### 2.12.3 单卡运行命令

```bash
python train/build_preference.py \
  --model_path $MODEL_7B --stage2_dir $STAGE2_ADAPTER \
  --data_dir $DATA_DIR --output_dir $WORK_DIR \
  --max_pairs 50 --max_samples 50 \
  --max_new_tokens 96 --temperature 0.9 --use_4bit
```

烟测 50 对 ~5min；全量 2.5万对 预估 7-9h。

#### 2.12.4 保留 sharding 代码的价值

- **设计一致性**：本项目三个 stage 都用一套 DDP 架构、多进程同步；build_preference 选单卡，但
  代码内部保留虚拟同步逻辑，风格统一。
- **未来场景重用**：如果后续换成 vLLM 多进程推理，`.rank*` + barrier + 原子 rename 仍可直接复用，
  只需修改 `main()` 启动部分。

#### 2.12.5 三层 flush 语义（补充，面向将来 vLLM 场景）

```
Python	buffer（随个函数调用）          ❌ 不能跨进程
↓  f.close() / f.flush()
OS	page cache（跨进程可见）             ✅ 同主机其他进程可读
↓  os.fsync()
disk	物理磁盘（跨主机可见）            ✅ NFS / 共享存储跨主机可读
```

**为什么现在不需要 fsync**：同主机内存可见 → page cache 同步足够。**为防 NFS/跨主机 / 异常路径，加
`os.fsync(f.fileno())` 才是万无一失**——代码里已加（try/except 包住，败了不中断、barrier 仍同步）。

#### 2.12.6 面试可讲

**生成式与训练式脚本的区别**：
- 训练：`loss.backward()` + `optimizer.step()` → DDP 反向同步 → 多卡有价值（倍加速比）。
- 生成：`model.generate()` → 不反传 → 多卡理论上可但需合理设计（vLLM 连续批 / tensor parallel），
  而 DDP 启动只会造成 4× 显存、无加速。

**踩坑点**：“误用 DDP 启动生成脚本”不仅是浪费，还会造成 OOM → SIGTERM → 误导性错误。代码入口加
hard assert 的价值在于：错误出现时使用者能立刻定位“误用了 torchrun”，而不是被 “未找到 .rank* 文件”
误导去查 sharding 代码。

### 2.13 公共环境跑长任务：setsid + < /dev/null + disown 三件套

#### 2.13.1 问题场景

**代中课题组 GPU 节点上跑 5h+ 的训练，跑到 76% 报错**：

```
[进度] step 3475/4528 (76.7%) | loss=0.4942 | 已用 270.6min | ETA 1.4h | 显存 alloc=8.7G reserved=29.9G peak=18.9G free=30.6G
...
Received Signals.SIGHUP death signal, shutting down workers
torch.distributed.elastic.multiprocessing.api.SignalException: Process 724727 got signal: 1
```

环境上依然有 step 3000 的 checkpoint（上一轮保存）——但手动重跑**还是可能被杀**。

#### 2.13.2 什么是 SIGHUP

SIGHUP（信号编号 1）是 POSIX 设计为**通知终端会话退出**的信号：

- SSH 断线 → 内核给会话内所有进程发 SIGHUP
- 父 shell 被 `exit` / `Ctrl+D` 退出 → 同 SID 下的子进程收到 SIGHUP
- `nvidia-smi --loop` / `watch` / systemd-logind 等 watchodg 脚本**有时会批量发 SIGHUP**
- 与之对比：SIGTERM(15) 是“清求终止”、SIGKILL(9) 是“强杀”、SIGINT(2) 是“Ctrl+C”

**信号编号表**：

| 编号 | 名称 | 默认行为 | 触发场景 |
|---|---|---|---|
| 1 | SIGHUP | 终止 | 终端退出 |
| 2 | SIGINT | 终止 | Ctrl+C |
| 9 | SIGKILL | 终止（不可捕） | OOM Killer / 手动 kill -9 |
| 15 | SIGTERM | 终止 | kill 默认、systemd shutdown |

**教训**：报错里只看到 “got signal: 1”，要立刻想到 “SIGHUP”，检查是不是会话退出。

#### 2.13.3 三件套是什么

```
cd $WORK_DIR
mkdir -p logs
LOGFILE=$WORK_DIR/logs/stage2_s1_$(date +%Y%m%d_%H%M).log

setsid torchrun --nproc_per_node=4 $CODE_DIR/stage2_sft.py \
  ... \
  > $LOGFILE 2>&1 < /dev/null &

PID=$!
disown
echo "[$(date)] torchrun 启动, PID=$PID, 日志=$LOGFILE"
```

**`setsid`——开新会话**：进程进全新的 session ID 与 process group ID，脱离原控制终端。
父 shell 被退出 / SSH 断线 时，内核**仅会**给同 SID 下的进程发 SIGHUP。setsid 启动的进程不在你 shell 的 SID 下 → SIGHUP 传不过去。

**`< /dev/null`——切断 stdin**：setsid 启动的进程 stdin 仍可能指向原 tty。后台进程读 stdin 可能阻塞或报 EIO。重定向到 /dev/null 后所有 read() 立刻返回 EOF。**防御性写法**，torchrun 本体不需要，但防止 debug traceback 卡住。

**`disown`——从 bash jobs 表移除**：bash 启动的后台进程会记在内部 jobs 表里，bash 退出时会给每个 job 发 SIGHUP（除非已 setsid/nohup）。disown 从表中抹除，bash exit 时不再尝试发信号。**严格说 setsid 已经免疫 SIGHUP**，disown 是 belt + suspenders，同时避免 fg/bg 误撞。

**`echo "PID: $!"`**：\$! 是 bash 最后一个后台进程 PID，记下供诊断（消失后可用 `dmesg` 查、或者 `ps -ef | grep <PID>`）。

#### 2.13.4 三者对比

| 工具 | 防的是谁 | 作用层级 | 不防什么 |
|---|---|---|---|
| `setsid` | 父 shell 退出 / SSH 断 → SIGHUP | 进程组 + 会话 | 手动 kill、OOM Killer |
| `< /dev/null` | 进程卡在 stdin 读 | 文件描述符 | 其他信号 |
| `disown` | bash exit 时给 job 发 SIGHUP | bash 内部 jobs 表 | 同 setsid |
| `echo "PID: $!"` | 无 | 仅记录 | — |

**核心是 setsid**。其他三个是防御性补丁。

#### 2.13.5 为什不使用 nohup / tmux / screen

- `nohup`：只忽略 SIGHUP，**不能脱离原 SID**。父 shell 退出后其他信号还能传进去；多个窗口开 nohup 命令之间互不隔离。不如 setsid 干净。
- `tmux` / `screen`：重量级，需要进入交互界面。项目使用 setsid + & 更轻量。但是仅个人调试时 tmux **是最容易调试的**（能随时 reattach 看进度）。**二者不互斥**：可以在 tmux 里执行 setsid 命令。

#### 2.13.6 代码侧补丁：信号转发

公共环境的 watchodg 不会被 setsid 挡住（不是从同一个 SID 发的）。代码侧同时加一层防御——在 `common.py::setup_env()` 里转发 SIGHUP/SIGINT/SIGTERM 为 SIGTERM，让 transformers Trainer 的 `_train_signal_handler` 走 on_train_end 保存 checkpoint：

```python
import signal as _signal
def _forward_signal_to_sigterm(signum, frame):
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[signal] 收到信号 {signum}({_signal.Signals(signum).name})，转发为 SIGTERM 触发 Trainer 优雅退出", flush=True)
    _signal.signal(_signal.SIGTERM, _signal.SIG_DFL)
    os.kill(os.getpid(), _signal.SIGTERM)

for _sig in (_signal.SIGHUP, _signal.SIGINT, _signal.SIGTERM):
    try:
        _signal.signal(_sig, _forward_signal_to_sigterm)
    except (ValueError, OSError):
        pass
```

**为什么不是 `SIG_IGN`**：忽略信号会让进程失去保存 checkpoint 的机会，**会被下一个 SIGKILL 强杀时丢全部进度**。SIGTERM 会让 transformers Trainer 走完整退出逻辑——保存 adapter → 退出 → 下次启动从 checkpoint-* 恢复。

#### 2.13.7 重入与续跑

Trainer 默认 `trainer.train()` 不传参时**自动从 `--output_dir` 里最新的 `checkpoint-*` 续跑**。只要：

1. 不动 `--output_dir` 路径
2. 不删 `checkpoint-*` 目录
3. Trainer 见到了 checkpoint-* 则 resume；见不到则从 step 1 重跑

所以重跑同一个命令**不需要传任何额外参数**。**别忘了 trainer.save_steps=500（默认）+ save_total_limit=2：坏了只会丢最多 500 步（≈ 30-40 分钟）的进度**，而不是 5h。

【面试可讲】公共环境长任务三件套：setsid（开新 session）+ < /dev/null（断 stdin）+ disown（从 bash jobs 移除）；代码侧信号转发（SIGHUP → SIGTERM）让 Trainer 优雅退出保存 checkpoint。**两者叠加 = 损失最多 500 步进度**，而不是全部丢失。

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
- 开 checkpoint（model + ref 都调 `enable_grad_checkpointing`）：激活降一个量级 → **实测 peak ≈26.8GiB/卡** ✓

**为什么 DPO 还需多一步 logits 优化**：除了“激活线性减倍”，logits在 loss 计算中会临时升为 fp32
（`(B, T, V)` ≈ 1.5GiB × 2 序列 = ≈3GiB·fp32 中间表 × 2 模型 = **8GiB** 峰值）。
`log_softmax(bf16_logits)` 原生支持 bf16 → 立刻 gather 到 (B, T-1) fp32 token_logp → 避开该峰值。
**这是 stage3 已应用的优化，面试时可重点讲**。“优化不是加项减项，是避开隐性峰。”

`model.gradient_checkpointing_enable()`（Trainer 模型通用 API，DPO 需对 `model` 和 `ref_model` 都调一次）就是告诉模型“前向不全存激活、反向重算”。

### 3.3 面试可讲的一句话

- **DPO 双模型**：同一权重的两份副本（一份训练、一份冻结作参考基线），实现"相对偏好"学习；4 次前向导致激活 ×4
- **gradient checkpointing**：用时间换显存，前向不全存激活、反向时重算，激活降约一个量级、训练慢约 33%
- **两者结合**：DPO 激活 ×4 必须开 checkpoint，否则大模型 DPO 必爆显存
