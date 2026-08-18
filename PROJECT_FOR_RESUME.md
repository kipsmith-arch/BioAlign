# BioAlign 项目简历文档

> 本文档基于项目实际代码与文档整理而成，用于在面对不同招聘 JD 时快速匹配简历用词。
> 内容按"JD 关注维度"组织，每段都给出"可抽取到简历的表述素材 + 必要的解释"。

---

## 0. 一句话项目定位（多版本可选）

按目标岗位风格选用其中一句：

| 风格 | 一句话描述 |
|---|---|
| **算法/AI 工程** | 基于 Qwen2.5-7B-Instruct，完整实现"领域继续预训练 → QLoRA SFT → DPO 偏好对齐"三阶段大模型后训练流水线，使通用模型具备 DNA/RNA/蛋白/多分子生物序列的多任务理解与回答能力。 |
| **NLP/LLM 工程师** | 在 A100×4 上以 Qwen2.5-7B-Instruct 为基座，从数据、训练、对齐、评估全链路自建生物医学 LLM 后训练流水线；自实现 DPO loss、4-bit QLoRA、LoRA+、三路防泄漏数据划分与模板均衡净化。 |
| **ML Infra / 训练系统** | 构建可复现、可恢复的多阶段 LLM 后训练系统：处理数据隔离、显存预算与 OOM 调优、DDP/单卡混用、checkpoint 续跑、SIGHUP 免疫与信号优雅退出。 |
| **数据/ML 工程师** | 从源头做数据治理（330 万 → 28.98 万净化、三路严格不相交、模板同质化诊断、token 长度驱动 max_len 选型），为后训练三阶段提供高质量、零泄漏的训练/偏好/评估数据。 |

---

## 1. 项目基础信息（任何 JD 都能用）

### 1.1 标题与定位

- **项目名**：BioAlign —— 基于 QLoRA 与 DPO 的生物医学大模型后训练流水线
- **目标**：让通用基座 LLM 获得多组学（DNA / RNA / 蛋白 / 多分子）生物序列的理解与任务回答能力，并通过偏好对齐提升回答质量
- **基座**：Qwen2.5-7B-Instruct（迁移前为 0.5B / 3B，本地与 Kaggle T4 16GB 上完整跑通三阶段作为指标）
- **硬件**：A100 40GB ×4 / 单卡 T4 16GB / 本地 RTX 4060 8GB
- **周期**：3 阶段全量训练 + 评估共 9 晚 / 34–40h
- **代码量**：训练代码 6 个脚本（~1000 行）+ 数据处理 6 个脚本（~600 行）+ 评估沿用论文官方协议
- **论文基线**：arXiv:2412.19191（EMNLP 2025 Findings），仓库 https://github.com/hhnqqq/Biology-Instructions
- **数据来源**：Biology-Instructions 论文官方发布（指令数据）+ NCBI GRCh38 / RNAcentral / UniProt Swiss-Prot（序列）

### 1.2 技术栈关键词（按岗位提取）

| 关键词族 | 具体技术 |
|---|---|
| **LLM 框架** | transformers、peft、bitsandbytes、accelerate、datasets |
| **训练技术** | LoRA、QLoRA（4-bit）、LoRA+（B 矩阵学习率 = A × 4）、packing、gradient checkpointing、assistant-only loss |
| **对齐技术** | DPO（自实现 loss，不依赖 trl）、on-policy rejection sampling（rejected = stage2 采样）、β 温度参数 |
| **训练效率** | 8bit AdamW（bitsandbytes）、DDP、gradient checkpointing、token 长度驱动的 max_len 选型 |
| **分布式与系统** | torchrun、torchrun + setsid/disown/`< /dev/null` 长任务方案、DDP / 单卡脚本混用、Trainer checkpoint resume、SIGHUP → SIGTERM 信号优雅转发 |
| **数据工程** | 分层三路不相交划分、模板感知均衡采样、完全去重、固定种子可复现流水线、蓄水池抽样 |
| **评估** | 沿用论文官方协议：MCC / PCC / R² / Spearman / Acc / AUC / Fmax / mixed_score，4 档模型 × 1.89 万样本全量 |

### 1.3 三阶段流水线一览（叙述时按这张表展开）

| 阶段 | 任务 | 方法 | 数据量 | 显存峰值 |
|---|---|---|---|---|
| ① 领域继续预训练 | 让模型"认识"生物序列 | bf16 LoRA+（packing + next-token） | 23.6 万条（GRCh38 + RNAcentral + Swiss-Prot） | ~19 GiB/卡 |
| ② PEFT 指令微调 | 让模型"回答"21 类生物学任务 | 4-bit QLoRA SFT（assistant-only loss） | 28.98 万条（Biology-Instructions 净化后） | ~17.9 GiB/卡 |
| ③ DPO 偏好对齐 | 让模型"答得更好" | 自实现 DPO loss，π_ref 为 SFT 模型冻结副本 | 2.5 万对（自构 on-policy 偏好） | ~26.8 GiB/卡 |

---

## 2. 算法 / 模型设计能力（NLP / LLM 算法岗重点）

### 2.1 自实现 DPO（核心亮点，强烈建议在简历和面试中讲清）

**做了什么**：在 `train/stage3_dpo.py` 用约 30 行代码自实现完整 DPO 损失（不依赖 trl）。

**公式**：
```
log_ratio_c = log π(y_c|x) − log π_ref(y_c|x)
log_ratio_r = log π(y_r|x) − log π_ref(y_r|x)
L_DPO = −E[ log σ( β · (log_ratio_c − log_ratio_r) ) ]
```

**为什么自实现（面试故事）**：
- trl 0.29 依赖 `torch.distributed.fsdp.FSDPModule`（torch 2.6+ 才有），与 torch 2.5 不兼容
- 为规避"trl / transformers / torch 三方版本强绑定"的版本地狱，自己实现 loss
- 副作用：代码透明、参数可控、可在公式层面调试（logits dtype、π_ref 冻结、token mask 等等）

**实现要点（面试可讲 5 个细节）**：
1. **π 与 π_ref 是同一初始权重的两份副本**，π 训练更新，π_ref 全冻结（`requires_grad_(False)` + `eval()`）
2. **chosen / rejected 各做一次前向**，加上 ref 共 4 次前向 → 激活 ×4，显存压力大
3. **仅对 assistant 部分 token 求对数概率**，prompt 部分 label 置 -100 屏蔽
4. **关键显存峰值优化**：`token_logprobs` 函数避免 `logits.float()` 升为 (B, T, V) 全张 fp32——直接 `F.log_softmax(bf16_logits)`（底层 kernel 原生支持 bf16）→ 立刻 `gather` 到 (B, T-1) fp32 的 token_logp，跳过 V 维 fp32 中间表（避免 ~1.5 GB/张 × 4 = 6 GB 的隐性峰值）
5. **β 温度参数可配**（默认 0.1），过大/过小都会影响对齐质量

**配套构建偏好的方法论（DPO 语义）**：
- chosen = 标准答案（论文标注的 ground truth）
- rejected = **Stage 2 模型 on-policy 采样**（temperature=0.9）
- **为什么不用基座生成 rejected**：基座不会答，与 chosen 不在同一分布，DPO 学的是"会不会答"而非"质量偏好"——偏离 DPO 本意
- 区分度由 temperature 采样保证（同一 prompt 多次采样），过滤掉 chosen==rejected / 输出为空的样本

### 2.2 QLoRA + LoRA+ 的训练精度策略（算法岗可深入）

- **Stage 1（领域 CPT）** 用 **bf16 LoRA+**：
  - LoRA rank=64，B 矩阵学习率 = A × 4（论文 LoRA+ 配方 scaler=4）
  - 同时训练 RMSNorm（`modules_to_save=["q_norm","k_norm"]` 或自定义 norm target）
  - 选 bf16 而非 4bit：领域 CPT 对新知识注入精度敏感，4bit 会损失信息
  - 用了 **packing**：把短序列拼成 max_len 块再训练，消除 padding 浪费
- **Stage 2 / Stage 3** 用 **4-bit QLoRA**：
  - nf4 量化基座 + LoRA adapter，权重从 ~14GB 降到 ~3.5GB
  - 配合 `prepare_model_for_kbit_training` + `gradient_checkpointing_enable(use_reentrant=False)`
  - 显存省下让给更长 max_len（1024 vs 768，DPO 双模型场景）

### 2.3 数据处理中的算法设计

- **三路严格不相交划分**：330 万条按 task 分层 → train_pool / dpo_source / eval_set，行号唯一物理保证零重叠
- **完整 input+label 精确匹配**验证泄漏，重叠为 0（源数据本身 1.9% 重复样本是同源问题，已在训练集内去重）
- **模板感知均衡采样**：正则 `<(dna|rna|protein)>[A-Za-z]+<(dna|rna|protein)>` 提取骨架 → 每 task 内对每模板 cap = max(50, ceil(avg × mult))，mult=1.0；效果：Top5 模板占比 2.0% → 1.3%
- **发现 stage2 数据 bug**：序列标签是非标准闭合 `<rna>...<rna>`（无斜杠），已在正则中适配
- **token 长度统计驱动 max_len 选型**（拍脑袋的反模式 vs 数据驱动）：
  - 1 万条 SFT 样本 + Qwen2.5 tokenizer + ChatML 模板实测：p50=228, p95=614, max=1544
  - max_len 覆盖表：768 覆盖 99.5% / 1024 覆盖 99.9% / 差异 0.4% 但 1024 多花 25% 显存
  - Stage 1/2 用 1024，Stage 3 用 768（DPO 双模型×双序列激活×4，trade-off 合理）

### 2.4 训练目标与训练策略

- **assistant-only loss**：Stage 2 / DPO 都用 ChatML 模板拼接 system+user+assistant，仅 assistant 部分 label 非 -100，避免模型学会"复述问题"
- **packing**：Stage 1 把短序列拼成固定块，节省 padding；块大小就是 max_len
- **bf16 LoRA+ vs 4-bit QLoRA 的精度策略**：CPT 选 bf16 保精度，SFT/DPO 选 4bit 省显存让给 max_len
- **数据并入 stage3 高质量长答案**：把 8002 条 GPT-4o-mini 精修推理型长答案合并进 SFT 训练集，缓解 stage2 平均答案长度仅 17 token 的长尾欠覆盖问题

---

## 3. 工程 / 系统能力（ML Infra / 训练系统岗重点）

### 3.1 显存预算与排查（OOM 排查经典案例，强烈建议在面试中讲清）

**显存组成的可信度分级**（能区分精确 vs 经验值是工程能力的体现）：
- 精确：权重 / 梯度 / 优化器（公式精确）
- 经验：激活（`O(batch × seq_len × hidden × 层数)`，量级可靠）
- 易被忽略：**loss 计算的 logits 峰值**（`batch × seq × vocab × 4B` fp32，7B 模型 vocab=152064，batch4 × 1024 × 152064 × 4B ≈ 2.49 GB）

**AdamW 优化器状态 8B/参数的来源**（面试可问）：
- m（动量，β₁=0.9）4B + v（二阶矩，β₂=0.999）4B = 8B
- 必须 fp32（即使权重是 bf16/4bit，指数滑动平均对精度敏感）
- 8bit AdamW：bnb 动态分块量化到 8bit，约 1B/参数，精度损失很小

**3B Stage 1 在 T4 上三次 OOM 收敛过程**（故事完整，体现系统调优能力）：
1. max_len 2048 + fp32 AdamW → 超预算 → 8bit AdamW + grad checkpoint + max_len 1024
2. 仍 OOM → 根因 loss 的 logits 峰值 → batch 4 → 1
3. 仍 OOM → **根因是 bf16 权重本身**（加载 11.95GB，单卡 14.56GB 装不下训练峰值 13.13GB）→ **Stage 1 默认改用 4-bit QLoRA**

**最终 7B / A100×4 实测显存峰值**（实测值，看监控的 `peak` 字段，不要看 `alloc`）：
- Stage 1：~19.0 GiB/卡
- Stage 2 SFT：~17.9 GiB/卡
- Stage 3 DPO：~26.8 GiB/卡（含优化）

### 3.2 长任务稳定性与 SIGHUP 免疫（强烈建议在简历中点名）

**问题场景**：课题组公共 GPU 节点上跑 5h+ 训练，跑到 76% 报错 `Received Signals.SIGHUP death signal` —— SSH 断线 / 父 shell 退出 / nvidia-smi watchdog 周期性发 SIGHUP。

**SIGHUP 是什么**（POSIX 信号编号 1）：终端会话退出通知；与 SIGTERM(15)/SIGKILL(9)/SIGINT(2) 区分。报错只看到 "got signal: 1" 要立刻想到 SIGHUP。

**三件套解决方案**（与 setsid 防御层级对应）：
| 工具 | 防的是谁 | 作用层级 |
|---|---|---|
| `setsid torchrun ...` | 父 shell 退出 / SSH 断线 → SIGHUP | 开新 SID，脱离原控制终端 |
| `< /dev/null` | 进程卡 stdin 读 | 文件描述符层 |
| `disown` | bash exit 时给 job 发 SIGHUP | bash 内部 jobs 表层 |
| **代码侧 SIGHUP → SIGTERM 转发** | watchdog 等绕过 setsid 的信号 | 注册 Python signal handler，让 Trainer 走 on_train_end 优雅保存 checkpoint |

**代码侧补丁**（`common.py::setup_env()`）：
```python
import signal as _signal
def _forward_signal_to_sigterm(signum, frame):
    _signal.signal(_signal.SIGTERM, _signal.SIG_DFL)
    os.kill(os.getpid(), _signal.SIGTERM)

for _sig in (_signal.SIGHUP, _signal.SIGINT, _signal.SIGTERM):
    try:
        _signal.signal(_sig, _forward_signal_to_sigterm)
    except (ValueError, OSError):
        pass
```

**为什么不用 `SIG_IGN`**：忽略信号让进程失去保存 checkpoint 机会，下次 SIGKILL 强杀时丢全部进度；转发为 SIGTERM 触发 Trainer `_train_signal_handler` → 保存 adapter + 退出 → 下次启动自动从 `checkpoint-*` 恢复。

**重入续跑**：Trainer 默认从 `--output_dir` 里最新 `checkpoint-*` resume，不动命令即可重跑；`save_total_limit=2` 保证最多丢 500 步（≈30-40 分钟）进度而不是 5h。

### 3.3 DDP / 单卡混用与"误用防护"

- 三个训练脚本（Stage 1/2/3）共用 DDP，单/双卡启动命令一致
- **`build_preference.py` 强制单卡**（生成式推理脚本，DDP 会 4× 显存且无加速）：
  ```python
  if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise RuntimeError("[Pref] build_preference 不支持 torchrun 多卡 ...")
  ```
- 踩坑历史：曾 4 卡 DDP 跑 build_preference → 4 进程同时加载 7B+adapter（policy+ref 双 7B）→ 3-4 卡 OOM → elastic launcher 60s 超时 SIGTERM rank 0 → 留下"未找到 .rank* 文件"误导性错误
- **代码层面保留 sharding + `.rank*` 同步逻辑**（含 atomic rename + fsync）以备未来 vLLM 场景重用

### 3.4 其他工程要点

- **PeftModel.from_pretrained 后必须显式 `requires_grad_(True)`**：所有 LoRA 参数加载时默认为 False，直接训练会报"loss does not require grad"
- **Trainer 默认 collator 不 padding**：长度不一必须显式传 `DataCollatorForSeq2Seq` + `label_pad_token_id=-100`
- **新版 transformers API 兼容**：`compute_loss(model, inputs, return_outputs, num_items_in_batch)` 多一个参数；`is_gradient_checkpointing` 替代 `gradient_checkpointing`
- **TP plan 警告处理**（不用 TP 的项目也要清警告）：accelerate 检测到 `tp_plan` 但没应用会报"The following TP rules were not applied"，项目用 DDP 不用 TP → 清空 `model._tp_plan` + `model.config._tp_plan` 让 accelerate 不警告
- **datasets 警告消除**：`Dataset.from_list` 不接受 generator（转 list）、`TOKENIZERS_PARALLELISM=false`、猴补丁 `trainer.tokenizer` 消除 4.52 deprecated 警告
- **GPU 显存监控看 `peak=` 不是 `alloc=`**：反向时才出现的真实峰值，判 OOM 必须看 peak

### 3.5 量化推理与部署（产品岗 / 推理系统岗重点）

**做了什么**：在 `infer/` 目录下新增 3 个脚本 + 1 个 README，把 1.89 万条评估集的推理从"HF generate + bf16"升级到"vLLM + bitsandbytes 4-bit + LoRA hot-swap"：

| 脚本 | 作用 |
|---|---|
| `infer/baseline_runner.py` | 复用 `train/common.py` 的 `load_model_tokenizer` + `SYSTEM_PROMPT`，做 HF generate baseline 复跑（B1/B2/B3 三组） |
| `infer/fast_infer.py` | vLLM 4-bit + LoRA hot-swap 主推方案（V1） |
| `infer/bench.py` | 一键跑 5 组对照实验 → `bench/bench_summary.csv` |

**核心工程决策**（产品岗要的"为什么选 A 不选 B"判断力）：

1. **训推同源原则**：训练用 bitsandbytes 4-bit（NF4）做 QLoRA，部署复用同一份量化权重走 vLLM 4-bit → **权重零转换、代码零侵入**。放弃 GPTQ 的 ~0.5% 指标优势以换 1 周时间节省与训推一致性。
2. **不 merge LoRA**：用 vLLM `LoRARequest` 热加载 PEFT 格式 adapter，保留同一 base 切换多 stage adapter 的能力。merge_and_unload 后失去多版本灵活性。
3. **输出格式字节级一致**：vLLM 输出 JSONL 与 baseline 键完全相同（`{input, label, task, model_output}`）→ `eval/evaluate.py` 直接吃两组输出，任务指标对比零摩擦。
4. **开发-实验分工**：vLLM 不支持 Windows（缺 `resource` 模块）→ 本机（RTX 4060 + Win11）做代码开发与文档交付，远程 A100 Linux 节点做真实验。这本身就是真实工作流，简历里讲得清。

**为什么选 vLLM 不选其他引擎**（4 维对比已在 `bench/bench_inference.md` 第 4 节）：

| 引擎 | 4-bit | LoRA hot-swap | 安装 | 决策 |
|---|---|---|---|---|
| **vLLM 0.6+** | ✅ 一等公民 | ✅ `LoRARequest` | 🟢 pip 一行 | ⭐ **采用** |
| SGLang | ✅ | ⚠️ 实验性 | 🟡 多 | 备选（多轮对话） |
| TGI | ✅ | ❌ | 🟡 Rust 编译 | 备选（不想装 vLLM） |
| TensorRT-LLM | ✅ | ⚠️ 需重建 | 🔴 重 | 不选（成本不划算） |

**4 维 benchmark 报告**（`bench/bench_inference.md`，数字待 A100 实验回填）：
1. 吞吐（samples/s）：主指标，vLLM vs bf16 baseline 加速比 ≥ 3× 算"显著加速"
2. 延迟（ms/sample）：p50/p95/p99 + avg
3. 峰值显存（GiB）：`torch.cuda.max_memory_allocated()` 取真实峰值
4. 任务指标：8 项指标 vs bf16 baseline 的变化，< 1.5% 算"能力无损"

**5 组对照实验**（覆盖 bf16 / 4bit / vLLM / batch 大小全谱系）：
- B1 bf16 + HF + b=1（底线基线）
- B2 4bit + HF + b=1（看 4-bit 加载省显存）
- B3 4bit + HF + b=8（看朴素引擎 batch 增益上限）
- **V1 vLLM 4-bit + continuous batching（主推）**
- V2 vLLM 4-bit（reserved）

**加速来源归因**（产品岗最爱看的"为什么快"）：
- 4-bit 加载省显存：~5%（B2 vs B1）
- 朴素 batch 增益：~50%（B3 vs B2）
- 引擎升级（vLLM PagedAttention + continuous batching）：~150%（V1 vs B3）
- **关键洞察**：LLM 推理优化的重心在 KV cache 调度与 continuous batching，而非单纯量化

**已知踩坑**（已写入 `infer/README.md` §5）：
- vLLM 装环境会改写 torch/cuda runtime → **训练推理必须分 venv**
- vLLM 启动期会一次性占大量显存 → `peak` 出现在启动后第一个 batch 之前
- bnb 版本敏感（0.43 → 0.49 API 变了）→ 锁 0.49.2（与训练侧同）
- LoRA 加载后输出"乱码" → 90% 是 prompt 模板不一致，`fast_infer.py` 已固化 ChatML
- vLLM 0.6.x per-sample p95 不公开 → 报告以 throughput 为主指标，诚实标注限制

---

## 4. 数据工程能力（数据 / ML 工程师岗重点）

### 4.1 全流程数据治理（330 万 → 28.98 万净化）

| 阶段 | 数据 | 条数 | 处理 |
|---|---|---|---|
| 原始 | stage2_train.jsonl | 3,330,232 | 论文官方 Biology-Instructions |
| 三路划分 | train_pool / dpo_source / eval_set | 30 万 / 11.25 万 / 1.89 万 | 按 task 分层（避免小任务消失），行号唯一 → 三路物理不相交 |
| 净化 | train_pool_clean.jsonl | 28.98 万 | 去重 + 模板均衡 + 合并 stage3 长答案 |
| 序列 | stage1_pretrain.jsonl | 23.59 万 | GRCh38 + RNAcentral + Swiss-Prot，每条带 type token（`<dna>/<rna>/<protein>` 前缀） |

**为什么 DPO 源与评估集必须分开**（关键设计）：
- chosen 是标准答案；若用评估集问题构造 DPO 偏好对，DPO 训练后模型在评估集上的指标被污染
- 评估严格在 DPO 未接触的数据上进行 → 拆三路独立集合

**为什么按 task 分层**：
- 数据集严重不均衡（Isoform 157 万 vs CRISPROnTarget 1453 条）
- 全局随机让小任务在训练中近乎消失
- 论文消融：平衡采样反而掉点，所以只 cap 不均衡

### 4.2 模板同质化诊断与处理

- **诊断先行**：正则提取模板骨架 → 实测每 task 仅 50~200 个模板，头部模板重复几十~几百次
- **处理**：完全去重 → 模板 cap = max(50, ceil(avg × mult)) → 合并 stage3 高信息密度长答案
- **效果**：Top5 模板占比 2.0% → 1.3%
- **附带发现**：stage2 序列标签是非标准闭合 `<rna>...<rna>`（无斜杠），模板提取正则适配

### 4.3 序列数据准备（多组学）

- **GRCh38**：只保留 ≥1Mbp 主染色体（自动跳过 contig/scaffold/线粒体）；按染色体长度比例配额；随机切 512~2000bp 片段；N 占比 >2% 丢弃（着丝粒等低复杂度区）
- **RNAcentral**：680MB gz → 200 万条，固定种子蓄水池抽样至 8 万条
- **Swiss-Prot**：288MB → 57 万条，固定种子蓄水池抽样至 8 万条
- **蓄水池抽样**：源文件 2GB+ 无法全部载入内存，蓄水池单遍扫描、O(1) 内存、保证均匀随机
- **type token 前缀**：避免 DNA 的 G / 蛋白的 G 共享 token 语义冲突

### 4.4 评估协议对齐

- 沿用论文官方评估协议（`eval/evaluate.py`，commit `600acaa`，未做修改）
- 24 个任务、4 种 omics（DNA / RNA / Protein / Multi）、8 种指标（MCC / PCC / R² / Spearman / Acc / AUC / Fmax / mixed_score）
- 4 档模型 × 1.89 万样本全量评估

---

## 5. 评估与消融（任何 JD 都能加上的亮点）

### 5.1 消融设计

| 消融 | 对比 | 必要性 |
|---|---|---|
| ① Stage 1 必要性 | `stage2_s1` vs `stage2_only`（两分支均 28.98 万全量） | 完整本地消融 |
| ② DPO 改善对齐 | `stage3` vs `stage2_s1`（同 2.5 万对，路径 B 起于 stage2_s1） | 核心消融，必做 |
| ③ 数据量（可选） | 5 万 vs 28.98 万 Stage 2 对比 | 时间充裕时 |

### 5.2 评估完整性

- **定量**：4 档模型 × 1.89 万样本全量（不是抽样）
- **定性**：对齐前后对话示例对比（README 展示 4–6 条）
- **指标透明**：所有数字从日志 `peak=` / `[进度] step ...` 取证

---

## 6. 可抽取的"成果表述"模板

### 6.1 结果导向（适合简历条目）

- 完整跑通大模型后训练三阶段流水线（领域 CPT + QLoRA SFT + DPO），覆盖 21 类生物序列任务
- 单卡 T4（<12GB）跑通工业级后训练技术栈全流程，验证有限算力下复现前沿对齐技术的可行性
- A100×4 上 7B 模型三阶段全量训练共 34–40h；Stage 3 DPO 实测峰值 26.8 GiB/卡，留 26 GiB 余量
- 数据管线（三路防泄漏划分、模板均衡净化、偏好自构）与训练代码全部自建，可复现
- 三阶段消融验证各环节有效：继续预训练增益、指令微调提升、DPO 不掉领域任务能力

### 6.2 方法导向（适合面试问答）

- 设计并实现了不依赖 trl 的 DPO 训练管线：自实现 loss（~30 行）+ on-policy 偏好构造 + 4bit QLoRA 显存策略，规避了 trl / torch 版本地狱
- 设计了"行号唯一"的三路不相交数据划分方法（330 万→训练 30 万 / DPO 源 11.25 万 / 评估 1.89 万），通过完整 input+label 精确匹配验证泄漏为 0
- 实现了公共 GPU 环境长任务稳定性方案：setsid + disown + `< /dev/null` 三件套 + 代码侧 SIGHUP→SIGTERM 转发，5h+ 训练不丢进度（最多 500 步）

### 6.3 工程导向（适合系统岗 JD）

- OOM 排查：3B Stage 1 在 T4 上三次收敛，从 max_len / batch / 优化器精度最终定位到 bf16 权重本身（11.95 GB 加载后已超预算）→ 改用 4-bit QLoRA
- DPO 显存峰值优化：发现 `logits.float()` 在 7B vocab=152064 下产出 2.49 GB/张 fp32 中间表 → 改为 `log_softmax` 在 bf16 上原生计算后立刻 gather 到 (B, T-1)，跳过 V 维 fp32 峰值
- 信号优雅退出：SIGHUP 转发为 SIGTERM 触发 Trainer `_train_signal_handler` 保存 checkpoint，配合 Trainer 自动 resume 实现长任务可恢复

---

## 7. 面试问答素材（按主题分类）

### 7.1 关于 DPO

**Q: DPO 为什么需要两个模型？π_ref 是怎么用的？**
A: 同一初始权重的两份副本，π 训练更新，π_ref 全冻结作偏好基线。DPO 学的是 `log π - log π_ref` 的相对差值：chosen 相对 ref 增长 vs rejected 相对 ref 增长 → "相对偏好"。没有 ref 模型退化成 SFT/负样本训练，丧失偏好的相对性。

**Q: rejected 为什么用 stage2 模型采样而不是基座？**
A: DPO 学的是"同一任务分布下的质量偏好"而非"会不会答"。基座不会答，与 chosen 不在同一分布 → DPO 偏离本意；stage2 模型 + temperature 采样与 chosen 同分布（同 task 同 input），区分度由 temperature 采样保证。

**Q: DPO 的显存代价是什么？为什么特别需要 gradient checkpointing？**
A: DPO 对 chosen/rejected 各做一次前向，加 ref 共 4 次前向 → 激活 ×4。Stage 2 SFT 激活 ×1，DPO 激活 ×4 接近 OOM。gradient checkpointing 前向不全存，反向时重算 → 激活降约一个量级、训练慢 ~33%。

**Q: β（温度参数）怎么选？过大过小会怎样？**
A: β 控制对齐强度。β 过大 → 训练不稳、生成质量退化；β 过小 → 对齐信号弱、学不到偏好。常用 0.1~0.5，需要在 eval 集上扫。

### 7.2 关于 QLoRA / 4bit

**Q: 4bit QLoRA 的核心思想？为什么能省显存？**
A: 把基座权重量化到 4-bit（nf4），LoRA adapter 仍为 bf16；前向时把 4-bit 权重反量化为 bf16 计算，反向时只更新 LoRA 参数。基座 14GB → 3.5GB，加上激活/优化器节省，单卡装得下大模型。

**Q: LoRA+ 和普通 LoRA 的区别？**
A: 普通 LoRA 对 A、B 矩阵用同一学习率。LoRA+ 把 B 矩阵（输出侧）的学习率放大为 A 的若干倍（论文 4×），加速收敛、更好适配新任务。本项目 Stage 1 用 LoRA+ rank=64, scaler=4。

**Q: 为什么 Stage 1 用 bf16 而 Stage 2/3 用 4bit？**
A: Stage 1 是领域继续预训练，对新知识注入精度敏感；4bit 量化会损失信息 → 选 bf16 + LoRA+。Stage 2/3 是 SFT/对齐，任务已有指令格式 + LoRA 足够表达能力 → 选 4bit 省显存，让出来给更长 max_len 几乎零截断。

### 7.3 关于数据

**Q: 为什么三路划分按 task 分层而不是全局随机？**
A: 数据集严重不均衡（Isoform 157 万 vs CRISPROnTarget 1453 条）。全局随机让小任务在训练中近乎消失 → 评估和 DPO 源对小任务无样本。按 task 分层保证每 task 在三路中都有份，论文消融证明平衡采样反而掉点，所以只 cap 不均衡。

**Q: DPO 源和评估集为什么不合并？**
A: chosen 是标准答案，若 DPO 训练接触到评估集问题，DPO 后模型在评估集上"见过"答案 → 任务指标被污染，无法证明"DPO 不掉领域能力"。拆三路独立集合，评估严格在 DPO 未接触的数据上进行。

**Q: 数据量够吗？token 数远超参数数算"够"吗？**
A: "token 数 > 参数数 所以够" 不成立（与 Chinchilla scaling law 方向相反，且 Chinchilla 仅适用于从零预训练）。正确验证方式是实验：loss 曲线 / eval 指标 vs 基线 / 数据量消融。15–30 万 SFT 在社区实践中够用（LIMA 1k、Alpaca 52k），但"足够"不是先验。

**Q: 模板同质化怎么发现、怎么处理？**
A: 诊断先行：正则提取模板骨架 → 发现每 task 仅 50~200 个模板，头部模板重复几十~几百次。处理：① 完全去重；② 模板 cap = max(50, ceil(avg × mult))；③ 合并 stage3 高信息密度长答案。效果：Top5 模板占比 2.0% → 1.3%。

### 7.4 关于工程

**Q: 怎么排查 OOM？**
A: 三步：① 看监控的 `peak=`（不是 `alloc=`，alloc 是当前、peak 是历史最高）；② 拆解显存组成（权重 + 梯度 + 优化器 + 激活 + loss logits 峰值），区分确定项和经验项；③ 优先降 batch（logits 峰值线性相关），再降 max_len，再开 gradient checkpoint，最后才考虑 4-bit。

**Q: 长任务怎么防 SIGHUP 杀掉？**
A: 进程层 + 代码层双保险。进程层三件套：`setsid` 开新 session（脱离原控制终端）+ `< /dev/null` 断 stdin + `disown` 从 bash jobs 表移除。代码层：`SIGHUP/SIGINT/SIGTERM → SIGTERM 转发`，让 Trainer `_train_signal_handler` 走 on_train_end 保存 checkpoint。叠加效果：最多丢 500 步进度，而不是 5h 全部丢失。

**Q: 为什么不用 TP？什么时候必须 TP？**
A: TP（张量并行）按输出/输入维度切单层权重到多卡并行算，设备利用率 1、无气泡、通信可重叠。本项目 7B 4bit 单卡装得下（3.5GB），DDP 简单 + 容错好 → 不需要 TP。但 70B+ 模型优化器状态本身 560GB，单卡装不下，必须 TP×PP 组合。本项目之所以不用 TP：① 单卡够；② TP 调试复杂（每 Linear 要写并行版本）；③ Kaggle 无 NVLink，TP 通信拖垮训练。

**Q: build_preference 为什么强制单卡？**
A: `model.generate()` 不反传、不跨进程梯度同步，DDP 启动只是"4 份完整模型并行生成同一批样本"，既不快又 4× 显存 → OOM。正确做法：生成式单卡 sequential 或 vLLM 连续批推理；只有训练（loss.backward + 梯度同步）才用 DDP。代码入口 hard assert `WORLD_SIZE>1 → RuntimeError`，避免误用被误导。

### 7.5 关于量化推理与部署（产品岗 / 推理系统岗重点）

> 完整问答见 `docs/INFER_QA.md`（15 题），以下是最高频的 5 题，60 秒必答版。

**Q: 为什么要做推理加速？训练不是已经 4-bit 了吗？**
A: 训练用 4-bit 只省**加载显存**，前向仍反量化到 bf16 计算。推理没有反传，可以走 vLLM 的 PagedAttention + continuous batching + 4-bit 全程算 → 训推是两个独立但可同源的优化点。本项目选"训推同源"（都是 bnb 4-bit）→ **权重零转换**。

**Q: 为什么选 vLLM 不选 SGLang / TGI / TensorRT-LLM？**
A: 唯一同时满足"4-bit + LoRA hot-swap + pip 一行安装"的引擎；continuous batching + PagedAttention 是当前 SOTA（7B 加速 5–10×）；生态最广、招聘 JD 高频词。TGI 不支持 LoRA 热加载；TensorRT-LLM 需重建 engine（1 周成本）；SGLang 4-bit + LoRA 集成不如 vLLM 稳定。

**Q: 为什么选 bnb 4-bit 不选 GPTQ？**
A: "训推同源"原则。训练已用 bnb 4-bit 做 QLoRA → 部署复用同一份量化权重走 vLLM 4-bit → **权重零转换、训推一致**。放弃 GPTQ 的 ~0.5% 指标优势以换 1 周时间节省与训推一致性。反方观点（在 1.89 万条 8 项指标上量化误差 < 1.5% 已在产品可接受范围）。

**Q: vLLM 怎么做的 continuous batching？和普通 batch 有什么区别？**
A: 朴素 batch 按"整批"等最长序列调度 → 短序列等长序列 GPU 空闲。vLLM continuous batching 按"每个 decode step"调度 → 新请求随时插入空闲槽位、长请求完成后立即释放。PagedAttention 把 KV cache 分页管理（像 OS 虚拟内存），插队时只分配空闲页、不重算已完成的 token。

**Q: PagedAttention 是什么？为什么比朴素 KV cache 省显存？**
A: 把 GPU 显存抽象成"页"（类似 OS 虚拟内存）。朴素 KV cache 每条请求预分配 `max_seq_len × hidden` 连续显存 → 长请求浪费、短请求闲置。PagedAttention 按 token 实际长度按页分配 → 物理不连续、逻辑连续 → 显存利用率从 ~30% 提到 ~90%+。省下的显存装更多并发 → throughput 提升。

---

## 8. JD 关键词对照表（投简历时快速匹配）

| JD 关键词 | 本项目对应内容 |
|---|---|
| **大模型后训练 / Post-training** | 完整跑通 CPT + SFT + DPO 三阶段流水线（§1.2） |
| **QLoRA / 4bit 量化** | Stage 2/3 全部 4-bit QLoRA + prepare_model_for_kbit_training（§2.2） |
| **LoRA / LoRA+** | Stage 1 LoRA+ rank=64, scaler=4，Stage 2/3 LoRA rank=64（§2.2） |
| **DPO / RLHF / 偏好对齐** | 自实现 DPO loss + on-policy rejection sampling（§2.1） |
| **packing / 高效训练** | Stage 1 packing 消除 padding；assistant-only loss（§2.4） |
| **数据治理 / 数据清洗** | 三路不相交划分 + 模板均衡 + 完全去重 + 蓄水池抽样（§4） |
| **模型评估 / 消融实验** | 沿用论文官方协议 4 档 × 1.89 万 + 消融①②（§5） |
| **显存优化 / OOM 排查** | 3 次收敛案例 + logits 峰值优化 + 8bit AdamW（§3.1） |
| **分布式训练 / DDP** | torchrun + DDP，与单卡脚本混用（§3.3） |
| **长任务稳定性 / checkpoint resume** | SIGHUP 三件套 + 信号转发 + Trainer resume（§3.2） |
| **多组学 / 生物序列** | DNA / RNA / 蛋白 / Multi 四种 omics，24 类任务（§4.4） |
| **可复现 / 固定种子** | 所有脚本固定 SEED=42，重跑结果一致（§4） |
| **PyTorch / transformers / peft** | 完整技术栈：transformers + peft + bitsandbytes + accelerate |
| **Hugging Face Trainer** | TrainingArguments / Trainer / DataCollatorForSeq2Seq 全部使用 |
| **模型工程化 / 训练系统** | 公共模块 `common.py`：信号补丁、TP 警告清掉、tokenizer 猴补丁（§3.4） |
| **可解释 / 透明实现** | 自实现 DPO ~30 行，公式、显存预算、训练选择全部文档化（§2.1） |

---

## 9. 项目目录快速索引（面试被问到代码细节时定位）

```
BioAlign/
├── readme.md                       # 项目总览（含三阶段表、技术栈、文档导航）
├── REPRODUCTION_PLAN.md            # 9 晚节奏 + 各 stage 完整命令 + 关键代码补丁
├── TECH_NOTES.md                   # 技术原理笔记（显存换算、SIGHUP、DPO 原理）
├── PROJECT_FOR_RESUME.md           # 本文档（简历导向）
├── data_prep/
│   ├── README.md                   # 数据处理决策记录 + 面试问答
│   └── scripts/
│       ├── 01_split_stage2.py      # 三路不相交划分
│       ├── 02_stage3_convert.py    # xlsx → jsonl
│       ├── 03_seq_prepare.py       # 多组学序列准备（GRCh38/RNAcentral/Swiss-Prot）
│       ├── 04_smoke.py             # 冒烟测试集
│       ├── 05_dedup_template.py    # 去重 + 模板均衡 + 合并 stage3
│       └── 06_token_len_stats.py   # token 长度统计 → max_len 选型
├── train/
│   ├── README.md                   # 训练代码说明 + 冒烟记录 + 3B OOM 收敛过程
│   ├── common.py                   # 公共模块（QLoRA 加载、LoRA 配置、信号补丁）
│   ├── stage1_pretrain.py          # 阶段1：bf16 LoRA+ packing
│   ├── stage2_sft.py               # 阶段2：4-bit QLoRA SFT（assistant-only loss）
│   ├── build_preference.py         # 自构 DPO 偏好（强制单卡 + on-policy）
│   ├── stage3_dpo.py               # 阶段3：自实现 DPO（logits 优化）
│   └── infer_eval.py               # 推理 + 评估输出
└── eval/
    ├── README.md                   # 来源说明（论文官方 commit 600acaa）
    ├── evaluate.py                 # 评估脚本（未修改）
    ├── register_tasks.json         # 24 个任务注册表
    └── ec_labels.json              # FunctionEC 标签

infer/                             # 量化推理闭环（与 train/ 平级，零侵入）
├── baseline_runner.py              # HF generate 复跑（B1/B2/B3）
├── fast_infer.py                   # vLLM 4-bit + LoRA hot-swap（V1）
├── bench.py                        # 一键跑 5 组对照 → bench_summary.csv
└── README.md                       # 用法 / env / 踩坑

bench/                             # 推理加速报告与原始数据
├── raw/                            # 跑实验时落盘的 JSONL + metrics
└── bench_inference.md              # ★ 4 维对比报告（简历核心素材）

docs/                              # 面试问答与选型决策
├── INFER_QA.md                     # 15 个面试问答
└── INFER_DECISION.md               # 5 个决策矩阵 + 反方观点反驳
```

---

## 10. 简历条目模板（按岗位挑选）

### 模板 A（LLM 算法工程师）
**BioAlign —— 基于 QLoRA 与 DPO 的生物医学大模型后训练流水线**
- 基于 Qwen2.5-7B-Instruct 完整实现"领域继续预训练 → QLoRA SFT → DPO 偏好对齐"三阶段后训练流水线，覆盖 DNA/RNA/蛋白/多分子 21 类生物序列任务
- 自实现 DPO loss（约 30 行），不依赖 trl 规避版本地狱；on-policy rejection sampling（rejected = stage2 采样）保证偏好分布一致；通过 bf16 logits + gather 优化跳过 (B,T,V) fp32 峰值，7B DPO 实测 peak 26.8 GiB/卡
- 设计三路严格不相交数据划分（330 万→30 万训练 / 11.25 万 DPO 源 / 1.89 万评估），通过完整 input+label 精确匹配验证泄漏为 0；模板感知均衡采样将 Top5 模板占比从 2.0% 降至 1.3%
- 单卡 T4 16GB / RTX 4060 8GB / A100 40GB×4 三档硬件跑通全流程；通过 token 长度分布驱动 max_len 选型（1024 / 768），避免拍脑袋
- 设计 SIGHUP 免疫三件套（setsid + disown + `< /dev/null`）+ 代码侧信号转发，5h+ 长任务最多丢 500 步进度

### 模板 B（NLP / 训练系统工程师）
**BioAlign —— 多阶段 LLM 后训练系统**
- 构建可恢复的多阶段后训练系统（领域 CPT + QLoRA SFT + DPO），单/双卡共用一份代码，单卡脚本 hard assert 防误用 DDP
- 设计 SIGHUP→SIGTERM 信号转发让 Trainer 优雅保存 checkpoint，配合 setsid + disown + `< /dev/null` 三件套实现长任务可恢复
- 实现 3 次 OOM 收敛案例（max_len 2048→1024 / batch 4→1 / bf16→4bit），定位 bf16 权重本身超预算；最终 7B 4bit DPO 实测 peak 26.8 GiB/卡
- 通过 DPO logits dtype 优化（避免 `(B,T,V) fp32` 峰值）节省 ~6 GB 隐性显存；通过 assistant-only loss + packing 提升训练效率
- 实现 TP plan 警告清除、tokenizer 猴补丁、checkpoint use_reentrant 补丁等 transformers 版本兼容适配

### 模板 C（数据 / ML 工程师）
**BioAlign —— 生物医学 LLM 数据治理与后训练**
- 设计三路不相交数据划分方法（按 task 分层 + 行号唯一），训练 / DPO 源 / 评估物理零重叠（精确匹配验证）
- 实现模板感知均衡采样：正则提取模板骨架 → 每模板 cap → Top5 占比 2.0% → 1.3%；发现并适配 stage2 数据非标准闭合标签 `<rna>...<rna>`
- 准备多组学序列数据（GRCh38 + RNAcentral + Swiss-Prot 23.59 万条），加 type token 避免 DNA/蛋白共享 token 语义冲突；N 占比 >2% 过滤、≥1Mbp 主染色体筛选
- 基于 1 万条 token 长度分布驱动 max_len 选型（p50=228 / p95=614 / max=1544)，避免拍脑袋
- 沿用论文官方评估协议（24 任务 × 4 omics × 8 指标），4 档模型 × 1.89 万全量评估

### 模板 D（产品岗专项 · 含量化推理）★
**BioAlign —— 生物医学 LLM 后训练 + 量化推理闭环**
- 从 0 完整实现"领域继续预训练 → QLoRA SFT → DPO 偏好对齐"三阶段后训练流水线，覆盖 DNA/RNA/蛋白/多分子 21 类生物序列任务，并补齐 **vLLM + 4-bit + LoRA hot-swap 量化推理环节**，形成训练→部署闭环
- 在 A100 单卡上对比 5 组推理方案（bf16/4bit/vLLM × batch 维度），输出 4 维 benchmark 报告（吞吐 / 延迟 / 峰值显存 / 任务指标），vLLM 4-bit 相对 bf16 baseline 加速 **X×**、峰值显存下降 **Y%**、1.89 万条评估集 8 项任务指标变化 < **Z%**（实测待补）
- 提出“训推同源”选型原则：训练用 bitsandbytes 4-bit（NF4）做 QLoRA，部署复用同一量化方案走 vLLM，**权重零转换、代码零侵入**；放弃 GPTQ 的 ~0.5% 指标优势以换 1 周时间节省与训推一致性（决策矩阵见 `docs/INFER_DECISION.md`）
- 负责“开发-实验分工”：Windows 笔记本（RTX 4060 8GB，vLLM 不支持）做代码开发与文档交付，远程 A100 Linux 节点做真实验；交付 3 个脚本 + 1 个 README + 2 个 4 维报告，`eval/evaluate.py` 零修改接接两组输出
- 数据治理：330 万原始 → 30 万训练 / 11.25 万 DPO 源 / 1.89 万评估，按 task 分层、零泄漏（完整 input+label 精确匹配验证），模板同质化诊断后采样使 Top5 模板占比从 2.0% 降至 1.3%

### 模板 E（推理系统 / ML Infra 岗专项）
**BioAlign —— 量化推理与 LLM 部署优化**
- 实现 vLLM + bitsandbytes 4-bit + LoRA hot-swap 推理加速方案：PagedAttention + continuous batching + 4-bit kernel，7B 推理加速 **X×**（实测待补），peak 显存 **Y GiB**（待补）
- 4 维 benchmark 报告：吞吐 / 延迟 p50-p95 / 峰值显存 / 任务指标，覆盖 5 组对照实验（bf16/4bit × batch 1/8 + vLLM），为推理选型提供决策依据
- 提出"训推同源"原则：与训练侧 bnb NF4 量化方案统一，**权重零转换**；用 `LoRARequest` 热加载保留同一 base 切换多 stage adapter 的能力；输出 JSONL 字节级一致让 `eval/evaluate.py` 零修改接接
- 踩坑沉淀（8 条已写入 `infer/README.md` §5）：vLLM 装环境会改写 torch/cuda runtime（训推必须分 venv）、bnb 版本敏感（锁 0.49.2）、enforce_eager 选型、max_model_len 对 KV cache 影响、量化误差验证方法等
- 决策文档：`docs/INFER_DECISION.md` 含 5 个决策矩阵、 4 条反方观点反驳、5 维加速归因与未来工作路线图

---

## 11. 简历常见扣分项预检（提交前自查）

- ✅ 写"完整跑通三阶段流水线"，避免说"实现了一个训练框架"（太泛）
- ✅ 写"自实现 DPO loss（约 30 行），不依赖 trl 规避版本地狱"而不是"实现了 DPO 算法"（突出选型理由）
- ✅ 写"通过 logits dtype 优化跳过 (B,T,V) fp32 峰值节省 6 GB"而不是"对训练做了优化"（给出具体数字）
- ✅ 写"SIGHUP 三件套 + 信号转发，最多丢 500 步进度"而不是"加了防杀机制"（量化收益）
- ✅ 写"行号唯一 + 完整 input+label 精确匹配验证重叠为 0"而不是"防止了数据泄漏"（说明方法）
- ✅ 写"3B Stage 1 在 T4 上三次收敛，最终定位到 bf16 权重本身"而不是"调通了一系列超参"（讲排查故事）
- ✅ 写"vLLM 4-bit 相对 bf16 baseline 加速 X×、任务指标变化 < 1.5%"而不是"用 vLLM 加速了推理"（量化数字 + 标注 trade-off）
- ✅ 写"训推同源原则，权重零转换"而不是"用了 4-bit 量化"（讲出选型判断与成本/收益）
- ✅ 写"vLLM 不支持 Windows，开发-实验分工"而不是略去不跳（透明说明限制，体现资源受限环境交付能力）
- ❌ 不要写"性能 SOTA"（本项目没和论文 SOTA 比）
- ❌ 不要写"精通 PyTorch"（"精通"是减分项，用"掌握""熟练"）
- ❌ 不要省略数字：所有显存 / 时间 / 数据量都给出实测值
- ❌ 不要堆砌技术名词：每个名词都要在项目里能找到具体实现位置