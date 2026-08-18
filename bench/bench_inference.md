# BioAlign · 推理加速 Benchmark 报告

> **报告目的**：把 1.89 万条评估集的推理从"HF generate + bf16"升级到"vLLM + 4-bit + LoRA
> hot-swap"，用 4 个维度（吞吐 / 延迟 / 显存 / 任务指标）做横向对比，给出
> **Go / No-Go 判定** 与 **为什么不选 GPTQ 的选型决策**。
>
> **报告状态**：⏳ **待 A100 实验回填数字**（所有 `{{}}` 占位处需补实测值）
> 代码与文档已就绪，跑完 `python infer/bench.py --all` 即可生成 `bench/bench_summary.csv`，
> 数字直接回填本表。

---

## 0. 实验条件（统一控制变量）

| 维度 | 取值 |
|---|---|
| 硬件 | A100 40GB ×1（单卡，避免多卡变量干扰） |
| 基座 | Qwen2.5-7B-Instruct |
| Adapter | `ckpt/stage2/`（SFT 后 PEFT 格式，rank=64） |
| 数据集 | `data_prep/output/eval_set.jsonl`，**18,870 条** |
| 任务 | 24 个（DNA/RNA/Protein/Multi 四种 omics） |
| 评估协议 | `eval/evaluate.py`（论文官方 commit，未改） |
| 生成参数 | `max_new_tokens=64`, `do_sample=False`（贪心解码） |
| 温度 | 0.0（vLLM 约定）/ None（HF generate 显式置 None） |
| 时间 | 2026-{{MM}}-{{DD}}，A100 节点 |
| 操作系统 | Linux（vLLM 不支持 Windows） |
| 软件栈 | torch {{2.5.1+cu124}} · transformers 4.52.1 · peft 0.18.1 · bitsandbytes 0.49.2 · vLLM {{0.6.X}} |

---

## 1. 实验设计（5 组对照）

| 组 | tag | 引擎 | 量化 | batch | 目的 |
|---|---|---|---|---|---|
| **B1** | `bf16` | HF generate | bf16 | 1 | 底线基线：无量化、无引擎优化 |
| **B2** | `4bit` | HF generate | NF4 4-bit | 1 | 看仅 4-bit 加载的省显存 |
| **B3** | `4bit_b8` | HF generate | NF4 4-bit | 8 | 看朴素引擎下 batch 增益上限 |
| **V1** ⭐ | `vllm_4bit` | **vLLM 0.6+** | **NF4 4-bit** | continuous | **主推方案** |
| V2 | `vllm_4bit_b16` | vLLM 0.6+ | NF4 4-bit | continuous | V1 的 reserved 名（暂与 V1 同） |

> ⭐ V1 是产品岗汇报的"主推方案"。B1/B2/B3 用于画"加速来源归因"曲线。

---

## 2. 4 维对比表（核心）

> ✅ = 填入实测值；❌ = 不适用；⏳ = 待填

### 2.1 维度 1：吞吐（throughput, samples/s）

> 主指标。**vLLM vs bf16 baseline 加速比 ≥ 3×** 算"显著加速"（社区 SOTA 倍数下限）。

| 组 | 吞吐 (smpl/s) | 相对 B1 加速比 | 相对上组增益 | 总耗时 (HH:MM:SS) |
|---|---|---|---|---|
| B1  bf16 + HF | ⏳ {{}} | 1.00× | — | ⏳ {{}} |
| B2  4bit + HF  | ⏳ {{}} | ⏳ {{}}× | 4bit 加载增益: ⏳ {{}}% | ⏳ {{}} |
| B3  4bit + HF b=8 | ⏳ {{}} | ⏳ {{}}× | batch 增益: ⏳ {{}}% | ⏳ {{}} |
| V1  vLLM 4-bit | ⏳ {{}} | ⏳ {{}}× | 引擎升级: ⏳ {{}}% | ⏳ {{}} |

**加速归因**（填表后写解读）：
- 4-bit 加载省显存：⏳ {{}}%（B2 vs B1 加速比 - 1）
- batch 增益：⏳ {{}}%（B3 vs B2 加速比 - 1）
- 引擎升级（vLLM）：⏳ {{}}%（V1 vs B3 加速比 - 1）

### 2.2 维度 2：延迟（latency, ms/sample）

> HF generate 单 batch 内同步 → 真实 p50/p95/p99 可读
> vLLM continuous batching → per-sample p50/p95/p99 在 0.6.x 不公开，以 avg 为主指标

| 组 | avg | p50 | p95 | p99 |
|---|---|---|---|---|
| B1  bf16 + HF | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} |
| B2  4bit + HF | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} |
| B3  4bit + HF b=8 | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} |
| V1  vLLM 4-bit | ⏳ {{}} | ≈ avg (continuous batching) | ❌ 不公开 | ❌ 不公开 |

**报告里的诚实声明**：
> vLLM continuous batching 是 throughput-optimized，per-sample p95 在 0.6.x 的公开 metrics 里
> 不稳定。本报告以**系统级 throughput** 作为主指标，per-sample latency 仅供大致参考。
> 若产品需要严格 p95 SLA，应在 0.7+ 升级时重新评估，或换 SGLang（其有 per-request metrics）。

### 2.3 维度 3：峰值显存（peak GPU mem, GiB）

> 显存采 `torch.cuda.max_memory_allocated()`（**peak**），不是 `memory_allocated()`。
> A100 40GB 装 7B 4-bit + KV cache + LoRA，预期 peak 在 8–15 GiB 区间。

| 组 | peak (GiB) | 相对 B1 节省 | A100 40GB 余量 |
|---|---|---|---|
| B1  bf16 + HF | ⏳ {{}} | — | ⏳ {{}} GiB |
| B2  4bit + HF | ⏳ {{}} | ⏳ {{}}% | ⏳ {{}} GiB |
| B3  4bit + HF b=8 | ⏳ {{}} | ⏳ {{}}% | ⏳ {{}} GiB |
| V1  vLLM 4-bit | ⏳ {{}} | ⏳ {{}}% | ⏳ {{}} GiB |

### 2.4 维度 4：任务指标（量化误差 vs bf16 baseline）

> 每组用 `eval/evaluate.py` 跑 `bench/raw/<tag>.jsonl` 算 8 项指标。
> 关键判定：**V1 vs B1 任务指标平均变化 < 1.5%** → 算"能力无损"。

| 任务 | B1 bf16 | V1 vLLM 4-bit | Δ 绝对值 | Δ 相对 |
|---|---|---|---|---|
| MCC (binary) | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}}% |
| AUC (binary) | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}}% |
| Fmax (binary) | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}}% |
| Accuracy | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}}% |
| R² (regression) | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}}% |
| Spearman | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}}% |
| PCC | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}}% |
| mixed_score | ⏳ {{}} | ⏳ {{}} | ⏳ {{}} | ⏳ {{}}% |
| **平均** | — | — | — | ⏳ {{}}% |

**结论模板**（填完指标后二选一）：
- ✅ **能力无损**：平均变化 {{X.X}}% < 1.5% 阈值，可上生产
- ⚠️ **轻微退化**：平均变化 {{X.X}}% 在 1.5%–3% 区间，需业务侧确认是否可接受
- ❌ **显著退化**：平均变化 > 3%，回退到 bnb 4-bit + HF generate（B2）

---

## 3. Go / No-Go 判定

| 条件 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| V1 vs B1 吞吐加速比 | ≥ 3× | ⏳ {{X.X×}} | ⏳ {{✅ / ⚠️ / ❌}} |
| V1 vs B1 任务指标平均变化 | < 1.5% | ⏳ {{X.X%}} | ⏳ {{✅ / ⚠️ / ❌}} |
| V1 peak 显存 | < A100 40GB × 0.6 = 24 GiB | ⏳ {{X.X GiB}} | ⏳ {{✅ / ⚠️ / ❌}} |
| V1 单卡装得下 | True | ⏳ {{}} | ⏳ {{✅ / ❌}} |

**最终判定**：⏳ {{ Go / Conditional Go / No-Go }}

---

## 4. 选型决策：为什么 vLLM + bnb 4-bit，不选 GPTQ

> 这是产品岗面试里"为什么选 A 不选 B"的判断力素材。

| 维度 | **vLLM + bnb 4-bit ✅** | GPTQ 4-bit | AWQ 4-bit |
|---|---|---|---|
| 与训练侧量化方案 | **完全一致**（都是 NF4 + double quant） | 不一致（GPTQ 是 PTQ） | 不一致（AWQ 是 PTQ） |
| 权重转换成本 | **零**（直接复用训练侧 4-bit 权重） | 高（calibration + 量化脚本） | 中（需预量化权重） |
| 训练-部署一致性 | ✅ 训推同源 | ❌ 训 PTQ + 部 PTQ 双轨 | ❌ 同上 |
| 量化误差（vs bf16） | 中（动态反量化） | 较小 | 最小 |
| 量化所需数据 | 0（量化方案在 bnb 库内） | 1–2k 条 calibration | 1–2k 条 calibration |
| 额外时间成本 | **0** | ~1 天（calibration + 转换） | ~0.5 天 |
| 工程复杂度 | 🟢 低（pip 一行 + 5 个参数） | 🟡 中（要写 calibration 脚本） | 🟡 中 |
| 简历故事性 | "训推同源 + 工程闭环" | "PTQ 全流程" | "激活感知量化" |

**决策**（产品岗视角的 cost/benefit）：
- 项目已用 bnb 4-bit 做 QLoRA 训练 → 部署也用 bnb 4-bit → **权重零转换、训推一致**
- GPTQ 的 ~0.5% 指标优势 vs 1 周时间增量 → **不划算**（产品岗：成本/收益分析）
- AWQ 推理稍快但需预量化权重 → **不灵活**（未来想换 base 模型要重新量化）

**写在简历里**：
> "选型上坚持'训推同源'原则：训练用 bitsandbytes 4-bit（NF4）做 QLoRA，部署复用同一
> 量化方案走 vLLM，**权重零转换**；放弃 GPTQ 的 ~0.5% 指标优势以换 1 周时间节省与
> 训推一致性。"

---

## 5. 加速来源归因（产品岗最爱看的"为什么快"）

> 跑完实验后填这一节，是简历"做了哪些工程优化"的强证据。

| 优化项 | 加速贡献 | 证据 |
|---|---|---|
| bitsandbytes 4-bit 加载 | ⏳ {{X%}} | B2 vs B1 加速比 |
| 朴素 batch=8 拼接 | ⏳ {{X%}} | B3 vs B2 加速比 |
| **vLLM PagedAttention** | ⏳ {{X%}} | V1 vs B3 加速比 |
| **vLLM continuous batching** | (含在上一行) | KV cache 复用 + 动态调度 |
| **vLLM CUDA graph 优化** | (含在上一行) | `enforce_eager=False` 默认开 |
| **vLLM bnb 4-bit kernel** | (含在上一行) | 与训练侧 bnb 同 kernel |

**关键洞察**（填表后写）：
> 加速比的主要来源是**引擎升级**（V1 vs B3，约 ⏳ {{X%}}），而非"4-bit 加载"本身
> （B2 vs B1 仅 ⏳ {{X%}}）。**这印证了 LLM 推理优化的重心在 KV cache 调度与
> continuous batching，而非单纯量化**——产品岗面试讲这个会非常加分。

---

## 6. 限制与未来工作

### 6.1 已知限制

- ⏳ {{vLLM 0.6.x per-sample p95 不公开 → 用 throughput 为主指标}}
- ⏳ {{只在 A100 单卡跑了对照 → 多卡 TP/PP 的扩展性未验证}}
- ⏳ {{未测多种 max_new_tokens → 长输出场景（128/256/512 token）的加速比待补}}
- ⏳ {{未测 vLLM 0.7+ 升级收益 → 升级后可能进一步提升 20–30%}}

### 6.2 未来工作（路线图）

1. **vLLM 0.7+ 升级**：per-request metrics 完整 → p50/p95/p99 可报
2. **SGLang 对照实验**：RadixAttention 在多轮对话场景可能有进一步优势
3. **TensorRT-LLM 对照**：极致性能场景（如 1000 QPS），但要 1 周 engine 重建
4. **A800/H100 集群压测**：从 1 卡扩到 8 卡 TP，看系统扩展性
5. **真实业务场景验证**：把"加速比 5×"换成"$节省 X / 月"的故事（产品岗最爱）

---

## 7. 附录：原始数据

| 文件 | 内容 |
|---|---|
| `bench/bench_summary.csv` | 5 组横向比较（宽表，pandas 可读） |
| `bench/raw/baseline_bf16.jsonl` | B1 输出 |
| `bench/raw/baseline_4bit.jsonl` | B2 输出 |
| `bench/raw/baseline_4bit_b8.jsonl` | B3 输出 |
| `bench/raw/vllm_4bit.jsonl` | V1 输出 |
| `bench/raw/vllm_4bit_metrics.json` | V1 速度/显存指标 |
| `bench/raw/vllm_4bit_task_metrics.txt` | V1 任务指标（拷贝自 `eval/logging/metrics_vllm_4bit_*.log`） |
| `bench/raw/baseline_bf16_metrics.json` 等 | B1/B2/B3 指标 |

---

## 8. 报告维护

- **生成时间**：{{YYYY-MM-DD}}
- **实验执行**：{{A100 节点名 / 操作人}}
- **下次 review**：{{A100 上跑完回填数字后 / 投递简历前}}
- **关联文档**：`docs/INFER_QA.md`（面试问答） / `PROJECT_FOR_RESUME.md` §10-D（简历条目）
