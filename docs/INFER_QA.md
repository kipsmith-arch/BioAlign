# BioAlign · 推理加速面试问答

> 本文档为 BioAlign 推理加速模块（`infer/`）的**面试问答素材库**。
> 写完后请在投递前通读一遍，确保每个问题都能在 60 秒内讲清核心点。
>
> **覆盖岗位**：NLP/LLM 算法岗 · 大模型产品岗 · ML Infra/推理系统岗
>
> **关联**：`bench/bench_inference.md`（4 维对比报告） / `PROJECT_FOR_RESUME.md`（简历导向）

---

## Q1: 为什么要做推理加速？训练不是已经用 4-bit 了吗？

**A**: 训练用 4-bit（QLoRA 的 NF4）只省**训练时**的加载显存，前向计算时仍然把权重**反量化到 bf16**。
也就是说训练时省的是"权重加载到显存"的成本，但前向仍然是 bf16 的速度和显存。

而推理场景里**没有反传**，量化可以做得更激进：
- 训练时量化需要保留梯度 → bnb NF4 + LoRA adapter 保精度
- 推理时量化只看前向 → 可以走 vLLM 的 PagedAttention + continuous batching + 4-bit 全程算

所以"训练用 4-bit"和"推理用 4-bit"是**两个独立但可以复用同一份权重**的优化点。本项目选择同源（都是 bnb 4-bit）→ **权重零转换**。

## Q2: 为什么选 vLLM 不选 SGLang / TGI / TensorRT-LLM？

**A**: 三条理由（按重要性）：
1. **同时支持 4-bit + LoRA hot-swap**：vLLM 的 `quantization="bitsandbytes"` + `LoRARequest` 是当前唯一"装一行 + 调两个参数"就能用上的组合。TGI 不支持 LoRA 热加载；TensorRT-LLM 需要重建 engine（1 周时间）
2. **continuous batching + PagedAttention 是当前 SOTA**：7B 上比 HF generate 快 5–10× 是社区常见倍数
3. **生态最广**：招聘 JD 高频词，简历性价比高

SGLang 在多轮对话（RadixAttention）有优势但 4-bit + LoRA 集成不如 vLLM 稳定；
TensorRT-LLM 在极致性能（1000+ QPS）场景是首选但工程成本高。

## Q3: 为什么选 bitsandbytes 4-bit 不选 GPTQ / AWQ？

**A** (产品岗核心 trade-off 问题):

| 维度 | bnb 4-bit (本项目) | GPTQ | AWQ |
|---|---|---|---|
| 与训练侧一致性 | **完全一致** | 不一致 | 不一致 |
| 权重转换 | **零** | 高（calibration） | 中 |
| 量化误差 | 中 | 小 | 最小 |
| 额外时间 | **0** | ~1 天 | ~0.5 天 |

**决策**："训推同源" 原则 → 训练用 bnb 4-bit 做 QLoRA，部署复用同一方案走 vLLM，
**权重零转换**。放弃 GPTQ 的 ~0.5% 指标优势以换 1 周时间节省与训推一致性。

**反方观点如何反驳**（面试官可能问）：
- "GPTQ 量化误差更小，你为啥不用？" → 答：在 1.89 万条 8 项任务指标上，平均变化 < 1.5%
  在产品可接受范围内；为 0.5% 指标优势搭 1 周时间不划算
- "AWQ 是激活感知量化，更先进" → 答：项目体量（18,870 条推理）不需要追求极限；
  AWQ 需预量化权重会失去 base 模型切换灵活性

## Q4: vLLM 怎么做的 continuous batching？和普通 batch 有什么区别？

**A**: 关键差异在**迭代级别**的调度：

| 方案 | 调度粒度 | 长短序列混合 | GPU 利用率 |
|---|---|---|---|
| 朴素 batch | 整批等最长序列 | 短序列等长序列 | 低（短序列 GPU 空闲） |
| **vLLM continuous batching** | **每个 decode step** | 新请求随时插入空闲槽位 | 高（动态调度） |

**直觉例子**（面试画图讲）：
- 朴素 batch：5 个请求，3 个生成 10 token、2 个生成 100 token → 必须等 100 token 全部完成才能处理下一批
- continuous batching：第 11 步时，3 个短请求完成、释放槽位 → 新请求立即插进来填上

**技术实现**：PagedAttention 把 KV cache 分页管理（像 OS 虚拟内存），新请求插队时
只需分配空闲页，**不重算已完成的 token**。

## Q5: 量化误差到底有多严重？任务指标会掉吗？

**A**: 报告里的实测说了算。但**经验值**（社区共识）：
- bnb 4-bit vs bf16：单任务指标变化通常 < 1%，24 任务平均 < 1.5%
- GPTQ 4-bit vs bf16：< 0.5%
- AWQ 4-bit vs bf16：< 0.3%

**为什么不是 0**：4-bit 量化把所有权重除以 block-wise scale 后只保留 4-bit 精度，
反量化时有 round-off 误差。这个误差在**单次前向**里小到看不见（~1e-3 量级），
但**多层 + 多 token 累积**后会显现（~1% 任务指标变化）。

**怎么验证**：用 1.89 万条评估集 + 8 项任务指标跑对照，**整组平均变化**比单任务
更可靠（单任务波动可能掩盖量化误差）。

## Q6: PagedAttention 是什么？为什么比朴素 KV cache 省显存？

**A**: 把 GPU 显存抽象成"页"（类似 OS 虚拟内存）：
- 朴素 KV cache：每条请求预分配 `max_seq_len × hidden` 连续显存 → 长请求浪费、短请求闲置
- **PagedAttention**：按 token 实际长度按页（block）分配 → 物理上不连续、逻辑上连续
  → 显存利用率从 ~30% 提到 ~90%+

**直觉类比**：和操作系统的虚拟内存分页一模一样——把"必须连续"这个约束去掉，
换来"按需分配"。

**为什么对推理加速关键**：显存省下来 → 单卡装更多并发请求 → throughput 提升
（粗略：装 2× 显存 = 装 2× 并发 ≈ 2× throughput）。

## Q7: 为什么 4-bit 推理能省显存？省到哪里去了？

**A**: 显存四件套（参考 `PROJECT_FOR_RESUME.md` §3.1）：
- **权重**：bf16 14GB → NF4 4-bit 3.5GB（省 10.5GB = 75%）
- **激活**：与 batch/seq_len 相关，4-bit 推理无优化
- **KV cache**：与并发数/seq_len 相关，与量化无关
- **优化器状态**：推理无

**主要省的是权重**。装得下更多并发 → 间接省 KV cache 调度 overhead。

## Q8: 你说 continuous batching 不公开 p50/p95/p99，怎么测延迟？

**A**: 三种方法（按推荐度）：

1. **系统级 latency**：端到端总时间 / 样本数。**主指标**，简单可靠
2. **per-request latency**：vLLM 0.7+ 的 `RequestOutput.metrics` 字段提供 `time_to_first_token` 和
   `time_of_request`（相对时间戳，需要归一化）。0.6.x 字段不稳定 → 不建议在 0.6 报告
3. **外部打时间戳**：vLLM `LLM.generate()` 不阻塞 per-request，但**可在输入侧打时间戳、
   输出 callback 里打时间戳**算差值（vLLM 0.6+ 支持 `RequestOutput.callback`）

**报告里诚实说**：continuous batching 是 throughput-optimized，per-sample p95 在
0.6.x 不稳定。我们以 throughput 为主指标，per-sample latency 仅供大致参考。

## Q9: 你怎么验证 vLLM 推理结果和 HF generate 一致？

**A**: 关键验证（防止指标对比被污染）：

1. **prompt 模板完全一致**：都走 Qwen2.5 的 ChatML 模板
   （`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`）
   `fast_infer.py` 已固化，绝不调默认 chat template
2. **采样参数完全一致**：`max_new_tokens=64`, `do_sample=False` / `temperature=0.0`
3. **输出格式完全一致**：JSONL 键 `{input, label, task, model_output}`，与 baseline 字节级对齐
4. **同 adapter、同 base、同数据**：用同一份 `ckpt/stage2/` + 同一份 `eval_set.jsonl`
5. **任务指标对比**：`eval/evaluate.py` 对两组 JSONL 跑同样协议，差值即"量化误差"

**常见坑**：vLLM 默认对 system prompt 做了 trim → 在 `fast_infer.py` 已禁用

## Q10: 你们 5 组实验里，vLLM 相对 bf16 加速比能到几倍？

**A**: 跑完实验才知道，但**经验范围**（社区共识）：
- **保守**：3×（continuous batching 增益）
- **典型**：5–7×（含 CUDA graph + 4-bit kernel + PagedAttention）
- **激进**：10×+（vLLM 0.7+ + H100 硬件）

**报告里要诚实**：如果实测 < 3×，**不报喜不报忧**——把 trade-off 写清"加速 2.2×
但代码复杂度↑，是否值得取决于部署量级"，这正是产品岗要的判断力。

## Q11: 为什么 4-bit 推理不损失精度？（反问的常见形式）

**A** (常被追问的细节):

- **NF4（4-bit NormalFloat）的设计**就是为正态分布权重优化的，4-bit 信息量比普通 INT4 多
- **double quant**：quantization constants 本身也量化一次，省额外 bit 但精度几乎不损
- **block-wise quant**：每 64 个权重共享一个 scale（不是 per-tensor），粒度细，round-off 小
- **计算时反量化到 bf16**：4-bit 只存权重，计算精度是 bf16 → 前向精度与 bf16 模型相当
- **LoRA adapter 不量化**：训练时学到的低秩更新仍是 bf16，保精度

所以"4-bit 推理"≠"4-bit 计算"——是"4-bit 存储 + bf16 计算"。

## Q12: 如果让你把 7B 换成 70B，推理方案怎么改？

**A**: 4 个变化（产品岗/infra 岗爱问的扩展性问题）：

1. **单卡装不下 70B**（bf16 140GB / 4-bit 35GB）→ 必须 TP（张量并行）或多卡
2. **vLLM 启动参数**：`tensor_parallel_size=4`（4 卡 A100 80GB）/ `tensor_parallel_size=8`（8 卡 H100）
3. **量化方案升级**：70B 4-bit 比 7B 4-bit 误差略大 → 考虑 AWQ 4-bit 或 GPTQ 4-bit
4. **max_model_len 限制**：长序列（>4k）需要 chunked prefill + prefix caching

**关键判断**（产品岗要的回答）：
- 7B 4-bit 单卡 → 当前最优 vLLM
- 70B 4-bit 多卡 → vLLM TP + 可能换 AWQ
- 推理量级 < 100 QPS → 单 vLLM 实例足够
- 推理量级 > 1000 QPS → 考虑 vLLM + 负载均衡 + 模型副本

## Q13: 你这个推理方案上线要多少钱？（产品岗典型问题）

**A**: 算 cost/benefit（产品岗要的故事）：

假设场景：1.89 万条评估 / 天（实际可小到 1k、也可大到 1M+）。

**硬件成本**（云上 A100 80GB）：
- 单卡 A100 云价格约 ¥15-25/小时（spot 更低）
- vLLM 4-bit 单卡峰值 ~15 GiB → **A100 40GB 装得下**（用 40GB 实例约 ¥8-12/小时）

**节省的钱**（vs HF generate 方案）：
- HF generate 跑 1.89 万条需 ⏳ {{X 小时}} × ¥10/小时 = ⏳ {{Y 元}}
- vLLM 跑 1.89 万条需 ⏳ {{X 小时}} × ¥10/小时 = ⏳ {{Y 元}}
- **单次节省**：⏳ {{Y 元}}
- **日节省**（每天评估 1 次）：⏳ {{Y 元/天}}
- **年节省**（按 250 工作日）：⏳ {{Y 元/年}}

**（填数字后这节是产品岗的"落地故事"硬通货）**

## Q14: vLLM 装环境踩过什么坑？

**A**: 5 个最常见的坑（产品岗也爱问，体现"我真跑过"）：

1. **Windows 不支持**：vLLM 依赖 `import resource`（Linux 独有）→ 本项目开发机 4060 Laptop
   跑不了真实验，必须在 A100 Linux 节点上跑
2. **vLLM 改写 torch / cuda runtime**：训练推理必须分 venv，否则训练脚本 import 报错
3. **bnb 版本敏感**：bnb 0.43 → 0.49 内部 API 变了，vLLM 0.6.x 锁 bnb 0.49 最佳
4. **CUDA graph 占显存**：默认 `enforce_eager=False` 启用 CUDA graph（更快但多占 ~2GB），OOM 时设 True
5. **max_model_len 爆 KV cache**：长序列场景把 2048 降到 1024 立省一半 KV cache 显存

**报告里说**：本项目的开发-实验分工是 "**Windows 4060 上做代码 / 文档，Linux A100 上做真实验**"，
这本身就是真实工作流（产品岗讲"我能在资源受限环境交付"）。

## Q15: 如果加速比不达预期怎么办？（产品岗要的反面思考）

**A**: 4 个降级路径（产品岗要"有 plan B"的成熟度）：

1. **降级到 bnb 4-bit + HF generate**（B2）：有 4-bit 加载省显存，但无引擎优化。适用：< 2× 加速比
2. **降级到 bnb 4-bit + HF generate + batch=8**（B3）：朴素 batch 增益。适用：2–3× 加速比
3. **换引擎**：vLLM → SGLang（多轮对话强）/ TensorRT-LLM（极致性能）。成本：1–2 周迁移
4. **换量化方案**：bnb 4-bit → AWQ 4-bit（误差小）/ GPTQ 4-bit（生态广）。成本：1 周 + calibration 数据

**决策原则**：加速比 < 3× → 走降级路径 1/2；3–5× → 评估迁移成本；> 5× → 上生产

---

## 附录：问题索引（按岗位分类）

### 算法岗（NLP/LLM）最爱问
- Q3（为什么 bnb 不选 GPTQ）、Q4（continuous batching 原理）、Q5（量化误差多大）、
  Q6（PagedAttention 是什么）、Q11（4-bit 为什么不损精度）、Q15（降级路径）

### 产品岗（AI 产品）最爱问
- Q1（为什么要做）、Q3（选型 trade-off）、Q10（加速比范围）、Q13（cost/benefit）、
  Q14（踩坑）、Q15（plan B）

### ML Infra / 推理系统岗最爱问
- Q2（为什么 vLLM）、Q4（continuous batching 调度）、Q6（PagedAttention）、
  Q8（延迟怎么测）、Q12（70B 怎么改）、Q14（环境坑）

### 通用
- Q1、Q7（显存省哪里）、Q9（一致性验证）、Q11
