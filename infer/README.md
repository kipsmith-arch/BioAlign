# BioAlign · 推理加速模块（infer/）

本目录实现 **BioAlign 训练 → 推理 闭环** 的推理侧：在不修改训练流水线、adapter
格式、评估协议的前提下，把 1.89 万条评估集的推理从"HF generate + bf16"升级到
"vLLM + 4-bit + LoRA hot-swap"，并产出 4 维对比报告。

## 1. 模块结构

```
infer/
├── baseline_runner.py    # 复用 train/infer_eval.py 逻辑，做 HF generate 复跑（B1/B2/B3）
├── fast_infer.py         # vLLM 4-bit + LoRA 主推方案（V1/V2）
├── bench.py              # 一键跑 5 组对照实验 → bench/raw/ + bench_summary.csv
└── README.md             # 本文档
```

输出与现有 `train/infer_eval.py` **字节级一致**：每行一个 JSON，
键 = `{input, label, task, model_output}`，可直接被 `eval/evaluate.py` 评估。

## 2. 快速上手

### 2.1 单组实验

```bash
# B1：bf16 + HF generate + batch=1（底线基线）
python infer/baseline_runner.py \
    --tag bf16 \
    --model_path /path/to/Qwen2.5-7B-Instruct \
    --ckpt_dir ckpt/stage2 \
    --in_file data_prep/output/eval_set.jsonl \
    --out_file bench/raw/baseline_bf16.jsonl \
    --metrics_file bench/raw/baseline_bf16_metrics.json \
    --no_4bit --batch_size 1

# B2：4bit + HF generate + batch=1（看仅 4-bit 加载的省显存）
python infer/baseline_runner.py \
    --tag 4bit \
    --model_path /path/to/Qwen2.5-7B-Instruct \
    --ckpt_dir ckpt/stage2 \
    --in_file data_prep/output/eval_set.jsonl \
    --out_file bench/raw/baseline_4bit.jsonl \
    --metrics_file bench/raw/baseline_4bit_metrics.json \
    --batch_size 1

# V1：vLLM 4-bit + LoRA + continuous batching（主推方案）
python infer/fast_infer.py \
    --tag vllm_4bit \
    --model_path /path/to/Qwen2.5-7B-Instruct \
    --ckpt_dir ckpt/stage2 \
    --in_file data_prep/output/eval_set.jsonl \
    --out_file bench/raw/vllm_4bit.jsonl \
    --metrics_file bench/raw/vllm_4bit_metrics.json
```

### 2.2 跑全 5 组

```bash
# A100 上一次性跑完 5 组对照
python infer/bench.py --all

# 冒烟（每组 100 条）—— 验流程 + vLLM 装环境是否正常
python infer/bench.py --all --smoke
```

每组完成后落盘：
- `bench/raw/<tag>.jsonl` —— 推理输出
- `bench/raw/<tag>_metrics.json` —— 速度/显存/latency 指标
- `bench/bench_summary.csv` —— 自动追加 5 组横向比较（宽表）

### 2.3 评估任务指标

每组 JSONL 输出后，调用项目原有的 `eval/evaluate.py` 算 8 项任务指标：

```bash
python eval/evaluate.py \
    --model_name <tag> \
    --OMICS all_omics \
    --input_file_path bench/raw/<tag>.jsonl
```

> `eval/evaluate.py` 会在 `logging/` 目录写 `metrics_<tag>_all_omics_<timestamp>.log`，
> 把 8 项任务指标落盘。把每组的这个 log 拷到 `bench/raw/<tag>_task_metrics.txt`，
> 即可在报告里横向对比"任务指标 vs 加速比 vs 显存"的 trade-off。

## 3. 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | ≥ 3.9 | 3.10/3.11 稳定 |
| torch | ≥ 2.3 | 与 transformers 4.52 / vLLM 0.6+ 兼容 |
| transformers | 4.52.x | 与训练侧同版本（`train/common.py` 用 4.52.1） |
| peft | 0.18.x | 与训练侧同版本（用 0.18.1） |
| bitsandbytes | **0.49.2** | 训练侧同版本；vLLM bnb 路径对版本敏感 |
| **vLLM** | **≥ 0.6.4** | 7B bnb 4-bit + LoRA 稳定版本 |
| tqdm | 任意 | 可选，缺则降级为 print 进度 |

### 3.1 安装（Linux + A100 / A800）

```bash
# 与训练侧 venv 分开（vLLM 会装一套自己的 torch / cuda runtime）
conda create -n bioalign-infer python=3.10 -y
conda activate bioalign-infer

# 1) 装 PyTorch（CUDA 12.4 适配 vLLM 0.6+）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 2) 装训练侧对齐的 transformers / peft / bnb
pip install transformers==4.52.1 peft==0.18.1 bitsandbytes==0.49.2

# 3) 装 vLLM（推荐 pip 自动解析兼容版本）
pip install vllm>=0.6.4

# 4) 可选
pip install tqdm pandas
```

### 3.2 ⚠️ 不支持 Windows

vLLM 依赖 `import resource`（Linux 独有），Windows 上 import 即报错：

```
ModuleNotFoundError: No module named 'resource'
```

**本项目开发机（RTX 4060 Laptop + Windows 11）无法跑 vLLM 真实验**。
实际推理实验必须在 **A100 Linux 节点** 上跑。

代码本机可读、可改、可 lint，但**真实验只能 A100**。这是产品岗汇报里要讲清的开发-实验分工。

## 4. 关键设计决策

### 4.1 为什么选 vLLM 不选 SGLang / TGI / TensorRT-LLM

| 引擎 | 7B 4-bit 支持 | LoRA hot-swap | 安装难度 | 文档完整度 |
|---|---|---|---|---|
| **vLLM 0.6+** | ✅ 一等公民 | ✅ `LoRARequest` API | 🟢 pip 一行 | 🟢 完整 |
| SGLang | ✅ 支持 | ⚠️ 实验性 | 🟡 多 | 🟡 增长中 |
| TGI (HF) | ✅ bitsandbytes | ❌ 不支持 | 🟡 Rust 编译 | 🟢 完整 |
| TensorRT-LLM | ✅ INT4/INT8 | ⚠️ 需 engine 重建 | 🔴 重（需 trtllm-build） | 🟡 |

**选 vLLM 的理由**（产品岗的"选型决策"素材）：
1. 唯一同时满足"4-bit + LoRA hot-swap + 装一行"的引擎
2. continuous batching 是当前 SOTA，7B 上比 HF generate 快 5–10× 是社区常见倍数
3. 与训练侧 bitsandbytes 同方案，**权重零转换**（不用 GPTQ 那种重新 calibration）
4. 生态最广，招聘 JD 高频词

### 4.2 为什么选 bitsandbytes 4-bit 不选 GPTQ / AWQ

| 维度 | bitsandbytes 4-bit (NF4) | GPTQ 4-bit | AWQ 4-bit |
|---|---|---|---|
| 训练时量化感知 | ✅ QLoRA 训练就用 | ❌ PTQ 静态 | ❌ PTQ 静态 |
| 部署时权重转换 | ✅ **零转换** | ❌ 需 calibration + 量化 | ❌ 需预量化权重 |
| 量化误差 | 略大（动态反量化） | 较小 | 最小 |
| 与训练侧兼容性 | **完全复用** | 不兼容 | 不兼容 |
| 简历故事性 | "训推同源" | "PTQ 全流程" | "激活感知量化" |

**结论**（已写入 `docs/INFER_QA.md`）：
- 项目已用 bnb 4-bit 做 QLoRA 训练 → 部署也用 bnb 4-bit → **权重零转换、训推一致**
- GPTQ / AWQ 需要单独 calibration 数据 + 量化脚本，**1 周时间增量收益不划算**（产品岗：成本/收益分析）
- 报告里**诚实标注**"为追求训推同源，放弃 GPTQ 的 ~0.5% 指标优势"

### 4.3 为什么 LoRA 不 merge 进 base

| 方案 | 优点 | 缺点 |
|---|---|---|
| **不 merge（LoRARequest）** | 同一 base 切换多 adapter | 微小推理开销 |
| merge_and_unload 后量化 | 推理更快 | 失去多版本灵活性；merge 慢；重量化慢 |

项目可能需要同时跑 stage2 / stage3 两个 adapter 对比 → **保留 PEFT 格式 + LoRARequest**
是更工程友好的选择。

### 4.4 为什么输出格式与 baseline 字节级一致

保证 `eval/evaluate.py` 直接吃两组输出，**任务指标横向对比零摩擦**。
任何格式差异都会污染指标对比（"vLLM 输出比 baseline 多了个空格？"这种坑产品岗最怕）。

## 5. 已知踩坑（vLLM 0.6.x 实测）

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'resource'` | Windows / macOS | 换 Linux 节点 |
| `ImportError: undefined symbol: ...` 装 vLLM 后训练脚本报错 | vLLM 改写了 torch / cuda runtime | **训练推理分 venv**（强烈建议） |
| `OutOfMemory` 启动期 | `gpu_memory_utilization=0.95` 太高 | 降到 0.9 → 0.85 |
| `OutOfMemory` 长序列 | KV cache 爆 | `max_model_len` 从 2048 降到 1024；或 `enforce_eager=True` 省 CUDA graph 显存 |
| LoRA 加载后输出"乱码" | prompt 模板与训练时不一致 | 用与 `train/common.py` 完全相同的 ChatML 模板（`fast_infer.py` 已固化） |
| `bitsandbytes` 版本不匹配 | bnb 0.43 → 0.49 API 微变 | 锁 0.49.2（与训练侧同） |
| 评估指标比 baseline 低 > 2% | vLLM 0.6.x 早期 bnb 4-bit 数值误差 | 升级 vLLM 至 0.6.4+；如还低，报告里诚实标注 trade-off |

## 6. 与现有项目代码的边界

| 已有模块 | 本目录是否修改 | 说明 |
|---|---|---|
| `train/common.py` | ❌ 不改 | 复用 `load_model_tokenizer` / `SYSTEM_PROMPT` / `read_jsonl` |
| `train/infer_eval.py` | ❌ 不改 | 保留作 0.5B 冒烟/小规模用；正式加速走 `infer/` |
| `eval/evaluate.py` | ❌ 不改 | 沿用官方协议 |
| `eval/register_tasks.json` | ❌ 不改 | 24 任务注册表 |
| `ckpt/stage2/...` | ❌ 不读 | 训练时 adapter，PEFT 格式直接喂 vLLM |

**零侵入设计**：本目录纯增量，对训练 / 数据 / 评估三个既有模块**完全无影响**。
这本身是产品岗要的"在不破坏现有系统的前提下做增量"的能力展示。

## 7. 验证 checklist（跑完实验后自查）

- [ ] 5 组实验的 `bench/raw/<tag>.jsonl` 都有产出，文件大小与样本数成正比
- [ ] 5 组 `bench_summary.csv` 5 行齐全，无 NaN / "?"
- [ ] vLLM 4-bit 相对 bf16 加速比 **≥ 3×**（社区 SOTA 倍数下限）
- [ ] vLLM 4-bit 相对 4bit + batch=8 加速比 ≥ 1.5×（continuous batching 增益）
- [ ] vLLM 4-bit 任务指标相对 bf16 baseline 平均变化 < 1.5%
- [ ] peak 显存 vLLM 4-bit < baseline bf16 30%+（4-bit 加载 + PagedAttention 双重省）
- [ ] `bench/bench_inference.md` 4 维表填完，Go/No-Go 判定清晰
- [ ] `docs/INFER_QA.md` 至少 10 个问答写完

## 8. 简历 / 面试引用路径

- 简历条目 → `PROJECT_FOR_RESUME.md` §10 模板 D
- 面试问答 → `docs/INFER_QA.md`
- 4 维对比报告 → `bench/bench_inference.md`
- 技术选型决策 → `docs/INFER_DECISION.md`（与 INFER_QA 互补：决策矩阵 + trade-off 表格）
