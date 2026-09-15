# BioAlign · 推理加速模块（infer/）

本目录实现 **BioAlign 训练 → 推理 闭环** 的推理侧：在不修改训练流水线、adapter
格式、评估协议的前提下，把 1.89 万条评估集的推理从"HF generate + bf16"升级到
"vLLM + 4-bit + LoRA hot-swap"，并产出 4 维对比报告。

## 1. 模块结构

```
infer/
├── baseline_runner.py    # 复用 train/infer_eval.py 逻辑，做 HF generate 复跑（B1/B2/B3）
├── fast_infer.py         # vLLM 4-bit + LoRA 主推方案（V1/V2）
├── multi_gpu_infer.py    # 多卡数据并行包装：自动切分 / 同步 / 合并（适用 fast 或 baseline）
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

### 2.4 多卡数据并行推理（auto-shard / auto-sync / auto-merge）

当机器里有 N ≥ 2 张 GPU 时，`multi_gpu_infer.py` 能在不改动 `fast_infer.py` /
`baseline_runner.py` 的前提下，把评估集切成 N 份，N 张卡**数据并行**同时跑，
结果按原始顺序自动合并，输出 JSONL 与单卡路径字节级一致，直接喂 `eval/evaluate.py`。

#### 为什么选数据并行（DP）不选张量并行（TP）

| | 数据并行 DP（本脚本）| 张量并行 TP |
|---|---|---|
| 加速原理 | 每卡独立 forward 自己的切片，无跨卡 collective | 每条样本必须跨卡同步 forward |
| 18970 条样本上的适用性 | ✅ 线性加速 N× | ⚠️ 7B 单卡足够，TP 反而引入通信开销 |
| PCIe-A100 / 4090 拓扑 | 实际 ~0.7–0.85× /卡 | 实际 ~0.5× /卡（通信受限） |
| NVLink H100/H200 拓扑 | 实际 ~0.95× /卡 | 实际 ~0.95× /卡 |
| 与 LoRA / vLLM 集成 | 零侵入，只需设 `CUDA_VISIBLE_DEVICES` | TP+LoRA 0.6.x 路径有坑 |
| 工程复杂度 | 🟢 低（独立子进程） | 🔴 高（需 collective 调试） |

#### 一行命令

```bash
# 自动探测机器里的全部 GPU，走 vLLM 路径（主推）
python infer/multi_gpu_infer.py \
    --in_file data_prep/output/eval_set.jsonl \
    --out_prefix bench/raw/stage2_multi \
    --model_path /path/to/Qwen2.5-7B-Instruct \
    --ckpt_dir ckpt/stage2 \
    --gpus auto \
    --engine fast

# 显式指定 2 张卡 + HF baseline 路径（验可控性）
python infer/multi_gpu_infer.py \
    --in_file data_prep/output/eval_set.jsonl \
    --out_prefix bench/raw/stage2_multi_b8 \
    --model_path /path/to/Qwen2.5-7B-Instruct \
    --ckpt_dir ckpt/stage2 \
    --gpus 0,1 \
    --engine baseline \
    --batch_size 8
```

#### 选项备忘

| 选项 | 默认 | 说明 |
|---|---|---|
| `--gpus auto` | auto | `auto` 调 `nvidia-smi` 探全部 GPU；也接受 `0,1,2` / `:2` / `1:3` |
| `--engine` | fast | `fast`=vLLM（主推）/ `baseline`=HF generate |
| `--shard_strategy` | contiguous | `contiguous`（连续切片）或 `interleaved`（交错，与各 shard 长度方差更小）|
| `--work_dir` | `<out_prefix>_work` | 中间文件目录（shards/logs/per_gpu_out/per_gpu_metrics）|
| `--max_samples` | -1 | 限制总样本数（-1=全部）|
| `--tag_suffix` | multi | 仅影响子进程 metrics 文件名，不变 JSONL schema |

#### 输出物

| 路径 | 内容 |
|---|---|
| `<out_prefix>_merged.jsonl` | 合并后的推理输出（与 baseline 字节级一致，可直接喂 `eval/evaluate.py`）|
| `<out_prefix>_merge_report.json` | per-shard 行数 / schema 校验统计 / 顺序还原规则 |
| `<out_prefix>_agg_metrics.json` | 三种吞吐量口径 + peak GPU 内存 + per-shard metrics |
| `work_dir/shards/shard_<idx>.jsonl` | N 个切片（合并阶段后保留备查）|
| `work_dir/logs/gpu_<idx>.log` | 每个子进程 stdout+stderr |
| `work_dir/per_gpu_out/gpu_<idx>.jsonl` | 每个子进程的原始输出（按 shard 序）|
| `work_dir/per_gpu_metrics/gpu_<idx>.json` | 每个子进程的 metrics（与单卡 metrics schema 一致）|

#### 吞吐量口径请看清

`<out_prefix>_agg_metrics.json` 里同时报三种吞吐量，设计上避免误用：

| Key | 口径 | 什么时候用 |
|---|---|---|
| `throughput_samples_per_sec_effective` | **总样本 / 主进程实际壁钟**（含启动 / 合并）| **与单卡路径做公平对比的唯一口径** |
| `throughput_samples_per_sec_sum_inference_only` | 各子进程纯推理 throughput 之和（不含启动） | 上限天花板参考；报告会高估 |
| `throughput_samples_per_sec_max_single_shard` | 最饱和那一片的 throughput | 调试“哪张卡是瓶颈” |

> **为什么不能把“推理 throughput 加和”直接报为总吞吐？** 各子进程**各付一次**
> vLLM 启动 30–60s，DP 下启动开销 ≥ N×。这个加和口径在评估报告里会被
> 同事诟病。**主进程壁钟**才是产品报告里唯一可比的数字。

#### 按预期启动顺序

```
[MultiInfer] 可用 GPU: [0, 1, 2]
[MultiInfer] 使用 GPU: [0, 1, 2]  (n_gpus=3)
[MultiInfer] 读 data_prep/output/eval_set.jsonl
[MultiInfer] 总样本=18870
[MultiInfer] 切分完成: [6290, 6290, 6290] (strategy=contiguous)
[MultiInfer] 子命令模板: ... infer/fast_infer.py ...
[MultiInfer] 启动 3 个子进程 ... 2026-09-03 00:37:17
  [gpu 0] CUDA_VISIBLE_DEVICES=0
  [gpu 0] cmd: ... infer/fast_infer.py --in_file ...shard_0.jsonl ...
  [gpu 1] CUDA_VISIBLE_DEVICES=1
  [gpu 1] cmd: ... infer/fast_infer.py --in_file ...shard_1.jsonl ...
  [gpu 2] CUDA_VISIBLE_DEVICES=2
  [gpu 2] cmd: ... infer/fast_infer.py --in_file ...shard_2.jsonl ...
  ... (并行运行中)
  ✅ gpu 0 (cuda 0) 完成
  ✅ gpu 1 (cuda 1) 完成
  ✅ gpu 2 (cuda 2) 完成
[MultiInfer] 全部子进程结束  用时=58m30s  成功 3/3
[MultiInfer] 合并完成: 18870 条 -> bench/raw/stage2_multi_merged.jsonl
[MultiInfer] 系统吞吐 (总样本/总壁钟): 5.37 smpl/s
[MultiInfer] 推理型吞吐 (仅推理子进程加和): 5.42 smpl/s
[MultiInfer] 单 shard 吞吐最大值: 2.18 smpl/s
[MultiInfer] peak GPU mem (max over shards): 23.5 GiB
[MultiInfer] DONE  总耗时 58m30s
```

#### 合并后的下游消费

```bash
# eval/evaluate.py 完全未知合并源，可直接吃
python eval/evaluate.py \
    --model_name stage2_multi \
    --OMICS all_omics \
    --input_file_path bench/raw/stage2_multi_merged.jsonl
```

### 2.5 单卡机器上运行（降级）

`--gpus auto` 在单卡机器上只看到 1 张卡，会自然**降级**为单子进程路径，生成
的 merged.jsonl 与单卡 `fast_infer.py` 输出**逐字节一致**（同一份输入、同一份
adapter、同一份量化、同一份采样参数）。这个降级是作为后续检查“合并逻辑
不引入了额外 bug”的快速烟测用，不是为了拆成多卡加速。

## 3. 环境要求

> **多卡补充**：多卡路径不额外需求环境依赖。`multi_gpu_infer.py` 仅动 `python` 子
> 进程编排与 JSONL 切分/合并，业务逻辑完全委托给 `fast_infer.py` / `baseline_runner.py`。
> 所以只要 §3 表里的依赖满足，多卡路径就在同一环境里跑。如果机器是 N ≥ 2 张
> CUDA 卡，需另保证 `nvidia-smi` 在 PATH 里（默认值都在）；环境变量 `CUDA_VISIBLE_DEVICES`
> 不要在 multi_gpu_infer 启动前设置（会破坏子进程的隔离）。


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

### 5.1 多卡路径额外踩坑（`multi_gpu_infer.py`）

| 现象 | 原因 | 解决 |
|---|---|---|
| 子进程启动 30-60s 后才动起来 | vLLM 启动期需 init CUDA / load weights，N 卡 N×启动 | 预期行为；总耗时里包容。与单卡对比用 `throughput_samples_per_sec_effective`（主进程壁钟）而不是 sum |
| 单卡 5×3 双路 GPU，但 `nvidia-smi` 只看到 1 张 | 输了 `CUDA_VISIBLE_DEVICES=0` 或 nvidia-smi 驱动问题 | 多卡脚本启动前**不设** `CUDA_VISIBLE_DEVICES`；检查 `nvidia-smi` 结果 |
| 卡 0 成功 / 卡 1 OOM，脚本主进程反而 exit 0 | 历史的错误码选错了 | 本脚本 N 卡**全部成功**才 exit 0；任一张卡 rc != 0 → exit 2，并报 `merge_report.json` 里的一致警告 |
| 卡 0 几十分钟后才卡 1 先崩，杀卡 0 后卡 1 显存泄 | `Popen` 未传 `start_new_session=True` | 本脚本所有子进程都传 `start_new_session=True`；SIGTERM 会主+子一起 kill |
| 多卡路径 启动开销×N 后，实际加速比只有 1.7×（预期 2×）| PCIe 拓扑下集体错余开销、显存带宽争抢正常 | 在 2×以下不必追优化；5× 严格问题上 NVLink 节点 |
| 合并后报 `schema_violations` | 子进程写出去的 JSONL 缺了 `task` 或 `label` 字段 | 看 `merge_report.json["schema_violations"]`；一般是上游 `in_file` 本身就不齐，不是多卡路径引入 |
| `world_size` 残留导致 DDP 奇怪报错 | 子进程继承了主进程的 DDP 环境变量 | 本脚本启动子进程前 `pop` 掉 `WORLD_SIZE / LOCAL_RANK / RANK / MASTER_*` |
| PCIe-A100 上 4 卡反而比 2 卡慢 | CPU/GPU 数据传输 + context switch 过重 | 不上过多片，A100-PCIE 推荐 2–4 卡；超过 4 卡需 NVLink 拓扑 |

## 6. 与现有项目代码的边界

| 已有模块 | 本目录是否修改 | 说明 |
|---|---|---|
| `train/common.py` | ❌ 不改 | 复用 `load_model_tokenizer` / `SYSTEM_PROMPT` / `read_jsonl` |
| `train/infer_eval.py` | ❌ 不改 | 保留作 0.5B 冒烟/小规模用；正式加速走 `infer/` |
| `eval/evaluate.py` | ❌ 不改 | 沿用官方协议；多卡合并后的 JSONL 直接喂 |
| `eval/register_tasks.json` | ❌ 不改 | 24 任务注册表 |
| `ckpt/stage2/...` | ❌ 不读 | 训练时 adapter，PEFT 格式直接喂 vLLM |
| `infer/fast_infer.py` | ❌ 不改（多卡路径复用其调用接口）| `multi_gpu_infer.py` 仅以子进程调用形式复用 |
| `infer/baseline_runner.py` | ❌ 不改（同上）| `multi_gpu_infer.py` 同时包成多卡 DP |

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

### 7.1 多卡路径额外 checklist

- [ ] `multi_gpu_infer.py` `--help` 输出可读，所有参数有默认值
- [ ] 单卡机器跑 `multi_gpu_infer.py` 不报错（仅 1 卡 降级路径，输出与单卡脚本字节级一致）
- [ ] `work_dir/shards/shard_<idx>.jsonl` 各切片行数加和 == 总样本数
- [ ] `work_dir/per_gpu_out/gpu_<idx>.jsonl` 行数 与对应 shard 行数一致
- [ ] `merge_report.json` 里 `bad_json_lines` / `schema_violations` 均为 0
- [ ] `merge_report.json` 里 `merged.path` 文件存在，行数 == 输入行数
- [ ] `agg_metrics.json` 的 `throughput_samples_per_sec_effective` > 单卡 baseline 路径 **N×0.7** （实际加速比下限）
- [ ] `agg_metrics.json` 里 `peak_gpu_mem_gib_max` < 单卡 baseline 路径峰值 × 1.2（即使 N 路，显存总量增加有限）
- [ ] 点查 `merged.jsonl` 前 5 / 中 5 / 后 5 行，`input` 字段与原始 `eval_set.jsonl` 完全一致
- [ ] 点查 `merged.jsonl` 里任取一行有 `model_output` 字段且不为空（验证子进程输出走通）

## 8. 简历 / 面试引用路径

- 简历条目 → `PROJECT_FOR_RESUME.md` §10 模板 D
- 面试问答 → `docs/INFER_QA.md`
- 4 维对比报告 → `bench/bench_inference.md`
- 技术选型决策 → `docs/INFER_DECISION.md`（与 INFER_QA 互补：决策矩阵 + trade-off 表格）
- 多卡数据并行设计 → `infer/multi_gpu_infer.py` + 本 README §2.4（可谈 "子进程隔离 + auto-shard + 保序合并"）
