# -*- coding: utf-8 -*-
"""
multi_gpu_infer.py —— 多卡数据并行推理
==========================================

目的
----
把 18870 条评估集在 N 张 GPU 上**数据并行**推理。每张卡独立 Python 进程跑
自己的 1/N 片，结果按原始顺序合并，schema 与 `infer/baseline_runner.py` /
`infer/fast_infer.py` 字节级一致，可直接喂 `eval/evaluate.py`。

为什么数据并行（DP）而非 tensor parallel（TP）
---------------------------------------------
Qwen2.5-7B 4-bit 加载后约 5–6 GB / 卡，A100-40GB 单卡吞吐 7–10 smpl/s 已可观。
- **TP**：每条样本必须跨卡 collective，本项目 PCIE-A100 互联带宽 ~32 GB/s，
  通信成本 > forward 增益，且 vLLM TP+LoRA 路径在 0.6.x 仍有坑。
- **DP**：每张卡独立 forward 自己的切片，无 collective，可线性加速到 N 卡，
  是 18870 条评估集最直接的正解。

自动切分 / 自动同步 / 自动合并
-----------------------------
1. **切分（auto-shard）**
   - 默认按"均匀切 N 份"分配到 `--gpus` 指定的卡
   - 支持 `--shard_strategy contiguous|interleaved`
     - contiguous（默认）：行 [0, M/N) 给卡0，M/N+1 ... 给卡1
       ← 与 `distributed_concat` 兼容，便于按切片位置回写
     - interleaved：行 i % N 给卡 i % N
       ← 在样本长度极度偏斜时（罕见）可让各卡 batch 长度方差更小
   - 每片写 `shards/shard_<idx>.jsonl`，**带原始行号**（用 `__line__` 内嵌
     字段或直接读取 JSON 顺序），合并时不依赖文件名顺序

2. **启动（auto-launch）**
   - `--gpus auto`：自动调 `nvidia-smi` 探测可用 GPU 索引
     （兼容 PyTorch CUDA_VISIBLE_DEVICES 限制）
   - `--gpus 0,1,2`：显式指定
   - 每张卡一个**独立 Python 子进程**，`CUDA_VISIBLE_DEVICES=<idx>`
   - 不依赖 `torchrun` / `accelerate launch`，门槛最低

3. **同步（auto-sync）**
   - `subprocess.Popen` 并行启动
   - 主循环 `wait()` 全部完成才退出 → 跑完后看总耗时
   - 任意一张子进程 returncode != 0 → 主流程 rc != 0，但**先等全部完成**
     再判失败（防止"卡 0 杀进程，卡 1 显存泄漏"）
   - 主进程可被 SIGTERM / SIGINT 安全 kill；杀掉时一并杀子进程
   - 子进程 stdout/stderr 落 `logs/gpu_<idx>.log`，方便后续 debug

4. **合并（auto-merge）**
   - 读各卡 `out_files/gpu_<idx>.jsonl`
   - 按 `__line__` 字段升序回写 `merged.jsonl`
   - schema 校验：保证与 baseline 一致（input/label/task/model_output），
     缺字段时显式报警
   - 写 `merge_report.json`：N 成功 / 失败行数 / 总耗时

用法
----
    # 自动检测 GPU 数（用所有 GPU）
    python infer/multi_gpu_infer.py \\
        --model_path /path/to/Qwen2.5-7B-Instruct \\
        --ckpt_dir ckpt/stage2 \\
        --in_file eval_set.jsonl \\
        --out_prefix bench/raw/stage2_multi \\
        --engine fast \\
        --gpus auto

    # 显式指定卡 + contiguous 切分
    python infer/multi_gpu_infer.py \\
        --model_path /path/to/Qwen2.5-7B-Instruct \\
        --ckpt_dir ckpt/stage2 \\
        --in_file eval_set.jsonl \\
        --out_prefix bench/raw/stage2_multi \\
        --engine fast \\
        --gpus 0,1,2 \\
        --shard_strategy contiguous

    # 用 baseline（HF generate）路径走多卡
    python infer/multi_gpu_infer.py ... --engine baseline --batch_size 8

环境
----
- 任何 N 卡机器（DGX / 多卡服务器 / WSL2 多卡均 OK）
- vLLM 路径：**Linux + vllm>=0.6.4**（vLLM 不支持 Windows）
- baseline 路径：与 baseline_runner 一致

注意
----
- 子进程并行启动的代价是"重复付一次 vLLM 初始化时间 ~30-60s"。每个子进程
  都跑 `LLM(...)` 一次是 DP 的固有成本，换来线性加速比。
- 上限：理论 N 张卡速度 N×，实际 0.85–0.95×（取决于 CPU/GPU 间数据传
  输和 PCIe 拓扑）。A100-PCIE 多卡实测常见 0.7–0.85×
  （PCIE 带宽受限）；NVLink 拓扑（H100/H200）可达 0.95×。
- 与 `bench.py` 配套：本脚本输出的 per-GPU metrics 在 merge_report.json 里聚
  合。可直接 append 到 bench/bench_summary.csv。
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _fmt_seconds(sec: float) -> str:
    if sec < 0 or sec != sec:
        return "--"
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# =============================================================================
# GPU 探测
# =============================================================================

def detect_gpus() -> list:
    """自动探测机器里的 GPU 索引列表。

    优先级：
      1) `nvidia-smi --query-gpu=index --format=csv,noheader` 直接拿物理 index
      2) `torch.cuda.device_count()`（兜底，但拿不到物理 index，只能用 0..N-1）

    返回：
      GPU 物理 index 列表（如 [0, 1, 2]）；失败返回 []
    """
    # 方法 1：nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
                timeout=10, stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="ignore").strip()
            ids = [int(x.strip()) for x in out.splitlines() if x.strip()]
            if ids:
                return ids
        except Exception:
            pass
    # 方法 2：torch fallback（CUDA_VISIBLE_DEVICES 影响下仍返回相对 idx）
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    return []


def parse_gpus_arg(spec: str, available: list) -> list:
    """解析 --gpus 参数：
        auto     -> available（全部）
        0,1,2    -> 字面 GPU index 列表（需要存在于 available）
        :2       -> available[:2]
        1:3      -> available[1:3]

    返回 GPU 物理 index 列表；不合法直接 sys.exit(2)。
    """
    spec = (spec or "").strip()
    if spec in ("", "auto"):
        if not available:
            print(f"❌ --gpus auto 但未探测到任何 GPU", file=sys.stderr)
            sys.exit(2)
        return list(available)
    # 支持纯列表 / slice
    if "," in spec:
        out = []
        for x in spec.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                out.append(int(x))
            except ValueError:
                print(f"❌ --gpus 列表项无效: {x!r}", file=sys.stderr)
                sys.exit(2)
        return _validate_gpu_subset(out, available)
    if ":" in spec:
        try:
            s, e = spec.split(":", 1)
            s_i = int(s) if s else 0
            e_i = int(e) if e else len(available)
            sub = available[s_i:e_i]
        except ValueError:
            print(f"❌ --gpus slice 解析失败: {spec!r}", file=sys.stderr)
            sys.exit(2)
        return sub
    # 单数字
    try:
        return _validate_gpu_subset([int(spec)], available)
    except ValueError:
        print(f"❌ --gpus 值无法解析: {spec!r}", file=sys.stderr)
        sys.exit(2)


def _validate_gpu_subset(indices: list, available: list) -> list:
    if not indices:
        print(f"❌ --gpus 结果为空", file=sys.stderr)
        sys.exit(2)
    if not available:
        print(f"❌ 机器里没有任何可用 GPU（CUDA 不可用）", file=sys.stderr)
        sys.exit(2)
    unknown = [i for i in indices if i not in available]
    if unknown:
        print(f"❌ --gpus 指定了不可用的 GPU: {unknown}; 机器可用: {available}", file=sys.stderr)
        sys.exit(2)
    # 去重保序
    seen, uniq = set(), []
    for i in indices:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


# =============================================================================
# 数据切分
# =============================================================================

def shard_rows(rows: list, n_shards: int, strategy: str) -> list:
    """按策略切分为 n_shards 个列表。"""
    n = len(rows)
    if n_shards <= 0:
        raise ValueError("n_shards 必须 > 0")
    if strategy == "contiguous":
        # 行 [a, b) 给 shard idx，区间均匀切
        shard_size = (n + n_shards - 1) // n_shards
        out = []
        for i in range(n_shards):
            s = i * shard_size
            e = min(s + shard_size, n)
            out.append(rows[s:e])
        return out
    if strategy == "interleaved":
        out = [[] for _ in range(n_shards)]
        for i, r in enumerate(rows):
            out[i % n_shards].append(r)
        return out
    raise ValueError(f"未知 shard_strategy: {strategy}")


def write_shards(rows: list, n_shards: int, strategy: str, shard_dir: Path) -> list:
    """切分并落盘 shards/shard_<idx>.jsonl，返回路径列表。"""
    shard_dir.mkdir(parents=True, exist_ok=True)
    # 清掉旧的 shard_* 避免和上轮混淆
    for old in shard_dir.glob("shard_*.jsonl"):
        old.unlink()

    splits = shard_rows(rows, n_shards, strategy)
    paths = []
    for i, split_rows in enumerate(splits):
        p = shard_dir / f"shard_{i}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths.append(str(p))
    return paths


# =============================================================================
# 子进程编排
# =============================================================================

def build_child_cmd(engine: str, args) -> list:
    """拼出每张卡要调用的子命令。"""
    if engine == "fast":
        cmd = [
            sys.executable, "infer/fast_infer.py",
            "--tag", f"multi_gpu_{args.tag_suffix}",
            "--model_path", args.model_path,
            "--in_file", "{SHARD_PATH}",  # 占位，运行时替换
            "--out_file", "{OUT_PATH}",
            "--metrics_file", "{METRICS_PATH}",
            "--max_new_tokens", str(args.max_new_tokens),
        ]
        if args.ckpt_dir:
            cmd += ["--ckpt_dir", args.ckpt_dir]
        if args.max_samples and args.max_samples > 0:
            cmd += ["--max_samples", str(args.max_samples)]
    elif engine == "baseline":
        cmd = [
            sys.executable, "infer/baseline_runner.py",
            "--tag", f"multi_gpu_{args.tag_suffix}",
            "--model_path", args.model_path,
            "--in_file", "{SHARD_PATH}",
            "--out_file", "{OUT_PATH}",
            "--metrics_file", "{METRICS_PATH}",
            "--max_new_tokens", str(args.max_new_tokens),
            "--batch_size", str(args.batch_size),
            "--max_len", str(args.max_len),
        ]
        if not args.use_4bit:
            cmd += ["--no_4bit"]
        if args.ckpt_dir:
            cmd += ["--ckpt_dir", args.ckpt_dir]
        if args.max_samples and args.max_samples > 0:
            cmd += ["--max_samples", str(args.max_samples)]
    else:
        raise ValueError(f"未知 engine: {engine}")
    return cmd


def spawn_child(idx: int, gpu_id: int, cmd_template: list, shard_path: str,
                out_path: str, metrics_path: str, log_path: str) -> subprocess.Popen:
    """以 CUDA_VISIBLE_DEVICES=<gpu_id> 启动一个子进程。

    注意：
      - WORLD_SIZE / LOCAL_RANK / RANK / MASTER_* 全部清掉，避免 DDP 残变
      - 子进程 stdout/stderr 都重定向到 log 文件
      - start_new_session=True → 本进程 Ctrl-C 时一并杀子进程
    """
    # 替换占位
    cmd = []
    for x in cmd_template:
        if x == "{SHARD_PATH}":
            cmd.append(shard_path)
        elif x == "{OUT_PATH}":
            cmd.append(out_path)
        elif x == "{METRICS_PATH}":
            cmd.append(metrics_path)
        else:
            cmd.append(x)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("WORLD_SIZE", None)
    env.pop("LOCAL_RANK", None)
    env.pop("RANK", None)
    env.pop("MASTER_ADDR", None)
    env.pop("MASTER_PORT", None)
    # PYTHONUNBUFFERED 让子进程日志实时落盘（不是被 buffer 在 pipe 里）
    env["PYTHONUNBUFFERED"] = "1"

    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    print(f"  [gpu {idx}] CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
    print(f"  [gpu {idx}] log -> {log_path}", flush=True)
    print(f"  [gpu {idx}] cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.Popen(
        cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


# =============================================================================
# 合并
# =============================================================================

def merge_outputs(out_paths: list, merged_path: Path, merge_report_path: Path,
                  expected_keys: set, shard_strategy: str = "contiguous") -> dict:
    """把各 GPU shard 的输出按原始顺序回写 merged.jsonl，schema 校验。

    保序策略
    --------
    不依赖子进程写 `__line__`（子进程代码不认这个字段）。
    顺序以主进程的 shard 划分位置为准：
      contiguous: shard_i 的输出 = 原始行 [offset_i, offset_i + count_i)
      interleaved: 第 i 个输出的原始位置 = i % n_shards 所属 shard 的第 i // n_shards 行

    返回 merge_report dict（含 schema 校验警告）。
    """
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    per_shard_count = {}
    bad_json = 0
    schema_violations = 0
    total_raw = 0
    per_shard_rows = []  # 每个 shard 的行列表（保序）

    for p in out_paths:
        shard_name = os.path.basename(p)
        per_shard_count[shard_name] = 0
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                total_raw += 1
                try:
                    r = json.loads(line)
                except Exception:
                    bad_json += 1
                    continue
                # schema 校验（input/label/task/model_output 必须有）
                if expected_keys and not expected_keys.issubset(
                    set(r.keys()) & expected_keys
                ):
                    schema_violations += 1
                rows.append(r)
                per_shard_count[shard_name] += 1
        per_shard_rows.append(rows)

    # 按 shard_strategy 重排为原始顺序
    n_shards = len(per_shard_rows)
    if shard_strategy == "contiguous":
        ordered = []
        for rows in per_shard_rows:
            ordered.extend(rows)
    elif shard_strategy == "interleaved":
        # 子进程 i 内的第 k 行对应原始行号 (k * n_shards + i)
        # 交错合并
        max_len = max(len(r) for r in per_shard_rows) if per_shard_rows else 0
        ordered = []
        for k in range(max_len):
            for i in range(n_shards):
                if k < len(per_shard_rows[i]):
                    ordered.append(per_shard_rows[i][k])
    else:
        ordered = []
        for rows in per_shard_rows:
            ordered.extend(rows)

    # 写出 merged（不显式带 __line__，eval/evaluate.py 不认这字段）
    n_written = 0
    with open(merged_path, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_written += 1

    report = {
        "merged_path": str(merged_path),
        "n_merged": n_written,
        "per_shard_count": per_shard_count,
        "total_raw_lines": total_raw,
        "bad_json_lines": bad_json,
        "schema_violations": schema_violations,
        "shard_strategy": shard_strategy,
        "merge_order_rule": (
            "contiguous: shard_i => rows [offset_i, offset_i+count_i); "
            "interleaved: per-shard k-th row => original line k*n_shards+i"
        ),
    }
    merge_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merge_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def aggregate_metrics(metrics_paths: list, agg_path: Path) -> dict:
    """聚合各卡 metrics 成 1 个 JSON。"""
    metrics = []
    for p in metrics_paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            m["__source__"] = p
            metrics.append(m)
        except Exception as e:
            print(f"  ⚠️ 读 metrics 失败 {p}: {e}", flush=True)

    if not metrics:
        agg = {"n_shards": 0, "error": "no metrics file produced"}
    else:
        # 聚合
        total_samples = sum(m.get("n_samples", 0) for m in metrics)
        # DP 下: 各 shard 同时并行，谁最慢谁是瓶颈 → wall clock = max(elapsed)
        wall_clock_max = max(m.get("elapsed_sec", 0.0) for m in metrics)
        sum_elapsed = sum(m.get("elapsed_sec", 0.0) for m in metrics)
        # 纯推理子进程 throughput 之和（不含 vLLM 启动：启动各付一次 已 overhead）
        # 这不是一个诚实的系统级 throughput，只供 "纯推理容争" 参考
        sum_thr = sum(m.get("throughput_samples_per_sec", 0.0) for m in metrics)
        # 有效系统吞吐 = 总样本 / (主进程实际壁钟) —— 是公平下报数
        peak_mem = max((m.get("peak_gpu_mem_gib", 0.0) or 0.0) for m in metrics)
        agg = {
            "n_shards": len(metrics),
            "sum_n_samples": total_samples,
            # 主进程实际壁钟（包含启动/读文件/合并）：公平下报数
            "wall_clock_sec_max": round(wall_clock_max, 2),
            "wall_clock_sec_sum": round(sum_elapsed, 2),
            # ========== 吞吐量三种口径（请看 NOTES） ==========
            # === [A] 系统 throughput（默认报告） ===
            # 总样本除以总壁钟；代表端到端处理能力，含所有启动开销。
            # 这是论文 / bench 里与单卡对比的唯一诚卖口径。
            "throughput_samples_per_sec_effective": round(
                total_samples / max(wall_clock_max, 1e-6), 3
            ),
            # === [B] 推理型 throughput (仅推理阶段，不是系统总吞吐) ===
            # 各 shard 纯推理 throughput 之和：不含子进程启动、不含主进程调度。
            # 不同子进程 启动负载各付一次，DP 下启动启动开销 ≥ N×。
            # 重要：这个值会被高估，上限是 启动开销×N 下的理论 N×，实际往往只
            # 比单卡高 N×0.7-0.95×。常作 "上限天花板参考"。
            "throughput_samples_per_sec_sum_inference_only": round(sum_thr, 3),
            # === [C] 最大单 shard throughput ===
            # 仅看最饱 shard 的吞吐。调试用。
            "throughput_samples_per_sec_max_single_shard": round(
                max(m.get("throughput_samples_per_sec", 0.0) for m in metrics), 3
            ),
            "peak_gpu_mem_gib_max": round(peak_mem, 2),
            "per_shard": metrics,
            # 明确告诉下游报告者哪个口径是公平对比口径
            "_metric_legend": {
                "throughput_samples_per_sec_effective": "主进程壁钟口径；公平对比口径",
                "throughput_samples_per_sec_sum_inference_only": "仅推理阶段加和；上限天花板上限",
                "throughput_samples_per_sec_max_single_shard": "调试用",
            },
        }
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    return agg


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="多卡数据并行推理 — 自动切分/同步/合并",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --gpus auto --engine fast --model_path /path/to/base --in_file eval_set.jsonl \\
           --out_prefix bench/raw/stage2_multi --ckpt_dir ckpt/stage2
""")
    # ---- 数据 / 输出 ----
    parser.add_argument("--in_file", type=str, required=True,
                        help="输入 JSONL（绝对路径或相对 cwd）")
    parser.add_argument("--out_prefix", type=str, required=True,
                        help="输出前缀；最终合并 -> <prefix>_merged.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="限制总样本数（-1=全部）")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="shards/logs/per-gpu-out 中间文件目录；默认 <out_prefix>_work")
    # ---- 模型 / 引擎 ----
    parser.add_argument("--model_path", type=str, required=True, help="基座模型路径")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="adapter 目录（PEFT 格式）")
    parser.add_argument("--engine", choices=["fast", "baseline"], default="fast",
                        help="fast=infer/fast_infer.py (vLLM)；baseline=infer/baseline_runner.py")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="baseline 引擎 batch size")
    parser.add_argument("--max_len", type=int, default=1024,
                        help="baseline 引擎 max_len")
    parser.add_argument("--use_4bit", dest="use_4bit", action="store_true", default=True,
                        help="baseline 引擎是否 4bit 量化（默认 True）")
    parser.add_argument("--no_4bit", dest="use_4bit", action="store_false",
                        help="baseline 引擎 bf16")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    # ---- 多卡 ----
    parser.add_argument("--gpus", type=str, default="auto",
                        help="GPU 列表：auto / 0,1,2 / :2 / 1:3")
    parser.add_argument("--shard_strategy", choices=["contiguous", "interleaved"],
                        default="contiguous", help="数据切分策略")
    # ---- 其他 ----
    parser.add_argument("--tag_suffix", type=str, default="multi",
                        help="per-GPU run 的 tag 后缀（影响子进程 metrics 文件名）")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    return main_with_args(args)


def main_with_args(args):
    """从 main() 抽出来的核心流程，方便 monkeFPatch & 单元测试。"""

    # ---- 1. 探测 GPU + 解析 --gpus ----
    available = detect_gpus()
    gpus = parse_gpus_arg(args.gpus, available)
    n_gpus = len(gpus)
    print(f"[MultiInfer] 可用 GPU: {available}")
    print(f"[MultiInfer] 使用 GPU: {gpus}  (n_gpus={n_gpus})")

    # ---- 2. 准备 work dir ----
    work_dir = Path(args.work_dir) if args.work_dir else Path(f"{args.out_prefix}_work")
    shard_dir = work_dir / "shards"
    log_dir = work_dir / "logs"
    out_dir = work_dir / "per_gpu_out"
    metrics_dir = work_dir / "per_gpu_metrics"
    for d in (shard_dir, log_dir, out_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 3. 读数据 + 切分 ----
    print(f"[MultiInfer] 读 {args.in_file}")
    rows = []
    with open(args.in_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    n_total = len(rows)
    if args.max_samples and args.max_samples > 0:
        rows = rows[:args.max_samples]
        n_total = len(rows)
    print(f"[MultiInfer] 总样本={n_total}")

    # 给每行加 __line__ 字段（合并时按它排序；不让 eval 看到也行，它不认这 key）
    for i, r in enumerate(rows):
        r["__line__"] = i

    # 切分
    shard_paths = write_shards(rows, n_gpus, args.shard_strategy, shard_dir)
    per_shard_counts = [sum(1 for _ in open(p)) for p in shard_paths]
    print(f"[MultiInfer] 切分完成: {per_shard_counts} (strategy={args.shard_strategy})")

    # ---- 4. 拼子命令模板 ----
    cmd_template = build_child_cmd(args.engine, args)
    print(f"[MultiInfer] 子命令模板: {' '.join(shlex.quote(c) for c in cmd_template)}")
    print(f"[MultiInfer] 引擎={args.engine}  模型={args.model_path}  ckpt={args.ckpt_dir}")

    # ---- 5. 并行启动子进程 ----
    print(f"[MultiInfer] 启动 {n_gpus} 个子进程 ... {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_start = time.time()
    procs = []
    for idx, gpu_id in enumerate(gpus):
        sp = shard_paths[idx]
        op = str(out_dir / f"gpu_{idx}.jsonl")
        mp = str(metrics_dir / f"gpu_{idx}.json")
        lp = str(log_dir / f"gpu_{idx}.log")
        p = spawn_child(idx, gpu_id, cmd_template, sp, op, mp, lp)
        procs.append((idx, gpu_id, p))

    # ---- 6. 同步等全部完成 ----
    # 阻塞等到所有子进程返回；用 Popen.wait()，主循环打印 periodic 进度
    rc_all = []
    last_print = time.time()
    try:
        while True:
            all_done = all(p.poll() is not None for _, _, p in procs)
            now = time.time()
            if not all_done:
                if now - last_print > 30:
                    elapsed = now - t_start
                    running = [(idx, g, p.pid) for idx, g, p in procs if p.poll() is None]
                    print(f"  [{_fmt_seconds(elapsed)}] 运行中: {running}", flush=True)
                    last_print = now
                time.sleep(0.5)
                continue
            # 全部完成
            for idx, gpu_id, p in procs:
                rc = p.returncode
                rc_all.append((idx, gpu_id, rc))
                if rc != 0:
                    print(f"  ❌ gpu {idx} (cuda {gpu_id}) returncode={rc}  log={log_dir / f'gpu_{idx}.log'}", flush=True)
                else:
                    print(f"  ✅ gpu {idx} (cuda {gpu_id}) 完成", flush=True)
            break
    except KeyboardInterrupt:
        print(f"\n[MultiInfer] KeyboardInterrupt → 杀掉所有子进程 ...", flush=True)
        for _, _, p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        sys.exit(1)

    elapsed_total = time.time() - t_start
    n_success = sum(1 for _, _, rc in rc_all if rc == 0)
    print(f"[MultiInfer] 全部子进程结束  rc_all={rc_all}  "
          f"用时={_fmt_seconds(elapsed_total)}  成功 {n_success}/{n_gpus}", flush=True)

    # ---- 7. 合并 ----
    if n_success == n_gpus:
        out_paths = [str(out_dir / f"gpu_{idx}.jsonl") for idx in range(n_gpus)]
        merged_path = Path(f"{args.out_prefix}_merged.jsonl")
        merge_report_path = Path(f"{args.out_prefix}_merge_report.json")
        # 期望 schema（与 baseline 一致）
        expected_keys = {"input", "label", "task", "model_output"}
        report = merge_outputs(out_paths, merged_path, merge_report_path,
                               expected_keys, shard_strategy=args.shard_strategy)
        print(f"[MultiInfer] 合并完成: {report['n_merged']} 条 -> {merged_path}")
        if report["bad_json_lines"] or report["schema_violations"]:
            print(f"[MultiInfer] ⚠️ 合并报告警告: {report}", flush=True)

        # 聚合 metrics
        metrics_paths = [str(metrics_dir / f"gpu_{idx}.json") for idx in range(n_gpus)]
        agg_path = Path(f"{args.out_prefix}_agg_metrics.json")
        agg = aggregate_metrics(metrics_paths, agg_path)
        print(f"[MultiInfer] 聚合指标 -> {agg_path}")
        if "throughput_samples_per_sec_effective" in agg:
            print(f"[MultiInfer] 系统吞吐 (总样本/总壁钟): {agg['throughput_samples_per_sec_effective']} smpl/s")
            print(f"[MultiInfer] 推理型吞吐 (仅推理子进程加和): {agg['throughput_samples_per_sec_sum_inference_only']} smpl/s")
            print(f"[MultiInfer] 单 shard 吞吐最大值: {agg['throughput_samples_per_sec_max_single_shard']} smpl/s")
            print(f"[MultiInfer] peak GPU mem (max over shards): {agg['peak_gpu_mem_gib_max']} GiB")
    else:
        print(f"[MultiInfer] ❌ {n_gpus - n_success}/{n_gpus} 张卡失败，未合并", flush=True)
        sys.exit(2)

    print(f"[MultiInfer] DONE  总耗时 {_fmt_seconds(elapsed_total)}")
    return rc_all


if __name__ == "__main__":
    rc_all = main()
    if not rc_all:
        sys.exit(2)
    sys.exit(0 if all(rc == 0 for _, _, rc in rc_all) else 2)
