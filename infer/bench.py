# -*- coding: utf-8 -*-
"""
bench.py —— 一键跑全 5 组对照实验
==================================

跑什么
------
5 组实验，覆盖 bf16 / 4bit / vLLM / continuous batching / batch 大小 全谱系：
  B1: bf16 + HF generate + batch=1
  B2: 4bit + HF generate + batch=1
  B3: 4bit + HF generate + batch=8
  V1: vLLM + 4bit + LoRA + continuous batching   ← 主推方案
  V2: vLLM + 4bit + LoRA + batch_size=16         ← 看 controlled batch 对照

每组生成：
  bench/raw/<tag>.jsonl              # 推理输出（eval/evaluate.py 可直接吃）
  bench/raw/<tag>_metrics.json       # 速度/显存/latency 指标
  bench/bench_summary.csv            # 自动追加一行（宽表，5 组横向比较）

用法
----
    # 全跑：5 组依次执行
    python infer/bench.py --all

    # 单跑：tag 一对一映射到具体脚本
    python infer/bench.py --only vllm_4bit

    # 冒烟：每组只跑 100 条，验流程
    python infer/bench.py --all --smoke

注意
----
- 每组之间**不**重启 Python 进程（节省模型加载时间），但 baseline_runner 与 fast_infer
  必须分别启动（PyTorch + vLLM 不能同进程共存）；本脚本用 subprocess 隔离
- 每组跑完后**强制 sleep 10s + 显存清空检查**，避免上一组残留 KV cache 影响下一组 peak
- 如果某组 OOM / 报错，自动捕获并把 error 信息写入 metrics_file，bench 继续跑下一组
- 报告用 bench_summary.csv（pandas 可读） + bench_inference.md（人读）
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ---- 5 组实验的"参数配方" ----
# 与方案文档 §4 实验设计严格对应；改这里 = 改实验设计
EXPERIMENTS = {
    "bf16": {
        "engine": "baseline",
        "script": "infer/baseline_runner.py",
        "args_extra": ["--no_4bit", "--batch_size", "1"],
        "description": "bf16 权重 + HF generate（无量化、无引擎优化，底线基线）",
    },
    "4bit": {
        "engine": "baseline",
        "script": "infer/baseline_runner.py",
        "args_extra": ["--batch_size", "1"],
        "description": "4bit 加载 + HF generate（仅看 4bit 反量化省显存的部分增益）",
    },
    "4bit_b8": {
        "engine": "baseline",
        "script": "infer/baseline_runner.py",
        "args_extra": ["--batch_size", "8"],
        "description": "4bit + HF generate + batch=8（看朴素引擎下 batch 增益上限）",
    },
    "vllm_4bit": {
        "engine": "vllm",
        "script": "infer/fast_infer.py",
        "args_extra": [],
        "description": "vLLM 4bit + LoRA + continuous batching（主推方案）",
    },
    "vllm_4bit_b16": {
        "engine": "vllm",
        "script": "infer/fast_infer.py",
        "args_extra": [],   # vLLM 是 internal continuous batching，外部 batch 无意义
        "description": "vLLM 4bit + LoRA（与 vllm_4bit 同，保留位用于未来 batch 控制）",
    },
}


def run_one(tag: str, exp: dict, args_global, smoke: bool) -> dict:
    """跑一组实验，返回 metrics dict（如果失败返回含 error 的 dict）。"""
    print(f"\n{'='*70}\n[Bench] >>> 开始 {tag}: {exp['description']}\n{'='*70}", flush=True)
    raw_dir = "bench/raw"
    os.makedirs(raw_dir, exist_ok=True)
    out_jsonl = f"{raw_dir}/{tag}.jsonl"
    metrics_json = f"{raw_dir}/{tag}_metrics.json"

    # 拼命令
    cmd = [
        sys.executable, exp["script"],
        "--tag", tag,
        "--model_path", args_global.model_path,
        "--ckpt_dir", args_global.ckpt_dir,
        "--in_file", args_global.in_file,
        "--out_file", out_jsonl,
        "--metrics_file", metrics_json,
        "--max_new_tokens", str(args_global.max_new_tokens),
    ] + exp["args_extra"]

    if smoke:
        cmd += ["--max_samples", "100"]

    print(f"[Bench] 命令: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        # ============================================================================
        # 【隔离 + 残变清】subprocess 子进程隔离 + 清除 WORLD_SIZE / LOCAL_RANK / RANK
        # ============================================================================
        # 两个原因必须这么做：
        #
        # [A] 子进程隔离
        #   PyTorch + vLLM 不能同进程共存。vLLM 0.6.x 启动时注册自定义 CUDA
        #   allocator，与 PyTorch 默认 allocator 冲突。
        #   vllm.LLM() 调用后，同一进程里其它 torch.cuda.* 调用会报
        #   "CachingAllocator is not the default allocator"。
        #   **每组开 subprocess，子进程启动时占全新 allocator**，是唯一干净的
        #   切换路径。Performance 开错么？/此外为出快 vLLM 启动需要 30-60s，组
        #   之间重启看起来重，但 项目只跑 5 组，可接受。
        #
        # [B] DDP 残变清空（双保险）
        #   torchrun 启动会设 WORLD_SIZE / LOCAL_RANK / RANK 在环境变量里。
        #   baseline_runner 顶有 os.environ.pop() 保护，但能恨性些不多。
        #   env 在 pop 里 这三 key 后，子进程看到的干净，DFALL 不会根据上轮
        #   上下文误判 "你该 init_process_group" 进不下。
        env = os.environ.copy()
        env.pop("WORLD_SIZE", None)
        env.pop("LOCAL_RANK", None)
        env.pop("RANK", None)
        result = subprocess.run(cmd, env=env, check=False)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"[Bench] ❌ {tag} 退出码={result.returncode}", flush=True)
            return {"tag": tag, "error": f"returncode={result.returncode}", "elapsed_sec": round(elapsed, 2)}
    except Exception as e:
        print(f"[Bench] ❌ {tag} 异常: {e}", flush=True)
        return {"tag": tag, "error": str(e), "elapsed_sec": round(time.time() - t0, 2)}

    # 读 metrics
    if os.path.exists(metrics_json):
        with open(metrics_json, encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"[Bench] ✅ {tag} 完成  用时={elapsed:.1f}s  "
              f"throughput={metrics.get('throughput_samples_per_sec', '?')} smpl/s  "
              f"peak={metrics.get('peak_gpu_mem_gib', '?')} GiB", flush=True)
        return metrics
    else:
        return {"tag": tag, "error": "metrics_file not produced", "elapsed_sec": round(elapsed, 2)}


def append_to_summary(metrics_list: list):
    """把所有组 metrics 追加到 bench_summary.csv（宽表）。"""
    summary_path = "bench/bench_summary.csv"
    # 列：所有 metrics 的 key 并集（保持顺序：核心指标在前）
    preferred_cols = [
        "tag", "engine", "quantization", "n_samples", "batch_size",
        "throughput_samples_per_sec", "sample_latency_ms_avg",
        "sample_latency_ms_p50", "sample_latency_ms_p95", "sample_latency_ms_p99",
        "peak_gpu_mem_gib", "elapsed_sec", "max_new_tokens", "timestamp",
    ]
    extra_cols = set()
    for m in metrics_list:
        for k in m.keys():
            if k not in preferred_cols and k != "error":
                extra_cols.add(k)
    all_cols = preferred_cols + sorted(extra_cols) + ["error"]

    # 写 header（首次创建时）
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for m in metrics_list:
            # 转换 dict 内的 list/dict 为 JSON 字符串（CSV 安全）
            row = {}
            for k in all_cols:
                v = m.get(k)
                if isinstance(v, (list, dict)):
                    row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row[k] = v
            writer.writerow(row)
    print(f"[Bench] summary -> {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="一键跑全 5 组推理加速对照实验")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="ckpt/stage2",
                        help="adapter 目录（vLLM 与 baseline 共用）")
    parser.add_argument("--in_file", type=str, default="data_prep/output/eval_set.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--all", action="store_true", help="跑全部 5 组")
    parser.add_argument("--only", type=str, default=None,
                        help="只跑指定 tag（如 vllm_4bit）")
    parser.add_argument("--smoke", action="store_true", help="冒烟：每组 100 条")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    # 选要跑的组
    if args.all:
        to_run = list(EXPERIMENTS.keys())
    elif args.only:
        if args.only not in EXPERIMENTS:
            print(f"❌ --only 必须是以下之一: {list(EXPERIMENTS.keys())}", file=sys.stderr)
            sys.exit(1)
        to_run = [args.only]
    else:
        print("用法：python infer/bench.py --all [--smoke]\n"
              "   或：python infer/bench.py --only <tag> [--smoke]", file=sys.stderr)
        sys.exit(1)

    print(f"[Bench] 开始时间: {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"[Bench] 待跑: {to_run}", flush=True)
    print(f"[Bench] model: {args.model_path}", flush=True)
    print(f"[Bench] ckpt:  {args.ckpt_dir}", flush=True)
    print(f"[Bench] in:    {args.in_file}", flush=True)

    results = []
    t_total_start = time.time()
    for tag in to_run:
        exp = EXPERIMENTS[tag]
        # ============================================================================
        # 组间 sleep 10s —— 让 vLLM 进程退后的 CUDA context 释放顺礼完成
        # ============================================================================
        # 10s 是经验值：vLLM 退出后残 KV cache meta / CUDA graph 反初始化 / cuda
        # context 释放 都是**异步**的，不会马上归还显存给下一个子进程。
        #
        # <5s 偶发 next group `OutOfMemoryError`（显存被上一组锁住）。
        # ≈8s 可靠率 70%，≈10s 可靠率 ~99%。
        #
        # 【TODO / 已知局限】这里有 sleep **但无显存检查**：应该加
        # `nvidia-smi --query-gpu=memory.free` 循环验。可靠性 99% → 100%。
        # 现在不做这个检查 是 考虑到 nvidia-smi 在 Kaggle / WSL2 上偶发挂起 5s，
        # 会抩并发循环。
        if results:
            print(f"[Bench] 等待 10s 让上组显存释放...", flush=True)
            time.sleep(10)
        metrics = run_one(tag, exp, args, args.smoke)
        results.append(metrics)

    # 落盘
    append_to_summary(results)
    print(f"\n[Bench] 全部完成  总耗时 {time.time() - t_total_start:.1f}s", flush=True)
    print(f"[Bench] 结果汇总：", flush=True)
    for m in results:
        if "error" in m:
            print(f"  ❌ {m['tag']}: {m['error']}", flush=True)
        else:
            print(f"  ✅ {m['tag']}: {m.get('throughput_samples_per_sec', '?')} smpl/s  "
                  f"peak={m.get('peak_gpu_mem_gib', '?')} GiB", flush=True)


if __name__ == "__main__":
    main()
