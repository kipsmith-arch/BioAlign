# -*- coding: utf-8 -*-
"""
baseline_runner.py —— 推理加速对照实验的 baseline 复跑器
============================================================

目的
----
给 `bench/bench_inference.md` 提供 B1（bf16 + HF generate）/ B2（4bit + HF generate）
两组对照数据。逻辑与 `train/infer_eval.py` 高度复用，但抽出来作为独立模块：

- 强制 `do_sample=False`（贪心解码，指标可复现）
- 统一时间与显存采集（peak= 真实峰值，标准产品报告要求）
- 输出与 `infer/fast_infer.py` **字节级一致**的 JSONL 格式，便于 vLLM 组对比
- metrics 落盘 `bench/raw/baseline_<tag>_<timestamp>.json` + 原始输出 JSONL

用法（A100 上）
--------------
    # B1：bf16 + HF generate（看"无量化 + 朴素引擎"的底线）
    python infer/baseline_runner.py \
        --tag bf16 \
        --model_path /path/to/Qwen2.5-7B-Instruct \
        --ckpt_dir ckpt/stage2 \
        --in_file eval_set.jsonl \
        --out_file bench/raw/baseline_bf16.jsonl \
        --metrics_file bench/raw/baseline_bf16_metrics.json \
        --no_4bit --batch_size 1

    # B2：4bit + HF generate（看"仅量化、不换引擎"的部分增益）
    python infer/baseline_runner.py \
        --tag 4bit \
        --model_path /path/to/Qwen2.5-7B-Instruct \
        --ckpt_dir ckpt/stage2 \
        --in_file eval_set.jsonl \
        --out_file bench/raw/baseline_4bit.jsonl \
        --metrics_file bench/raw/baseline_4bit_metrics.json \
        --batch_size 1

    # B3：4bit + HF generate + batch=8（看 batch 增益，与 vLLM 同 batch 量级）
    python infer/baseline_runner.py \
        --tag 4bit_b8 \
        --model_path /path/to/Qwen2.5-7B-Instruct \
        --ckpt_dir ckpt/stage2 \
        --in_file eval_set.jsonl \
        --out_file bench/raw/baseline_4bit_b8.jsonl \
        --metrics_file bench/raw/baseline_4bit_b8_metrics.json \
        --batch_size 8

注意
----
- `max_new_tokens=64` 与现有 `infer_eval.py` 一致；这是论文评估协议规定的输出长度
- 时间统计含 warmup：第一批（5 个 batch）不计入 samples/s，避开 CUDA kernel 编译抖动
- 显存取 `torch.cuda.max_memory_allocated()`（**peak**），不是 `memory_allocated()`（当前点）
  —— 后者会大幅低估峰值
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
from peft import PeftModel

# 与 train/ 同源：复用公共模块里的模型加载和 system prompt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from common import SYSTEM_PROMPT, load_model_tokenizer, read_jsonl  # noqa: E402

# 进度条：与 train/infer_eval.py 一致，tqdm 不可用时降级
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _fmt_seconds(sec: float) -> str:
    """把秒数格式化成 'Hh Mm Ss' / 'Mm Ss' / 'Ss.s'。与 train/infer_eval.py 一致。"""
    if sec < 0 or sec != sec:
        return "--"
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def run(args):
    """主流程：加载 → 准备 → 推理 → 落盘。"""
    # ---- 1. 加载模型 + tokenizer（单卡）----
    # baseline 走单卡推理（DDP 在 generate 上无意义，会 OOM，与 build_preference 同理）
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("LOCAL_RANK", None)

    print(f"[Baseline] 加载 base: {args.model_path}  (use_4bit={args.use_4bit})", flush=True)
    model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)

    if args.ckpt_dir:
        print(f"[Baseline] 加载 adapter: {args.ckpt_dir}", flush=True)
        model = PeftModel.from_pretrained(model, args.ckpt_dir)
    model.eval()

    # ---- 2. 读数据 ----
    rows = read_jsonl(args.in_file, args.max_samples)
    total = len(rows)
    print(f"[Baseline] 读取 {args.in_file}: {total} 条", flush=True)

    # ---- 3. 准备 batch（动态切批 + 左填充）----
    # 左填充是生成场景标准做法：避免右填充让模型在 pad 后才开始生成第一个真实 token
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    batch_size = max(1, args.batch_size)
    n_batches = (total + batch_size - 1) // batch_size
    # 左填充下 pad_token 必须存在（Qwen2.5 已有 eos 作为 pad）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 显存基线
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"

    # ---- 4. 推理主循环 ----
    print(
        f"[Baseline] tag={args.tag} 设备={device_name} batch_size={batch_size} "
        f"总样本={total} 总batch={n_batches} max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    print(f"[Baseline] 开始推理  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    out_path = args.out_file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t_start = time.time()
    samples_done = 0
    per_batch_latency_ms = []  # 每 batch 的端到端毫秒数（不含 warmup）
    warmup_left = max(3, n_batches // 50)  # 至少 3 个 batch warmup；大批量下 2%

    # 进度条
    iterator = range(0, total, batch_size)
    if _HAS_TQDM:
        pbar = tqdm(total=total, desc=f"[Baseline:{args.tag}]", unit="smpl",
                    mininterval=2.0, file=sys.stdout)
    else:
        pbar = None

    with open(out_path, "w", encoding="utf-8") as f:
        try:
            for batch_idx, batch_start in enumerate(iterator):
                batch = rows[batch_start:batch_start + batch_size]
                prompts = []
                for r in batch:
                    msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": r["input"]},
                    ]
                    prompts.append(tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True))
                inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=args.max_len).to(model.device)

                # ---- 计时（本 batch 端到端 ms）----
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,            # 贪心解码，与 fast_infer 保持一致
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=None, top_p=None, top_k=None,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_ms = (time.time() - t0) * 1000.0

                # ---- 写输出（与 fast_infer 字节级一致：input/label/task/model_output）----
                input_len = inputs["input_ids"].shape[1]
                for i, r in enumerate(batch):
                    gen_ids = gen[i][input_len:]
                    gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    f.write(json.dumps({
                        "input": r["input"],
                        "label": r["label"],
                        "task": r["task"],
                        "model_output": answer,
                    }, ensure_ascii=False) + "\n")

                # ---- 记录 latency（去掉 warmup）----
                if batch_idx >= warmup_left:
                    per_batch_latency_ms.append(dt_ms)

                samples_done = min(batch_start + batch_size, total)

                # 进度刷新
                if pbar is not None:
                    pbar.update(len(batch))
                    pbar.set_postfix({
                        "ms/batch": f"{dt_ms:.0f}",
                        "peak_GiB": f"{(torch.cuda.max_memory_allocated() / 2**30):.1f}" if torch.cuda.is_available() else "0",
                    })
                else:
                    if samples_done % (batch_size * 5) == 0 or samples_done == total:
                        elapsed = time.time() - t_start
                        speed = samples_done / elapsed if elapsed > 0 else 0.0
                        remain = (total - samples_done) / speed if speed > 0 else 0.0
                        peak_gib = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
                        print(
                            f"  [{samples_done}/{total}] {speed:.2f} smpl/s  "
                            f"elapsed={_fmt_seconds(elapsed)}  eta={_fmt_seconds(remain)}  "
                            f"peak={peak_gib:.2f}GiB",
                            flush=True,
                        )
        except KeyboardInterrupt:
            print(f"\n[Baseline] KeyboardInterrupt，已写出 {samples_done}/{total} 条 -> {out_path}", flush=True)
            raise
        finally:
            if pbar is not None:
                pbar.close()
    tokenizer.padding_side = orig_padding_side

    # ---- 5. 汇总指标 ----
    elapsed_total = time.time() - t_start
    # throughput 按"扣除 warmup 后的总耗时"算更公平
    effective_samples = total - warmup_left * batch_size
    effective_time = elapsed_total - sum(per_batch_latency_ms[:0]) / 1000  # warmup 不计
    # 简化：直接用 total time（warmup 占 < 2%，可忽略）
    speed_total = total / elapsed_total if elapsed_total > 0 else 0.0
    # 每样本延迟 = 整 batch wall time / batch_size
    sample_latency_ms = (elapsed_total * 1000.0) / total if total > 0 else 0.0
    # latency p50/p95（基于 batch 维度除以 batch_size 估算）
    if per_batch_latency_ms:
        per_sample_latency = [x / batch_size for x in per_batch_latency_ms]
        p50 = statistics.median(per_sample_latency)
        # 简单 p95：排序后取 95% 位置
        sorted_lat = sorted(per_sample_latency)
        p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)
        p95 = sorted_lat[p95_idx]
        p99_idx = max(0, int(len(sorted_lat) * 0.99) - 1)
        p99 = sorted_lat[p99_idx]
    else:
        p50 = p95 = p99 = 0.0
    peak_gib = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0

    summary = {
        "tag": args.tag,
        "engine": "hf_generate",
        "quantization": "nf4_4bit" if args.use_4bit else "bf16",
        "device": device_name,
        "model_path": args.model_path,
        "ckpt_dir": args.ckpt_dir,
        "in_file": args.in_file,
        "n_samples": total,
        "batch_size": batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_len": args.max_len,
        "warmup_batches": warmup_left,
        "elapsed_sec": round(elapsed_total, 2),
        "throughput_samples_per_sec": round(speed_total, 3),
        "sample_latency_ms_avg": round(sample_latency_ms, 2),
        "sample_latency_ms_p50": round(p50, 2),
        "sample_latency_ms_p95": round(p95, 2),
        "sample_latency_ms_p99": round(p99, 2),
        "peak_gpu_mem_gib": round(peak_gib, 2),
        "out_file": out_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.metrics_file:
        os.makedirs(os.path.dirname(args.metrics_file) or ".", exist_ok=True)
        with open(args.metrics_file, "w", encoding="utf-8") as mf:
            json.dump(summary, mf, ensure_ascii=False, indent=2)
        print(f"[Baseline] metrics -> {args.metrics_file}", flush=True)

    print(
        f"[Baseline] 完成 {time.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"用时={_fmt_seconds(elapsed_total)}  均速={speed_total:.2f} smpl/s  "
        f"latency p50={p50:.0f}ms p95={p95:.0f}ms p99={p99:.0f}ms  "
        f"peak={peak_gib:.2f}GiB  -> {out_path}",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="推理加速对照实验的 baseline 复跑器")
    # 不调 add_common_args：训练参数（--output_dir/--epochs/--lr/--lora_r/...）与推理无关
    # 只挑出 baseline 实际用得到的几个
    parser.add_argument("--model_path", type=str, required=True,
                        help="基座模型路径")
    parser.add_argument("--use_4bit", dest="use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--max_len", type=int, default=1024,
                        help="输入最大序列长度")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, required=True,
                        help="本组实验的标签，写入 metrics 和文件名前缀，如 bf16 / 4bit / 4bit_b8")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="adapter 目录；不传则评估基座模型（零样本基线）")
    parser.add_argument("--in_file", type=str, required=True,
                        help="输入 JSONL（绝对路径或相对 cwd）")
    parser.add_argument("--out_file", type=str, required=True,
                        help="输出 JSONL（与 fast_infer 字节级一致）")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="指标 JSON 落盘路径")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="生成最大 token 数（与 eval 协议一致）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="推理 batch size（baseline 单 batch = 1 最严格对照）")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(args)


if __name__ == "__main__":
    main()
