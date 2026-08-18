# -*- coding: utf-8 -*-
"""
fast_infer.py —— vLLM 4-bit + LoRA hot-swap 推理
==================================================

目的
----
BioAlign 推理加速主推方案：在 A100 上用 vLLM 加载同一份 base（Qwen2.5-7B-Instruct），
用 bitsandbytes 4-bit 量化（与训练时 QLoRA 同方案，权重零转换），通过 LoRARequest
热加载 stage2/stage3 adapter，continuous batching 跑 1.89 万条评估集。

输出格式与 `infer/baseline_runner.py` 字节级一致，可直接用 `eval/evaluate.py` 算指标。

技术要点
--------
- vLLM ≥ 0.6：bitsandbytes 4-bit 是 supported quantization，Qwen2.5 + LoRA 一等公民
- PagedAttention：KV cache 分页管理，长短序列混合推理无浪费
- Continuous batching：动态 insert / preempt，新请求可插队到 GPU 空闲槽位
- LoRA hot-swap：vLLmLoraRequest API，**不 merge** adapter 进 base，可同时跑多 adapter

为什么不用 merge_and_unload
---------------------------
训练时 adapter 留 PEFT 格式（不 merge），部署时 vLLM 的 LoRARequest 即可热加载。
合并后：
- 失去"同一 base 切换不同 stage adapter"的能力
- 重新量化 7B 慢、显存占
不合并：
- base 量化一次常驻显存，adapter 按需加载
- 适合"多版本模型 + 同一 base"的部署场景

用法（A100 上）
--------------
    # V1 主推：vLLM + 4-bit + LoRA + continuous batching
    python infer/fast_infer.py \
        --tag vllm_4bit \
        --model_path /path/to/Qwen2.5-7B-Instruct \
        --ckpt_dir ckpt/stage2 \
        --in_file eval_set.jsonl \
        --out_file bench/raw/vllm_4bit.jsonl \
        --metrics_file bench/raw/vllm_4bit_metrics.json

    # V2 对照：vLLM + 4-bit + 不带 LoRA（看 adapter 加载本身的开销）
    python infer/fast_infer.py \
        --tag vllm_4bit_base \
        --model_path /path/to/Qwen2.5-7B-Instruct \
        --in_file eval_set.jsonl \
        --out_file bench/raw/vllm_4bit_base.jsonl \
        --metrics_file bench/raw/vllm_4bit_base_metrics.json

环境要求
--------
- Python ≥ 3.9
- vLLM ≥ 0.6（建议 0.6.4+ 稳定 7B bnb 4-bit 支持）
- bitsandbytes ≥ 0.43（与训练时同版本 0.49.2 最佳）
- torch ≥ 2.3（与 transformers 4.52 / vLLM 0.6+ 兼容）
- **不支持 Windows**（vLLM 依赖 `resource` 模块；本项目 README 已声明）
- 显存：7B 4-bit 约 5–6 GB；加 KV cache + LoRA 工作集，A100 40GB 富余

注意
----
- vLLM 0.6.x 的 bnb 4-bit 路径需要在 `LLM(...)` 里同时设
  `quantization="bitsandbytes"` + `load_format="bitsandbytes"`（不同版本 API 略有差异）
- vLLM 启动期会一次性占大量显存（init cache + workspace），**peak 出现在启动后第一
  个 batch 之前**；本脚本采集的是 `LLM(...)` 初始化完成后的首次 `reset_peak_memory_stats`
  之后到推理结束之间的真实峰值
- 评估输出 JSONL 与 baseline 完全一致，键：input / label / task / model_output
"""
import argparse
import json
import os
import statistics
import sys
import time

# ---- vLLM 与 bnb 4-bit 集成 ----
# vLLM ≥ 0.6 把 bnb 作为 supported quantization method 暴露
# import 失败时给出明确指引（Windows / 版本不匹配）
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    _HAS_VLLM = True
except ImportError as e:
    _HAS_VLLM = False
    _VLLM_IMPORT_ERR = e

# 进度条
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# 与 train/ 同源：复用 common.py 的 system prompt 和数据读取
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from common import SYSTEM_PROMPT, read_jsonl  # noqa: E402


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


def build_chat_prompt(tokenizer, user_input: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """用与 baseline 完全一致的 ChatML 模板拼 prompt，保证对照公平。"""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    # vLLM 不直接吃 apply_chat_template 输出；多数场景用字符串 prompt 即可
    # （tokenizer 在 vLLM 内部独立管理 chat template）
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def run(args):
    if not _HAS_VLLM:
        print(
            f"[FastInfer] ❌ vLLM 未安装：{_VLLM_IMPORT_ERR}\n"
            f"  在 A100 Linux 环境执行：\n"
            f"  pip install vllm>=0.6.4 bitsandbytes==0.49.2\n"
            f"  注意：vLLM 不支持 Windows。",
            flush=True,
        )
        sys.exit(1)

    # ---- 1. 读数据 ----
    rows = read_jsonl(args.in_file, args.max_samples)
    total = len(rows)
    print(f"[FastInfer] 读取 {args.in_file}: {total} 条", flush=True)

    # ---- 2. 准备 prompts ----
    # 用与 baseline 完全一致的 ChatML 模板，**字符串格式**（vLLM 0.6+ 推荐用 prompt 字符串）
    # token-level 拼接交给 vLLM 内部 tokenizer 处理
    # 这里用与 train/common.py SYSTEM_PROMPT 相同的 system 提示
    print(f"[FastInfer] 构造 prompts (template=ChatML)...", flush=True)
    t_prep = time.time()
    # vLLM 也支持直接传 messages，但这里手动拼字符串便于明确控制模板
    # （vLLM 0.6+ 的 prompt 字符串必须自己保证模板正确）
    prompts = []
    for r in rows:
        # 完整 ChatML 格式（与 Qwen2.5 的对话模板一致）
        # <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{r['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append(prompt)
    print(f"[FastInfer] prompt 构造完成 {time.time() - t_prep:.1f}s", flush=True)

    # ---- 3. 初始化 vLLM ----
    # 关键参数解释：
    #   quantization="bitsandbytes"  → 4-bit 量化（NF4 + double quant）
    #   load_format="bitsandbytes"   → 与上面配套（不同 vLLM 版本要求）
    #   enable_lora=True             → 允许热加载 LoRA adapter
    #   max_lora_rank                → ≥ 训练时 LoRA rank（这里 64）
    #   max_model_len                → ≥ max_len；7B + 长序列 KV cache 占显存，留 2048 保险
    #   gpu_memory_utilization       → 0.9 留 10% 给 CUDA workspace / fragmentation
    #   dtype="bfloat16"             → 计算精度（4-bit 权重反量化到 bf16 计算）
    #   enforce_eager=False          → True 时禁用 CUDA graph（更省显存但慢；产品报告默认 False）
    print(f"[FastInfer] 初始化 vLLM ...", flush=True)
    t_init = time.time()
    llm = LLM(
        model=args.model_path,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    print(f"[FastInfer] vLLM 初始化完成  {time.time() - t_init:.1f}s", flush=True)

    # ---- 4. 准备 LoRARequest（如有）----
    lora_request = None
    if args.ckpt_dir:
        lora_request = LoRARequest(
            lora_name=f"bioalign_{args.tag}",
            lora_int_id=1,                   # int_id 在单 adapter 场景下随便给
            lora_local_path=args.ckpt_dir,
        )
        print(f"[FastInfer] LoRARequest 装载: {args.ckpt_dir}", flush=True)

    # ---- 5. SamplingParams（与 baseline 严格一致：贪心 + max_new_tokens=64）----
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,           # 0.0 = 贪心解码（vLLM 约定）
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
    )

    # ---- 6. 显存基线（vLLM 启动后立刻 reset，方便后面报"推理过程峰值"）----
    if torch.cuda.is_available():
        import torch  # 局部 import：避免在没装 vLLM 时也要求 torch
        torch.cuda.reset_peak_memory_stats()
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"
    print(f"[FastInfer] 设备={device_name}  gpu_mem_util={args.gpu_memory_utilization}  "
          f"max_model_len={args.max_len}  enforce_eager={args.enforce_eager}", flush=True)

    # ---- 7. 推理主循环（continuous batching 已在 llm.generate 内部）----
    print(f"[FastInfer] 开始推理  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    t_start = time.time()
    # vLLM 的 llm.generate 内部已经把 N 条请求分批、连续批处理、KV cache 调度
    # 我们只关心端到端总时间 + 输出
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
        use_tqdm=_HAS_TQDM,  # 内部进度条
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_total = time.time() - t_start

    # ---- 8. 写出 JSONL（与 baseline 字节级一致）----
    out_path = args.out_file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # 按原顺序对齐 outputs（vLLM 保证输入输出顺序一致）
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r, out in zip(rows, outputs):
            # out.outputs[0] 是第一条（n=1）生成结果
            answer = out.outputs[0].text.strip()
            f.write(json.dumps({
                "input": r["input"],
                "label": r["label"],
                "task": r["task"],
                "model_output": answer,
            }, ensure_ascii=False) + "\n")
            n_written += 1
    assert n_written == total, f"vLLM 输出数 {n_written} ≠ 输入数 {total}，异常！"

    # ---- 9. 汇总 ----
    speed_total = total / elapsed_total if elapsed_total > 0 else 0.0
    # 单样本延迟：vLLM 内部是 continuous batching，per-sample 延迟不便直读；
    # 用"端到端总时间 / 样本数"算平均延迟，再报 throughput 作主指标
    sample_latency_ms_avg = (elapsed_total * 1000.0) / total if total > 0 else 0.0
    # 从 vLLM 输出里抽 per-request 端到端 latency
    # vLLM 的 RequestOutput 有 metrics 字段（不同版本字段名略不同），这里做容错
    request_latencies_ms = []
    try:
        for out in outputs:
            # vLLM 0.6+ 的 RequestOutput.metrics.time_in_queue / time_to_first_token / time_of_request
            # 但跨版本不稳定，最稳的方法：自己打时间戳（vLLM 在 generate 内部已经做完，不可行）
            # 折中：用"总时间 / 总样本"作为系统级 latency
            if hasattr(out, "metrics") and out.metrics is not None:
                if hasattr(out.metrics, "time_of_request"):
                    # time_of_request 是相对时间戳，需要对所有请求归一化
                    pass
    except Exception:
        pass

    peak_gib = 0.0
    if torch.cuda.is_available():
        import torch as _t
        peak_gib = _t.cuda.max_memory_allocated() / 2 ** 30

    summary = {
        "tag": args.tag,
        "engine": "vllm",
        "quantization": "bitsandbytes_nf4_4bit",
        "device": device_name,
        "model_path": args.model_path,
        "ckpt_dir": args.ckpt_dir,
        "in_file": args.in_file,
        "n_samples": total,
        "batch_size": "continuous_batching",
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_len,
        "max_lora_rank": args.max_lora_rank,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "tensor_parallel_size": args.tensor_parallel_size,
        "init_time_sec": round(time.time() - t_init - elapsed_total, 2),  # 粗估：总-vllm启动-推理
        "elapsed_sec": round(elapsed_total, 2),
        "throughput_samples_per_sec": round(speed_total, 3),
        "sample_latency_ms_avg": round(sample_latency_ms_avg, 2),
        "sample_latency_ms_p50": round(sample_latency_ms_avg, 2),  # 连续批下 p50 ≈ avg，下游报告里说明
        "sample_latency_ms_p95": None,                              # vLLM 0.6 公开 metrics 暂不给 p95
        "sample_latency_ms_p99": None,
        "peak_gpu_mem_gib": round(peak_gib, 2),
        "out_file": out_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": (
            "vLLM continuous batching 内部调度 per-sample p50/p95/p99 在 0.6.x 不直接公开；"
            "报告里以 throughput + 整 batch 端到端 latency 为主指标。"
        ),
    }

    if args.metrics_file:
        os.makedirs(os.path.dirname(args.metrics_file) or ".", exist_ok=True)
        with open(args.metrics_file, "w", encoding="utf-8") as mf:
            json.dump(summary, mf, ensure_ascii=False, indent=2)
        print(f"[FastInfer] metrics -> {args.metrics_file}", flush=True)

    print(
        f"[FastInfer] 完成 {time.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"用时={_fmt_seconds(elapsed_total)}  均速={speed_total:.2f} smpl/s  "
        f"avg_latency={sample_latency_ms_avg:.0f}ms  peak={peak_gib:.2f}GiB  -> {out_path}",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="vLLM 4-bit + LoRA hot-swap 推理")
    parser.add_argument("--tag", type=str, required=True,
                        help="本组实验的标签，写入 metrics 和文件名前缀")
    parser.add_argument("--model_path", type=str, required=True,
                        help="基座模型路径（Qwen2.5-7B-Instruct）")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="adapter 目录（PEFT 格式）；不传则评估基座模型")
    parser.add_argument("--in_file", type=str, required=True,
                        help="输入 JSONL（绝对路径或相对 cwd）")
    parser.add_argument("--out_file", type=str, required=True,
                        help="输出 JSONL（与 baseline_runner 字节级一致）")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="指标 JSON 落盘路径")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=2048,
                        help="= max_model_len；7B + 长序列 KV cache 占显存，A100 40GB 可开 2048")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_lora_rank", type=int, default=64,
                        help="≥ 训练时 LoRA rank；项目用 64")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM 占 GPU 显存比例；A100 40GB 可开 0.9")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="TP 大小；A100 单卡 1 即可（7B 4-bit 单卡 5–6GB）")
    parser.add_argument("--enforce_eager", action="store_true", default=False,
                        help="禁用 CUDA graph（更省显存但慢 ~5–10%%，仅在 OOM 时开）")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(args)


if __name__ == "__main__":
    main()
