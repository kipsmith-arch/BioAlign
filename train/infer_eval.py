# -*- coding: utf-8 -*-
"""
infer_eval.py —— 推理 + 评估
==============================
加载任意阶段 checkpoint（base + adapter），对 eval_set.jsonl（或其他含
input/label/task 的 jsonl）生成回答，输出 evaluate.py 兼容格式：
  {"input": ..., "label": ..., "task": ..., "model_output": ...}

然后可选调用 eval/evaluate.py 计算各任务指标。

用法（本地 0.5B 冒烟）：
  python train/infer_eval.py \
    --model_path D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct \
    --ckpt_dir ckpt/stage2 --data_dir data_prep/output \
    --in_file eval_set.jsonl --out_file eval_outputs_stage2.jsonl \
    --max_new_tokens 64 --max_samples 20 --use_4bit
"""
import argparse
import json
import subprocess
import sys
import time

import torch
from peft import PeftModel

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import SYSTEM_PROMPT, add_common_args, load_model_tokenizer, read_jsonl

# 进度条：tqdm 不可用时降级到 None（保持原有 print 行为，不破坏依赖）
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _fmt_seconds(sec: float) -> str:
    """把秒数格式化成 'Hh Mm Ss' / 'Mm Ss' / 'Ss.s'。"""
    if sec < 0 or sec != sec:  # NaN / 负数
        return "--"
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="待评估 checkpoint adapter 目录；不传则评估基座模型（零样本基线）")
    parser.add_argument("--in_file", type=str, default="eval_set.jsonl")
    parser.add_argument("--out_file", type=str, default="eval_outputs.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="推理 batch size（batch 1 = 原单样本；默认 8 提速约 5×）")
    parser.add_argument("--run_eval", action="store_true", help="推理后调用 eval/evaluate.py")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.ckpt_dir:
        print(f"[Infer] 加载 base: {args.model_path} + adapter: {args.ckpt_dir}")
        model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
        model = PeftModel.from_pretrained(model, args.ckpt_dir)
    else:
        print(f"[Infer] 评估基座模型（无 adapter）: {args.model_path}")
        model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
    model.eval()

    rows = read_jsonl(f"{args.data_dir}/{args.in_file}", args.max_samples)
    print(f"[Infer] 读取 {args.in_file}: {len(rows)} 条")

    out_path = f"{args.output_dir}/{args.out_file}"
    # batch generation：动态切批 + 左填充（生成场景标准做法，避免右填充让模型看到 pad 后再生成）
    # 单卡 1.89万 × 4 档 = 63h 加 batch 后约 8-13h，实测请按显存调整 batch_size
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    batch_size = max(1, args.batch_size)
    total = len(rows)
    n_batches = (total + batch_size - 1) // batch_size

    # 显存基线（推理开始时清一次 peak，方便后面报"推理过程峰值"）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        device_name = torch.cuda.get_device_name(model.device)
    else:
        device_name = "cpu"

    print(f"[Infer] 设备={device_name}  batch_size={batch_size}  总样本={total}  总batch={n_batches}  max_new_tokens={args.max_new_tokens}")
    print(f"[Infer] 开始推理  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_start = time.time()
    samples_done = 0
    # 进度条 / 降级 print
    # tqdm 模式：每 batch 刷新一次，自带 ETA + 速度；非 tqdm 模式：每 5 个 batch 打印一行
    iterator = range(0, total, batch_size)
    if _HAS_TQDM:
        pbar = tqdm(total=total, desc="[Infer]", unit="smpl",
                    mininterval=2.0,  # 至少 2s 刷一次，避免 I/O 风暴
                    file=sys.stdout)
    else:
        pbar = None

    with open(out_path, "w", encoding="utf-8") as f:
        try:
            for batch_start in iterator:
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
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,           # 贪婪解码，指标可复现
                        pad_token_id=tokenizer.pad_token_id,
                        # 显式置 None，屏蔽 generation_config 里残留的 Qwen 默认采样参数
                        # （否则会触发 "temperature/top_p/top_k 被忽略" 的 warning）
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )
                # 左填充下 input_ids 全部对齐到统一长度，generated tokens 出现在末尾
                input_len = inputs["input_ids"].shape[1]
                for i, r in enumerate(batch):
                    gen_ids = gen[i][input_len:]
                    # 防御性去末尾 pad（贪心 + 左填充下理论上不会有）
                    gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    f.write(json.dumps({
                        "input": r["input"],
                        "label": r["label"],
                        "task": r["task"],
                        "model_output": answer,
                    }, ensure_ascii=False) + "\n")
                samples_done = min(batch_start + batch_size, total)

                # 进度更新
                if pbar is not None:
                    pbar.update(len(batch))
                else:
                    if samples_done % (batch_size * 5) == 0 or samples_done == total:
                        elapsed = time.time() - t_start
                        speed = samples_done / elapsed if elapsed > 0 else 0.0
                        remain = (total - samples_done) / speed if speed > 0 else 0.0
                        peak_gib = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
                        print(f"  [{samples_done}/{total}] {speed:.2f} smpl/s  "
                              f"elapsed={_fmt_seconds(elapsed)}  eta={_fmt_seconds(remain)}  "
                              f"peak={peak_gib:.2f}GiB")
        except KeyboardInterrupt:
            # 用户 Ctrl-C：保留已写出文件，给出明确提示
            print(f"\n[Infer] KeyboardInterrupt，已写出 {samples_done}/{total} 条 -> {out_path}")
            raise
        finally:
            if pbar is not None:
                pbar.close()
    tokenizer.padding_side = orig_padding_side

    # 汇总
    elapsed_total = time.time() - t_start
    speed_total = total / elapsed_total if elapsed_total > 0 else 0.0
    peak_gib = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
    print(f"[Infer] 完成 {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"用时={_fmt_seconds(elapsed_total)}  均速={speed_total:.2f} smpl/s  "
          f"peak={peak_gib:.2f}GiB  -> {out_path}")

    if args.run_eval:
        print("[Infer] 调用 eval/evaluate.py ...")
        subprocess.run([
            sys.executable, "eval/evaluate.py",
            "--model_name", args.out_file.replace(".jsonl", ""),
            "--OMICS", "all_omics",
            "--input_file_path", out_path,
        ], check=True)


if __name__ == "__main__":
    main()
