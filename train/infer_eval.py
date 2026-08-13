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

import torch
from peft import PeftModel

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import SYSTEM_PROMPT, add_common_args, load_model_tokenizer, read_jsonl


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
    with open(out_path, "w", encoding="utf-8") as f:
        for batch_start in range(0, len(rows), batch_size):
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
            done = min(batch_start + batch_size, len(rows))
            if done % (batch_size * 20) == 0 or done == len(rows):
                print(f"  已推理 {done}/{len(rows)}")
    tokenizer.padding_side = orig_padding_side

    print(f"[Infer] 完成 -> {out_path}")

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
