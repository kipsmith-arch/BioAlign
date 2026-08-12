# -*- coding: utf-8 -*-
"""
build_preference.py —— 自构 DPO 偏好数据（方案 A）
====================================================
从 dpo_source.jsonl（SFT 不可见的独立集合）构造 chosen/rejected 对：

  chosen   = 该样本的标准答案（output 字段，权威标注）
  rejected = **Stage 2 模型**对同一问题采样的回答（on-policy，同一"会答"分布）

设计依据（DPO 语义）：偏好对应编码"同一任务分布下的质量偏好"，而非
"会不会答"。因此 rejected 不用未微调的基座（胡编/不会答，偏离 DPO 本意），
而用 stage2 模型采样——chosen（标准答案，精炼准确）与 stage2 采样输出
（同分布但质量较低）的差异即质量维度偏好。区分度通过 temperature 采样保证。

输出 trl DPOTrainer 格式：
  {"prompt": [{"role":"user","content": input}],
   "chosen": [{"role":"assistant","content": output}],
   "rejected":[{"role":"assistant","content": <stage2 采样>}]}

质量控制：
- rejected 用 stage2 模型 + 较高温度采样
- 过滤 rejected 与 chosen 相同 / 输出为空的样本（无区分度）

用法（本地 0.5B 冒烟）：
  python train/build_preference.py \
    --model_path D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct \
    --stage2_dir ckpt/smoke_stage2 --data_dir data_prep/output --output_dir data_prep/output \
    --max_pairs 50 --max_new_tokens 96 --temperature 0.9 --use_4bit
"""
import argparse
import json
import sys

import torch
from peft import PeftModel

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import SYSTEM_PROMPT, add_common_args, load_model_tokenizer, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    # build_preference 覆藇 common 的 --output_dir required 默认 = --data_dir
    # （让 stage3 能在同目录下找到 dpo_pairs.jsonl，避免忘填跳坑）
    parser.set_defaults(output_dir=None)
    parser.add_argument("--stage2_dir", type=str, required=True,
                        help="Stage 2 模型 adapter（生成 rejected 用，on-policy）")
    # output_dir 不再用 common 的 required，改用下面的默认（=data_dir），避免忘填
    parser.add_argument("--max_pairs", type=int, default=25000, help="构造偏好对数量")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--in_file", type=str, default="dpo_source.jsonl")
    parser.add_argument("--out_file", type=str, default="dpo_pairs.jsonl")
    args = parser.parse_args()
    # 默认 = data_dir（与 stage3_dpo --data_dir 一致，能在同目录找到产出）
    if args.output_dir is None:
        args.output_dir = args.data_dir
        print(f"[Pref] --output_dir 未传，默认 = --data_dir = {args.output_dir}")
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"[Pref] 加载 base: {args.model_path} + stage2 adapter: {args.stage2_dir}（生成 rejected 用）")
    model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
    model = PeftModel.from_pretrained(model, args.stage2_dir)
    model.eval()

    rows = read_jsonl(f"{args.data_dir}/{args.in_file}", args.max_pairs)
    print(f"[Pref] 读取 {args.in_file}: {len(rows)} 条")

    out_path = f"{args.output_dir}/{args.out_file}"
    written, skipped = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            # 构造 prompt（system + user，generate 时不加 assistant 前缀，让模型直接续写）
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["input"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                               max_length=args.max_len).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = gen[0][inputs["input_ids"].shape[1]:]
            rejected = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            chosen = r["output"]
            # 过滤无区分度样本（rejected 与 chosen 相同 / 空输出）
            if not rejected or rejected == chosen:
                skipped += 1
                continue
            f.write(json.dumps({
                "prompt": [{"role": "user", "content": r["input"]}],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }, ensure_ascii=False) + "\n")
            written += 1
            if (i + 1) % 50 == 0:
                print(f"  已处理 {i+1}/{len(rows)}，有效 {written}，跳过 {skipped}")

    print(f"[Pref] 完成: 写入 {written} 对 -> {out_path}（跳过 {skipped}）")


if __name__ == "__main__":
    main()
