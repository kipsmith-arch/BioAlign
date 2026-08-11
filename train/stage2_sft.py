# -*- coding: utf-8 -*-
"""
stage2_sft.py —— Stage 2: PEFT（4-bit QLoRA 指令微调）
==========================================================
在 Biology-Instructions 抽样指令数据（train_pool.jsonl）上做 SFT。
关键处理：**只对 assistant 部分算 loss**（user 输入与 system prompt 的 token
label 置为 -100），防止模型学会"复述问题"。

数据格式：{"input": 问题(含序列), "output": 标准答案, ...}
-> Qwen2.5 ChatML 模板：[system] Psc | [user] input | [assistant] output

用法（本地 0.5B 冒烟）：
  python train/stage2_sft.py \
    --model_path D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct \
    --data_dir data_prep/output --output_dir ckpt/stage2 \
    --max_len 1024 --per_device_batch 4 --grad_accum 4 \
    --lr 2e-4 --max_steps 60 --max_samples 1000 --use_4bit
"""
import argparse
import sys

from datasets import Dataset
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import (SYSTEM_PROMPT, add_common_args, add_lora,
                    load_model_tokenizer, read_jsonl, setup_output_dir)


def encode_sft(item, tokenizer, max_len, system_prompt):
    """编码为 (input_ids, labels)，assistant 部分才参与 loss。"""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item["input"]},
        {"role": "assistant", "content": item["output"]},
    ]
    full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    # prompt = system + user + assistant 头（不含 assistant 内容）
    prompt_text = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True)
    ids_full = tokenizer.encode(full_text, add_special_tokens=False)
    ids_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)

    # 截断保护
    if len(ids_full) > max_len:
        ids_full = ids_full[:max_len]
    n_prompt = min(len(ids_prompt), len(ids_full))
    labels = [-100] * n_prompt + ids_full[n_prompt:]
    return {"input_ids": ids_full, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--train_file", type=str, default="train_pool_clean.jsonl",
                        help="SFT 训练数据（默认：净化后含 stage3 合并版）")
    parser.add_argument("--resume_adapter", type=str, default=None,
                        help="从 Stage 1 的 LoRA adapter 继续训练（stage1+stage2 分支）；"
                             "不传则从基座直接训（stage2-only 分支）")
    parser.set_defaults(lora_r=64)  # 两分支统一 rank=64，保证消融①只差“有无 Stage1”
    args = parser.parse_args()
    setup_output_dir(args.output_dir)

    print(f"[Stage2] 加载模型: {args.model_path} (4bit={args.use_4bit})")
    model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
    if args.resume_adapter:
        print(f"[Stage2] 从 Stage1 adapter 继续: {args.resume_adapter}")
        model = PeftModel.from_pretrained(model, args.resume_adapter)
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad_(True)
    else:
        model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    print(f"[Stage2] 读取指令数据: {args.data_dir}/{args.train_file}")
    rows = read_jsonl(f"{args.data_dir}/{args.train_file}", args.max_samples)
    print(f"[Stage2] 样本数: {len(rows)}")

    dataset = Dataset.from_list(
        [encode_sft(r, tokenizer, args.max_len, SYSTEM_PROMPT) for r in rows])
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.max_steps if args.max_steps > 0 else 500,
        save_total_limit=2,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=[],
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset,
                      data_collator=DataCollatorForSeq2Seq(
                          tokenizer, padding=True, label_pad_token_id=-100))
    print("[Stage2] 开始训练 ...")
    trainer.train()
    print(f"[Stage2] 保存 adapter 到 {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[Stage2] 完成。")


if __name__ == "__main__":
    main()
