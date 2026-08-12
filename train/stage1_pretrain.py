# -*- coding: utf-8 -*-
"""
stage1_pretrain.py —— Stage 1: 领域继续预训练（bf16 LoRA+）
=============================================================
在未标注多组学序列（stage1_pretrain.jsonl）上做 next-token 预训练（causal LM）。
关键处理：**packing** —— 把短序列拼成固定 max_len 块，消除 padding 浪费。

训练方法（与论文 Stage 1 一致）：
- **bf16 LoRA+**：LoRA rank=64，B 权重学习率 = A×4（LoRA+，论文用 scaler=4），
  同时训练 RMSNorm 层 —— 论文原配置是 24×A100 + LoRA+ rank128；
  T4 上 bf16 版本可复现其方法（4bit 版本作为备选 --use_4bit）。
- 默认 bf16（不用 4bit）：继续预训练对新知识注入的精度要求高于 SFT，
  4bit 量化会损失部分信息，bf16 + LoRA 是 T4 单卡能装下的最大精度方案。

用法（本地 0.5B 冒烟，默认 bf16 + LoRA+）：
  python train/stage1_pretrain.py \
    --model_path D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct \
    --data_dir data_prep/output --output_dir ckpt/smoke_stage1 \
    --max_len 512 --per_device_batch 4 --grad_accum 4 \
    --lr 1e-4 --max_steps 30 --max_samples 2000
"""
import argparse
import sys
import os
IS_MAIN = int(os.environ.get("LOCAL_RANK", "0")) == 0

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import (ProgressCallback, add_common_args, add_lora, build_lora_plus_optimizer,
                    load_model_tokenizer, read_jsonl, setup_env, setup_output_dir, enable_grad_checkpointing)


def pack_texts(texts, tokenizer, max_len):
    """把文本序列 token 化后拼成固定长度块（packing）。"""
    all_ids = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        all_ids.extend(ids)
    blocks = []
    for i in range(0, len(all_ids) - max_len + 1, max_len):
        block = all_ids[i:i + max_len]
        blocks.append({"input_ids": block, "labels": block})
    return blocks


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--lora_plus_scaler", type=float, default=4.0,
                        help="LoRA+：B 权重学习率 = A×scaler（论文用 4）")
    parser.add_argument("--train_norm", action="store_true", default=True,
                        help="同时训练 RMSNorm 层（论文 Stage 1 做法）")
    parser.add_argument("--optim", type=str, default="adamw8bit",
                        choices=["adamw8bit", "adamw"],
                        help="优化器：adamw8bit 省显存（推荐 T4），adamw 为 fp32 全精度")
    # Stage 1 默认 4bit：T4 单卡 bf16 3B 训练峰值装不下（加载后已 ~12GB）
    # 4bit 省下的显存让给 max_len=2048（论文 2000 字符≈1200 token，几乎零截断）
    parser.set_defaults(use_4bit=True, lora_r=64, max_len=2048, per_device_batch=1)
    args = parser.parse_args()
    setup_env()
    setup_output_dir(args.output_dir)

    print(f"[Stage1] 加载模型: {args.model_path} (4bit={args.use_4bit}, "
          f"LoRA+ scaler={args.lora_plus_scaler}, train_norm={args.train_norm}, "
          f"optim={args.optim})")
    model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                     dropout=args.lora_dropout, train_norm=args.train_norm)
    # 显式开启 gradient checkpointing（bf16 3B 长序列下激活是显存大头）
    enable_grad_checkpointing(model)
    if torch.cuda.is_available() and IS_MAIN:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # 新版 transformers 属性名为 is_gradient_checkpointing（旧版 gradient_checkpointing）
        gc = getattr(model, 'is_gradient_checkpointing', None)
        if gc is None:
            gc = getattr(model, 'gradient_checkpointing', None)
        # max_memory_allocated 更接近 nvidia-smi 口径（含 PyTorch caching allocator 预留）
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        import transformers as _tf
        print(f"[Stage1] gradient_checkpointing={gc}")
        print(f"[Stage1] trainable={trainable:,} 峰值显存={mem:.2f}GiB")
        print(f"[Stage1] transformers={_tf.__version__}")

    if IS_MAIN:
        print(f"[Stage1] 读取序列数据: {args.data_dir}/stage1_pretrain.jsonl")
    rows = read_jsonl(f"{args.data_dir}/stage1_pretrain.jsonl", args.max_samples)
    texts = [r["text"] for r in rows]
    if IS_MAIN:
        print(f"[Stage1] 序列条数: {len(texts)}")

    blocks = pack_texts(texts, tokenizer, args.max_len)
    if IS_MAIN:
        print(f"[Stage1] packing 后块数: {len(blocks)} (max_len={args.max_len})")
    dataset = Dataset.from_list(blocks)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        bf16=True,
        logging_steps=25,
        disable_tqdm=True,
        log_on_each_node=False,
        label_names=[],
        save_strategy="steps",
        save_steps=args.max_steps if args.max_steps > 0 else 500,
        save_total_limit=2,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=[],
        ddp_find_unused_parameters=False,
    )
    # LoRA+ 优化器：B 组 lr = base×scaler，A 组与 RMSNorm 用 base lr
    optimizer = build_lora_plus_optimizer(model, args.lr, args.lora_plus_scaler,
                                          use_8bit=(args.optim == "adamw8bit"))
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset,
                      optimizers=(optimizer, None))
    trainer.add_callback(ProgressCallback())
    if IS_MAIN:
        print("[Stage1] 开始训练 ...")
    trainer.train()
    if IS_MAIN:
        print(f"[Stage1] 保存 adapter 到 {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if IS_MAIN:
        print("[Stage1] 完成。")


if __name__ == "__main__":
    main()
