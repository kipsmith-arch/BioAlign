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

import sklearn
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import (ProgressCallback, add_common_args, add_lora, build_lora_plus_optimizer,
                    load_model_tokenizer, read_jsonl, setup_env, setup_output_dir, enable_grad_checkpointing)


def pack_texts(texts, tokenizer, max_len):
    """【自定义 packing】把独文本序列 token 化后拼成固定长度块。

    ============================================================================
    packing 在 stage1 里是核心优化：
      - bio序列平均长度远小于 max_len（多数 50-200 bp vs max_len=2048）。
      - 不 packing 时，批次里 70%是 pad 0 token（模型不学，但占计算量 70%）。
      - packing 把多个独序列拼满 max_len 块 → **全部 token 都是监督信号**。
      代价仅是“丢齄”几个 token（多个序列拼接处的隔断）。

    而本实现三点反直觉设计需要注释：

    [1] `max_length=max_len, truncation=False`：
        看似"为什么传 max_length 但不 truncation"——这看起来是 bug，实则不是。
        - `max_length=max_len` 仅仅是为了让 tokenizer **不报"超过 model_max_length 警告"**。
          （Qwen2.5 默认 model_max_length=32768，远超样本，不限制反而会报上上游警告。）
        - `truncation=False` 是设计要求：样本可能 >max_len（极个别长的），**后面
          Python 切 max_len 块时会截除**；不在 tokenize 阶段截以避免：
            (a) Qwen tokenizer 内部默认从中间截未个 caption 起算（不是末尾）；
            (b) 双层截断与训练侧 labels mask 不一致。
        选错在这里会遇到"训练看起来收敛但下游 eval 崩坏"的幽眼问题。

    [2] 所有 sample token 拼成一个长序列（all_ids）再切 max_len 块，
        不是循环 per-sample 看 〈len+拼》：上一个样本不会跟下一个样本拼接被 shield，
        表征能里 token 预测下一个 token 是 8/7 合理的（不是有在联邦 分隔）。

    [3] `blocks.append({"input_ids": block, "labels": block})`：labels = input_ids
        （CLM 套路，人人都能输入 token 能预测同一 token）。
        Pack 上没有 attention mask：所有块都能全 token 看见，这是 packing 的
        **已知伪缺陷**（人人为对照乙序列 contamination 影响极小，论文如此设计）。
    """
    all_ids = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False, max_length=max_len, truncation=False)
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
