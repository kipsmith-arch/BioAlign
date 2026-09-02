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
import gc
import random
import sys
import os

import numpy as np
import sklearn
import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import (ProgressCallback, SYSTEM_PROMPT, add_common_args, add_lora,
                    build_lora_plus_optimizer, enable_grad_checkpointing,
                    load_model_tokenizer, read_jsonl, setup_env,
                    setup_output_dir)


# ============================================================
# task_kind() + prefix logic + class_weight
# ============================================================
# BioAlign 论文 §4.2 建议：
#   "we randomly select 30 percent of the training data and prepend a task label in
#    the format '[Classification/Regression:task name]' at the beginning of each
#    question. This method effectively aids the model in identifying different
#    tasks and output objectives."
# 实现记录：
#   1) task_kind(r) → 'classification' / 'regression'
#   2) maybe_add_prefix(r, ratio, rng) → 随机选 30% 样本，在 input 前拼前缀
#   3) compute_sample_weight(r, weights) → per-sample weight，用于 Trainer.compute_loss
#      加权 binary 任务的 minority class，逆转 stage2 坍缩到 dominant class 的趋势
#
# Class_weight 说明：
#   本文以 binary 任务为 target（emp-/tf-/pd/cpd/promoter_enhancer/rna_protein/
#   antibody_antigen/Solubility）；regression/multiclass/dict 任务不加权。
#   这些任务训练数据本身接近 50/50 平衡，balanced weight 接近 1，所以
#   class_weight 仅能提供温和修正。但加上是“零成本安全网”（不影响收敛）。
# ============================================================

# Binary task 名集合——以 label ∈ {positive, negative} 为标志
BINARY_TASKS = frozenset({
    "rna_protein_interaction",
    "antibody_antigen",
    "Solubility",
    "tf-h-0", "tf-h-1", "tf-h-2", "tf-h-3", "tf-h-4",
    "tf-m-0", "tf-m-1", "tf-m-2", "tf-m-3", "tf-m-4",
    "emp-H3", "emp-H3K4me1", "emp-H3K4me2", "emp-H3K4me3",
    "emp-H3K9ac", "emp-H3K14ac", "emp-H3K36me3", "emp-H3K79me3",
    "emp-H4", "emp-H4ac",
    "pd-prom_300_all", "pd-prom_300_notata", "pd-prom_300_tata",
    "cpd-prom_core_all", "cpd-prom_core_notata", "cpd-prom_core_tata",
    "promoter_enhancer_interaction-K562", "promoter_enhancer_interaction-GM12878",
    "promoter_enhancer_interaction-HeLa-S3", "promoter_enhancer_interaction-HUVEC",
    "promoter_enhancer_interaction-IMR90", "promoter_enhancer_interaction-NHEK",
})


def task_kind(rec: dict) -> str:
    """判断任务类型。Binary 任务走特殊 prefix（'Classification'）。"""
    task = rec.get("task", "")
    label = rec.get("label", "")
    if task in BINARY_TASKS or label in ("positive", "negative"):
        return "Classification"
    # multi-value regression (dict) 也属于 regression
    if isinstance(label, dict):
        return "Regression"
    try:
        float(label)
        return "Regression"
    except Exception:
        return "Classification"  # multiclass 也归为 Classification


def maybe_add_prefix(rec: dict, ratio: float, rng: random.Random) -> dict:
    """论文§4.2: 以 ratio 概率为样本加 task prefix。返回新 dict（不修改原 rec）。"""
    if rng.random() >= ratio:
        return rec
    kind = task_kind(rec)
    task = rec.get("task", "unknown")
    new_input = f"[{kind}: {task}] " + rec["input"]
    return {**rec, "input": new_input}


def compute_task_weights(rows: list[dict], power: float = 0.5) -> dict[str, float]:
    """任务间平衡：小任务样本权重高。

    weight[task] = (max_count / count[task]) ** power
    - power=0   → 全部 1.0（关闭）
    - power=0.5 → 温和：小任务权重 = sqrt(7.8x) ≈ 2.8x，大任务被轻微压制
    - power=1.0 → 完全 balanced：小任务 7.8x，大任务被强压制

    动机：train_pool_clean.jsonl 任务间样本量差 7.8x（CRISPROnTarget 1234 条
    vs sirnaEfficiency 9684 条）；emp 一族占 27.5%。模型会过度关注样本量大的
    任务，小任务（promoter_enhancer / CRISPROnTarget 等）训练信号不足。
    """
    from collections import Counter
    counts = Counter(r.get("task", "") for r in rows)
    if not counts:
        return {}
    max_c = max(counts.values())
    return {t: (max_c / c) ** power for t, c in counts.items()}


def compute_binary_class_weights(rows: list[dict]) -> dict[tuple[str, str], float]:
    """对 binary 任务计算每个 (task, label) 的 balanced weight。
    仅针对二元分类任务；其它任务返回空 dict。
    """
    from sklearn.utils.class_weight import compute_class_weight
    weights = {}
    # 按 task 聚合 labels
    by_task: dict[str, list[str]] = {}
    for r in rows:
        if task_kind(r) != "Classification":
            continue
        # 过滤掉 multiclass 的（label 不是 positive/negative 的）
        if r.get("label") not in ("positive", "negative"):
            continue
        by_task.setdefault(r["task"], []).append(r["label"])
    for task, labels in by_task.items():
        unique = sorted(set(labels))
        if len(unique) < 2:
            continue  # 单类不需要加权
        w = compute_class_weight("balanced", classes=np.array(unique), y=np.array(labels))
        for cls, wi in zip(unique, w):
            weights[(task, cls)] = float(wi)
    return weights


def encode_sft(item, tokenizer, max_len, system_prompt):
    """编码为 (input_ids, labels)，assistant 部分才参与 loss。

    【重要修复】必须用 ``apply_chat_template(tokenize=True)`` 直接拿到 ChatML 完整 ids，
    而不是先 string 再 ``tokenizer.encode()``——后者会把 ChatML 框架 token
    (``<|im_start|>`` / ``<|im_end|>`` 等) 当作普通文本切分，让模型无法识别
    prompt/response 边界，导致 ``model(input_ids)`` 看到的输入与训练时不匹配，
    loss 在第一步后就掉到 ~0 (实际是模型在拟合被错位的 token)。

    正确流程：
      1. ``apply_chat_template(..., tokenize=True)`` 一次性拿到 ChatML 完整 ids
      2. 手动截断到 ``max_len`` (apply_chat_template 的 truncation 参数在某些
         transformers 版本里对 list-of-ints 不生效)
      3. 用 prompt ids 的长度 mask 掉前 ``n_prompt`` 个 token 让 assistant
         部分才贡献 loss
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item["input"]},
        {"role": "assistant", "content": item["output"]},
    ]
    # 【修复】直接 tokenize=True,保留 ChatML 框架 token
    ids_full = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False)
    ids_prompt = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=True, add_generation_prompt=True)

    # 【修复】apply_chat_template 返回的可能不是纯 int 列表(老版本会包 list[list[int]])
    # 统一展平成 List[int]
    def _flatten(xs):
        out = []
        for x in xs:
            if isinstance(x, (list, tuple)):
                out.extend(_flatten(x))
            else:
                out.append(int(x))
        return out

    ids_full = _flatten(ids_full)
    ids_prompt = _flatten(ids_prompt)

    # 截断保护：硬截 max_len,但要确保不会把 <|im_end|> 中间切断
    if len(ids_full) > max_len:
        ids_full = ids_full[:max_len]
        # 防御：如果末尾正好是 <|im_end|>(151645 for Qwen2.5)就保留，否则补一个
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id \
                and ids_full[-1] != im_end_id:
            ids_full[-1] = im_end_id
    n_prompt = min(len(ids_prompt), len(ids_full))
    labels = [-100] * n_prompt + ids_full[n_prompt:]
    return {"input_ids": ids_full, "labels": labels}


class WeightedSFTCollator:
    """在 DataCollatorForSeq2Seq 的基础上多塞一个 weight 字段，供 Trainer.compute_loss 使用。

    Trainer 默认 collator 不会保留额外字段，这里手动接力：
      1) 先用 DataCollatorForSeq2Seq 生成 input_ids / labels / attention_mask
      2) 再从原 batch 取 weight 字段拼回去
    """

    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seq2seq = DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100, max_length=max_len)

    def __call__(self, features):
        weights = [float(f.get("weight", 1.0)) for f in features]
        batch = self.seq2seq(features)
        # seq2seq 保留 input_ids/labels/attention_mask，外加 weight
        batch["weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedSFTTrainer(Trainer):
    """自定义 Trainer：实现 per-sample 加权 loss。

    设计概要：
    - Stage 2 任务间样本量严重不均（7.8×是参考）。论文推荐用 sample weight
      修正：weight[task] ∝ max_count / count[task]^power，power=0.5。
    - 因为 task prefix 是文本注入而非独立 embedding，决定了不能用 logits mask
      做 "task 平衡采样"——只能用 loss function 加权。
    - 为了让 weight**不退化损失量级**，采归一化到 mean(w) = 1 的策略：全 1 weight
      时严格退化为标准 token-CE（在 token 数相同时）。
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if weights is None:
            # 未启用 task weights 时的标准 CE 路径——走 cross_entropy 的
            # 默认 reduction="mean"，等价于在所有非 -100 token 上取平均。
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        else:
            # ============================================================================
            # per-sample 加权 loss 三阶段流程
            # ============================================================================
            # [Stage A] per-token CE，reduction="none" 跳出默认的 mean 归约；
            #           形状 (B, T)，保留每个 (b, t) 的独立 CE 值。
            # [Stage B] per-sample 平均：在每个 b 上把 loss × (labels != -100) 抹
            #           掉 padding / 不了 token 位置，再对 t 求和 / token_cnt。
            #           得到 (B,) 的 per_sample loss  —— 与 batch 里 token 数
            #           不同无关，完全对齐"一个样本一个 loss”的语义。
            # [Stage C] per-sample weight 加权：太鲁棒太鲁棒的 mean(weight)=1
            #           归一化是为了严格退回为 mean(per_sample)，与未加权时
            #           同量级。
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape)
            mask = (labels != -100).float()
            token_cnt = mask.sum(dim=-1).clamp_min(1.0)
            per_sample = (losses * mask).sum(dim=-1) / token_cnt
            w = weights.to(per_sample.device).to(per_sample.dtype)
            # 归一化：mean(w)=1 → weight=1 时 loss = mean(per_sample)，等于标准 CE（在 token 数相同时）
            w = w * (per_sample.numel() / w.sum().clamp_min(1e-9))
            loss = (per_sample * w).sum() / per_sample.numel()
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--train_file", type=str, default="train_pool_clean.jsonl",
                        help="SFT 训练数据（默认：净化后含 stage3 合并版）")
    parser.add_argument("--resume_adapter", type=str, default=None,
                        help="从 Stage 1 的 LoRA adapter 继续训练（stage1+stage2 分支）；"
                             "不传则从基座直接训（stage2-only 分支）")
    # 【P1】论文§4.2 task prefix + class_weight + task_weight 控制
    parser.add_argument("--task_prefix_ratio", type=float, default=0.30,
                        help="论文§4.2 推荐的 task prefix 概率（默认 0.30 = 论文原值）；设为 0 关闭")
    parser.add_argument("--use_class_weight", action="store_true", default=False,
                        help="是否对 binary 任务启用 balanced class_weight（计算 sample-level weight）")
    parser.add_argument("--task_weight_power", type=float, default=0.0,
                        help="任务间平衡强度：weight[task]=(max_count/count[task])**power。"
                             "0=关闭（默认）；0.5=温和平衡；1.0=完全 balanced。"
                             "复用 WeightedSFTTrainer 的 per-sample weight 机制。")
    parser.set_defaults(lora_r=64)  # 两分支统一 rank=64，保证消融①只差“有无 Stage1”
    args = parser.parse_args()
    setup_env()
    setup_output_dir(args.output_dir)

    # 所有诊断 print 只在主进程输出，避免 DDP 双进程重复日志
    IS_MAIN = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if IS_MAIN:
        print(f"[Stage2] 加载模型: {args.model_path} (4bit={args.use_4bit})")
    model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
    # 【修复 DDP prepare OOM】4bit BNB 加载后，allocator cache 里会有大量未返回的小 block；
    # DDP 在 _sync_module_states 阶段需要一次性 broadcast ~2 GiB 连续 buffer，
    # 碎片化会直接 OutOfMemoryError。提前清理一次，把碎片退掉、合并到 reserved。
    # 每张卡上独立打印，避免主进程单一汇报导致误判"所有卡都缺"。
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if IS_MAIN:
            print(f"[Stage2-pre] post-load-base alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G",
                  flush=True)
    if args.resume_adapter:
        if IS_MAIN:
            print(f"[Stage2] 从 Stage1 adapter 继续: {args.resume_adapter}")
        # 【修复 OOM】resume 路径必须复用 QLoRA 标配：
        #   1) prepare_model_for_kbit_training → enable_input_require_grads + cast LayerNorm
        #      不开这个，配合 gradient checkpoint 时 LoRA→base 的梯度链断裂，前向要么失败
        #      要么 PyTorch 强制保留整图，导致 base 7B 的激活占满显存。
        #   2) gradient_checkpointing_enable(use_reentrant=False) → 用时间换显存，
        #      把激活从 O(L·H) 降到 O(sqrt(L·H))，4bit 7B + 1024 序列的关键。
        model = PeftModel.from_pretrained(model, args.resume_adapter)
        model = prepare_model_for_kbit_training(
            model, gradient_checkpointing_kwargs={"use_reentrant": False})
        enable_grad_checkpointing(model)
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad_(True)
    else:
        model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        # 【修复 DDP prepare OOM】PEFT add_lora 会调 get_peft_model/get_peft_model_state_dict，
        # 内部会对 base 做一次 cast/拷贝以注入 LoRA wrapper，BNB 4bit 路径下会产生临时 fp16 张量。
        # 套完 LoRA 后立刻 gc + empty_cache，把临时 fp16 镜像 release，避免 DDP broadcast 阶段找不到连续 ~2 GiB。
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            if IS_MAIN:
                print(f"[Stage2-pre] post-add_lora alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
                      f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G",
                      flush=True)

    if IS_MAIN:
        print(f"[Stage2] 读取指令数据: {args.data_dir}/{args.train_file}")
    rows = read_jsonl(f"{args.data_dir}/{args.train_file}", args.max_samples)
    if IS_MAIN:
        print(f"[Stage2] 样本数: {len(rows)}")

    # 【P1-a】论文§4.2 task prefix：以 30% 概率为样本加 [Classification/Regression: task] 前缀
    rng = random.Random(args.seed)
    rows_prefixed = [maybe_add_prefix(r, args.task_prefix_ratio, rng) for r in rows]
    n_prefixed = sum(1 for a, b in zip(rows_prefixed, rows) if a.get("input") != b.get("input"))
    if IS_MAIN:
        print(f"[Stage2] task prefix: {n_prefixed}/{len(rows)} ({n_prefixed/len(rows)*100:.1f}%)")
    rows = rows_prefixed

    # 【P1-b】class_weight：针对 binary 任务计算 balanced sample weight
    cw = compute_binary_class_weights(rows) if args.use_class_weight else {}
    if IS_MAIN:
        if cw:
            sample_weights = list(cw.values())
            print(f"[Stage2] class_weight: {len(cw)} (task, label) pairs, "
                  f"weight range [{min(sample_weights):.3f}, {max(sample_weights):.3f}]")
        else:
            print(f"[Stage2] class_weight: disabled (use_class_weight=False or no binary task)")

    # 【P1-c】task_weight：任务间平衡（小任务样本权重高）
    tw = compute_task_weights(rows, power=args.task_weight_power) if args.task_weight_power > 0 else {}
    if IS_MAIN and tw:
        vals = list(tw.values())
        print(f"[Stage2] task_weight: power={args.task_weight_power}, "
              f"weight range [{min(vals):.3f}, {max(vals):.3f}]")

    # 编码 + 为每个 sample 打上 weight（class_weight × task_weight 相乘）
    encoded = []
    for r in rows:
        enc = encode_sft(r, tokenizer, args.max_len, SYSTEM_PROMPT)
        if len(enc["input_ids"]) > 0:
            w = 1.0
            # class_weight：仅对 (task, str label) 有效；dict/其它 label 跳过
            label = r.get("label")
            if isinstance(label, str):
                w *= cw.get((r["task"], label), 1.0)
            # task_weight：对所有任务生效
            if tw:
                w *= tw.get(r.get("task", ""), 1.0)
            enc["weight"] = w
            encoded.append(enc)
    dataset = Dataset.from_list(encoded)

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
    # 【省显存】用 bitsandbytes 8bit AdamW，fp32 优化器状态 → 1B/参数，8B model 的 LoRA
    # 训练省 ~5GiB 优化器状态显存；不影响收敛。
    optimizer = build_lora_plus_optimizer(model, base_lr=args.lr)
    # class_weight / task_weight 任一开启 → 用 WeightedSFTTrainer（per-sample weight）
    use_weighted = args.use_class_weight or args.task_weight_power > 0
    TrainerCls = WeightedSFTTrainer if use_weighted else Trainer
    if use_weighted:
        collator = WeightedSFTCollator(tokenizer, max_len=args.max_len)
    else:
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer = TrainerCls(model=model, args=train_args, train_dataset=dataset,
                         data_collator=collator,
                         optimizers=(optimizer, None))
    trainer.add_callback(ProgressCallback())
    if IS_MAIN:
        print("[Stage2] 开始训练 ...")
    # 【重入安全】Trainer.train() 不传参时默认从 output_dir 里的 checkpoint-* 最新一个恢复。
    # 公共环境 SIGHUP/SIGTERM 转发后优雅退出产生的 checkpoint-* 会被自动加载。
    # save_total_limit=2 只保留 2 个最近的 checkpoint 避免占爆硬盘。
    trainer.train()
    if IS_MAIN:
        print(f"[Stage2] 保存 adapter 到 {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if IS_MAIN:
        print("[Stage2] 完成。")


if __name__ == "__main__":
    main()
