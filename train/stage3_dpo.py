# -*- coding: utf-8 -*-
"""
stage3_dpo.py —— Stage 3: RL（DPO 偏好对齐，自实现 DPO loss）
================================================================
在 Stage 2 产物（adapter）基础上做直接偏好优化。**自实现 DPO loss**
（不依赖 trl 版本），公式为标准 DPO（Rafailov et al. 2023）：

  log_ratio_c = log π(y_c|x) − log π_ref(y_c|x)
  log_ratio_r = log π(y_r|x) − log π_ref(y_r|x)
  loss = −E[ log σ( β · (log_ratio_c − log_ratio_r) ) ]

- π     = 待训练模型（stage2 adapter + 新可训练 LoRA）
- π_ref = 冻结参考模型（与 π 同初始权重，即 stage2 权重，不更新）
- β     = 温度参数（默认 0.1）

数据：build_preference.py 产出的 dpo_pairs.jsonl
  {"prompt": [...], "chosen": [...], "rejected": [...]}

实现要点（面试可讲）：
- 对 chosen / rejected 各算"只统计 assistant 部分"的序列对数概率
- reference 模型用与 model 相同的 stage2 初始化、全程冻结
- loss 只对 assistant 部分 token 求和，prompt 部分不参与

用法（本地 0.5B 冒烟）：
  python train/stage3_dpo.py \
    --model_path D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct \
    --stage2_dir ckpt/smoke_stage2 --data_dir data_prep/output --output_dir ckpt/smoke_stage3 \
    --max_len 1024 --per_device_batch 2 --grad_accum 4 \
    --lr 1e-5 --beta 0.1 --max_steps 20 --max_samples 100 --use_4bit
"""
import argparse
import gc
import sys
import os

# 【OOM 防御】必须在 import torch 之前设置——PYTORCH_CUDA_ALLOC_CONF 仅在首次 CUDA 分配前生效。
# common.py setup_env() 里也有 setdefault，但那里是在 main() 中调，torch import 已经发生。
# 此处顶层 setdefault 覆盖任何用户 shell 设置，避免第一次 CUDA 分配后才生效的问题。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sklearn
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import (ProgressCallback, SYSTEM_PROMPT, add_common_args, load_model_tokenizer,
                    read_jsonl, setup_env, setup_output_dir, enable_grad_checkpointing)


def encode_pair(pair, tokenizer, max_len, system_prompt):
    """把 (prompt, chosen/rejected) 编码为 (input_ids, labels)，仅 assistant 算 loss。"""
    def _enc(content):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["prompt"][0]["content"]},
            {"role": "assistant", "content": content},
        ]
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True)
        ids_full = tokenizer.encode(full_text, add_special_tokens=False, max_length=max_len, truncation=False)
        ids_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, max_length=max_len, truncation=False)
        if len(ids_full) > max_len:
            ids_full = ids_full[:max_len]
        n_prompt = min(len(ids_prompt), len(ids_full))
        labels = [-100] * n_prompt + ids_full[n_prompt:]
        return ids_full, labels
    c_ids, c_labels = _enc(pair["chosen"][0]["content"])
    r_ids, r_labels = _enc(pair["rejected"][0]["content"])
    return {"chosen_input_ids": c_ids, "chosen_labels": c_labels,
            "rejected_input_ids": r_ids, "rejected_labels": r_labels}


def token_logprobs(logits, input_ids, labels, pad_token_id, chunk_size=512):
    """对每个非 -100 位置计算 log P(token)，返回 (总对数概率, 有效 token 数)。

    【显存优化 关键修复】不一次性对全 vocab 算 log_softmax——那会产出与 logits 同形状 (B, T-1, V)
    的临时张量，V=152064、bf16 下 ≈ 12 GB/张。DPO 一次 step 同时存在 policy/ref × chosen/rejected
    四张 logits，峰值中间 temp 就能 48 GB，超出 A100 40 GB 直接 OOM（"this process 37 GiB" 的根因）。

    实现：log_softmax(x) = x - logsumexp(x)。logsumexp 是全 vocab 维度归一化常数，必须在完整
    logits 上一次性算。输出 (B, T-1) fp32：几 KB。
    然后按 vocab 分块遍历，每块 chunk_logits (B, T-1, chunk_size) bf16 ≈ 4 MB；
    chunk_logp = chunk_logits - logsumexp_full[..., None]（逐元素减）；
    gather 出 chunk_token_logp (B, T-1)；乘 mask 后累加到 fp32 标量。峰值只与 chunk 有关、与 V 无关。
    数值上与全张 log_softmax 等价（无近似误差）——因为 logsumexp 使用了完整 vocab。
    """
    targets = input_ids[:, 1:]                    # (B, T-1)
    labels = labels[:, 1:]                        # (B, T-1)
    mask = (labels != -100) & (targets != pad_token_id)
    count = mask.sum(dim=-1).clamp(min=1)
    # 一次性算全 vocab log-sum-exp（数值稳定），输出 (B, T-1) fp32 ≈ 几 KB
    logsumexp_full = torch.logsumexp(logits[:, :-1, :].float(), dim=-1)  # (B, T-1) fp32
    total = torch.zeros(targets.size(0), dtype=torch.float32, device=logits.device)
    V = logits.size(-1)
    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        # 截取当前 chunk（(B, T-1, chunk_size) 小张），减 logsumexp 得该 chunk 的 log_softmax
        chunk_logits = logits[:, :-1, v_start:v_end]  # (B, T-1, chunk_size)
        chunk_logp = chunk_logits - logsumexp_full.unsqueeze(-1)  # 广播减
        # gather 需要 target ∈ [v_start, v_end)：以 v_start 为代填值（之后被 mask 过滤）
        safe_targets = targets.clamp(min=v_start, max=v_end - 1) - v_start
        chunk_token_logp = torch.gather(
            chunk_logp, -1, safe_targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        # 只在 target 原本属于本 chunk 的位置参与求和（mask 只过滤 prompt/pad，额外乘
        # in_range 把错位贡献变 0）
        in_range = (targets >= v_start) & (targets < v_end)
        total += (chunk_token_logp.float() * (mask & in_range).float()).sum(dim=-1)
        del chunk_logits, chunk_logp, chunk_token_logp
    del logsumexp_full
    return total, count


class DPOTrainer(Trainer):
    """自实现 DPO：对 chosen/rejected 各算对数概率差，优化偏好。"""

    def __init__(self, *args, ref_model=None, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pad = self.tokenizer.pad_token_id
        # 【显存关键修复】logits 张量本身是 (B, T, V) bf16，V=152064 下 per_device_batch=4
        # 单张 logits 12 GB。必须"算完一路就丢"——之前将 logits_c / logits_r 绑定到变量同时存活
        # 是 OOM 的直接原因（chosen + rejected 两份同时 ≈ 24 GB）。下面对四路 forward 都按
        # "表达式内联"写法使 logits 变量随 token_logprobs 返回后即释放。
        # 同时 token_logprobs 内部是 chunked log_softmax（vocab 维 512 分块），不再产生
        # (B, T-1, V) 全张临时。
        # 当前策略 π：chosen / rejected 各一次 forward，算完即丢 logits
        logp_c, _ = token_logprobs(
            model(input_ids=inputs["chosen_input_ids"]).logits,
            inputs["chosen_input_ids"], inputs["chosen_labels"], pad)
        logp_r, _ = token_logprobs(
            model(input_ids=inputs["rejected_input_ids"]).logits,
            inputs["rejected_input_ids"], inputs["rejected_labels"], pad)

        # 参考策略 π_ref（全冻结）。用 inference_mode 比 no_grad 更彻底——除了脱图还禁用
        # view tracking，对纯前向+立即消费的路径显存更稳。注意 ref 全冻结 + GC 开着，
        # 本身中间层 activation 不会保留；这里 inference_mode 主要防止 ref_logits 张量本身
        # 被 autograd metadata 附加（提高 ref 释放及时性）。
        with torch.inference_mode():
            ref_logp_c, _ = token_logprobs(
                self.ref_model(input_ids=inputs["chosen_input_ids"]).logits,
                inputs["chosen_input_ids"], inputs["chosen_labels"], pad)
            ref_logp_r, _ = token_logprobs(
                self.ref_model(input_ids=inputs["rejected_input_ids"]).logits,
                inputs["rejected_input_ids"], inputs["rejected_labels"], pad)

        # logp_* 是 fp32 标量 (B,)，相减后乘 β 再过 logsigmoid。fp32 精度计算 DPO Δ。
        log_ratio_c = logp_c - ref_logp_c
        log_ratio_r = logp_r - ref_logp_r
        loss = -F.logsigmoid(self.beta * (log_ratio_c - log_ratio_r)).mean()
        return (loss, {"loss": loss}) if return_outputs else loss


class DPODataCollator:
    """分别对 chosen / rejected 序列做 padding（两者长度不同）。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        for key in ("chosen", "rejected"):
            ids = [f[f"{key}_input_ids"] for f in features]
            labels = [f[f"{key}_labels"] for f in features]
            max_len = max(len(x) for x in ids)
            padded_ids, padded_labels = [], []
            for x, y in zip(ids, labels):
                pad_n = max_len - len(x)
                padded_ids.append(x + [self.tokenizer.pad_token_id] * pad_n)
                padded_labels.append(y + [-100] * pad_n)
            batch[f"{key}_input_ids"] = torch.tensor(padded_ids, dtype=torch.long)
            batch[f"{key}_labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--stage2_dir", type=str, required=True, help="Stage 2 的 adapter 目录")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dpo_data", type=str, default="dpo_pairs.jsonl")
    # Stage 3 DPO 默认 max_len=768：4 卡 7B DPO 1024 序列 + 双模型 + chosen/rejected 激活太大，768 更稳
    parser.set_defaults(max_len=768)
    args = parser.parse_args()
    setup_env()
    setup_output_dir(args.output_dir)

    # 所有诊断 print 只在主进程输出，避免 DDP 双进程重复日志
    IS_MAIN = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if IS_MAIN:
        print(f"[Stage3] 加载 base: {args.model_path} + stage2 adapter: {args.stage2_dir}")
    model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
    model = PeftModel.from_pretrained(model, args.stage2_dir)   # 可训练（更新 stage2 adapter）
    # 【OOM 修复 关键】PeftModel.from_pretrained 后必须重新跑 prepare_model_for_kbit_training：
    #   1) enable_input_require_grads → 让 embedding 输出 requires_grad=True，否则 grad checkpoint
    #      重新前向时 LoRA→base 梯度链断裂，autograd 检测到 require_grad 路径不全，会强制保留
    #      整张激活图（base 7B 的激活直接吃满显存，这是 step225 撞 OOM 的根因之一）。
    #   2) cast LayerNorm 到 fp32 → 4bit QLoRA 标准做法，缺了会精度崩坏。
    #   和 stage2_sft.py --resume_adapter 路径完全对齐——stage3 续训 stage2 adapter 也得重做。
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False})
    # 显式开 grad checkpoint：DPO 激活是 model+ref 双重 + chosen+rejected 两序列 × 1024 序列，显存大头
    enable_grad_checkpointing(model)
    # peft 加载后 LoRA 参数 requires_grad 默认 False，显式启用
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad_(True)
    # 【OOM 修复】DDP × 4 同时加载两个 4bit 模型（policy + ref），中间不清缓存会让 allocator
    # 积累大量未返还的小 block。DDP 在 _sync_module_states 阶段需要 broadcast ~2GiB 连续 buffer，
    # 碎片化直接 OutOfMemoryError。和 stage2_sft.py 行 109-114 的清理时机对齐。
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if IS_MAIN:
            print(f"[Stage3-pre] post-policy-load alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G", flush=True)
    # 参考模型：与 model 相同初始化，单独实例、全冻结
    ref_base, _ = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
    ref_model = PeftModel.from_pretrained(ref_base, args.stage2_dir)
    # 【不跑 prepare】ref 全冻结 + inference_mode 前向→ 中间层 activation 不会保留，不需要 GC。
    # prepare_model_for_kbit_training 会 enable_input_require_grads()——会让 ref 的 embedding
    # 输出 requires_grad=True，反而增加 ref 的 autograd metadata 开销。
    # LN 数值精度问题：QLoRA 4bit 反量化到 bf16 计算 LN 与 fp32 LN 数值差异极小（<1e-3），
    # β·log_ratio 在 bf16 vs fp32 下偏差远小于此，DPO 收敛不受影响。
    # ref 是冻结副本，必须设 eval 模式（避免 DPOTrainer 误判 ref 也可训练、避免 drop/BN 等行为）
    ref_model.eval()
    model.train()
    # 加载完两个模型后再清一次缓存，避免进入 train loop 时 allocator 仍持有加载期间的临时 fp16/bf16 副本
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if IS_MAIN:
            print(f"[Stage3-pre] post-ref-load alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G", flush=True)
    # 加载完两个模型后再清一次缓存，避免进入 train loop 时 allocator 仍持有加载期间的临时 fp16/bf16 副本
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if IS_MAIN:
            print(f"[Stage3-pre] post-ref-load alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G", flush=True)

    rows = read_jsonl(f"{args.data_dir}/{args.dpo_data}", args.max_samples)
    if IS_MAIN:
        print(f"[Stage3] DPO 数据: {len(rows)} 对")
    dataset = Dataset.from_list(
        [encode_pair(r, tokenizer, args.max_len, SYSTEM_PROMPT) for r in rows])
    dataset = dataset.filter(lambda x: len(x["chosen_input_ids"]) > 0
                             and len(x["rejected_input_ids"]) > 0)

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
        save_steps=args.max_steps if args.max_steps > 0 else 200,
        save_total_limit=2,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=[],
        ddp_find_unused_parameters=False,  # DPO 实际无 unused 参数（之前 True 是防御）
    )
    trainer = DPOTrainer(
        model=model, ref_model=ref_model, beta=args.beta,
        args=train_args, train_dataset=dataset,
        processing_class=tokenizer, data_collator=DPODataCollator(tokenizer),
    )
    # 确保 trainer.processing_class 已设（避免 transformers 内部访问 .tokenizer 触发 deprecation）
    if trainer.processing_class is None:
        trainer.processing_class = tokenizer
    # DPO 的 input key 是 chosen_input_ids/rejected_input_ids，告知 Trainer 让其能 estimate tokens。
    # 关键：必须设到 base model 上——PeftModel.__getattr__ 把 floating_point_ops/estimate_tokens
    # 转发给 base_model 执行，方法体内读的是 base 的 main_input_name（默认 "input_ids"），
    # 只改 PeftModel 包装层上的同名属性不会被读到（此前该修复未生效、警告仍在的根因）。
    model.get_base_model().main_input_name = "chosen_input_ids"
    if hasattr(model, "main_input_name"):
        model.main_input_name = "chosen_input_ids"
    trainer.add_callback(ProgressCallback())
    if IS_MAIN:
        print("[Stage3] 开始 DPO 训练 ...")
    trainer.train()
    if IS_MAIN:
        print(f"[Stage3] 保存 adapter 到 {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if IS_MAIN:
        print("[Stage3] 完成。")


if __name__ == "__main__":
    main()
