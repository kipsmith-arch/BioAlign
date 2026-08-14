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
import sys
import os

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
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


def token_logprobs(logits, input_ids, labels, pad_token_id):
    """对每个非 -100 位置计算 log P(token)，返回 (总对数概率, 有效 token 数)。

    【显存优化】不 .float() 转为整张 (B, T-1, V) fp32——那在 V=152064 词表下光是 chosen 方向就是
    ~1.2GiB (2·1024·152064·4B)，与 logits[下一个]、输入的中间表示加在一起同时张在峰值。
    改为：log_softmax 在原精度上算（bfloat16 下数学等价、手枝 log_softmax 原生支持）→立刻 gather
    到 (B, T-1) fp32 的 token_logp 来做后续 sum。B T V 全张的中间张量不再出现。
    """
    # 避免 logits 跨 forward 被反向（拒绝 explicit .float() 保证 loss 断图）
    targets = input_ids[:, 1:]                    # (B, T-1)
    labels = labels[:, 1:]                        # (B, T-1)
    # log_softmax 默认在原 dtype 上算（bf16/fp16 下不需手动 .float()，底层有 kernel）
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)，原 dtype
    token_logp = torch.gather(logp, -1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    mask = (labels != -100) & (targets != pad_token_id)
    # sum 转为 fp32 只为数值稳定（全宽度 python 运算在泪为上不会能节省多少）
    total = (token_logp.float() * mask).sum(dim=-1)     # (B,)
    count = mask.sum(dim=-1).clamp(min=1)
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
        # 当前策略 π：分别对 chosen / rejected 前向
        logits_c = model(input_ids=inputs["chosen_input_ids"]).logits
        logits_r = model(input_ids=inputs["rejected_input_ids"]).logits
        # 【防御】显式保留梯度路径（某些 peft/量化组合下从 logits 派生 loss 会失梯度，
        # 以logits在计算图里的情况下显式打上 requires_grad 保证保险。
        # 但 为了避免同时在峰值中保留 2×(B,T,V) 的 fp32 副本，这里不 .float()。原有代码
        # 会 .float() 升为 fp32，全张 (2, 1024, 152064) fp32 ≈ 2·5GiB 显存爆炸。为节约峰值
        # 保证重构在 token_logprobs 内部。原始代码已删除 .float() 调用。
        logits_c.requires_grad_(True)
        logits_r.requires_grad_(True)
        logp_c, _ = token_logprobs(logits_c, inputs["chosen_input_ids"], inputs["chosen_labels"], pad)
        logp_r, _ = token_logprobs(logits_r, inputs["rejected_input_ids"], inputs["rejected_labels"], pad)

        # 参考策略 π_ref（冻结）。ref_model 已被设 .eval() 且有 GC，这里进一步明：
        # ①关掉反向以免 ref_logits 保留包含中间层的 autograd graph（全是死内存），
        # ②不需要 .requires_grad_(True)——torch.no_grad() 自动会造成 ref_logits 脱图。
        with torch.no_grad():
            ref_logits_c = self.ref_model(input_ids=inputs["chosen_input_ids"]).logits
            ref_logits_r = self.ref_model(input_ids=inputs["rejected_input_ids"]).logits
            ref_logp_c, _ = token_logprobs(ref_logits_c, inputs["chosen_input_ids"], inputs["chosen_labels"], pad)
            ref_logp_r, _ = token_logprobs(ref_logits_r, inputs["rejected_input_ids"], inputs["rejected_labels"], pad)

        # 【显存优化】降精度——logp_c/r、ref_logp_c/r 原 dtype 都是 bf16。.float() 后相减
        # fp32 精度使用。可控的多例化显存节约点。
        log_ratio_c = (logp_c - ref_logp_c).float()
        log_ratio_r = (logp_r - ref_logp_r).float()
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
    # peft 加载后 LoRA 参数 requires_grad 默认 False，显式启用
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad_(True)
    # 参考模型：与 model 相同初始化，单独实例、全冻结
    ref_base, _ = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
    ref_model = PeftModel.from_pretrained(ref_base, args.stage2_dir)
    # 显式开 grad checkpoint：DPO 激活是 model+ref 双重 + chosen+rejected 两序列 × 1024 序列，显存大头
    enable_grad_checkpointing(model)
    enable_grad_checkpointing(ref_model)
    # ref 是冻结副本，必须设 eval 模式（避免 DPOTrainer 误判 ref 也可训练、避免 drop/BN 等行为）
    ref_model.eval()
    model.train()

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
