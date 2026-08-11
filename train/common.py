# -*- coding: utf-8 -*-
"""
common.py —— 三段式流水线公共模块
=================================
QLoRA 模型加载、LoRA 配置、通用参数、数据加载工具。

单卡/双卡兼容：device_map 按 LOCAL_RANK 环境变量分配
（本地单卡 = 卡0；Kaggle torchrun 双卡 = 进程自动分配到卡0/卡1），
因此本地开发与 Kaggle 双卡训练使用同一份代码。
"""
import argparse
import json
import os
import sys

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 论文 Stage 2 system prompt（Psc）
SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful biology assistant. "
    "Please answer my biology sequence-related questions clearly and concisely. "
    "For regression tasks, please return a number."
)


def add_common_args(parser: argparse.ArgumentParser):
    """通用训练参数（所有 stage 共用）。"""
    parser.add_argument("--model_path", type=str, required=True,
                        help="基座模型路径（本地 0.5B / Kaggle 3B）")
    parser.add_argument("--data_dir", type=str, default="data_prep/output",
                        help="数据目录（data_prep/output）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="checkpoint 输出目录")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--per_device_batch", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help=">0 时覆盖 epochs（冒烟测试用）")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help=">0 时只取前 N 条数据（冒烟测试用）")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_model_tokenizer(model_path: str, use_4bit: bool = True, max_len: int = 1024):
    """加载 base 模型 + tokenizer。device_map 兼容单卡/多卡 DDP。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_len

    device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config,
            device_map=device_map, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device_map, trust_remote_code=True)
    model.config.use_cache = False
    return model, tokenizer


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, train_norm: bool = False):
    """给模型套 LoRA（QLoRA 标准做法）。train_norm=True 时同时训练 RMSNorm 层（论文 Stage 1 做法）。"""
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=QWEN_TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    if train_norm:
        # 启用 RMSNorm 训练（Qwen2.5: input_layernorm / post_attention_layernorm / norm）
        for n, p in model.named_parameters():
            if "layernorm" in n.lower() or n.endswith(".norm.weight"):
                p.requires_grad_(True)
    model.print_trainable_parameters()
    return model


def build_lora_plus_optimizer(model, base_lr: float, scaler: float = 4.0, weight_decay: float = 0.0):
    """LoRA+ 优化器（Hayou et al. 2024）：B 权重的学习率 = A 的 scaler 倍。
    论文使用 scaler=4（原论文建议 16，继续预训练中 16 不稳定）。
    分组：lora_A / lora_B / 其他可训练参数（如 RMSNorm）。"""
    from torch.optim import AdamW
    group_a, group_b, group_other = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in n:
            group_a.append(p)
        elif "lora_B" in n:
            group_b.append(p)
        else:
            group_other.append(p)
    return AdamW([
        {"params": group_a, "lr": base_lr},
        {"params": group_b, "lr": base_lr * scaler},
        {"params": group_other, "lr": base_lr},
    ], weight_decay=weight_decay)


def read_jsonl(path: str, max_samples: int = -1):
    """读取 jsonl，逐条返回 dict 列表。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if max_samples > 0 and len(out) >= max_samples:
                break
    return out


def setup_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
