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
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

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


def setup_env():
    """训练环境清理：消除 Kaggle Debugger 重复警告、tokenizer 并行警告、checkpoint shards 重复、
    DDP 进程组未关闭警告。必须在任何 transformers/torch 导入前调用。"""
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # 降低 transformers/peft 库的日志级别（避免 "Loading checkpoint shards" 等重复噪音）
    # 注意：accelerate 的 TP warnings 不抑制——那是真问题，需在加载后清空 _tp_plan
    import logging
    for name in ("transformers.modeling_utils", "transformers.tokenization_utils_base",
                 "transformers.trainer", "peft", "peft.utils", "peft.tuners.tuners_utils"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # 注册 DDP 进程组清理函数：异常退出（OOM 强杀/KeyboardInterrupt）时也能执行
    # 消除 "destroy_process_group() was not called before program exit" 警告
    import atexit
    import torch.distributed as dist
    def _cleanup_pg():
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    atexit.register(_cleanup_pg)


class ProgressCallback(TrainerCallback):
    """训练进度日志（替代不可用的 tqdm）：步数/进度%/loss/显存/ETA。
    仅 rank 0 打印，避免 DDP 双进程重复日志；显式 flush 保证 commit 模式可见。"""

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0

    def on_log(self, args, state, control, logs, **kwargs):
        if not self.is_main:
            return
        gs = state.global_step
        total = state.max_steps if (state.max_steps and state.max_steps > 0) else None
        elapsed = time.time() - self.t0
        if total:
            pct = f"{100 * gs / total:.1f}%"
            eta = elapsed / gs * (total - gs) if gs > 0 else None
            eta_s = f"{eta / 3600:.1f}h" if eta else "?"
            step_s = f"{gs}/{total}"
        else:
            pct, eta_s, step_s = "?", "?", str(gs)
        mem = torch.cuda.memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
        loss = logs.get("loss", "?")
        loss_s = f"{loss:.4f}" if isinstance(loss, float) else loss
        print(f"[进度] step {step_s} ({pct}) | loss={loss_s} | "
              f"已用 {elapsed / 60:.1f}min | ETA {eta_s} | 显存 {mem:.1f}GiB",
              flush=True)


def add_common_args(parser: argparse.ArgumentParser):
    """通用训练参数（所有 stage 共用）。"""
    parser.add_argument("--model_path", type=str, required=True,
                        help="基座模型路径（本地 0.5B / Kaggle 3B）")
    parser.add_argument("--data_dir", type=str, default="data_prep/output",
                        help="数据目录（data_prep/output）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="checkpoint 输出目录")
    parser.add_argument("--max_len", type=int, default=2048,
                        help="最大序列长度：4bit 下 T4 可支持 2048（覆盖论文 2000 字符序列，避免截断）")
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
    # 真修 TP warnings：accelerate 的 check_tp_plan 检测到模型有 TP 规则但我们用 DDP/device_map 不应用
    # 清空模型 _tp_plan / config._tp_plan = 告诉 accelerate "我们不打算做 TP"（根因消除，非抑制日志）
    for attr in ("_tp_plan",):
        if hasattr(model, attr):
            setattr(model, attr, None)
    if hasattr(model.config, "_tp_plan"):
        model.config._tp_plan = None
    return model, tokenizer


def enable_grad_checkpointing(model):
    """统一开启 grad checkpoint，显式传 use_reentrant=False（消除 PyTorch 2.5+ 警告）"""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})


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


def build_lora_plus_optimizer(model, base_lr: float, scaler: float = 4.0, weight_decay: float = 0.0, use_8bit: bool = True):
    """LoRA+ 优化器（Hayou et al. 2024）：B 权重的学习率 = A 的 scaler 倍。
    论文使用 scaler=4（原论文建议 16，继续预训练中 16 不稳定）。
    分组：lora_A / lora_B / 其他可训练参数（如 RMSNorm）。
    use_8bit：bitsandbytes 8bit AdamW，大幅节省优化器状态显存（fp32 8B/参数 → 1B/参数）。"""
    if use_8bit:
        from bitsandbytes.optim import AdamW8bit
        Opt = AdamW8bit
    else:
        from torch.optim import AdamW
        Opt = AdamW
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
    return Opt([
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
