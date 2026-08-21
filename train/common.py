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



# Stage 2 system prompt (Reason + Answer version)
# 设计动机: 原 prompt 只说了 "regression 返回数字", 没说 classification 怎么答,
# 导致模型学到自然语言描述, 与 eval parser 期望的关键词格式脱节。
# 下面这个 prompt 明确要求结构化输出: 先 Reason 后 Answer,
# Answer 严格使用 label 字符串 (parser 能 100% 提取)。
SYSTEM_PROMPT = """You are a knowledgeable and helpful biology assistant. \
Please answer my biology sequence-related questions clearly and concisely.

FORMAT: Every response MUST contain exactly two sections in this order:
  1. <reason> - a brief justification (1-3 sentences).
  2. <ans> - the final answer, one of:
       * binary classification: positive or negative
       * multi-class classification: the class name (e.g. IRES, EC2.4.1.-, m6A, leader)
       * regression: a single numeric value (e.g. 3.14, -0.5)
       * multi-value regression: hk=0.12, dev=-0.34 or ON=0.3, OFF=0.4, ON_OFF=0.7

Example (binary classification):
  <reason>
  The RNA contains AU-rich elements matching the protein RRM domain.
  </reason>
  <ans>
  positive
  </ans>

Example (regression):
  <reason>
  Based on sequence composition, predicted thermostability is around 51.
  </reason>
  <ans>
  51.09
  </ans>
"""

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
    # Monkey-patch checkpoint 自动补 use_reentrant=False：PyTorch 2.5 强制要求传 use_reentrant，
    # 否则每次调用都报 deprecation 警告。注意两个坑（此前 patch 未生效的根因）：
    #   1) transformers 在 import 时执行 `from torch.utils.checkpoint import checkpoint`，
    #      手里握的是原函数对象——只改 torch 模块属性拦不住它，必须连 modeling_utils
    #      模块里那个引用一起换。
    #   2) 有的调用方显式传 use_reentrant=None，`if "use_reentrant" not in kwargs` 漏判，
    #      要用 setdefault 兜住 None。
    import torch.utils.checkpoint as _ckpt_mod
    if not getattr(_ckpt_mod.checkpoint, "_patched_use_reentrant", False):
        _orig_ckpt = _ckpt_mod.checkpoint
        def _ckpt_patched(function, *args, **kwargs):
            kwargs.setdefault("use_reentrant", False)
            return _orig_ckpt(function, *args, **kwargs)
        _ckpt_patched._patched_use_reentrant = True
        _ckpt_mod.checkpoint = _ckpt_patched
        import transformers.modeling_utils as _mu
        if _mu.checkpoint is not _ckpt_patched:
            _mu.checkpoint = _ckpt_patched
    # TP 检查补丁：transformers 在 from_pretrained **内部**调用 verify_tp_plan
    # （modeling_utils.py ~5025，按 logger.level >= WARNING 门控），加载完成后才清
    # model._tp_plan 为时已晚——警告必然在加载期打出。项目用 DDP 从不 TP，这个检查
    # 对我们无意义，把 modeling_utils 持有的 verify_tp_plan 引用换成 no-op（根因消除，非抑制日志）。
    import transformers.modeling_utils as _mu
    if not getattr(_mu, "_patched_verify_tp_plan", False):
        _mu.verify_tp_plan = lambda *a, **k: None
        _mu._patched_verify_tp_plan = True
    # Monkey-patch Trainer.tokenizer property：消除 "Trainer.tokenizer is now deprecated" 警告
    # transformers 4.52 的 @property 在每次访问 trainer.tokenizer 时都 warn——覆盖为直接返回 processing_class
    from transformers import Trainer as _HfTrainer
    if not getattr(_HfTrainer, "_patched_no_tokenizer_warning", False):
        _HfTrainer.tokenizer = property(lambda self: self.processing_class)
        _HfTrainer._patched_no_tokenizer_warning = True
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

    # 【公共环境信号补丁】将 SIGHUP / SIGINT / SIGTERM 转发成统一的 SIGTERM 逻辑。
    # transformers Trainer 内部在 4.52+ 注册了 _train_signal_handler 负责优雅退出：
    # 走 on_train_end callback（保存 checkpoint + adapter）→ 正常退出。
    # 公共环境（课题组 GPU 节点）会被 nvidia-smi/watch/systemd-logind 周期发 SIGHUP，
    # 默认行为是“主进程被发后调 handler 退出 + torchrun 转发给 workers"，但 elastic agent
    # 有时会将信号标记为 death_signal = SIGHUP 然后 torchrun 会报 SignalException(1)。
    # 转发为 SIGTERM 让 elastic 当作正常 shutdown，下次重启 Trainer 会从 checkpoint-1 resume。
    #
    # 【为什么不是 signal.SIG_IGN】：忽略信号会让 Trainer 进程在被 SIGKILL 强杀前
    # 没有机会走 checkpoint 保存 → 丢进度。转发为 SIGTERM 是弹性策略：信号传递 + 能保存。
    import signal as _signal
    def _forward_signal_to_sigterm(signum, frame):
        # 仅主进程调用；DDP worker 默认也会装这个 handler，因为 setup_env 在 worker 也调
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                f"[signal] 收到信号 {signum}({_signal.Signals(signum).name})，"
                f"转发为 SIGTERM 触发 Trainer 优雅退出",
                flush=True,
            )
        # 替换 handler 为默认（avoid recursion if SIGTERM handler call raise）
        _signal.signal(_signal.SIGTERM, _signal.SIG_DFL)
        # 重新发送给自己 → 走 transformers Trainer 的 _train_signal_handler
        os.kill(os.getpid(), _signal.SIGTERM)
    for _sig in (_signal.SIGHUP, _signal.SIGINT, _signal.SIGTERM):
        try:
            _signal.signal(_sig, _forward_signal_to_sigterm)
        except (ValueError, OSError):
            # SIGTERM 在子线程不可设，忽略；主线程都会成功
            pass


class ProgressCallback(TrainerCallback):
    """训练进度日志（替代不可用的 tqdm）：步数/进度%/loss/显存/ETA。
    仅 rank 0 打印，避免 DDP 双进程重复日志；显式 flush 保证 commit 模式可见。"""

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
        # 【准的】重置本次训练的 peak 采样器。
        # 在主进程上在 forward/backward 代码量都齐的环境才准确。
        # max_memory_allocated 本质是从该进程启动以来的高水堆，
        # 重启后是全重设。多次 on_train_begin 会读上次训 peak。
        if self.is_main and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
        # 【准的】原来这里只打 torch.cuda.memory_allocated()——只是那一调祥点，由 optimizer/loss
        # 反向/create 时不同。只看这个点判不开"OOM 高肤"。现在同时打：
        #   allocated  : 本进程当前 kernel 正持有的显存（实时）
        #   reserved   : allocator cache 中被多丒的 padding
        #   max_allocated : 从本进程启动以来为止最高点（含反向） 才是判 OOM 金标准
        #   free       : 设备物理剩余（可界估上下 OOM）
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 2 ** 30
            reserved = torch.cuda.memory_reserved() / 2 ** 30
            max_alloc = torch.cuda.max_memory_allocated() / 2 ** 30
            free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 2 ** 30
        else:
            alloc = reserved = max_alloc = free = 0
        loss = logs.get("loss", "?")
        loss_s = f"{loss:.4f}" if isinstance(loss, float) else loss
        print(f"[进度] step {step_s} ({pct}) | loss={loss_s} | "
              f"已用 {elapsed / 60:.1f}min | ETA {eta_s} | "
              f"显存 alloc={alloc:.1f}G reserved={reserved:.1f}G peak={max_alloc:.1f}G free={free:.1f}G",
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
    # TP warnings 的真修在 setup_env()（把 modeling_utils.verify_tp_plan 替换为 no-op）——
    # verify_tp_plan 是在 from_pretrained 加载期间被调用的，这里加载完才清 _tp_plan 已经晚了。
    # 下面的清空保留作防御（后续若有其他路径读 _tp_plan 也不至于误判我们在做 TP）。
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
    # 显式传 gradient_checkpointing_kwargs={"use_reentrant": False}：
    # peft 的 prepare_model_for_kbit_training 默认传 {}（空 dict 而非 None），transformers 4.52 的
    # 默认值 {"use_reentrant": True} 只在参数为 None 时生效 → 生成 bare partial(checkpoint)，
    # 每层前向都以 use_reentrant=None 调 torch checkpoint → 每步都报 deprecation 警告。
    # （stage1/3 加载后又显式 enable_grad_checkpointing() 才没暴露；stage2 只走这里，必须修）
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False})
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
    """读取 jsonl，逐条返回 dict 列表。捕获 JSONDecodeError 给出明确错误信息（文件路径 + 前 200 字节内容），
    帮助排查"空文件/损坏文件/路径错"等问题。"""
    import os
    out = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"read_jsonl: 文件不存在 → {path}")
    size = os.path.getsize(path)
    if size == 0:
        raise ValueError(f"read_jsonl: 文件为空（0 字节）→ {path}\n"
                         f"提示：build_preference 写出的文件是否被中断？或 --dpo_data 路径写错？")
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                # 报告出错行的实际字节（不是文件头）—— 上次 race 误导排查了一轮
                line_bytes = line.encode("utf-8")[:200]
                raise ValueError(
                    f"read_jsonl: JSON 解析失败 → {path}\n"
                    f"  错误行 {line_num}: {e}\n"
                    f"  出错行前 200 字节: {line_bytes!r}\n"
                    f"  文件总大小: {size} 字节"
                )
            if max_samples > 0 and len(out) >= max_samples:
                break
    if not out and size > 0:
        raise ValueError(f"read_jsonl: 文件 {size} 字节但无有效记录（可能全空行）→ {path}")
    return out


def setup_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
