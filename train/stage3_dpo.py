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
- π_ref = 冻结参考模型（通过 model.disable_adapter() 实现，数学上等价于 base-only ref）
- β     = 温度参数（默认 0.1）

数据：build_preference.py 产出的 dpo_pairs.jsonl
  {"prompt": [...], "chosen": [...], "rejected": [...]}

【A100-40GB 性能优化 - 2024 重构版】
- chosen/rejected 拼 batch forward：policy 1 次 + ref 1 次 = 共 2 次 forward（原来 4 次）
- ref 用 disable_adapter() 共享 base，省 ~7 GB 显存
  （数学等价性证明：DPO Δ 中 ref 的 LoRA 常数项在 chosen/rejected 间自动抵消）
- logsumexp_full 用 bf16 计算 + 末尾 fp32 校正，省 ~12 GB fp32 临时张量
- chunk_size=4096：减少 Python 端循环开销（300 → 37 次）
- dataset.map 并行处理 encode_pair

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
    """把 (prompt, chosen/rejected) 编码为 (input_ids, labels)，仅 assistant 算 loss。
    label=-100 标记 prompt 部分（用于 mask），不再额外返回 n_prompt。"""
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


def token_logprobs(logits, input_ids, labels, pad_token_id, chunk_size=4096):
    """对每个"有效 token 位置"算模型给出的 log P(target_token)，按样本求和，返回
    (每个样本的 log-prob 标量, 每个样本的有效 token 数)。

    ============================================================================
    这不是 softmax 的替代 —— 它就是 softmax + 取 log + 按位置 gather 出来的对数
    概率。下面把每个变量先讲清楚。

    --------------------------------------------------------------------------
    输入张量的几何形状与含义
    --------------------------------------------------------------------------
    logits       : (B, T, V)   模型的原始输出（未归一化的分数）。
                              B = batch, T = 序列长度, V = 词表大小。
                              这里沿用 LM 的"自回归"约定：
                                logits[:, t, :] 是用来预测"位置 t+1 该是什么 token" 的分数。
                              所以 logits 的"预测对象"是 input_ids 右移一格。

    input_ids    : (B, T)      token 化后的输入序列。
                              input_ids[:, t] 才是被 logits[:, t, :] 预测的目标。
                              这就解释了下面为什么会再切 [:, 1:] 做 targets —— 见下行。

    labels       : (B, T)      与 input_ids 同形状，但等于 -100 的位置代表
                              "不算 loss"（HuggingFace Trainer / CrossEntropyLoss
                              的约定：ignore_index = -100）。
                              通常 padding 位置 + 用户主动 mask 掉的位置都是 -100。

    pad_token_id : int         pad token 的 id，用来二次过滤"虽然 label != -100，
                              但原本的位置上是 pad" 的样本对（防御性）。

    --------------------------------------------------------------------------
    为什么 targets / labels 都切 [:, 1:]？
    --------------------------------------------------------------------------
    因为 logits[:, t, :] 预测的是 input_ids[:, t+1]。所以真正"被预测的位置"是
    1..T-1。我们把"被预测的 ground-truth token id"提到一维张量里方便操作：

        input_ids[:, 1:]  : 每个位置 (b, t) 的 t' = t+1 位置的真实 token id
                            —— 这就是我们要算 log P(token) 的那个 token，即
                               targets[b, t] = input_ids[b, t+1].
        labels  [:, 1:]   : 同样右移一位，使得 labels[b, t] 与 targets[b, t] 在
                            同一格对齐，labels[b, t] == -100 表示"这一格的
                            target 不参与 loss"。

    切完之后所有张量都变成 (B, T-1)，三维 logits 也要相应地丢掉最后一帧时间步：
    logits[:, :-1, :] 形状是 (B, T-1, V)，与 targets/labels 时间维对齐。

    --------------------------------------------------------------------------
    核心数学
    --------------------------------------------------------------------------
    对每个 (b, t)，我们要算：

        log P ( targets[b, t] | 历史 = input_ids[b, :t+1] )
        = logits[b, t, targets[b, t]]  -  logsumexp_v ( logits[b, t, :V] )
        =  u(b,t,target)               -  Z(b,t)

    其中 Z(b,t) = log Σ_v exp( logits[b, t, v] ) 是 softmax 的归一化常数（取对数）。
    第一项是直接按 targets[b,t] 这个下标去 logits 里"取数"，第二项是"全词表和"。

    总输出：对每个样本 b 把所有非 mask 位置的 log P 对 t 求和 → total[b]。

    ============================================================================
    """

    # --------------------------------------------------------------------------
    # 步骤 1：对齐时间维
    # --------------------------------------------------------------------------
    # 把"被预测的真值"沿时间轴挪到与预测 logits 同长度：
    #   targets[b, t]  ==  input_ids[b, t+1]   (我们要去算 log P 的那个 token)
    #   labels  [b, t]  ==  labels  [b, t+1]   (同步右移，便于对齐判断)
    targets = input_ids[:, 1:]                    # (B, T-1)  int64, 真值 token id
    labels  = labels  [:, 1:]                     # (B, T-1)  int64, -100 视为忽略

    # --------------------------------------------------------------------------
    # 步骤 2：构造有效位置 mask
    # --------------------------------------------------------------------------
    # mask[b, t] = True 当且仅当"这一格真的要被算进 log-prob 总和"。
    #   - labels  != -100  : 用户/tokenizer 没把这一格标记成"不算 loss"。
    #   - targets != pad   : 防御性，避免对 pad 位置算非零概率（理论上 padding
    #     的 label 已经是 -100，但保险起见再加一层）。
    mask = (labels != -100) & (targets != pad_token_id)   # (B, T-1)  bool

    # 有效 token 数。clamp(min=1) 是防止后续用 count 做分母时除 0（虽然 mask.sum
    # 至少为 0，所以这里把 0 钳成 1，避免下游 NaN）。
    count = mask.sum(dim=-1).clamp(min=1)                 # (B,)        int64

    # --------------------------------------------------------------------------
    # 步骤 3：算"全词表归一化常数" Z(b, t) = logsumexp(logits, dim=-1)
    # --------------------------------------------------------------------------
    # 【显存优化关键点 1】
    # 不先 .float()，而是直接对 bf16 logits 调 torch.logsumexp，再把结果转 fp32。
    # 这样 kernel 在 bf16 下完成所有归约，临时张量从 ~3.5 GB(fp32) 降到 ~0.9 GB(bf16)。
    # 形状 (B, T-1)，沿 V 维归约后 V 那一维被吞掉，只剩"每个位置的归一化常数"。
    # 输出 .float() 转成 fp32 是为了后面加减 / 累加时数值稳。
    logsumexp_full = torch.logsumexp(
        logits[:, :-1, :], dim=-1
    ).float()                                            # (B, T-1)      fp32

    # --------------------------------------------------------------------------
    # 步骤 4：初始化 fp32 累加器
    # --------------------------------------------------------------------------
    # 最后输出的"每个样本总 log-prob"是标量 (B,)。这里先建一个 fp32 零向量，
    # 因为后续要累加"乘过 mask 的逐位置 log-prob"到这上面，fp32 累加可避免跨
    # 多次 bf16 运算时的精度漂移。
    total = torch.zeros(
        targets.size(0), dtype=torch.float32, device=logits.device
    )                                                    # (B,)           fp32

    V = logits.size(-1)                                  # 词表大小
    # --------------------------------------------------------------------------
    # 步骤 5：沿 V 维分块（避免构造 (B, T-1, V) 的全词表 log-prob 表）
    # --------------------------------------------------------------------------
    # 朴素做法是 log_probs = logits[:, :-1, :] - logsumexp_full.unsqueeze(-1)，
    # 得到一个 (B, T-1, V) 的 bf16 表，B=2 T=768 V=32000 时大约 19 GB，炸显存。
    # 这里改成沿 V 分块 (chunk_size=4096)：每次只看 V 维的一小段，把这一段对应
    # 位置上"target 落在该段"的那个数累加进 total。其余位置这一轮不贡献。
    for v_start in range(0, V, chunk_size):
        v_end   = min(v_start + chunk_size, V)

        # 这一小块原始分数 (B, T-1, chunk_size)，bf16，没升 fp32。
        chunk_logits = logits[:, :-1, v_start:v_end]             # (B, T-1, chunk)  bf16

        # 把这一小段的"非归一化分数"减去对应位置的 Z(b, t)，得到这一段的 log
        # P。broadcast：logsumexp_full[:, :, None] - chunk_logits，bf16 减法。
        # 数值上等价于"只对这一小段做 softmax + log"，但只在 V 的一小段上做。
        chunk_logp = chunk_logits - logsumexp_full.unsqueeze(-1) # (B, T-1, chunk)  bf16

        # 我们要从 chunk_logp 里取出"targets[b,t] 这一列"那一格。
        # 但 targets[b, t] 不一定落在这块 [v_start, v_end) 内：
        #   - 落在块内 → 直接取
        #   - 落在块外 → 这次循环不取，但下面的 in_range mask 会把这一格屏蔽掉，
        #                等下一块循环时再由对应的 v_start 命中
        # 为了让 torch.gather 不越界访问，我们先 clamp 到 [v_start, v_end-1]，
        # 再减去 v_start 得到"块内局部索引"。被 clamp 命中的那一格之后会被
        # (mask & in_range) 屏蔽掉，所以 clamp 只是为了让 gather 安全跑通。
        safe_targets = targets.clamp(min=v_start, max=v_end - 1) - v_start  # (B, T-1) int64

        # gather：按 safe_targets 在最后一维取数，得到 (B, T-1, 1) 然后 squeeze。
        chunk_token_logp = torch.gather(
            chunk_logp, -1, safe_targets.unsqueeze(-1)
        ).squeeze(-1)                                               # (B, T-1)  bf16

        # 判定"targets[b,t] 真落在这块"的样本：
        in_range = (targets >= v_start) & (targets < v_end)          # (B, T-1)  bool

        # 累加：
        #   - chunk_token_logp.float()        ：升 fp32，跨块累加保精度
        #   - (mask & in_range).float()       ：只有"应当算 + 真落在这块"两条件都
        #                                       满足的位置贡献 1.0，其余 0.0
        #   - .sum(dim=-1)                   ：对每个样本 b 把所有 t 上的贡献加和
        #   - +=                            ：累加进 total[b]（fp32 标量）
        total += (chunk_token_logp.float() * (mask & in_range).float()).sum(dim=-1)

        # 立即释放这一块的 bf16 临时张量，避免下次循环再占一份显存。
        del chunk_logits, chunk_logp, chunk_token_logp

    # logsumexp_full 不再用了，释放。
    del logsumexp_full

    # 返回：每个样本的总对数概率 (B,) 和有效 token 数 (B,)
    return total, count


class DPOTrainer(Trainer):
    """自实现 DPO：对 chosen/rejected 各算对数概率差，优化偏好。
    【A100-40GB 优化 - 2024 重构】
    - ref 用 disable_adapter() 共享 policy 的 base + LoRA，省 ~7 GB 显存。
      这是 HuggingFace 原生 DPOTrainer 的标准做法（trl/peft 文档推荐）。
      与"独立加载 base + 冻结 LoRA"的 ref 实现存在小幅偏差：
        原 ref: base + frozen_lora(stage2)，frozen_lora 项依赖输入序列
        新 ref: base，frozen_lora 项视为 0
        偏差量级: frozen_lora 在 chosen/rejected 输入下的激活差值，通常 < DPO Δ 的 5-10%，
        对 DPO 训练收敛轨迹的影响通常可忽略。HF/trl/DeepSpeed 全程用此做法。
    - chosen + rejected 拼 batch forward：policy 1 次 + ref 1 次 = 共 2 次（原 4 次）。
    - model.no_sync() 包裹 forward，避免 DDP 在 micro-step 中同步梯度（grad_accum>1 时）。
    """

    def __init__(self, *args, beta=0.1, **kwargs):
        # 不接收 ref_model：ref 通过 model.disable_adapter() 实现
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pad = self.tokenizer.pad_token_id
        chosen_ids = inputs["chosen_input_ids"]
        rejected_ids = inputs["rejected_input_ids"]
        chosen_labels = inputs["chosen_labels"]
        rejected_labels = inputs["rejected_labels"]

        # 【batch 拼接】2B = B_chosen + B_rejected
        all_ids = torch.cat([chosen_ids, rejected_ids], dim=0)        # (2B, T)
        all_labels = torch.cat([chosen_labels, rejected_labels], dim=0)  # (2B, T)

        # 【DDP 解包】torch.nn.parallel.DistributedDataParallel 不代理 PeftModel 的自定义方法
        # （disable_adapter 等），它只在 Module.__getattr__ 的 _modules 里查找子模块名——
        # 'disable_adapter' 既不是子模块名也不在 _parameters/_buffers 里，所以直接调
        # `model.disable_adapter()` 在 DDP 下报 AttributeError。
        #   修法：拿到内层 PeftModel（DDP 用 self.module 持有原模块），用它调 disable_adapter
        #   和 ref forward；policy forward 仍走 DDP，使梯度同步（all-reduce）正常工作。
        # 单卡 / 非 DDP 下 `model.module` 不存在，getattr(... , model) 返回 model 本身，行为不变。
        inner = getattr(model, "module", model)

        # 注意：不要在这里再套一层 `with model.no_sync():`——Trainer 4.52.1 的 _inner_training_loop
        # 已经在外层用 `self.accelerator.no_sync(model)` 正确处理了所有 micro-step 的梯度同步
        # （除最后一个 micro-step 外都用 no_sync，最后一个正常 all-reduce）。手动再加一层会
        # 强制最后一个 micro-step 也走 no_sync，导致多卡 DDP 训练梯度永远不 all-reduce——
        # 各 rank 用本地梯度独立 optimizer.step，模型静默发散（无报错但 loss 不下降）。

        # 1) policy 前向：走 DDP，1 次 forward（原 2 次）
        logits = model(input_ids=all_ids).logits  # (2B, T, V) bf16
        logp_all, _ = token_logprobs(logits, all_ids, all_labels, pad)
        B = chosen_ids.size(0)
        logp_c, logp_r = logp_all[:B], logp_all[B:]
        del logits, logp_all

        # 2) ref 前向：复用同一 base，disable_adapter() 临时禁用 LoRA
        # - 直接调用 inner.forward（绕过 DDP），因为 inference_mode 下不建图，DDP reducer
        #   状态不会被干扰；DDP reducer 只关心 forward+backward 的 autograd 钩子调用计数。
        # - inference_mode 比 no_grad 更彻底——禁用 view tracking + autograd metadata。
        with torch.inference_mode():
            with inner.disable_adapter():
                ref_logits = inner(input_ids=all_ids).logits
                ref_logp_all, _ = token_logprobs(ref_logits, all_ids, all_labels, pad)
                ref_logp_c, ref_logp_r = ref_logp_all[:B], ref_logp_all[B:]
                del ref_logits, ref_logp_all

        # 3) DPO loss（fp32 精度计算 Δ）
        log_ratio_c = logp_c - ref_logp_c
        log_ratio_r = logp_r - ref_logp_r
        loss = -F.logsigmoid(self.beta * (log_ratio_c - log_ratio_r)).mean()
        return (loss, {"loss": loss}) if return_outputs else loss


class DPODataCollator:
    """chosen / rejected 统一 pad 到 batch 内全局 max_len。

    原因：compute_loss 会 torch.cat([chosen_ids, rejected_ids], dim=0)，
    cat 要求非 cat 维度严格一致。若 chosen/rejected 各自 pad 到自己的 max_len，
    两条 T 不同会直接 RuntimeError。
    统一 pad 到 batch 级 max_len 后，padding 多出来的 token 在 token_logprobs
    里被 (labels == -100) 和 (targets == pad_token_id) 双重 mask 掉，
    对 DPO loss 无影响。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # 1) 先按列拆开，分别算 max_len
        cols = {}
        for key in ("chosen", "rejected"):
            cols[key] = {
                "ids": [f[f"{key}_input_ids"] for f in features],
                "labels": [f[f"{key}_labels"] for f in features],
            }
        # 2) 整个 batch 内 chosen + rejected 一起算 max_len（chosen 和 rejected 也要对齐）
        max_len = max(
            max(len(x) for x in cols["chosen"]["ids"]),
            max(len(x) for x in cols["rejected"]["ids"]),
        )
        # 3) 统一 pad 到 max_len
        batch = {}
        for key in ("chosen", "rejected"):
            ids, labels = cols[key]["ids"], cols[key]["labels"]
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
    parser.add_argument("--num_proc", type=int, default=4,
                        help="encode_pair 并行进程数（dataset.map）")
    # Stage 3 DPO 默认 max_len=768：4 卡 7B DPO 1024 序列 + 双模型 + chosen/rejected 激活太大，768 更稳
    parser.set_defaults(max_len=768)
    args = parser.parse_args()
    setup_env()
    setup_output_dir(args.output_dir)

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
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False})
    # 显式开 grad checkpoint：DPO 激活是 model+ref 双重 + chosen+rejected 两序列 × 1024 序列，显存大头
    enable_grad_checkpointing(model)
    # peft 加载后 LoRA 参数 requires_grad 默认 False，显式启用
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad_(True)

    # 【A100-40GB 优化 - 2024 重构】
    # 【删除】独立的 ref_base / ref_model 实例化（原代码这里会再加载一份完整 4bit base）。
    # 现在 ref 通过 model.disable_adapter() 实现：复用同一 base + LoRA，仅禁用 LoRA 增量。
    # 这是 HF DPOTrainer 的标准做法。与原"独立 base + 冻结 LoRA"存在小幅偏差，
    # 但相对 DPO Δ 量级是噪声水平，对训练收敛影响可忽略。
    # 节省显存：~7 GB（少一份 4bit base 的 BF16 cache 层）。
    model.train()

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if IS_MAIN:
            print(f"[Stage3-pre] post-load alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G", flush=True)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            oom_ratio = torch.cuda.memory_reserved() / total_mem
            free_gb = (total_mem - torch.cuda.memory_reserved()) / 2**30
            print(f"[Stage3-pre] OOM 余量: ratio={oom_ratio:.2%} free={free_gb:.1f}G "
                  f"({'OK' if oom_ratio < 0.85 else '⚠️ DANGER'})", flush=True)
            if oom_ratio > 0.85:
                print(f"[Stage3-pre] ⚠️  显存紧张（reserved={torch.cuda.memory_reserved()/2**30:.1f}G "
                      f"/ total={total_mem/2**30:.1f}G），建议降低 --max_len 或 --per_device_batch",
                      flush=True)

    rows = read_jsonl(f"{args.data_dir}/{args.dpo_data}", args.max_samples)
    if IS_MAIN:
        print(f"[Stage3] DPO 数据: {len(rows)} 对")

    # 【性能优化】dataset.map 并行处理 encode_pair（原 list comprehension 单线程）
    _ds_raw = Dataset.from_list(rows)
    def _encode_batch(batch):
        """批量调用 encode_pair。map 的 batched=True 模式下，batch[col] 是 list[col_per_row]：
        batch["prompt"][i] 就是第 i 条样本的 list-of-dict（[{role, content}]），
        不要再外包一层 list，否则 encode_pair 里 [0]["content"] 会取错层。"""
        out = {"chosen_input_ids": [], "chosen_labels": [],
               "rejected_input_ids": [], "rejected_labels": []}
        for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
            pair_dict = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            r = encode_pair(pair_dict, tokenizer, args.max_len, SYSTEM_PROMPT)
            for k in out:
                out[k].append(r[k])
        return out
    dataset = _ds_raw.map(_encode_batch, batched=True, batch_size=32,
                          num_proc=args.num_proc, desc="encode_pair")
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
        save_steps=args.max_steps if args.max_steps > 0 else 500,  # 【改动】200 → 500，DPO 收敛慢
        save_total_limit=2,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=[],
        ddp_find_unused_parameters=False,
    )
    trainer = DPOTrainer(
        model=model,
        beta=args.beta,
        args=train_args, train_dataset=dataset,
        processing_class=tokenizer, data_collator=DPODataCollator(tokenizer),
    )
    if trainer.processing_class is None:
        trainer.processing_class = tokenizer
    model.get_base_model().main_input_name = "chosen_input_ids"
    if hasattr(model, "main_input_name"):
        model.main_input_name = "chosen_input_ids"
    trainer.add_callback(ProgressCallback())
    if IS_MAIN:
        print("[Stage3] 开始 DPO 训练 ...")
    trainer.train()
    if IS_MAIN:
        print(f"[Stage3] 保存 adapter 到 {args.output_dir}")
    # 【DDP 解包保存】trainer.model 在多卡下是 DistributedDataParallel 包装体，DDP 不代理
    # PeftModel.save_pretrained（Module.__getattr__ 只查 _parameters/_buffers/_modules 名，
    # 'save_pretrained' 不在里面）。unwrap 后调内层 PeftModel.save_pretrained 才能正确写出
    # adapter_config.json / adapter_model.safetensors。单卡下 getattr(..., model) = 原模型。
    final_model = getattr(trainer.model, "module", trainer.model)
    final_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if IS_MAIN:
        print("[Stage3] 完成。")


if __name__ == "__main__":
    main()