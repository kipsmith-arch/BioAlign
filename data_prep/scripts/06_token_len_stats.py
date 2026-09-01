# -*- coding: utf-8 -*-
"""
06_token_len_stats.py —— 统计 SFT/DPO 数据的 token 长度分布
==============================================================
基于实测长度决定 max_len（不再拍脑袋）。

统计：
  - SFT 输入长度（input 字段，Qwen ChatML 包裹）
  - SFT 输出长度（output 字段）
  - SFT 总长度（input + output，模拟 encode_sft）
  - DPO 总长度（prompt + chosen；prompt + rejected，模拟 encode_pair）
    chosen=output 字段；rejected=output 字段（这是 7B build_preference 之前，但可作为上界参考）

用法：python 06_token_len_stats.py
输出：print 分位数表 + 可选保存 stats.json
"""
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

TOK_PATH = os.environ.get("TOK_PATH", "D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct")
DATA_PATH = "../data_prep/output/train_pool_clean.jsonl"
SAMPLE_N = 10000
SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful biology assistant. "
    "Please answer my biology sequence-related questions clearly and concisely. "
    "For regression tasks, please return a number."
)


def stat(name, lens):
    arr = np.array(lens)
    p = np.percentile(arr, [50, 75, 90, 95, 99])
    print(f"  {name:<22} n={len(arr):>6} | mean={arr.mean():>6.0f} | "
          f"p50={p[0]:>5.0f} p75={p[1]:>5.0f} p90={p[2]:>5.0f} "
          f"p95={p[3]:>5.0f} p99={p[4]:>5.0f} max={arr.max():>5.0f}")
    return arr


def main():
    print(f"加载 tokenizer: {TOK_PATH}")
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 模拟 SFT 的 ChatML 模板（与 stage2_sft.encode_sft 一致）
    def sft_encode_len(input_text, output_text):
        """复刻 stage2_sft.encode_sft 的 ChatML 包裹，返回 (full_len, output_len)。

        为什么两次 apply_chat_template：
          - add_generation_prompt=False → 包含 assistant 完整消息 "full"；
          - add_generation_prompt=True  → 只到 "<|im_start|>assistant\\n" 处 "prompt"；
          - output_len = full - prompt = assistant 这部分的 token 数（近似，会
            受 `prompt_text` 末尾 `<|im_start|>assistant\n` 几个 token 差影响）。

        输出是 token **数**，不是 bytes；采样权重要点是：你那边看到的 SFT total
        是 "包含 ChatML 模板 + input + output" 的总长。max_len 调践时用 total。
        """
        msgs_full = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        msgs_prompt = msgs_full[:-1]
        full = tok.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
        full_ids = tok.encode(full, add_special_tokens=False)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        return len(full_ids), len(full_ids) - len(prompt_ids)

    sft_total, sft_output = [], []
    dpo_chosen_total, dpo_rejected_total = [], []

    print(f"读取 {SAMPLE_N} 条样本（从 {DATA_PATH}）...")
    n = 0
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            total, out = sft_encode_len(d["input"], d["output"])
            sft_total.append(total)
            sft_output.append(out)
            # ============================================================================
            # DPO 模拟 —— 这里是 7B build_preference **之前**用的上界
            # ============================================================================
            # chosen = output（生成的高质量答案）
            # rejected = output（**同样** output 作为上界）
            #
            # 【为什么是上界】
            #   实际 DPO pair 里 chosen 与 rejected 是两个不同答案（rejected 是被
            #   偏好的反例）；distinct rejected 通常与 chosen 长度接近但是偏长一点。
            #   本脚本在 build_preference 运行之前写，手上只有 chosen 字段，所以
            #   保守地拒绝 rejected_total = chosen_total —— 能估出**不会被截的样本**。
            #
            #   当 build_preference 产出后，应运行 07_dpo_token_len_stats.py，它
            #   才有真实 chosen/rejected 分组统计。
            # ============================================================================
            dpo_chosen_total.append(total)
            dpo_rejected_total.append(total)  # rejected 上界（同 prompt + 同长度答案）
            n += 1
            if n >= SAMPLE_N:
                break

    print(f"\n=== 统计结果（n={n}）===")
    print("\n[SFT input + output 长度（含 ChatML 模板）]")
    stat("SFT total tokens", sft_total)
    stat("SFT output tokens", sft_output)

    print("\n[DPO total 长度（prompt + chosen 或 prompt + rejected）]")
    stat("DPO chosen total", dpo_chosen_total)
    stat("DPO rejected total", dpo_rejected_total)

    # 给出 max_len 建议
    arr = np.array(sft_total)
    for thr in [512, 768, 1024, 1280, 1536, 2048]:
        ratio = (arr <= thr).mean()
        print(f"  max_len={thr:>4}: 保留 {(arr <= thr).mean()*100:>5.1f}% 样本（不截断）")
    print()


if __name__ == "__main__":
    main()
