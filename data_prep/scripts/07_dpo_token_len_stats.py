# -*- coding: utf-8 -*-
"""
07_dpo_token_len_stats.py —— 统计 DPO 数据 (dpo_pairs.jsonl) 的真实 token 长度分布
==================================================================================
复刻 stage3_dpo.py::encode_pair 的 ChatML 模板逻辑，保证统计与训练时 encode 一致。

注意：本脚本**故意不** from common import SYSTEM_PROMPT。
  train/common.py 顶层 import torch / peft / transformers / sklearn，会拖慢启动并污染环境。
  本脚本只依赖 transformers + numpy，已足够完成 token 统计。
  SYSTEM_PROMPT 的内容直接复制自 train/common.py 的 __main__ 常量（逐字对齐）。

统计口径：
  - prompt_len          : 单独算 prompt 部分 token 数
  - chosen_total_len    : prompt + chosen 整体 token 数（≤ max_len）
  - rejected_total_len  : prompt + rejected 整体 token 数（≤ max_len）
  - batch_max_len       : max(chosen_total, rejected_total)，即 collator 实际 pad 到的长度
                          训练时一个 micro-step 内的实际 activation/logits 尺寸由这个决定
输出：
  - 屏幕打印分位数表（p50/p75/p90/p95/p99/max）
  - 给出 max_len 在 512/768/1024/1280 时的样本保留率
  - 可选写 stats.json

用法：
  python data_prep/scripts/07_dpo_token_len_stats.py \
    --tok_path /path/to/Qwen2.5-7B-Instruct \
    --dpo_path input_data/dpo_pairs.jsonl \
    --max_len 1280 \
    [--out_json input_data/dpo_len_stats.json] \
    [--no_save_json]
"""
import argparse
import json
import os
import sys

import sklearn
import numpy as np
from transformers import AutoTokenizer

# ===== 与 train/common.py 第 34 行 SYSTEM_PROMPT 逐字一致 =====
# 不要在这里 from common import —— common.py 顶层 import torch/peft/transformers/sklearn，
# 会无谓拖慢启动并污染环境。下面是字面值拷贝，stage3_dpo.py 用的就是这一份。
SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful biology assistant. "
    "Please answer my biology sequence-related questions clearly and concisely. "
    "For regression tasks, please return a number."
)

sys.stdout.reconfigure(encoding="utf-8")


def stat(name, lens):
    arr = np.array(lens)
    p = np.percentile(arr, [50, 75, 90, 95, 99])
    print(f"  {name:<22} n={len(arr):>6} | mean={arr.mean():>6.1f} | "
          f"p50={p[0]:>5.0f} p75={p[1]:>5.0f} p90={p[2]:>5.0f} "
          f"p95={p[3]:>5.0f} p99={p[4]:>5.0f} max={arr.max():>5.0f}")
    return arr


def encode_len(pair, tokenizer, max_len, system_prompt):
    """与 stage3_dpo.py::encode_pair._enc 完全一致的 token 计数。
    返回 (chosen_total_len, rejected_total_len, prompt_len)"""
    def _enc(content):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["prompt"][0]["content"]},
            {"role": "assistant", "content": content},
        ]
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True)
        ids_full = tokenizer.encode(full_text, add_special_tokens=False,
                                    max_length=max_len, truncation=False)
        ids_prompt = tokenizer.encode(prompt_text, add_special_tokens=False,
                                      max_length=max_len, truncation=False)
        if len(ids_full) > max_len:
            ids_full = ids_full[:max_len]
        n_prompt = min(len(ids_prompt), len(ids_full))
        return len(ids_full), n_prompt

    c_total, prompt_len = _enc(pair["chosen"][0]["content"])
    r_total, _ = _enc(pair["rejected"][0]["content"])
    return c_total, r_total, prompt_len


def main():
    parser = argparse.ArgumentParser(
        description="统计 DPO 数据 (dpo_pairs.jsonl) 的 token 长度分布")
    parser.add_argument("--tok_path", type=str, required=True,
                        help="tokenizer 路径（必填，生产环境必须用 Qwen2.5-7B-Instruct）")
    parser.add_argument("--dpo_path", type=str, required=True,
                        help="DPO 数据 jsonl 路径（必填）")
    parser.add_argument("--max_len", type=int, required=True,
                        help="统计上限（必填，不截断；只是决定 encode 时不抛错的 token 上限）")
    parser.add_argument("--out_json", type=str,
                        default="input_data/dpo_len_stats.json",
                        help="分位数 + 保留率统计输出 json 路径")
    parser.add_argument("--no_save_json", action="store_true",
                        help="不写 json，只打印到屏幕")
    args = parser.parse_args()
    tok_path = args.tok_path
    data_path = args.dpo_path
    max_len_cap = args.max_len
    save_json = not args.no_save_json
    out_json = args.out_json

    print(f"加载 tokenizer: {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"读取 {data_path} ...")
    prompt_lens, chosen_totals, rejected_totals, batch_maxes = [], [], [], []
    n_total = 0
    n_truncated = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            try:
                c, r, p = encode_len(d, tok, max_len_cap, SYSTEM_PROMPT)
            except (KeyError, IndexError, TypeError):
                continue
            prompt_lens.append(p)
            chosen_totals.append(c)
            rejected_totals.append(r)
            batch_maxes.append(max(c, r))
            n_total += 1
            if c == max_len_cap or r == max_len_cap:
                n_truncated += 1

    print(f"\n=== 统计结果（n={n_total}，可能截断 {n_truncated} 条）===\n")
    print("[各部分长度分布]")
    stat("prompt_len", prompt_lens)
    stat("chosen_total", chosen_totals)
    stat("rejected_total", rejected_totals)
    stat("batch_max", batch_maxes)
    print(f"\n  截断率（>= {max_len_cap}）: "
          f"{n_truncated}/{n_total} = {n_truncated / max(n_total, 1) * 100:.2f}%\n")

    print("[max_len 选值建议 —— batch_max 在阈值内的保留率]")
    arr = np.array(batch_maxes)
    for thr in [512, 640, 768, 896, 1024, 1152, 1280, 1536]:
        ratio = (arr <= thr).mean()
        marker = "  ⭐ 推荐" if thr == 1024 else ""
        print(f"  max_len={thr:>4}: 保留 {ratio * 100:>5.1f}% 样本（不截断）{marker}")

    print("\n[显存粗略估计 —— per-GPU per micro-step 的 (2B, T, V) 关键张量]")
    print("  假设 per_device_batch=B（拼接后总 micro batch=2B），max_len=T，V=152064 (Qwen2.5-7B)")
    print("  policy bf16 logits = 2B·T·V·2B；ref 同尺寸（inference_mode 不建图但占用 alloc）")
    print("  + lm_head fp32 grad = 2B·T·V·4B  ← 反向时炸显存的真凶")
    print()
    for thr in [768, 1024]:
        for B in [2, 4]:
            T = thr
            logits_gb = 2 * B * T * 152064 * 2 / 2 ** 30
            grad_gb = 2 * B * T * 152064 * 4 / 2 ** 30
            print(f"  T={thr}, B={B} (→ 总 micro batch={2 * B}): "
                  f"logits≈{logits_gb:.1f}G + grad≈{grad_gb:.1f}G 合计 ≈ {logits_gb + grad_gb:.1f}G")

    if save_json:
        result = {
            "n": n_total,
            "n_truncated": n_truncated,
            "max_len_cap": max_len_cap,
            "prompt_len": {
                "mean": float(np.mean(prompt_lens)),
                "p50": float(np.percentile(prompt_lens, 50)),
                "p95": float(np.percentile(prompt_lens, 95)),
                "p99": float(np.percentile(prompt_lens, 99)),
                "max": int(max(prompt_lens))},
            "chosen_total": {
                "mean": float(np.mean(chosen_totals)),
                "p50": float(np.percentile(chosen_totals, 50)),
                "p95": float(np.percentile(chosen_totals, 95)),
                "p99": float(np.percentile(chosen_totals, 99)),
                "max": int(max(chosen_totals))},
            "rejected_total": {
                "mean": float(np.mean(rejected_totals)),
                "p50": float(np.percentile(rejected_totals, 50)),
                "p95": float(np.percentile(rejected_totals, 95)),
                "p99": float(np.percentile(rejected_totals, 99)),
                "max": int(max(rejected_totals))},
            "batch_max": {
                "mean": float(np.mean(batch_maxes)),
                "p50": float(np.percentile(batch_maxes, 50)),
                "p95": float(np.percentile(batch_maxes, 95)),
                "p99": float(np.percentile(batch_maxes, 99)),
                "max": int(max(batch_maxes))},
            "retention_by_max_len": {
                str(thr): float((np.array(batch_maxes) <= thr).mean())
                for thr in [512, 640, 768, 896, 1024, 1152, 1280, 1536]},
        }
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n→ 写入 {out_json}")


if __name__ == "__main__":
    main()