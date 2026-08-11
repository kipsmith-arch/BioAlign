# -*- coding: utf-8 -*-
"""
05_dedup_template.py —— 训练集净化：完全去重 + 模板感知均衡采样 + 合并 stage3
===============================================================================
应对模板同质化（诊断：每 task 仅 50~200 个模板骨架，头部模板重复套用几十~
几百条序列，如 emp 的 Top1 模板 533 条）。

三步处理：
  1) 完全去重：task+input+output 完全相同 → 删除（源数据 ~1.9% 重复）
  2) 模板感知均衡采样：按 task 提取"模板骨架"（挖掉序列后的问题句式），
     对每个模板的样本数做 cap —— 限制单一模板重复，提高信息密度。
     骨架提取用正则匹配 stage2 的**非标准闭合标签**（<rna>...<rna>，无斜杠）：
       r'<(?:dna|rna|protein)>[A-Za-z]+<(?:dna|rna|protein)>'  →  '<SEQ>'
     无标签可挖的 input 归为 "NO_TAG" 类，不裁剪（无法聚类）。
  3) 合并 stage3.jsonl（8002 条 GPT-4o-mini 精修推理长答案，增强长答案覆盖）

cap 规则（每 task 内）：per_template_cap = max(50, ceil(task_n / template_n × mult))
  mult=1.0 表示"每模板至少留平均值的 1 倍"，可调。

用法：python 05_dedup_template.py --template_mult 1.0
输出：output/train_pool_clean.jsonl, output/clean_stats.json
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

SEQ_RE = re.compile(r"<(?:dna|rna|protein)>[A-Za-z]+<(?:dna|rna|protein)>")


def skeleton(s: str) -> str:
    """提取模板骨架：序列替换为 <SEQ>。无标签返回 'NO_TAG'。"""
    out = SEQ_RE.sub("<SEQ>", s)
    return out if out != s else "NO_TAG"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="output")
    parser.add_argument("--train_file", default="train_pool.jsonl")
    parser.add_argument("--stage3_file", default="stage3.jsonl")
    parser.add_argument("--out_file", default="train_pool_clean.jsonl")
    parser.add_argument("--template_mult", type=float, default=1.0,
                        help="每模板 cap = max(50, ceil(avg×mult))")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    rows = []
    with open(f"{args.data_dir}/{args.train_file}", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    n_raw = len(rows)
    print(f"[05] 读入 train_pool: {n_raw} 条")

    # --- Step 1: 完全去重 ---
    seen = set()
    dedup = []
    dup_n = 0
    for d in rows:
        key = (d["task"], d["input"], d["output"])
        if key in seen:
            dup_n += 1
            continue
        seen.add(key)
        dedup.append(d)
    print(f"[05] 完全去重: 删除 {dup_n} 条重复 ({dup_n/n_raw:.2%})")

    # --- Step 2: 模板感知均衡采样（每 task 内每模板 cap）---
    by_task = defaultdict(list)
    for d in dedup:
        by_task[d["task"]].append(d)

    cleaned = []
    cap_stats = []
    for t, lst in by_task.items():
        # 模板分组
        groups = defaultdict(list)
        for d in lst:
            groups[skeleton(d["input"])].append(d)
        num_templates = len(groups)
        avg = len(lst) / num_templates
        cap = max(50, math.ceil(avg * args.template_mult))
        # 每模板打乱后取前 cap 条
        keep = 0
        for sk, items in groups.items():
            if sk == "NO_TAG" or len(items) <= cap:
                cleaned.extend(items)          # 无标签/未超 cap 全保留
                keep += len(items)
            else:
                rng.shuffle(items)
                cleaned.extend(items[:cap])
                keep += cap
        cap_stats.append((t, len(lst), num_templates, avg, cap, keep))
        print(f"    [{t}] 样本 {len(lst):>6} | 模板 {num_templates:>3} | "
              f"avg {avg:>5.0f} | cap {cap:>4} | 保留 {keep:>6}")

    # --- Step 3: 合并 stage3 ---
    n_stage3 = 0
    try:
        with open(f"{args.data_dir}/{args.stage3_file}", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cleaned.append(json.loads(line))
                    n_stage3 += 1
        print(f"[05] 合并 stage3: +{n_stage3} 条")
    except FileNotFoundError:
        print("[05] 未找到 stage3.jsonl，跳过合并")

    rng.shuffle(cleaned)
    with open(f"{args.data_dir}/{args.out_file}", "w", encoding="utf-8") as f:
        for d in cleaned:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[05] -> {args.out_file}: {len(cleaned)} 条")

    # --- 统计：净化前后模板分布对比 ---
    def top5_ratio(lst):
        c = Counter(skeleton(d["input"]) for d in lst)
        return sum(v for _, v in c.most_common(5)) / sum(c.values())

    before = top5_ratio(dedup)
    after = top5_ratio(cleaned)
    print(f"[05] Top5 模板占比: 净化前(去重后) {before:.1%} -> 净化后 {after:.1%}")

    stats = {
        "raw": n_raw, "dedup_removed": dup_n, "stage3_added": n_stage3,
        "final": len(cleaned), "template_mult": args.template_mult,
        "top5_ratio_before": round(before, 4), "top5_ratio_after": round(after, 4),
        "per_task": [{"task": t, "raw": r, "templates": nt, "cap": c, "kept": k}
                     for t, r, nt, avg, c, k in cap_stats],
    }
    with open(f"{args.data_dir}/clean_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("[05] 统计已写入 clean_stats.json")


if __name__ == "__main__":
    main()
