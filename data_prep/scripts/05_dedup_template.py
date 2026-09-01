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

# ============================================================================
# 模板骨架正则 —— 与 stage1/03_seq_prepare.py 使用的"非标准闭合标签"严格耦合
# ============================================================================
#
# 为什么这样写：03_seq_prepare.py 给序列加的包装是 `<rna>...<rna>`、
# `<dna>...<dna>`、`<protein>...<protein>` —— **没有斜杠**，不是标准 XML。
#
# 常见误判一：把 `<(dna|rna|protein)>` 当通用标签——错，会误吞普通 XML；
#            [A-Za-z]+ 强制要求标签里包了"序列字符"，所以是 `<dna>ACGT</dna>` 这种;
#            不会去吞 `<someml>` 这种纯标签。
# 常见误判二：写成 `<(?:dna|rna|protein)>.*?</(?:dna|rna|protein)>`（带斜杠）——
#            真数据里没有斜杠，会一个都匹配不到，全部进 NO_TAG，cap 失效。
#
# 性能：re.compile 一次，模块级缓存，10 万次 sub 也就 ~0.5s。
SEQ_RE = re.compile(r"<(?:dna|rna|protein)>[A-Za-z]+<(?:dna|rna|protein)>")


def skeleton(s: str) -> str:
    """提取模板骨架：把序列段替换为哨兵 <SEQ>，剩下的就是"问题句式模板"。

    --------------------------------------------------------------------------
    输入：原始 input 字符串（已经是 templated 后的句子，如
        "The <rna>ACGU...</rna> sequence is most consistent with the
         <dna>ACGT...</dna> family classification.")
    输出：模板骨架，如
        "The <SEQ> sequence is most consistent with the <SEQ> family classification."
         或  "NO_TAG"（若没有任何可识别的标签）。

    为什么 "NO_TAG" 是字符串而不是 None：
        - 后面按 sk 做分组（defaultdict 字典 key），用 None 当 key 会让 hash
          与字符串类混乱；
        - "NO_TAG" 是一个显眼的字符串，调试时打印就能看出"本任务没标签"。
    """
    out = SEQ_RE.sub("<SEQ>", s)
    # 没有发生任何替换意味着没有匹配 —— 返回 "NO_TAG" 让后面走"不裁剪"分支。
    # 注意：用户 input 里如果字面出现 "<SEQ>" 这个词字面量，会被误判为已挖；
    # 这是已知 corner case，靠 03_seq_prepare 不写 "序列字符 + 斜杠" 来回避。
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

    # ============================================================================
    # Step 1：完全去重
    # ============================================================================
    # key 选择：(task, input, output) 三元组 —— **故意不**包 `label`。
    #
    # 为什么选这三元组而不是更细的 hash：
    # - 不同 label（如 positive / negative）但 input/output 完全相同 —— 仍应
    #   被视为重复：对监督学习没增量信息（同一 prompt，同一答案）。
    # - 若按 (task, input, output, label) 去重，同一 input/output 下出现两种
    #   label 的"边界样本"会被全保留，但训练时模型看到的就是同一段 prompt
    #   配两种答案 → 损失震荡。**故意把 label 排除掉**。
    #
    # "截断 key / hash" 这种做法不行：如果用 md5(input[:100])，两条 prompt
    # 前 100 字符相同但长度悬殊仍会被判重复 → 误删源数据的 1.9% 重复是已确认
    # 的事实，跟 01_split_stage2 的"全精确匹配"口径保持一致更安全。
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

    # ============================================================================
    # Step 2：模板感知均衡采样 —— 自定义规则最密集的一段
    # ============================================================================
    #
    # 目标：对每个 task，把"同一个模板骨架下的样本数"截断到一个 cap 上，
    #       防止模型在头部模板上反复刷监督信号。
    #
    # 先按 task 切桶，避免跨任务混算（不同 task 的模板空间独立）。
    by_task = defaultdict(list)
    for d in dedup:
        by_task[d["task"]].append(d)

    cleaned = []
    cap_stats = []
    for t, lst in by_task.items():
        # ------------------------------------------------------------
        # 2a) 模板分组：每 task 内按 skeleton(input) 聚合。
        # ------------------------------------------------------------
        # 注意：分的是 *input* 而不是 output —— 不同任务的 output 模式差异大
        # （如有的就是 "positive" / "negative"，有的长答案），按 output 分会
        # 把"提问方式不同、答案相同"的样本错聚；按 input 分更贴近"模板同质化"
        # 的实际语义。
        groups = defaultdict(list)
        for d in lst:
            groups[skeleton(d["input"])].append(d)

        num_templates = len(groups)

        # ------------------------------------------------------------
        # 2b) 每模板 cap 的计算 —— 这是整段的自定义核心。
        # ------------------------------------------------------------
        # avg = 任务样本数 / 任务模板数 ≈ "平均每个模板有多少样本"。
        # cap = max(50, ceil(avg × mult))：
        #   - 下限 50：再少就失去"模板学习意义"了，模型连一个完整的
        #     few-shot 上下文覆盖都学不到；
        #   - 上限由 mult 控制：mult=1.0 表示"每模板至少留平均值的 1 倍"，
        #     意味着大模板会被显著裁、小模板不动；
        #   - mult 调大 → 等价于基本不裁（少数超大模板被 cap 住）；
        #   - mult 调小 → 所有模板被裁到只剩 50 条的下限（激进净化）。
        # 注意 ceil 不是 floor —— 51.000001 也算 52；下行口径要避免"ceil(49.9)=50"
        # 触发的"刚好等于 cap"误判；math.ceil 安全。
        avg = len(lst) / num_templates
        cap = max(50, math.ceil(avg * args.template_mult))

        # ------------------------------------------------------------
        # 2c) 每模板打乱后取前 cap 条。
        # ------------------------------------------------------------
        # NO_TAG 兜底：这条记录没有可识别的标签式序列，skeleton 返回
        # "NO_TAG"，全 task 共用同一组；直接全保留 —— 否则 "无可归类"
        # 那一桶会被裁掉。
        #
        # len(items) <= cap 的小模板：原样保留，不打乱（不打乱也无所谓是
        # 因为 shuffle 在大模板里才有意义；这里没必要）。
        keep = 0
        for sk, items in groups.items():
            if sk == "NO_TAG" or len(items) <= cap:
                cleaned.extend(items)          # 无标签/未超 cap 全保留
                keep += len(items)
            else:
                rng.shuffle(items)             # 大模板：随机抽 cap 条
                cleaned.extend(items[:cap])
                keep += cap
        cap_stats.append((t, len(lst), num_templates, avg, cap, keep))
        print(f"    [{t}] 样本 {len(lst):>6} | 模板 {num_templates:>3} | "
              f"avg {avg:>5.0f} | cap {cap:>4} | 保留 {keep:>6}")

    # ============================================================================
    # Step 3：合并 stage3.jsonl（GPT-4o-mini 精修的长答案补充）
    # ============================================================================
    # stage3 是用大模型给同一批 (input, label) 重写过的 *长* `<reason>+<ans>`
    # 回答，用来增强 SFT 时学到的"长答案覆盖"。
    # 这里直接 append 不查重（重复是设计内 —— 大模型给出不同表述同样有价值，
    # 训练时同一个 prompt 配多种风格答案反而提升泛化）。
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

    # ============================================================================
    # Step 4：落盘
    # ============================================================================
    # 整体再次 shuffle —— 防止 "前半全是 train_pool 净化 / 后半全是 stage3"，
    # 影响下游 trainer 的 prefetch 性能以及 batch 内任务均衡。
    rng.shuffle(cleaned)
    with open(f"{args.data_dir}/{args.out_file}", "w", encoding="utf-8") as f:
        for d in cleaned:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[05] -> {args.out_file}: {len(cleaned)} 条")

    # ============================================================================
    # 净化效果实证指标：Top5 模板占比对比
    # ============================================================================
    #
    # 选 Top5 而非全分布的理由：
    #   - 全分布分散，差异小到淹没在打印里；
    #   - 头部 5 个模板通常吃掉 40~60% 的样本，这个比值最能反映"信息密度"
    #     是否提高（净化后这个比值应该下降）。
    #
    # before = 净化前（仅 Step 1 去重后），after = 净化后（再 + 模板 cap 后）。
    # 数学上 after <= before，且差距越大说明 cap 越有效。
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
