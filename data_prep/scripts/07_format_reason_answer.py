# -*- coding: utf-8 -*-
"""
07_format_reason_answer.py
===========================
把训练数据的 `output` 字段从"自然语言描述"改成 "<REASON>...</REASON><ANS>label</ANS>" 结构化格式。

设计动机：
- 原训练数据 output 是英文自然语言描述（如 "The interaction is not predicted..."），
  eval parser 找 positive/negative/数字关键词时覆盖率只有 7.5% (base) ~ 100% (stage3)。
- 改成结构化格式后，模型学会输出 label 字符串，eval parser 可以 100% 提取。

格式:
    <reason>
    {原 output 的精简版本（保留 1-3 句解释）}
    </reason>
    <ans>
    {label}
    </ans>

label 类型映射：
- classification: label ∈ {"positive", "negative"} → 直接写入
- multi-class: 如 "IRES", "EC2.4.1.-", "m6A" → 直接写入
- regression: float → 保留 2~6 位小数
- multi-value regression (dict): 如 {"hk": ..., "dev": ...} → "hk=X, dev=Y"

输入：input_data/{train_pool_clean.jsonl, dpo_source.jsonl, stage3.jsonl}
输出：input_data/{train_pool_clean.jsonl, dpo_source.jsonl, stage3.jsonl}
       （原地覆盖，保留原 input/label/task 字段）

用法：python data_prep/scripts/07_format_reason_answer.py
"""
import json
import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

INPUT_DIR = Path('../../input_data')

# ============================================================================
# 多分类标签集合（按长度倒序）—— 避免子串误匹配
# ============================================================================
# 为什么倒序：
#   RNA_CLASSES 里 '5S_rRNA' 是 '5_8S_rRNA' 的子串。
#   如果用 in / re.search 不加边界，会把 '5_8S_rRNA' 误判为 '5S_rRNA'。
#   sorted(..., key=len, reverse=True) 后**检查顺序**先长后短：
#   '5_8S_rRNA' (8字符) 先匹配，命中即返回，'5S_rRNA' 永远不会被误匹配。
#
# 其它 label_kind（如 mod）按相同思路保证 'm6Am' / 'm6A' 等不互相误匹配。
#
# 数据耦合：这些集合与 eval/parser_v2.py 的 RNA_CLASSES / MODIFICATION_CLASSES
# 严格同步；改这边时要同步另一边。
RNA_CLASSES = sorted([
    '5S_rRNA', '5_8S_rRNA', 'tRNA', 'ribozyme', 'CD-box', 'miRNA',
    'Intron_gpI', 'Intron_gpII', 'HACA-box', 'riboswitch', 'IRES',
    'leader', 'scaRNA',
], key=len, reverse=True)
MOD_CLASSES = sorted([
    'm6Am', 'm1A', 'm5C', 'm5U', 'm6A', 'm7G', 'AtoI', 'Psi',
    'Am', 'Cm', 'Gm', 'Um', 'none',
], key=len, reverse=True)

# 回归任务的合理取值范围
REGRESSION_RANGES = {
    'Thermostability-Thermostability': (30, 80),
    'Stability-Stability': (-3, 3),
    'Fluorescence-Fluorescence': (0, 5),
    'sirnaEfficiency-sirnaEfficiency': (0, 150),
    'Isoform-Isoform': (0, 1),
    'MeanRibosomeLoading-MeanRibosomeLoading': (0, 10),
    'CRISPROnTarget-CRISPROnTarget': (0, 1),
}


def label_kind(label, task):
    """label 类型分发 —— 整段格式转换的 dispatch 中心。

    返回以下 kind：
      'dict'        多值回归（label 是 dict，如 {"hk": ..., "dev": ...}）
      'class'       二分类字符串 positive/negative
      'rna_family'  RNA 多分类标签（CD-box, IRES, leader 等）
      'mod'         单个 RNA modification（m6A, m5C, ...)
      'ec'          EC 号（"EC" 前缀 + 4 段数字 + 可选 '-' 占位）
      'mod_multi'   多个 modification 用逗号分隔
      'num'         数值（int / float 可转）
      'other_str'   其它字符串（目前未用，保留）
      'other'       兜底

    EC 正则 `^EC\\d+\\.\\d+\\.\\d+\\.\\-?\\d*$` 设计点：
      - 必须以 "EC" 开头；
      - 3 个 "\." 分隔的数字段；
      - 第 4 段 "\-?\d*"：可空可负（占位号是 "-" 如 EC2.4.1.-）。
    """
    if isinstance(label, dict):
        return 'dict'
    if isinstance(label, str):
        ls = label.strip()
        if ls.lower() in ('positive', 'negative'):
            return 'class'
        if ls in RNA_CLASSES:
            return 'rna_family'
        if ls in MOD_CLASSES:
            return 'mod'
        if re.match(r'^EC\d+\.\d+\.\d+\.\-?\d*$', ls):
            return 'ec'
        # 含逗号的多标签 modification
        if ',' in ls and any(m in ls for m in MOD_CLASSES):
            return 'mod_multi'
        return 'other_str'
    try:
        float(label)
        return 'num'
    except Exception:
        return 'other'


def compress_reason(original_output: str, max_sentences: int = 2) -> str:
    """压缩原始自然语言 output 为 1~max_sentences 句。

    切句正则 `r'(?<=[.!?])\\s+'`：用 **lookbehind**（向后视），匹配"在
    句末标点之后的空白处"切句。
    - 用 lookbehind 而非简单 split：避免标点被切走（"Hello." 切成 ["Hello"]
      而不是 ["Hello", "."]）；
    - 不切中文句号"。" —— 数据是英文。

    corner case：
    - \"\" / None → 返回 ''；
    - 无标点的长字符串 → 整个作为 1 句返回（即使 > 200 字符）；
    - max_sentences 默认 2 是行为上限，调大可放更长；2 是当前配置。
    """
    if not original_output:
        return ''
    text = original_output.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    keep = []
    for s in sentences:
        if not s.strip():
            continue
        keep.append(s.strip())
        if len(keep) >= max_sentences:
            break
    return ' '.join(keep)


def format_answer(label, task) -> str:
    """根据 label 类型返回 <ans> 标签内的字符串。

    数值分支（'num'）的关键设计：用 `f'{v:g}'` —— Python 自动选"最短表示"
    保留有效数字同时去掉无意义的尾零：
      - 0.1200000001 → 0.12（不会出现 0.1200000001 这种浮点噪声进 prompt）
      - 51.09       → 51.09（精度保留）
      - 1000000.0   → 1e+06（科学计数法兜底）
      - 0.0         → 0
      - 整数且 |v| 极小 < 1e-3 或 >= 1：直接 str 出来（如 51、-0.5）
      - 整数但落入 (1e-3, 1) 区间（如 0.5 是 1/2）：保留 2 位小数避免信息损失
        （防御性，多数情况下走 :g 已足够）

    多值回归（'dict'）分支：每项独立格式化用同一种 :g，
    保证 'hk=0.12, dev=0.34' 风格统一，与 stage2/fast_infer 输出格式对齐。
    """
    kind = label_kind(label, task)
    if kind == 'class':
        return label.strip().lower()
    if kind in ('rna_family', 'mod', 'ec', 'other_str'):
        return label.strip()
    if kind == 'mod_multi':
        # 多标签 modification: 按规范化逗号分隔
        return label.strip()
    if kind == 'num':
        try:
            v = float(label)
            # 保留 2-6 位有效数字，避免精度损失
            if v == int(v) and abs(v) < 1e6:
                return str(v) if 0 < abs(v) < 1e-3 or abs(v) >= 1 else f'{v:.2f}'
            return f'{v:g}'  # 自动选最短表示
        except Exception:
            return str(label)
    if kind == 'dict':
        # 多值回归: "hk=0.12, dev=-0.34"
        parts = []
        for k, v in label.items():
            try:
                vf = float(v)
                parts.append(f'{k}={vf:g}')
            except Exception:
                parts.append(f'{k}={v}')
        return ', '.join(parts)
    return str(label)


def convert_record(r: dict) -> dict:
    """转换单条训练样本：把 output 字段从自然语言描述换成结构化。

    关键设计：
    - 用 `{**r, 'output': new_output}` 而非显式列字段名：
      未来新增字段（如 'metadata'、'source_url'）不需要改本函数，避免后
      人加了字段被这里默默丢。
    - 输出格式里换行符 `\n` 必须保留：parser_v2 的 `extract_ans_block`
      用 `<ans>\s*(.+?)\s*</ans>` 的 DOTALL 模式依赖换行对齐。
    """
    task = r['task']
    label = r['label']
    original_output = r.get('output', '')
    reason = compress_reason(original_output, max_sentences=2)
    ans = format_answer(label, task)
    new_output = (
        f'<reason>\n{reason}\n</reason>\n'
        f'<ans>\n{ans}\n</ans>'
    )
    return {**r, 'output': new_output}


def main():
    targets = ['train_pool_clean.jsonl', 'dpo_source.jsonl', 'stage3.jsonl']
    stats_total = {}
    for fname in targets:
        fp = INPUT_DIR / fname
        if not fp.exists():
            print(f'  [skip] {fp} not found')
            continue
        print(f'[{fname}] processing...')
        n = 0
        n_changed = 0
        out_lines = []
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                new_r = convert_record(r)
                if new_r['output'] != r.get('output', ''):
                    n_changed += 1
                out_lines.append(json.dumps(new_r, ensure_ascii=False))
                n += 1
        # 备份原文件
        bak = fp.with_suffix('.jsonl.bak')
        if not bak.exists():
            fp.replace(bak)
            print(f'  backup -> {bak}')
            with open(fp, 'w', encoding='utf-8') as f:
                f.write('\n'.join(out_lines) + '\n')
        else:
            # 已有 backup，直接覆盖
            with open(fp, 'w', encoding='utf-8') as f:
                f.write('\n'.join(out_lines) + '\n')
        stats_total[fname] = (n, n_changed)
        print(f'  -> {n} rows, {n_changed} reformatted')

    print('\n=== summary ===')
    for k, (n, c) in stats_total.items():
        print(f'  {k}: {n} rows, {c} reformatted ({c/n*100:.1f}%)')

    # 显示一些样本
    print('\n=== samples (first 3 of train_pool_clean.jsonl) ===')
    with open(INPUT_DIR / 'train_pool_clean.jsonl', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            r = json.loads(line)
            print(f'\n[task: {r["task"]}] [label: {r["label"]}]')
            print(r['output'][:500])


if __name__ == '__main__':
    main()
