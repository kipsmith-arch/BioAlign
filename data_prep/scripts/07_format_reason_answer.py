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

# 多分类标签集合（按长度倒序，避免子串误匹配）
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
    """判断 label 的类型"""
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
    """把训练数据原始 output 压缩成 1~3 句 reason。
    策略：保留第一个句号前的内容 + 第一个句号后的第一句。"""
    if not original_output:
        return ''
    text = original_output.strip()
    # 切句
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
    """根据 label 类型返回 ANS 内的字符串"""
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
    """转换一条训练样本"""
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
