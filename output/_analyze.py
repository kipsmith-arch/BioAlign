# -*- coding: utf-8 -*-
"""分析 4 个 eval jsonl 的输出覆盖率和简单预测对比"""
import json, re, collections, math

FILES = ['eval_base.jsonl','eval_s2_only.jsonl','eval_s1_s2.jsonl','eval_stage3.jsonl']

# 关键字 / 数字匹配
YES_NO_RE  = re.compile(r"\b(yes|no|positive|negative|true|false|1|0)\b", re.I)
NUM_RE     = re.compile(r"-?\d+(?:\.\d+)?")
PCT_RE     = re.compile(r"-?\d+(?:\.\d+)?\s*%")

def classify_output(out: str) -> str:
    o = out.strip()
    if not o:
        return 'empty'
    lo = o.lower()
    # 提取出现的第一个 yes/no/positive/negative
    pos = lo.find('positive')
    neg = lo.find('negative')
    yes = lo.find(' yes')
    no  = lo.find(' no')
    # 头几个词作为快速判断
    head = lo[:60]
    if 'positive' in head or 'negative' in head or 'yes,' in head or 'no,' in head or re.match(r'\s*(yes|no)\b', head):
        return 'has_label_word'
    if pos != -1 or neg != -1 or yes != -1 or no != -1:
        return 'has_label_word'
    # 看是否有数字
    nums = NUM_RE.findall(o)
    if nums:
        return f'has_numbers(n={len(nums)})'
    return 'free_text'

def safe_label_parse(s):
    """对每个任务的 label 做规范化，返回 (kind, ref_value)
       kind: 'class' (positive/negative) | 'num' (float) | 'dict' | 'other'
    """
    if isinstance(s, dict):
        return ('dict', s)
    if isinstance(s, str) and s.lower() in ('positive','negative'):
        return ('class', s.lower())
    if isinstance(s, (int,float)):
        return ('num', float(s))
    # 试转 float
    try:
        return ('num', float(s))
    except Exception:
        return ('other', s)

# 任务 -> (kind)
TASK_KIND = {}
rows_ref = [json.loads(l) for l in open('eval_base.jsonl', encoding='utf-8')]
for r in rows_ref:
    kind, val = safe_label_parse(r['label'])
    TASK_KIND[r['task']] = kind

print('Task kinds:', collections.Counter(TASK_KIND.values()))
print()

# 统计每个任务在 4 个模型上输出类型
print(f"{'task':<45} {'kind':<6} {'base':<22} {'s2_only':<22} {'s1_s2':<22} {'stage3':<22}")
all_tasks = list(TASK_KIND.keys())
for task in all_tasks:
    kind = TASK_KIND[task]
    counts = []
    for fp in FILES:
        rows = [json.loads(l) for l in open(fp, encoding='utf-8') if l.strip()]
        c = collections.Counter()
        for r in rows:
            if r['task'] != task:
                continue
            c[classify_output(r['model_output'])] += 1
        total = sum(c.values()) or 1
        # 简化: 给出 has_label_word/number/free_text 占比
        def pct(k):
            v = c.get(k,0)
            return f"{k}={v/total*100:.0f}%"
        s = f"lbl={c.get('has_label_word',0)/total*100:.0f}% num={c.get('has_numbers(n=1)',0)/total*100:.0f}% txt={c.get('free_text',0)/total*100:.0f}%"
        counts.append(s)
    print(f"{task:<45} {kind:<6} {counts[0]:<22} {counts[1]:<22} {counts[2]:<22} {counts[3]:<22}")
