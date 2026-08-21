# -*- coding: utf-8 -*-
"""对每个模型输出做简单解析：分类任务 -> positive/negative; 数字任务 -> 第一个数字.
   计算分类准确率、spearman（如果 label 是数字）、overall coverage.
"""
import json, re, math, collections, statistics

FILES = ['eval_base.jsonl','eval_s2_only.jsonl','eval_s1_s2.jsonl','eval_stage3.jsonl']

# 把每个任务的 label kind 记录下来
def label_kind(label):
    if isinstance(label, dict):
        return 'dict'
    if isinstance(label, str) and label.lower() in ('positive','negative'):
        return 'class'
    try:
        float(label); return 'num'
    except Exception:
        return 'other'

# 从输出里提取分类
def parse_class(out: str):
    lo = out.lower()
    # 优先匹配行首的 "Yes." / "No." / "Positive." / "Negative."
    m = re.search(r"\b(yes|no|positive|negative|true|false)\b", lo[:80])
    if m:
        return m.group(1)
    # 再在全文搜索, 取第一个
    m = re.search(r"\b(yes|no|positive|negative|true|false)\b", lo)
    return m.group(1) if m else None

def parse_num(out: str):
    # 取文本中第一个合理的数字 (允许负数和小数)
    m = re.search(r"-?\d+(?:\.\d+)?", out)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

# 计算分类准确率
def cls_acc(preds, golds):
    correct = 0; total = 0
    for p,g in zip(preds,golds):
        if p is None or g is None: continue
        p2 = 'positive' if p in ('yes','positive','true') else 'negative' if p in ('no','negative','false') else None
        g2 = 'positive' if g in ('yes','positive','true') else 'negative' if g in ('no','negative','false') else None
        if p2 and g2:
            total += 1
            if p2 == g2:
                correct += 1
    return (correct, total)

# 计算 pearson (近似 spearman -> 用值相关)
def pearson(xs, ys):
    n = len(xs)
    if n < 2: return None
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    if dx == 0 or dy == 0: return None
    return num/(dx*dy)

# 收集各任务 label
def collect(fp):
    rows = [json.loads(l) for l in open(fp, encoding='utf-8')]
    bytask = collections.defaultdict(list)
    for r in rows:
        bytask[r['task']].append((r['model_output'], r['label']))
    return bytask

# 任务分组
TASK_NAMES = []
for fp in FILES:
    for r in (json.loads(l) for l in open(fp, encoding='utf-8')):
        if r['task'] not in TASK_NAMES:
            TASK_NAMES.append(r['task'])

def main():
    print(f"\n{'task':<45} {'kind':<5}", end='')
    for f in FILES:
        print(f" {f.replace('eval_','').replace('.jsonl',''):>10}", end='')
    print()
    print('-'*120)

    # 每任务的指标
    per_task_score = collections.defaultdict(dict)
    for fp in FILES:
        bt = collect(fp)
        name = fp.replace('eval_','').replace('.jsonl','')
        for t, items in bt.items():
            kind = label_kind(items[0][1])
            n_parsed = 0; cls_correct = 0; xs=[]; ys=[]
            for out, lab in items:
                if kind == 'class':
                    p = parse_class(out); g = lab.lower()
                    if p:
                        n_parsed += 1
                        p2 = 'positive' if p in ('yes','positive','true') else 'negative'
                        if p2 == g: cls_correct += 1
                elif kind == 'num':
                    p = parse_num(out)
                    try:
                        gv = float(lab)
                    except Exception:
                        continue
                    if p is not None:
                        n_parsed += 1
                        xs.append(p); ys.append(gv)
                else:
                    # 暂不解析
                    pass
            coverage = n_parsed / max(len(items),1)
            if kind == 'class':
                acc = cls_correct / max(n_parsed,1)
                score = acc * 100
            elif kind == 'num':
                score = pearson(xs, ys)
                score = score * 100 if score is not None else None
            else:
                score = None
            per_task_score[t][name] = (kind, coverage, score, n_parsed, len(items))

    for t in TASK_NAMES:
        kinds = set(per_task_score[t][f.replace('eval_','').replace('.jsonl','')][0] for f in FILES)
        kind = list(kinds)[0] if kinds else '?'
        print(f"{t:<45} {kind:<5}", end='')
        for f in FILES:
            name = f.replace('eval_','').replace('.jsonl','')
            kind, cov, score, np_, tot = per_task_score[t][name]
            if score is None:
                s = '-'
            elif kind == 'class':
                s = f"{score:>5.1f}% ({cov*100:.0f}%)"
            else:
                if math.isnan(score): s = 'nan'
                else: s = f"{score:>6.2f} ({cov*100:.0f}%)"
            print(f" {s:>10}", end='')
        print()

    # 汇总
    print('\n--- summary per model (overall coverage by kind) ---')
    summary = {f.replace('eval_','').replace('.jsonl',''): collections.Counter() for f in FILES}
    for t, d in per_task_score.items():
        for name, (kind, cov, score, np_, tot) in d.items():
            summary[name][f'{kind}_parsed'] += np_
            summary[name][f'{kind}_total'] += tot
    for name, c in summary.items():
        msg = []
        for kind in ['class','num','dict','other']:
            parsed = c[f'{kind}_parsed']; total = c[f'{kind}_total']
            if total > 0:
                msg.append(f"{kind}={parsed}/{total} ({parsed/total*100:.1f}%)")
        print(f"  {name:<12}", "  ".join(msg))

if __name__ == '__main__':
    main()
