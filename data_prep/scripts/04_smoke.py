# -*- coding: utf-8 -*-
"""
04_smoke.py
===========
冒烟测试集：从 train_pool.jsonl 每 task 抽少量样本（用于 10-20 分钟内
验证训练/评估全流程代码正确性，与正式训练同一模型同一套代码，只改数据量）。

用法：python 04_smoke.py
输出：output/smoke.jsonl
"""
import json
import random
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
PER_TASK = 100      # 每 task 最多抽 100 条
SRC = "output/train_pool.jsonl"
OUT = "output/smoke.jsonl"

rng = random.Random(SEED)

by_task = defaultdict(list)
with open(SRC, encoding="utf-8") as f:
    for line in f:
        by_task[json.loads(line)["task"]].append(line.strip())

out_lines = []
for t, lines in by_task.items():
    out_lines.extend(rng.sample(lines, min(PER_TASK, len(lines))))

rng.shuffle(out_lines)
with open(OUT, "w", encoding="utf-8") as f:
    for line in out_lines:
        f.write(line + "\n")

print(f"smoke.jsonl: {len(out_lines)} 条, {len(by_task)} 个 task")
