# -*- coding: utf-8 -*-
"""
02_stage3_convert.py
====================
将 stage3.xlsx（8002 条 GPT-4o-mini 精修的推理问答对）转为 jsonl。
stage3 数据可选并入 SFT 训练（增强推理长答案覆盖），chosen 也可用于 DPO 混合。

用法：python 02_stage3_convert.py
输出：output/stage3.jsonl
"""
import json
import sys
import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

SRC = "../dataset/stage3.xlsx"
OUT = "output/stage3.jsonl"

wb = openpyxl.load_workbook(SRC, read_only=True)
ws = wb["Sheet1"]  # 列: Unnamed:0, input, task, label, output
out = []
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[2] is None:  # input 为空则跳过
        continue
    out.append({"input": r[2], "task": r[3], "label": r[4], "output": r[5]})

with open(OUT, "w", encoding="utf-8") as f:
    for d in out:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

from collections import Counter
c = Counter(d["task"] for d in out)
print(f"stage3.jsonl: {len(out)} 条, {len(c)} 个 task")
print("task 分布(前 10):", c.most_common(10))
