# -*- coding: utf-8 -*-
"""
01_split_stage2.py (v2, deterministic)
======================================
将 stage2_train.jsonl（330 万条）按 task 分层划分为三路**严格不相交**的数据集：

  train_pool  -> SFT 训练（每 task cap，总目标 ~15 万条）
  dpo_source  -> DPO 偏好数据构造源（每 task 最多 5000 条）
  eval_set    -> 评估集（每 task 300~500 条，SFT/DPO 均不可见）

v2 变更（确定性抽样）：
- 不再用在线概率抽样，改为：收集每 task 的行号 -> 固定种子打乱 -> 前 N 行
  精确切分。三路数量严格等于配额，且互不重叠（行号唯一）。
- 重叠检查改用**完整 input 精确匹配**（源数据本身存在 ~1.9% 重复样本，
  用截断 key 会误报）。

用法：python 01_split_stage2.py
输出：output/train_pool.jsonl, output/dpo_source.jsonl, output/eval_set.jsonl, output/prep_stats.json
"""
import json
import math
import random
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
SRC = "../dataset/stage2_train.jsonl"

EVAL_PER_TASK_CAP = 500
EVAL_PER_TASK_MIN = 50
EVAL_RATIO = 0.05
DPO_PER_TASK_CAP = 5000
DPO_RATIO = 0.10
TRAIN_PER_TASK_CAP = 20000
TRAIN_TOTAL_TARGET = 300000
TRAIN_PER_TASK_FLOOR = 2000

rng = random.Random(SEED)

# ---------- Pass 1: 统计每 task 数量 ----------
print("[Pass 1] 统计每 task 数量 ...")
counts = defaultdict(int)
with open(SRC, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        counts[json.loads(line)["task"]] += 1
print(f"  共 {len(counts)} 个 task, 总条数 {sum(counts.values())}")

# ---------- 计算每 task 三路配额 ----------
eval_quota, dpo_quota, train_quota = {}, {}, {}
for t, n in counts.items():
    # 配额必须 <= 条数，且三路之和 <= n（小 task 评估优先）
    eval_quota[t] = min(EVAL_PER_TASK_CAP, max(EVAL_PER_TASK_MIN, math.ceil(n * EVAL_RATIO)), n)
    dpo_quota[t] = min(DPO_PER_TASK_CAP, math.ceil(n * DPO_RATIO), n - eval_quota[t])
    train_quota[t] = min(TRAIN_PER_TASK_CAP, n - eval_quota[t] - dpo_quota[t])

total_train = sum(train_quota.values())
if total_train > TRAIN_TOTAL_TARGET:
    scale = TRAIN_TOTAL_TARGET / total_train
    for t in train_quota:
        avail = counts[t] - eval_quota[t] - dpo_quota[t]
        floor_t = min(TRAIN_PER_TASK_FLOOR, avail)  # 小 task 保底不超过可用条数
        train_quota[t] = max(floor_t, math.floor(train_quota[t] * scale))
    deficit = sum(train_quota.values()) - TRAIN_TOTAL_TARGET
    for t in sorted(train_quota, key=lambda x: -train_quota[x]):
        if deficit <= 0:
            break
        avail = counts[t] - eval_quota[t] - dpo_quota[t]
        floor_t = min(TRAIN_PER_TASK_FLOOR, avail)
        cut = min(train_quota[t] - floor_t, deficit)
        train_quota[t] -= cut
        deficit -= cut

print(f"  训练配额合计 {sum(train_quota.values())}, "
      f"DPO 源配额 {sum(dpo_quota.values())}, 评估配额 {sum(eval_quota.values())}")

# ---------- Pass 2a: 收集每 task 的行号 ----------
print("[Pass 2a] 收集行号 ...")
task_lines = defaultdict(list)
with open(SRC, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        task_lines[json.loads(line)["task"]].append(idx)
print(f"  行号收集完成，共 {sum(len(v) for v in task_lines.values())} 行")

# ---------- Pass 2b: 每 task 打乱并精确切分 ----------
print("[Pass 2b] 打乱并切分 ...")
train_set, dpo_set, eval_set = set(), set(), set()
for t, lines in task_lines.items():
    rng.shuffle(lines)
    e, d, tr = eval_quota[t], dpo_quota[t], train_quota[t]
    assert e + d + tr <= len(lines), f"{t}: 配额超过条数"
    eval_set.update(lines[:e])
    dpo_set.update(lines[e:e + d])
    train_set.update(lines[e + d:e + d + tr])
print(f"  切分完成: train={len(train_set)}, dpo={len(dpo_set)}, eval={len(eval_set)}")

# ---------- Pass 2c: 重读源文件，按行号分发 ----------
print("[Pass 2c] 分发写入 ...")
out_paths = {
    "train": "output/train_pool.jsonl",
    "dpo": "output/dpo_source.jsonl",
    "eval": "output/eval_set.jsonl",
}
files = {k: open(p, "w", encoding="utf-8") for k, p in out_paths.items()}
with open(SRC, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue
        if idx in train_set:
            files["train"].write(line)
        elif idx in dpo_set:
            files["dpo"].write(line)
        elif idx in eval_set:
            files["eval"].write(line)
for fh in files.values():
    fh.close()

# ---------- 统计与重叠检查（完整 input 精确匹配） ----------
print("[统计] ...")
data = {}
for k, p in out_paths.items():
    lst = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            lst.append(json.loads(line))
    data[k] = lst
    print(f"  -> {p}: {len(lst)} 条, {len(set(d['task'] for d in lst))} 个 task")

def exact_keys(lst):
    return set(d["task"] + "|" + d["input"] + "|" + str(d["label"]) for d in lst)

overlap = {
    "train_x_dpo": len(exact_keys(data["train"]) & exact_keys(data["dpo"])),
    "train_x_eval": len(exact_keys(data["train"]) & exact_keys(data["eval"])),
    "dpo_x_eval": len(exact_keys(data["dpo"]) & exact_keys(data["eval"])),
}
print("三路精确重叠（应为 0）:", overlap)

stats = {
    "seed": SEED, "source": SRC, "overlap_exact": overlap,
    "source_total": sum(counts.values()), "source_tasks": len(counts),
}
for k in ("train", "dpo", "eval"):
    c = Counter(d["task"] for d in data[k])
    stats[k] = {"total": len(data[k]), "num_tasks": len(c), "per_task": dict(sorted(c.items()))}
with open("output/prep_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("统计已写入 output/prep_stats.json")
