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

# --------------------------------------------------------------------------
# 配额常量 —— 每 task 上下限（all use =min(cap, max(min, ceil(n*ratio)), n)）
# --------------------------------------------------------------------------
# EVAL_PER_TASK_CAP = 500  : 评估集每 task 最多 500 条（足够指标稳定）
# EVAL_PER_TASK_MIN = 50    : 小 task（< 1000 条）也至少分 50 条作评估，
#                            否则样本太分散、置信区间爆炸
# EVAL_RATIO = 0.05         : 默认 5% 进 eval；小 task 由 MIN 兜底
# DPO_PER_TASK_CAP = 5000   : DPO 偏好构造单 task 上限（超过会拖慢 build_preference）
# DPO_RATIO = 0.10          : 默认 10% 走 DPO
# TRAIN_PER_TASK_CAP = 20000: 单 task 训练数据上限（防止头重脚轻）
# TRAIN_TOTAL_TARGET = 300000 : 三路配额按此总目标做一次缩放（防止小 task 被
#                              吃太多）
# TRAIN_PER_TASK_FLOOR = 2000 : 缩放后小 task 仍保留至少 2000 条（任务覆盖）
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

# ============================================================================
# 自定义规则 1：每 task 三路配额计算（保证 e+d+tr ≤ n）
# ============================================================================
# 思路：
#   eval_quota = min(MAX, max(MIN, ceil(n*EVAL_RATIO)), n)
#            ↓           ↓               ↓
#           500        ≥50           5% 兜底，若 n<1000 时用 MIN
#                                且不能超过总条数
#
#   dpo_quota = min(5000, ceil(n*0.10), n - eval)
#             ↑            ↑           ↑
#            cap        10%        不能再吃 eval 的
#
#   train_quota = min(20000, n - eval - dpo)
#                ↑              ↑
#               cap          eval + dpo 都不能再吃
#
# 关键不变量：
#   eval + dpo + train ≤ n
#   eval 不被 dpo 吃，DPO 不被 train 吃 —— 三路严格不重叠的代数前提。
# 小 task (n<1000)：eval 拿 50（MIN），剩 n-50 给 dpo + train；通常 dpo 拿个位数，
# train 拿绝大多数。
eval_quota, dpo_quota, train_quota = {}, {}, {}
for t, n in counts.items():
    # 配额必须 <= 条数，且三路之和 <= n（小 task 评估优先）
    eval_quota[t] = min(EVAL_PER_TASK_CAP, max(EVAL_PER_TASK_MIN, math.ceil(n * EVAL_RATIO)), n)
    dpo_quota[t] = min(DPO_PER_TASK_CAP, math.ceil(n * DPO_RATIO), n - eval_quota[t])
    train_quota[t] = min(TRAIN_PER_TASK_CAP, n - eval_quota[t] - dpo_quota[t])

# --------------------------------------------------------------------------
# 总训练配额超目标时，二次缩放 + 小 task 保底 + 反向削减
# --------------------------------------------------------------------------
# 触发条件：sum(train_quota.values()) > 300000（说明头 task 吃太多）。
# 两阶段修正：
#   1) 等比缩放：scale = 300000 / total，使总和为 300000；
#      但 floor_t 保底（默认 2000）防止小 task 被截到接近 0 —— "全任务覆盖"。
#   2) 反向削减：等比缩放后总和可能仍超 300000（floor 抬高了下界），
#      此时按 train_quota[t] 倒序，逐 task 砍到 floor_t 为止，直到总和达标。
# 这套是"总额约束 + 单 task 上下界"标准回退算法。
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

# ============================================================================
# 自定义规则 2：Pass 2b —— 每 task 行号打乱 + 按配额精确切分到三路
# ============================================================================
# 算法：每 task 把行号列表 rng.shuffle()（固定种子，可复现），然后
#   eval  = lines[:e]                  前 e 个
#   dpo   = lines[e:e+d]               中间 d 个
#   train = lines[e+d:e+d+tr]          尾部 tr 个
#
# 关键性质：
#   - **切片是按 [0:e), [e:e+d), [e+d:e+d+tr) 顺序**，与配额等式 e+d+tr ≤ len
#     配套，三段不重叠；
#   - train 取尾部而非头部 —— 因为头部已被 eval 拿走，**故意不让 train 吃
#     评估样本**；以保证 eval 永远不进入 SFT 训练。
#   - rng.shuffle 保证行号均匀分布，不偏向源数据顺序。
#
# 最后用 set 保存行号是为了 Pass 2c 的 O(1) `idx in train_set` 查询
# （30 万行 × 三次 in 检查 ≈ 90 万次操作，set 远快于 list）。
train_set, dpo_set, eval_set = set(), set(), set()
for t, lines in task_lines.items():
    rng.shuffle(lines)
    e, d, tr = eval_quota[t], dpo_quota[t], train_quota[t]
    assert e + d + tr <= len(lines), f"{t}: 配额超过条数"
    eval_set.update(lines[:e])
    dpo_set.update(lines[e:e + d])
    train_set.update(lines[e + d:e + d + tr])
print(f"  切分完成: train={len(train_set)}, dpo={len(dpo_set)}, eval={len(eval_set)}")

# ============================================================================
# Pass 2c：按行号分发到三路写出
# ============================================================================
# 这里不复用 Pass 2a 读过的 JSON 对象 —— 直接按源文件原始文本 forward 到对应
# 输出文件，避免二次 json.dumps/dumps 的格式漂移（如 ensure_ascii）。
# 优先级 train > dpo > eval（互斥三选一，写多路时 train 优先），但因为行号
# 集合互不相交，顺序只影响写盘的写入次序，不影响内容。
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

# ============================================================================
# 自定义规则 3：重叠检查 —— 完整 input 精确匹配（不是 truncated hash）
# ============================================================================
# exact_keys 用 (task + "|" + input + "|" + label) 作为"完整精确匹配键"。
#
# 为什么是完整精确匹配而不是 md5 / 截断 hash：
#   - 源数据本身存在 ~1.9% 重复样本（同 task、同 input、同 label 多次出现）；
#     这些是数据源副本，去重应该交给 Step 1，否则 SFT 多次看到同一样本就是
#     **虚假 epoch 进度**。
#   - 如果用 md5(input[:200]) 这种截断哈希，长样本前半相同但后半不同（如
#     长 RNA 序列后段有差异）会被误判为重复 → **误报**；
#   - source 数据中常见 input 比 200 字符长得多，截断损失大。
#
# 三路重叠（应有严格为 0）：
#   train_x_dpo  : SFT 训练看到 DPO preference 候选 → 数据泄漏
#   train_x_eval : SFT 训练看到 eval 答案 → 指标虚高
#   dpo_x_eval   : DPO 偏好构造时拿到 eval 答案 → SFT 看不到但 DPO 看到
#   三者任意 > 0 都需重跑。
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
    """完整精确匹配的 task|input|label 拼接键。
    用 '|' 分隔（不可能出现在 input 里 —— input 是生物学序列+问题句式），
    保证三段无歧义。"""
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
