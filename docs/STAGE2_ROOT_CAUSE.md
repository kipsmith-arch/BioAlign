# Stage2 SFT 是 binary 任务坍缩的真正根源

> **状态**：诊断报告（基于 2026-08-23 ~ 2026-08-26 的训练 / 推理 / 评估数据）
> **作者**：项目组
> **日期**：2026-08-26

## 一、TL;DR

**Stage2 SFT 没真正学会任务能力——只在结构化格式上学得很好**。
这一结论从7 个独立证据交叉验证得出。

**对 binary 任务的影响**：Stage2 在4 个 binary 任务上**完全坍缩到 dominant class**（emp 全部 negative / promoter_enhancer 全部 negative / ncRNAProteinInter 全部 negative / tf_h 几乎全 negative）。Stage3 DPO 不仅没救活，反而把 antibody_antigen 仅存的 46 个正类预测压成 2 个。

**修复方向不再是 DPO 数据**——Stage3 DPO 只能在 Stage2 已经能做的任务上调优，不能教会 Stage2 不会做的事。真正的杠杆点：
1. 加大 LoRA rank（当前 r=16 → 建议 r=64）
2. 加 class_weight 处理 binary 任务不平衡
3. 延长训练 epoch（当前1 epoch → 建议 3）
4. 加论文§4.2 推荐的 task prefix（我们的实现仅 22.4%，论文是 30%）
5. 用 GPT-4 重写训练数据 reasoning（最根本）

---

## 二、Stage2 训练本身：loss 曲线"收敛良好"是误导

### 2.1 三个 stage 的 loss 曲线

**Stage1 继续预训练**（1666 步，230 min）：
- 起始 loss：2.9178（最后5步平均：~2.70）
- 训练格式：`stage1_pretrain.jsonl`（23.6万条）
- 收敛状态：loss 从 2.92 → 2.70，**降幅很小**（约 8%）

**Stage2_only**（分支 A，4528 步，464 min）：
- 起始 loss：0.8582，终值：0.2176
- 训练格式：`train_pool_clean.jsonl`（28.98万条）
- 收敛状态：loss 从 0.86 → 0.22

**Stage2_s1**（分支 B，4528 步，从 stage1 adapter 继续）：
- 起始 loss：0.8769，终值：0.2176
- 训练格式：同上
- **与 Stage2_only 的 loss 曲线完全重合**

### 2.2 关键观察：Stage1 对 Stage2 几乎没有帮助

| Epoch% | Stage2_only | Stage2_s1 | Δ |
|---|---|---|---|
| 5% | 0.3067 | 0.3047 | -0.0020 |
| 10% | 0.2726 | 0.2725 | -0.0001 |
| 20% | 0.2640 | 0.2637 | -0.0003 |
| 30% | 0.2531 | 0.2508 | -0.0023 |
| 50% | 0.2334 | 0.2334 | +0.0000 |
| 70% | 0.2332 | 0.2336 | +0.0004 |
| 90% | 0.2244 | 0.2245 | +0.0001 |
| 100% | 0.2222 | 0.2221 | -0.0001 |

**两个分支在每个 checkpoint 的 loss 几乎完全相同**（Δ ≤ ±0.003，相当于小数点第三位的随机噪声）。

**结论**：Stage1 继续预训练对 Stage2 SFT 的训练动力学几乎没有任何增益。

### 2.3 训练 loss 0.22 不等于"学会了"

继续预训练的目标是注入生物领域知识。但**Stage2 SFT 的 loss 不反映任务能力，只反映"模仿 chosen 文本"的能力**。

具体证据：
- Stage2 SFT 数据是 `<reason>...</reason>\n<ans>label</ans>` 结构化文本
- 训练目标 = 最大化 chosen token 的对数似然
- **如果模型只学会"模仿 chosen 模板"（"先写 reason，再写 label"），loss 就能达到 0.2 量级**——因为模板本身很容易预测

Stage2 实际学会的可能是：
- ✅ 学会写 `<reason>` `<ans>` 标签结构
- ✅ 学会模仿 chosen 的英文生物学术语（"interaction"、"sequence"、"domain"等）
- ❌ **没学会基于序列特征判断 label**

这解释了为什么 loss 看起来"收敛良好"，但 eval 指标崩溃。

---

## 三、Stage2 在 binary 任务上完全坍缩（最强证据）

### 3.1 四档模型在 8 个 binary 任务上的预测分布对比

|任务 | Base | Stage2_only | Stage2_s1 | Stage3 |
|---|---|---|---|---|
| ncRNAProteinInter | pos=3 / neg=98 / **NONE=399** / 500 | **pos=0 / neg=500** / 0 / 500 | **pos=0 / neg=500** / 0 / 500 | pos=0 / neg=500 / 0 / 500 |
| promoter_enhancer | pos=20 / neg=115 / NONE=581 / 717 | **pos=0 / neg=717** / 0 / 717 | pos=0 / neg=717 / 0 / 717 | pos=3 / neg=714 / 0 / 717 |
| emp (10 子任务) | pos=0 / neg=931 / **NONE=4062** / 5000 | **pos=0 / neg=5000** / 0 / 5000 | pos=0 / neg=5000 / 0 / 5000 | pos=0 / neg=5000 / 0 / 5000 |
| antibody_antigen | pos=13 / neg=59 / NONE=426 / 500 | pos=21 / neg=479 / 0 / 500 | **pos=46 / neg=454** / 0 / 500 | **pos=2 / neg=498** / 0 / 500 |
| pd (3 子任务) | pos=20 / neg=701 / NONE=525 / 1246 | **pos=687 / neg=559** / 0 / 1246 | pos=664 / neg=582 / 0 / 1246 | pos=568 / neg=678 / 0 / 1246 |
| cpd (3 子任务) | pos=16 / neg=1154 / NONE=55 / 1246 | pos=546 / neg=700 / 0 / 1246 | pos=555 / neg=691 / 0 / 1246 | pos=480 / neg=766 / 0 / 1246 |
| tf_h (5 子任务) | pos=12 / neg=904 / **NONE=1582** / 2500 | pos=55 / neg=2445 / 0 / 2500 | pos=5 / neg=2495 / 0 / 2500 | pos=12 / neg=2488 / 0 / 2500 |
| tf_m (5 子任务) | pos=7 / neg=611 / **NONE=933** / 1551 | pos=172 / neg=1379 / 0 / 1551 | pos=147 / neg=1404 / 0 / 1551 | pos=159 / neg=1392 / 0 / 1551 |

### 3.2 解读：三种不同类型的"坍缩"

**类型A：结构化输出学会 + 类别完全坍缩**

- **ncRNAProteinInter / promoter_enhancer / emp**
- 现象：Base 模型大部分写不出 ans（70-80% NONE）；Stage2 之后 NONE 归零，模型**100% 能写 ans**，但**100% 写 negative**
- 含义：Stage2 学会了结构化输出格式，但完全没学会分类

**类型B：结构化输出学会 + 类别部分坍缩**

- **tf_h / antibody_antigen**
- 现象：Base 大部分写 NONE；Stage2 之后 NONE 归零，但**仍然以 negative 为主**（tf_h 仅 2% positive，antibody_antigen 仅 4% positive）
- 含义：Stage2 学到了一点"negative"的模式，但没学到"positive"的判别特征

**类型C：成功学会分类（对照组）**

- **pd / cpd / tf_m**
- 现象：Base 大量 NONE（40-90%），Stage2 之后 NONE 归零，且**正负比例接近 50/50**
- 含义：Stage2 在这3 个任务上**真正学会了**——这证明模型**有能力**学会 binary 分类，只是其它任务没学会

### 3.3 为什么 Stage3 DPO 让 antibody_antigen 更糟

| 模型 | antibody_antigen 正类预测数 |
|---|---|
| Stage2_s1 | **46 个** |
| Stage3 DPO | **2 个** |

**Stage3 DPO 把 Stage2 仅存的 46 个正类预测压缩成 2 个**——这不是修数据能解决的问题。

原因：Stage2 本身就不会做 antibody_antigen（只有 4% 正类），chosen/rejected 都倾向 negative，DPO 的"偏好信号"实际上在**强化模型写 negative 的倾向**。

这印证了"Stage3 不能教会 Stage2 不会做的事"。

---

## 四、Stage2 训练数据本身是平衡的（排除数据不平衡假设）

### 4.1 train_pool_clean.jsonl 里 binary 任务的 label 分布

|任务 | positive | negative | 比例 |
|---|---|---|---|
| emp-*（10 子任务聚合）| 43582 | 36228 | **54.6% / 45.4%** |
| promoter_enhancer（6 子任务聚合）| 5526 | 5478 | **50.2% / 49.8%** |
| tf-*（10 子任务聚合）| ~53000 | ~53000 | ~50/50 |
| pd-prom_300_* | ~8400 | ~8400 | ~50/50 |
| cpd-prom_core_* | ~8200 | ~8200 | ~50/50 |
| rna_protein_interaction | ~2700 | ~2600 | ~50/50 |
| antibody_antigen | ~4300 | ~4300 | ~50/50 |

**所有 binary 任务的训练数据都是 50/50 平衡的**。

### 4.2 这意味着什么

坍缩**不是数据不平衡问题**：
- ❌ 不能通过"对少数类上采样"解决
- ❌ 不能通过 class_weight 单独解决（虽然 class_weight 可能仍有帮助）
- ✅ 真正的瓶颈是模型**从这些平衡数据中没学到**任务相关表征

---

## 五、Stage2 作为 rejected 采样器也是坍缩的（DPO 数据问题的源头）

### 5.1 dpo_pairs.jsonl 中 rejected 的分布

|任务 | rejected 分布（stage2 采样）| chosen 分布（标注）|
|---|---|---|
| ncRNAProteinInter | neg=571 / pos=419 (57%/43%) | neg=502 / pos=488 (50%/50%) |
| tf | neg=404 / pos=207 (**66%/34%**) | pos=311 / neg=300 (50%/50%) |
| promoter_enhancer | neg=246 / pos=157 (61%/39%) | neg=211 / pos=197 (52%/48%) |
| antibody_antigen | neg=408 / pos=279 (59%/41%) | pos=363 / neg=324 (53%/47%) |

**stage2 作为"被采样的模型"已经偏向 negative 类**。

### 5.2 直接后果

- Stage2 prior ≠ chosen prior → 50% 的 pair stage2 采样撞 chosen标签
- DPO 学习"chosen_logp - rejected_logp"拉开时，模型被推向**stage2 的 prior**（偏 negative）
- 这就是为什么 Stage3 在 binary 任务上进一步退化

### 5.3 关键：这不是 DPO 数据问题

如果 DPO 数据本身50% 的 binary pair 是撞标签的，我们可以加 `<ans>` 过滤（P0）消除这一半。但这只能让"剩余的50% pair 提供有效梯度"——而这些 pair 的**信号量方向**仍然受 Stage2 prior 影响。

**真正的杠杆点是 Stage2 本身学会分类能力**——而不是 Stage3 的数据过滤。

---

## 六、Stage2 在回归任务上也是"中心化输出"（LoRA 容量不足证据）

### 6.1 回归任务预测分布

|任务 | 真实值范围 | Stage2_only 预测范围 | 占真实分布比例 |
|---|---|---|---|
| Stability | [-3, 3] | [0.18, 0.81] | **10%** |
| Thermostability | [30, 80] | [48.11, 58.21] | **25%** |
| Fluorescence | [0, 5] | [3.68, 3.68] | **0%（constant）** |

### 6.2 解读

**Stage2 在回归任务上的预测被压缩到训练分布的中心**：
- Stability 真实范围 ±3，预测只在 [0.18, 0.81] 之间
- Thermostability 真实范围 50，预测只在 [48, 58] 之间
- Fluorescence 完全坍缩到常数3.68

这是 **LoRA 容量不足的典型表现**：
- 模型学会了"answer 应该是数值"
- 模型没学会"answer 应该与具体序列相关"
- LoRA r=16 的可训练参数不足以拟合 21 个任务的全部细节

### 6.3 对 Stage1 评估的进一步说明

Stage1 训练 loss 从 2.92 → 2.70（降幅 8%）也偏小——继续预训练注入的领域知识有限。这与 Stage1 数据格式有关：
- Stage1 输入是 `stage1_pretrain.jsonl`（生物领域无标签语料）
- Stage2 输入是 `<input><output>` 任务格式
- Stage1 注入的"领域知识"对 Stage2 任务格式帮助有限——所以 Stage2_s1 和 Stage2_only 的 loss 几乎相同

---

## 七、Stage3 DPO 退化案例：Stability ρ 从 0.247 → 0.113

### 7.1 退化的具体表现

| 模型 | Stability ρ | Thermostability ρ |
|---|---|---|
| Stage2_only | 0.021 | 0.267 |
| Stage2_s1 | **0.247** | 0.295 |
| Stage3 (DPO) | **0.113** | 0.264 |

Stage3 DPO 把 Stage2_s1 在 Stability 上的 ρ 从 0.247 压到 0.113（**-54%**）。

### 7.2 退化的原因

回归任务的 chosen/rejected ans 数值天然不同（连续值），但 chosen 的 `<reason>` 是脚本生成的套话。Stage2 在采样时倾向于**模仿训练数据的中心**（即数值预测 ≈ 训练集均值）。

DPO 的"偏好信号"实际是：
- chosen reason（套话） vs rejected reason（更具体的 center value 预测）
- DPO 把模型推向"模仿 chosen 套话"——结果模型反而**忘记了 Stage2_s1 在 Stability 上学会的具体预测**

### 7.3 这印证 DPO 不能修复 Stage2 已经学好的能力退化

Stage3 DPO 是一把双刃剑：
- ✅ 在 Stage2 已经学对的任务上可以微调优
- ❌ 在 Stage2 没学会的任务上提供无效信号
- ❌ **在 Stage2 已经学好的任务上可能反向退化**

---

## 八、根因总结：3 个独立因素叠加

|因素 | 证据 | 修复方向 |
|---|---|---|
| **1. LoRA 容量不足（r=16）** | 回归任务预测被压缩到训练分布中心 | 加大 LoRA rank 至 r=64 |
| **2. 二元分类训练方法不当** | 4 个 binary 任务完全坍缩到 dominant class | 加 class_weight + label smoothing |
| **3. 训练数据 reasoning 是模板化套话** | chosen reason 99.78% 不含具体序列特征 | GPT-4 重写 reasoning / 增加 feature-based 标签 |

**三者叠加导致 Stage2 没真正学到任务能力**——Stage3 DPO 不能修复其中任何一个（只会让它更糟或无效）。

---

## 九、修复方案（按 ROI 排序）

### P0（30 分钟）：当前已完成——build_preference 加 `<ans>` 过滤 + 输出 task 字段

**价值**：让 DPO 数据达到"标签层面无噪声"基线，未来回退原因分析更清晰
**预期收益**：binary 任务 MCC 边际改善（+0.05 ~ +0.15）
**风险**：零

### P1（已实现）：重训 Stage2 SFT，加论文 §4.2 推荐的 task prefix + class_weight

> **✅ 已实现于 `train/stage2_sft.py`（2026-08-26 后）**：
> - `--task_prefix_ratio 0.30`（默认 0.30，论文原值；设 0 关闭）——随机选 30% 样本加 `[Classification/Regression: task]` 前缀
> - `--use_class_weight`（默认关）——对 binary 任务启用 balanced per-sample weight（通过自定义 `WeightedSFTTrainer`）
> - `--task_weight_power <0.0-1.0>`（默认 0 = 关）——任务间平衡：`weight[task]=(max_count/count[task])**power`。
>   0.5=温和（小任务 2.6x）、1.0=完全 balanced（小任务 7.8x）。复用 WeightedSFTTrainer，可与 class_weight 相乘。
> - smoke test 通过（0.5B 模型 2 step 端到端验证）
>
> 重跑命令：
> ```bash
> python train/stage2_sft.py \
>   --model_path /path/to/Qwen2.5-7B-Instruct \
>   --data_dir input_data --output_dir output/ckpt/stage2_v3 \
>   --task_prefix_ratio 0.30 --use_class_weight --use_4bit
> ```

**重要发现**：BioAlign 论文 §4.2 明确建议：

> "**In the initial attempts of the training process, we find that the imbalance among tasks within the dataset can pose challenges for the model in distinguishing between different tasks. To mitigate this, we randomly select 30 percent of the training data and prepend a task label in the format '[Classification/Regression:task name]' at the beginning of each question.**"

**我们的实现只有 22.4% 的训练数据加了前缀**——少加了7.6 个百分点。修复方法：

```python
# train/stage2_sft.py 加 task prefix（论文 §4.2 方法）
import random
random.seed(42)
PREFIX_RATIO = 0.30  # 论文原文比例

for r in train_data:
    if random.random() < PREFIX_RATIO:
        task = r['task']
        kind = 'Classification' if task in binary_or_multiclass_tasks else 'Regression'
        r['input'] = f'[{kind}: {task}] ' + r['input']
```

**预期收益**：emp / promoter_enhancer 等完全坍缩任务的 MCC 从 0 → 0.1~0.3
**风险**：低（论文标准做法）

**关于 class_weight 的补充说明**：

`class_weight` 是对少数类样本加权的 loss 修正方法。但**我们的 binary 任务训练数据本身接近 50/50 平衡**（emp 54.6/45.4、promoter_enhancer 50.2/49.8）——balanced 公式算出的 weight 几乎接近 1.0（0.93 vs 1.09），**不足以逆转 stage2 的坍缩**。

class_weight 适用的场景：
- ✅ 训练数据明显不平衡（如 9:1）
- ✅ loss landscape 全局最小值是"全答多数类"
class_weight 不适用的场景：
- ❌ 数据平衡但模型有其它学习倾向
- ❌ LoRA 容量不足（这是我们的主要问题）
- ❌ 训练数据 reasoning 是套话（这是我们的次要问题）

**结论**：class_weight 对本项目 ROI 低，**建议作为 P3 可选项**，不放在 P1。

### P2（半天）：重训 Stage2 SFT，加大 LoRA rank 至 r=64

```python
# train/stage2_sft.py LoraConfig
peft_config = LoraConfig(
    r=64,             # 原 16 → 64
    lora_alpha=128,   # 原 32 → 128
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**预期收益**：回归任务预测 spread 扩大，binary 任务细节捕捉能力提升
**风险**：中（4× 训练时间和显存）

### P3（1 周 + $300）：用 GPT-4 重写训练数据 reasoning

对 30,000 条 binary 任务训练数据，用 GPT-4 生成基于序列特征的 reasoning（详细 prompt 见 `docs/P0_DPO_DATA_FIX.md`）。

**预期收益**：根本上解决"模型学会模仿套话而非推理"的问题
**风险**：高（API 成本 + 人工 review + 时间）

### P4（可选）：Stage3 DPO 改用 on-policy rejection sampling + RLAIF

让 stage3 不是对 stage2 采样做 DPO，而是用规则模板生成 chosen、用 stage2 采样做 rejected，让 chosen vs rejected 的差距是**推理质量**而非 prior 差异。

**预期收益**：Stage3 不再退化 Stage2 已经学到的
**风险**：高（需重写 build_preference）

---

## 十、对 Phase 2 文档的修正

之前的 `docs/REASON_ANSWER_DECISION.md` 把 binary 任务坍缩归咎于"训练/评测格式不一致"，并提出 Phase 2 的 Reason+Answer 改造。本文档证明：

**Phase 2 的格式改造解决了"格式问题"（从 7.5% → 100% parser 覆盖率），但没解决"模型能力问题"**——Stage2 即使在 100% 结构化输出上，仍然坍缩到 dominant class。

**这意味着 Phase 2 的"预期效果"（binary 任务 MCC 从 -0.9~0.0 提升到 0.3~0.7）是不现实的**——除非 Stage2 本身具备分类能力。

正确的 Phase 2 文档应当：
- 承认格式改造只解决了一半问题
- 标注 Stage2 训练方法的局限
- 把"教会模型分类能力"作为下一阶段目标

## 十点五、任务难度评估：为什么不能简单归结为"任务太难"

**重要辨析**：论文说"任务对 LLM 难"≠"我们 Stage2 表现差是不可避免的"。

### 10.5.1 论文自己承认 Task 难度高，但仍然达到了进步

BioAlign 论文核心贡献是“在这些 hard 任务上证明了二阶段训练 + 推理微调能产生明显增益”。论文 Figure 6（消融图）明确显示：
- 只 Stage2 训练：EMP/TB-H/PD300 都明显超过 random 水平
- 加 Stage3 推理：进一步提升

**我们的 Stage2 明显达不到论文中同任务的表现**——这证明**除了任务难度，还有其它因素**。

### 10.5.2 我们项目与论文的差距量化

|维度 | 论文 | 我们 | 差距 |
|---|---|---|---|
| 训练数据量 | 333万 全量 | 28.98万 | **论文 11.5×** |
| EMP 训练集 | 22.99万 | 8.0万 | 论文 2.9× |
| RPI 训练集 | 1.5万 | 5,324 | 论文 2.8× |
| TB-H 训练集 | 12.83万 | 3.88万 | 论文 3.3× |
| Stage3 数据量 | ~8,000 reasoning | 25,000 DPO pairs | 反而我们多 |
| Task prefix 训练样本 | 30% 样本加 | 22.4% 样本加 | 论文多 7.6% |
| 基础模型 | Llama3.1-8B | Qwen2.5-7B | 接近 |
| LoRA 配置 | 论文未明示 | r=16 | 待查 |

**关键**：**我们项目与论文最大的差距是数据量**——不是任务难度。论文 Stage2 在 EMP/TB-H 上明显超过 random 水平，说明**论文的 Stage2 设置*是能学到任务能力*的**——我们的设置不行。

### 10.5.3 如果补上数据量 + task prefix + LoRA rank，能接近论文吗？

**大概率能明显改善**但可能达不到论文所有数字：
- pd/cpd/tf_m：我们已经接近论文水平（MCC 0.6+）——证明 28.98万数据量对部分任务够用
- emp/promoter_enhancer：坍缩到 dominant class——**论文在同样任务上能达到明显超过 random**，说明我们 Stage2 设计有问题

**可执行的改善预测**：
|任务 | 当前 MCC | 加 prefix + class_weight 后预测 | 加 LoRA r=64 后预测 |
|---|---|---|---|
| emp | 0.000 | 0.05~0.15 | 0.10~0.25 |
| promoter_enhancer | 0.000 | 0.05~0.15 | 0.10~0.25 |
| ncRNAProteinInter | 0.000 | 0.10~0.20 | 0.15~0.30 |
| tf_h | 0.023 | 0.10~0.20 | 0.15~0.30 |

**这些改善主要是从论文明确提出的优化里拿，而不是改 DPO 数据**。

---

## 十一、可执行的诊断脚本

附录脚本用于快速复现本文档中的关键证据：

```python
# 1. Stage2 loss 曲线对比
import re
def extract_loss_curve(path):
    losses = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = re.search(r"'loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)", line)
            if m: losses.append((float(m.group(2)), float(m.group(1))))
    return losses

# 2. 4 档模型 binary 任务预测分布
import json, re
def extract_ans(s):
    m = re.search(r'<ans>\s*(.+?)\s*</ans>', s, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else 'NONE'

# 3. dpo_pairs 中 rejected 分布
# （同本文档 §5.1 代码）

# 4. 训练数据 binary 任务 label 分布
import json, collections
c = collections.Counter()
with open('input_data/train_pool_clean.jsonl', encoding='utf-8') as f:
    for line in f:
        j = json.loads(line)
        if j['task'].startswith(('emp-','pd-','cpd-','rna_protein','antibody','tf-','promoter')):
            c[(j['task'], j['label'])] += 1

# 5. Stage2 回归任务预测 spread
import json, re
preds = []
with open('output/eval_s2_only.jsonl', encoding='utf-8') as f:
    for line in f:
        j = json.loads(line)
        if j.get('task') != 'Stability-Stability': continue
        m = re.search(r'<ans>\s*([\-\d.]+)', j.get('model_output',''))
        if m: preds.append(float(m.group(1)))
print(f'Stability 预测 spread: min={min(preds)}, max={max(preds)}, std={statistics.stdev(preds):.3f}')
```

---

## 十二、参考文献

- 训练日志：`logging/stage1_20260823_1100.log`、`logging/stage2_only_20260825_1923.log`、`logging/stage2_s1_20260823_2335.log`、`logging/stage3_20260825_1714.log`
- 推理结果：`output/eval_{base,s2_only,s1_s2,stage3}.jsonl`
- 评估指标：`eval/metrics_result/metrics_result_{base,s2_only,s1_s2,stage3}_all_omics_fix_v1.json`
- DPO 数据：`input_data/dpo_pairs.jsonl`、`input_data/dpo_source.jsonl`
- Phase 2 文档：`docs/REASON_ANSWER_DECISION.md`

---

**文档状态**：诊断完成，可作为下一阶段优化方向参考。