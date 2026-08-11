# 数据准备记录（data_prep）

> 本目录包含三段式后训练流水线（领域继续预训练 → QLoRA SFT → DPO）所需的**全部处理脚本与产出数据**。
> 所有处理可复现（固定随机种子），全部决策记录如下，供复查与面试应答。

## 1. 产出文件总览

| 文件 | 条数 | 大小 | 用途 | 来源 |
|---|---|---|---|---|
| `train_pool.jsonl` | 300,000 | 188MB | Stage 2 SFT 原始采样（30 万） | stage2_train.jsonl 三路划分 |
| **`train_pool_clean.jsonl`** | **289,768** | 182MB | **Stage 2 SFT 正式训练数据**（去重 + 模板均衡 + 含 stage3） | [05] 净化产物 |
| `dpo_source.jsonl` | 112,472 | 61MB | DPO 偏好对构造源（SFT 不可见） | 同上 |
| `eval_set.jsonl` | 18,870 | 11MB | 评估集（SFT/DPO 均不可见） | 同上 |
| `stage3.jsonl` | 8,002 | 21MB | 已并入 train_pool_clean / DPO 混合 | stage3.xlsx |
| `stage1_pretrain.jsonl` | 235,890 | 216MB | Stage 1 继续预训练 | GRCh38 + RNAcentral + Swiss-Prot |
| `smoke.jsonl` | 4,700 | 3MB | 冒烟测试（全流程小数据验证） | train_pool 每 task 抽 100 |
| `prep_stats.json` / `clean_stats.json` / `seq_stats.json` | - | - | 划分/净化/序列统计 | - |

## 2. 数据来源（原始出处与获取方式）

| 原始数据 | 来源 | 版本/获取方式 | 许可 |
|---|---|---|---|
| `stage2_train.jsonl` | 论文官方发布（Biology-Instructions） | Google Drive: `1OC3VpPKSQ0VHd9ZeZhnxI8EA2wTdrBg5`（论文 README 链接） | 论文仓库未附 LICENSE，仅用于本项目学习/求职复现 |
| `stage3.xlsx` / `stage3_gemini_check.xlsx` | 论文官方 GitHub 仓库 | https://github.com/hhnqqq/Biology-Instructions（commit `600acaa`） | 同上 |
| `GCA_000001405.20_GRCh38.p5_genomic.fna` | NCBI 人类参考基因组 | assembly `GCA_000001405.20` = **GRCh38.p5**（GenBank 版），NCBI datasets 下载 | NCBI 序列数据属公共领域 |
| `homo_sapiens.fasta` | RNAcentral（EBI） | `ftp.ebi.ac.uk/.../RNAcentral/current_release/sequences/by-species/vertebrates/homo_sapiens.fasta.gz`（2025-07 下载，680MB gz 解压） | RNAcentral 数据 CC0 |
| `uniprot_sprot.fasta` | UniProt Swiss-Prot | `uniprot_sprot.fasta`（2025-03 下载，约 UniProt 2025_01 release） | Swiss-Prot 数据 CC-BY-4.0 |

来源链：**原始数据 → 本目录脚本处理 → `output/` 产出**。评估代码（`eval/`）同样来源自论文官方仓库，另见 `eval/README.md`。

## 3. 处理管线

```
stage2_train.jsonl (3,330,232)
   ├─[01] 按 task 分层三路划分（互不相交）
   │     ├── train_pool.jsonl       300,000（SFT 原始采样）
   │     ├── dpo_source.jsonl       112,472（DPO 偏好构造源）
   │     └── eval_set.jsonl          18,870（评估，SFT/DPO 均不可见）
   │
stage2 三路划分 ──[05] 去重 + 模板均衡 + 合并 stage3 ──> train_pool_clean.jsonl（289,768，SFT 正式训练）
stage3.xlsx ──[02] 转 jsonl ──> stage3.jsonl（8,002，并入 [05]）

GRCh38 (3.27GB, 染色体级) ──┐
RNAcentral (2.29GB, 200万条) ─┼─[03] 切片段/抽样 + type token ──> stage1_pretrain.jsonl
Swiss-Prot (288MB, 57万条) ──┘

train_pool ──[04] 每 task 抽 100 ──> smoke.jsonl
```

## 4. 处理步骤与决策

### 3.1 stage2 三路划分（scripts/01_split_stage2.py）

**目标**：把 330 万条按 task 分层切成训练/DPO 源/评估三份，三者**严格不相交**。

**为什么按 task 分层（而不是全局随机）**：数据集严重不均衡（Isoform 157 万 vs CRISPROnTarget 1,453 条）。全局随机会让小任务在训练中近乎消失。分层保证每个 task 在训练/评估/DPO 中都有份。论文附录明确"平衡采样反而掉点"，所以**不做**均匀化，只做 cap。

**每 task 配额规则**（小任务评估优先）：
- 评估：`min(500, max(50, ceil(n×5%)))` —— 每 task 至少 50 条、至多 500 条
- DPO 源：`min(5000, ceil(n×10%))` —— 每 task 至多 5000 条
- 训练：剩余部分，cap 20,000/task；总目标 300,000，超量按比例压缩、每 task 保底 2,000（保底不超过可用条数）

**为什么 DPO 源与评估集分开（关键设计）**：DPO 偏好数据的 chosen 是"标准答案"。若用评估集的问题构造 DPO 偏好对，DPO 训练后模型在评估集上的任务指标会被污染（模型"见过"这些问题的答案）→ 无法证明"DPO 不掉领域能力"。因此拆成两个独立集合，评估严格在 DPO 未接触的数据上进行。

**抽样方法**：确定性抽样——收集每 task 行号 → 固定种子打乱 → 前 N 行切分。行号唯一 → 三路物理上不可能重叠。重叠检查（完整 input+label 精确匹配）：三路交集均为 0（少量"重叠"来自源数据本身的重复样本——同问题不同答案的模板变体，见 3.4）。

### 3.5 训练集净化：去重 + 模板均衡（scripts/05_dedup_template.py）

**动机（模板同质化诊断）**：stage2 是模板化生成的（论文：每 task 100~300 个问题模板 × 大量序列）。实测 train_pool 中每 task 仅 50~200 个模板骨架，头部模板重复套用几十~几百条序列（如 emp Top1 模板 533 条、Top5 占比 12.8%）。同质数据的信息密度低，训练后期会导致模型**过拟合模板风格**而非学习任务。

**三步处理**：
1. **完全去重**：task+input+output 完全相同 → 删除
2. **模板感知均衡采样**：正则提取模板骨架（挖掉序列的问题句式）→ 每 task 内对每模板样本数 cap：`cap = max(50, ceil(avg × mult))`，`avg = task样本数/模板数`，默认 mult=1.0。无标签样本（`NO_TAG`）不裁剪
3. **合并 stage3.jsonl**（8,002 条 GPT-4o-mini 精修推理长答案，增强长答案/推理覆盖）

**效果**：30 万 → 289,768 条；Top5 模板占比 2.0% → 1.3%（mult 可调更激进）。

**关键发现：stage2 序列标签是非标准闭合**——格式为 `<rna>...<rna>`（闭合标签**无斜杠**），非 `<rna>...</rna>`。影响任何基于标签的解析（模板提取正则已适配：`<(?:dna|rna|protein)>[A-Za-z]+<(?:dna|rna|protein)>`），但不影响模型训练（整段文本喂入）。

### 3.2 stage3 转换（scripts/02_stage3_convert.py）

stage3.xlsx（8,002 条 GPT-4o-mini 精修的**推理型**长答案）转 jsonl，保留 input/task/label/output 四字段。用途：可选并入 SFT 增强长答案覆盖；其 output 可作为 DPO 的高质量 chosen 混合（区分度保障）。

### 3.3 序列准备（scripts/03_seq_prepare.py）

| 源 | 处理 | 产出 |
|---|---|---|
| GRCh38 基因组 (3.27GB) | 只保留长度 ≥1Mbp 的主染色体（49 条，自动跳过 contig/scaffold/线粒体）；按染色体长度比例配额；每条染色体随机切 512~2000bp 片段；**N 占比 >2% 的片段丢弃**（着丝粒/跑台区低复杂度区） | 75,890 片段 |
| RNAcentral 人类 ncRNA (2.29GB, 200万条) | 蓄水池在线抽样（固定种子） | 80,000 条 |
| Swiss-Prot 蛋白 (288MB, 57万条) | 蓄水池在线抽样 | 80,000 条 |

每条输出 `{"text": "<dna>序列</dna>"}` 等——**加 type token 前缀**。

**为什么加 `<dna>/<rna>/<protein>` 前缀**：DNA/RNA/蛋白字母表不同（DNA: ATCG；RNA: AUCG；蛋白: 20 种氨基酸），共享 token（如 "G" 在 DNA 是鸟嘌呤、在蛋白是甘氨酸）语义不同。加 type token 让模型显式区分 omics 语义，避免混合预训练时的语义冲突。这是对论文（Stage 1 用裸序列）的增强，SFT 数据本就带这些标签，保持一致。

**为什么 RNAcentral 序列含 T（不是 U）**：RNAcentral 官方 FASTA 以 DNA 形式存储，与论文使用的数据同源，保留原样。

**字母表观察**（见 seq_stats.json）：DNA 片段含少量 IUPAC 模糊码（K/Y/N，N 占比 ≤2%）；RNA 含较多模糊码（R/Y/S/W/M/D/K/X）；蛋白含 B/Z/X/U/O（模糊码/硒代半胱氨酸）。保留原分布，只过滤 N 密集区。

### 3.4 已知问题与说明

1. **源数据本身有重复样本**：stage2_train.jsonl 中 ~1.9% 的样本（同 task+input+label）出现多次，个别"同问题不同 output"。行号切分保证三路物理不重叠（"重叠检查"中的少量命中即来自此类源数据重复）；训练集内的完全重复由 [05] 步骤 1 去重。
2. **小 task 训练配额可能很少**：极小 task（如 tf-m-3 仅 8 条）几乎全部进入评估/DPO，训练中近乎无样本——这是"评估优先"的代价，可接受（这些任务在评估中仍可测，训练覆盖率问题由 cap 逻辑缓解）。
3. **GRCh38 片段未做低复杂度/重复区域屏蔽**：随机切片段天然包含基因间区/重复区（与论文做法一致，论文同样直接切 2000 字符）。

## 5. 面试问答准备

- **Q: 为什么不平衡采样？** A: 论文消融实验证明平衡采样导致下游指标下降（扭曲真实生物学分布）；采用分层 + cap。
- **Q: 评估集和 DPO 数据重叠怎么办？** A: 三路独立划分，DPO 偏好对从 dpo_source 构造、评估在 eval_set，行号唯一性保证不相交。
- **Q: 数据泄漏怎么防？** A: ① 训练/DPO/评估三路不相交；② 完整 input+label 精确匹配验证重叠为 0；③ 冒烟集从训练集抽取，仅用于代码验证。
- **Q: 为什么用蓄水池抽样？** A: 源文件 2GB+ 无法全部载入内存；蓄水池单遍扫描、O(1) 内存、保证均匀随机。
- **Q: 序列长度为什么 512~2000？** A: 论文上限 2000 字符（≈1200 token）；Kaggle 资源下缩短到 512 起步，片长随机以覆盖不同尺度模式。
- **Q: 数据量为什么是论文的 1/10~1/20？** A: 硬件约束（24×A100 1.5 天 vs T4×2 数小时）；数据下载与论文同源，训练抽样按资源压缩，消融实验证明方向性结论仍成立。
- **Q: 15 万条 SFT 数据量够吗？有量化依据吗？** A: **诚实回答：没有先验量化依据**。曾提出"token 数远超可训练参数数所以够"的论据，经核对是不成立的——该比例与 Chinchilla scaling law（预训练最优 token≈20×参数）方向相反，且 Chinchilla 仅适用于从零预训练，不适用于 SFT。15 万条是资源约束下的工程选择，落在社区 SFT 规模范围内（LIMA 仅 1k 条、Alpaca 52k 条，均非定律）；"低秩 LoRA → 数据需求低"亦无文献严格支撑。**充分性的正确验证方式是实验**：① 训练 loss 曲线（不降→欠拟合，加数据/epoch/rank）；② eval 指标 vs 基线（基座零样本、论文 SOTA）；③ 数据量消融（改 `TRAIN_TOTAL_TARGET` 重跑 `01_split_stage2.py`，几分钟生成 20-30 万条训练集，直接测边际效应）。

- **Q: 模板同质化怎么应对？** A: 诊断先行——按 task 提取模板骨架（挖掉序列的问题句式），实测每 task 仅 50~200 个模板、头部模板重复套用几十~几百条序列。应对：① 完全去重；② 模板感知均衡采样（每模板 cap = max(50, ceil(avg×mult))）；③ 合并 stage3 高信息密度长答案。效果：Top5 模板占比 2.0%→1.3%。附发现：stage2 的序列标签是非标准闭合 `<rna>...<rna>`（无斜杠），模板提取正则已适配。

## 6. 复现方式

```bash
cd data_prep
python -X utf8 scripts/01_split_stage2.py     # 需要 ../dataset/stage2_train.jsonl（TRAIN_TOTAL_TARGET=300000）
python -X utf8 scripts/02_stage3_convert.py   # 需要 ../dataset/stage3.xlsx
python -X utf8 scripts/03_seq_prepare.py      # 需要 ../seq/*.fasta/.fna
python -X utf8 scripts/05_dedup_template.py   # 需要 output/train_pool.jsonl + stage3.jsonl
python -X utf8 scripts/04_smoke.py            # 需要 output/train_pool.jsonl
```

所有脚本固定 `SEED=42`，重跑结果一致。
