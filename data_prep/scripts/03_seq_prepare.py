# -*- coding: utf-8 -*-
"""
03_seq_prepare.py
=================
Stage 1 继续预训练序列数据准备。三源未标注生物序列 -> stage1_pretrain.jsonl。

  1) 人类基因组 DNA：GRCh38 (GCA_000001405.20_GRCh38.p5_genomic.fna)
     染色体级 FASTA -> 随机切 512~2000bp 片段，过滤 N 占比 >2% 的片段
  2) 人类非编码 RNA：RNAcentral (homo_sapiens.fasta)
     200 万条 -> 固定种子随机抽 N 条（保留原始字母表，含 T）
  3) 蛋白序列：UniProt Swiss-Prot (uniprot_sprot.fasta)
     57 万条 -> 固定种子随机抽 N 条

每条输出为 {"text": "<dna>序列</dna>"}（type token 前缀帮助模型区分
三种 omics 语义，避免共享 token 的语义冲突；可配置关闭）。

设计要点（面试可讲）：
- GRCh38 只保留主染色体（长度 >= 1Mbp，自动跳过 contig/scaffold/线粒体）
- 片段随机长度 512~2000（论文为 2000 字符上限，Kaggle 资源压缩为 512 起步）
- N 过滤：片段内 N 占比 >2% 丢弃（着丝粒/跑台区等低复杂度区域）
- RNAcentral 序列为 DNA 形式存储（含 T 非 U），与论文同源，保留原样
- 三源均抽样至 target_per_source 条（默认 80000），总量可控

用法：python 03_seq_prepare.py
输出：output/stage1_pretrain.jsonl, output/seq_stats.json
"""
import json
import random
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
TARGET_PER_SOURCE = 80000      # 每源抽样条数
MIN_FRAGMENT = 512             # 片段最短长度
MAX_FRAGMENT = 2000            # 片段最长长度
MAX_N_RATIO = 0.02             # 片段内 N 占比上限
MIN_CHROM_LEN = 1_000_000      # 只处理主染色体（跳过 contig/scaffold/线粒体）
PREFIX = True                  # 是否加 <dna>/<rna>/<protein> type token

DNA_SRC = "../seq/GCA_000001405.20_GRCh38.p5_genomic.fna"
RNA_SRC = "../seq/homo_sapiens.fasta"
PROT_SRC = "../seq/uniprot_sprot.fasta"

rng = random.Random(SEED)


def fasta_sequences(path, skip_short=None):
    """逐条产出 (header, seq)，**流式**、内存 O(1)。

    skip_short 用于过滤掉 过短序列。GRCh38 传 1MB 是关键：人类基因组除主
    染色体以外还有几万个 contig/scaffold/线粒体/MHC 区段，这些都该跳过；
    留 min_chrom=1MB 是在 FASTA 源头过濾，最高效。

    FASTA 多行拼接：一条记录可能跨多行（有时 80 字符续行），这里在遇到
    `>` 才 flush 当前"header + seq"，并在循环结束末尾补一个 flush（最后
    一条记录不被遗漏）。

    参数 `errors="ignore"`：GRCh38 拼接初期 FASTA 出现过非法 ASCII 字符
    （0x00等），逐字节 ignore 是人畜无害的选择。
    """
    header, seq = None, []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    s = "".join(seq)
                    if skip_short is None or len(s) >= skip_short:
                        yield header, s
                header, seq = line, []
            else:
                seq.append(line.strip())
    if header is not None:
        s = "".join(seq)
        if skip_short is None or len(s) >= skip_short:
            yield header, s


# ============================================================================
# 自定义 1：GRCh38 按主染色体长度比例切片段
# ============================================================================
# 两遍扫描：
#   1) 先拿每条染色体长度 → main_lens[chrom] = bp 数；
#   2) 按比例 per_chrom = max(1, round(TARGET * len/total)) 切片段。
#
# 为什么走比例：染色体 1（2.5 亿 bp）和染色体 Y（5.7 千万 bp）长度差 5 倍，
# 染色体 22 比 Y 还短。如果按均匀 pos 切，染色体 22/Y 会**过度表示**。
# 按比例切出来各染色体的**覆盖度**均匀。
# `max(1, round(...))` 防止“染色体 22 算出 0 比例”被 round 截成 0 → 被跳过。
#
# 【生物学动机】N 占比 > 2% 主要是着丝粒/跑台区/假基因区；DNA 序列里那一段
# 没有建模意义。过滤 2% 不是随便打的，是与论文上下文一致。
print("[1/3] GRCh38 切片段 ...")
main_lens = {}
for header, seq in fasta_sequences(DNA_SRC, skip_short=MIN_CHROM_LEN):
    main_lens[header[1:].split()[0]] = len(seq)
total_main = sum(main_lens.values())
print(f"  主染色体 {len(main_lens)} 条, 总长 {total_main}")

dna_frags = []
skipped_n = 0
for header, seq in fasta_sequences(DNA_SRC, skip_short=MIN_CHROM_LEN):
    name = header[1:].split()[0]
    per_chrom = max(1, round(TARGET_PER_SOURCE * main_lens[name] / total_main))
    for _ in range(per_chrom):
        L = rng.randint(MIN_FRAGMENT, MAX_FRAGMENT)
        start = rng.randint(0, len(seq) - L)
        frag = seq[start:start + L].upper()
        if frag.count("N") / L > MAX_N_RATIO:
            skipped_n += 1
            continue
        dna_frags.append(frag)
if len(dna_frags) > TARGET_PER_SOURCE:
    dna_frags = rng.sample(dna_frags, TARGET_PER_SOURCE)
print(f"  GRCh38: 得到 {len(dna_frags)} 个片段 (N 过滤丢弃 {skipped_n})")

# ============================================================================
# 自定义 2：经典 Reservoir Sampling（在线 / O(TARGET) 内存）
# ============================================================================
# 两源 几十万→抽取 8万条：如果车集全量后调 random.sample 内存会爆。
# 这里用 Vitter 算法 (随机抽样版本)的**单遍**实现：
#
#   前 TARGET 条：直接扔进仓位。
#   第 i > TARGET 条：以概率 TARGET/i 与仓里随机位置 j 交换。
#
# 证明：每一条原始记录最终留在仓里的概率都是 TARGET/total。 
# （前 TARGET 条 以 P=1 进仓；后面第 i 条 以 P=TARGET/i 进仓；合起来 = TARGET/i。）
# 这样我们不需要事先 total_rna，就能保证估计出的 8万条是**无偏均匀抽**。
print("[2/3] RNAcentral 抽样 ...")
rna_seqs = []
total_rna = 0
for header, seq in fasta_sequences(RNA_SRC):
    total_rna += 1
    if len(rna_seqs) < TARGET_PER_SOURCE:
        rna_seqs.append(seq.upper())
    else:
        j = rng.randint(0, total_rna - 1)
        if j < TARGET_PER_SOURCE:
            rna_seqs[j] = seq.upper()
print(f"  RNAcentral: 扫描 {total_rna} 条, 抽得 {len(rna_seqs)} 条")

print("[3/3] Swiss-Prot 抽样 ...")
prot_seqs = []
total_prot = 0
for header, seq in fasta_sequences(PROT_SRC):
    total_prot += 1
    if len(prot_seqs) < TARGET_PER_SOURCE:
        prot_seqs.append(seq.upper())
    else:
        j = rng.randint(0, total_prot - 1)
        if j < TARGET_PER_SOURCE:
            prot_seqs[j] = seq.upper()
print(f"  Swiss-Prot: 扫描 {total_prot} 条, 抽得 {len(prot_seqs)} 条")

# ============================================================================
# 合并输出 —— type-token 前缀
# ============================================================================
# `f"<{omics}>{s}</{omics}>"` —— **非标准闭合标签**（无斜杠不是 XML），是
# 与 05_dedup_template 的 SEQ_RE（`<(dna|rna|protein)>[A-Za-z]+<(dna|rna|protein)>`）
# **强耦合**的。如果以后要改这里（例如改成 `<|dna|>...<|/dna|>`），必须同步
# 调整 05_dedup_template.SEQ_RE。
#
# 为什么需要标签：模型看到同一个 token "ACGT..." 在不同语义下出现（DNA?RNA?）
# 会冲突。type token 明确告知来源 omics，让 BPE 不会跨域 merge。
tags = {"dna": dna_frags, "rna": rna_seqs, "protein": prot_seqs}
out = []
for omics, seqs in tags.items():
    for s in seqs:
        if PREFIX:
            out.append({"text": f"<{omics}>{s}</{omics}>"})
        else:
            out.append({"text": s})

rng.shuffle(out)
with open("output/stage1_pretrain.jsonl", "w", encoding="utf-8") as f:
    for d in out:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

# 字母表统计（供复查）
alphabet = {}
for omics, seqs in tags.items():
    sset = set()
    for s in seqs:
        sset.update(s)
    alphabet[omics] = sorted(sset)

stats = {
    "seed": SEED, "prefix": PREFIX,
    "per_source": {k: len(v) for k, v in tags.items()},
    "total": len(out),
    "source_scanned": {"rna": total_rna, "protein": total_prot, "dna_main_chroms": len(main_lens), "dna_total_len": total_main},
    "n_skipped_fragments": skipped_n,
    "alphabet_observed": alphabet,
    "length_stats": {
        "dna": {"min": min(map(len, dna_frags)), "max": max(map(len, dna_frags))},
        "rna": {"min": min(map(len, rna_seqs)), "max": max(map(len, rna_seqs))},
        "protein": {"min": min(map(len, prot_seqs)), "max": max(map(len, prot_seqs))},
    },
}
with open("output/seq_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"-> output/stage1_pretrain.jsonl: {len(out)} 条")
print("字母表观察:", alphabet)
print("统计已写入 output/seq_stats.json")
