# -*- coding: utf-8 -*-
"""
parser_v2.py —— 鲁棒的评测输出解析器
========================================
替代 eval/evaluate.py 里脆弱的关键词/正则提取。

设计目标：
1. 离线、零成本、可复现（不依赖外部 LLM）
2. 对"自然语言描述"和"短语/标签"双向兼容
3. 与未来"Reason + Answer"结构化输出兼容
   （从 `Answer: positive` / `Classification: positive` / `Result: 3.14` 中提取）
4. 对 47 个任务的输出做任务特定的最优提取

公开接口：
- parse_binary_classification(output) -> {'class': 0/1, 'confidence': float, 'reason': str}
- parse_multiclass_classification(output, classes) -> {'class': str|None, 'reason': str}
- parse_modification_multilabel(output, classes) -> {'labels': List[str]}
- parse_regression_number(output, value_range=None) -> {'value': float|None, 'candidates': List[float]}
- parse_ec_numbers(output) -> {'ecs': List[str]}
- parse_enhancer_activity(output) -> {'hk': float|None, 'dev': float|None}
- parse_programmable_switch(output) -> {'ON': float|None, 'OFF': float|None, 'ON_OFF': float|None}

新增 Reason + Answer 支持 (v2.1, 2024-xx):
- extract_ans_block(output): 从 <ans>...</ans> 块提取
- extract_reason_block(output): 从 <reason>...</reason> 块提取
- extract_structured_field 优先取 <ans> 内容，再 fallback 到 Answer: 形式
"""
import re
from typing import List, Dict, Optional, Tuple, Any

# ============================================================
# 通用工具
# ============================================================

# Reason + Answer 格式专用
ANS_BLOCK_RE = re.compile(r"<ans>\s*(.+?)\s*</ans>", re.IGNORECASE | re.DOTALL)
REASON_BLOCK_RE = re.compile(r"<reason>\s*(.+?)\s*</reason>", re.IGNORECASE | re.DOTALL)

# 兼容老格式：Answer: positive / Classification: positive / Result: 3.14
STRUCT_PATTERNS = [
    re.compile(r"(?:answer|classification|class|label|result|prediction|output)\s*[:：]\s*([^\n\r]+)", re.IGNORECASE),
]

# 优先从 output 头部 200 字符中找标签词（更鲁棒，避免被文中"no evidence"误中）
HEAD_WINDOW = 200


def extract_ans_block(output: str) -> Optional[str]:
    """从 <ans>...</ans> 块提取，返回第一行非空内容（避免多标签 mod 引入子串匹配问题）"""
    if not output:
        return None
    m = ANS_BLOCK_RE.search(output)
    if not m:
        return None
    content = m.group(1).strip()
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line
    return content


def extract_reason_block(output: str) -> Optional[str]:
    """从 <reason>...</reason> 块提取，返回拼接后的内容"""
    if not output:
        return None
    m = REASON_BLOCK_RE.search(output)
    return m.group(1).strip() if m else None


def extract_structured_field(output: str) -> Optional[str]:
    """从多种结构化字段中提取预测值，按优先级：
       1) <ans>...</ans> 块 (Reason + Answer 格式)
       2) Answer: xxx / Classification: xxx / Result: xxx
       3) 返回 None - 让调用方走关键词 fallback
    """
    if not output:
        return None
    # 1) Reason + Answer 格式
    ans = extract_ans_block(output)
    if ans:
        return ans
    # 2) Answer: / Classification: 格式
    head = output[:HEAD_WINDOW * 2]
    for pat in STRUCT_PATTERNS:
        m = pat.search(head)
        if m:
            return m.group(1).strip().rstrip('.,;')
    return None


# ============================================================
# 1) Binary classification (yes/no, positive/negative)
# ============================================================

def predict_label_unfiltered(text: str) -> Optional[int]:
    """复刻 evaluate.py 的 classify_by_keywords: 只看 yes/no/absence 等关键词"""
    if not text: return None
    t = text.lower()
    pos_kw = ['yes']
    neg_kw = ['no','absence','not found','not detected','not associated','not inferred',
              'not linked','does not indicate','no evidence','not predicted','absent']
    unk_kw = ["don't know",'unknown','unsure','uncertain','not applicable']
    if any(k in t for k in pos_kw): return 1
    if any(k in t for k in neg_kw): return 0
    if any(k in t for k in unk_kw): return 'dont_know'
    return None


# 强 positive 信号：在文本"明确表达肯定"
STRONG_POSITIVE = [
    r"\byes[,.\s]",                  # yes / yes.
    r"\bconfirm\w*\s+that\b",        # confirm that
    r"\baffirm\w*\b",                # affirmative / affirm
    r"\bpositive\b",                 # positive
    r"\bpresent\b",                  # present
    r"\bdetected\b",                 # detected
    r"\binteraction\s+is\s+(?:likely|possible|expected|present)\b",
    r"\bsupport\w*\s+(?:the\s+)?(?:interaction|binding|prediction|presence)\b",
    r"\bexhibit\w*\s+(?:features|patterns|characteristics)\s+(?:of|that)\b",
    r"\binteraction\s+is\s+predicted\b",
    r"\binteraction\s+is\s+confirmed\b",
    r"\b(?:likely|probably|expected)\s+to\s+(?:interact|bind|be\s+present)\b",
    r"\bI\s+can\s+(?:positively\s+)?(?:identify|confirm)\b",
    r"\b(?:shows?|contain\w*|reveal\w*|indicat\w*|suggest\w*)\s+(?:signs?|evidence|presence|interaction|binding|motifs?|sites?|regions?)\b",
    r"\bcontains?\s+(?:a\s+|the\s+|motifs?\s+(?:known|that)\s+to\s+facilitate\s+)?(?:binding|interaction|promoter|enhancer|motifs?|sites?)\b",
    r"\b(?:motifs?\s+)?(?:known|that)\s+to\s+facilitate\s+\w+\s+binding\b",
    r"\b(?:sequences?\s+)?exhibit\w*\s+(?:features|patterns)\b",
    r"\bcorrelat\w*\s+with\b",
    r"\bsignificant\s+correlation\b",
]

# 强 negative 信号：明确表达否定
STRONG_NEGATIVE = [
    r"\bno[,.\s]",                                # no / no.
    r"\bnegative\b",
    r"\babsence\b",
    r"\bnot\s+(?:found|detected|present|associated|inferred|linked|predicted|identified|observed|confirmed|conducive)\b",
    r"\bdoes\s+not\s+(?:contain|support|show|indicate|reveal|exhibit|suggest|appear)\b",
    r"\bdo\s+not\s+(?:contain|support|show|indicate)\b",
    r"\b(?:lacks?|lacking)\b",
    r"\bcannot\s+(?:confirm|detect|identify|assert|affirm|find)\b",
    r"\bcan\s+not\s+(?:confirm|detect|identify)\b",
    r"\binsufficient\s+(?:evidence|molecular\s+basis|data)\b",
    r"\bno\s+(?:evidence|signs?|molecular\s+basis|indication|interaction)\b",
    r"\b(?:unlikely|not\s+likely|not\s+conduive|not\s+support)\b",
    r"\bhas\s+not\s+(?:been\s+)?(?:located|found|detected|identified|observed)\b",
    r"\bdoes\s+not\s+(?:point|tend)\s+(?:towards?|to)\b",
    r"\bdoes\s+not\s+(?:exhibit|show|display|reveal)\b",
    r"\bnot\s+(?:conduive|consistent)\s+(?:to|with)\b",
    r"\bno\s+core\s+promoter\b",
    r"\bafter\s+carefully\s+(?:looking|examining)[,\s]+my\s+answer\s+is\s+negative\b",
    r"\bmy\s+answer\s+is\s+no\b",
    r"\bcertainly\s+not\b",
    r"\bwithout\s+(?:evidence|signs?|detectable)\b",
]

# 中性 / 不知 / 跳过（不算预测）
UNKNOWN = [
    r"\bdon['']?t\s+know\b",
    r"\bunknown\b",
    r"\bunsure\b",
    r"\buncertain\b",
    r"\bnot\s+applicable\b",
    r"\bI\s+cannot\s+(?:answer|determine|assess)\b",
    # 犹疑 / 推卸型语言 —— 应判为不确定而非 negative
    r"\bwithout\s+(?:knowing|access\s+to|specific)\b",
    r"\bhowever[,\s]+without\b",
    r"\btypically\s+(?:use|look|require)\b",
    r"\bwould\s+(?:typically|usually)\s+(?:use|need|require)\b",
]

_NEG_PAT = re.compile("|".join(STRONG_NEGATIVE), re.IGNORECASE)
_POS_PAT = re.compile("|".join(STRONG_POSITIVE), re.IGNORECASE)
_UNK_PAT = re.compile("|".join(UNKNOWN), re.IGNORECASE)


def _score_binary(text: str) -> Tuple[int, float, str]:
    """返回 (pred_label 0/1, confidence 0~1, matched_rule)"""
    if not text or not text.strip():
        return None, 0.0, "empty"
    # 优先从结构化字段提取
    struct = extract_structured_field(text)
    if struct:
        struct_l = struct.lower().strip()
        # 短结构化字段直接判
        if struct_l in ("yes", "positive", "true", "1", "present", "detected"):
            return 1, 1.0, f"struct:{struct_l}"
        if struct_l in ("no", "negative", "false", "0", "absent", "not detected"):
            return 0, 1.0, f"struct:{struct_l}"
        # 字段长，可能包含更多上下文，对字段内部再跑一次关键词
        text = struct  # 替换为字段内容继续判

    head = text[:HEAD_WINDOW]
    full = text

    # 先判 unknown —— 犹疑型语言不算预测
    if _UNK_PAT.search(head):
        # 但如果后面有明确信号，仍可判
        neg_match = _NEG_PAT.search(head)
        pos_match = _POS_PAT.search(head)
        if not (pos_match or neg_match):
            return None, 0.0, "unknown"

    # 强 negative
    neg_match = _NEG_PAT.search(head)
    pos_match = _POS_PAT.search(head)

    # 解决 "yes" 误中 "yes/no 都不确定" 等：取最早出现的强信号
    pos_pos = head.lower().find("yes") if "yes" in head.lower() else -1
    neg_pos = -1
    m = _NEG_PAT.search(head)
    if m:
        neg_pos = m.start()

    # 先看 head 里的强 positive
    if pos_match and pos_match.start() < (neg_pos if neg_pos >= 0 else 1 << 30):
        return 1, 0.9, "positive_head"
    if neg_match:
        # 特殊：head 里有"yes"但也有 negative 关键词（如"yes, but no binding"）
        # 此时看 negative 出现位置是否在 positive 之后
        if pos_pos >= 0 and neg_pos > pos_pos:
            return 0, 0.7, "negative_after_yes"
        return 0, 0.9, "negative_head"

    # 退到全文
    pos_match = _POS_PAT.search(full)
    neg_match = _NEG_PAT.search(full)
    if pos_match and not neg_match:
        return 1, 0.6, "positive_body"
    if neg_match and not pos_match:
        return 0, 0.6, "negative_body"
    if pos_match and neg_match:
        # 取最早出现的
        if pos_match.start() < neg_match.start():
            return 1, 0.55, "positive_first_body"
        return 0, 0.55, "negative_first_body"

    return None, 0.0, "no_signal"


def parse_binary_classification(output: str) -> Dict[str, Any]:
    """返回 {class: 0|1|None, confidence: 0~1, rule: str}"""
    label, conf, rule = _score_binary(output or "")
    return {"class": label, "confidence": conf, "rule": rule}


# ============================================================
# 2) Multi-class classification (NoncodingRNAFamily)
# ============================================================

RNA_CLASSES = sorted([
    '5S_rRNA', '5_8S_rRNA', 'tRNA', 'ribozyme', 'CD-box', 'miRNA',
    'Intron_gpI', 'Intron_gpII', 'HACA-box', 'riboswitch', 'IRES',
    'leader', 'scaRNA',
], key=len, reverse=True)


def parse_multiclass_classification(output: str, classes: List[str]) -> Dict[str, Any]:
    """返回 {class: str|None, confidence: 0~1}"""
    if not output:
        return {"class": None, "confidence": 0.0}

    # 优先结构化字段
    struct = extract_structured_field(output)
    if struct:
        sl = struct.strip()
        # 精确匹配
        if sl in classes:
            return {"class": sl, "confidence": 1.0, "rule": "struct_exact"}
        # 忽略大小写
        for c in classes:
            if c.lower() == sl.lower():
                return {"class": c, "confidence": 1.0, "rule": "struct_case_insensitive"}

    # 全文搜索，按长度倒序（先长后短避免子串误匹配）
    head = output[:HEAD_WINDOW]
    full = output
    for c in classes:
        pat = re.compile(rf"\b{re.escape(c)}\b")
        if pat.search(head):
            return {"class": c, "confidence": 0.9, "rule": "head"}
        if pat.search(full):
            return {"class": c, "confidence": 0.6, "rule": "body"}

    return {"class": None, "confidence": 0.0, "rule": "no_match"}


# ============================================================
# 3) Multi-label classification (Modification)
# ============================================================

MODIFICATION_CLASSES = sorted([
    'm6Am', 'm1A', 'm5C', 'm5U', 'm6A', 'm7G', 'AtoI', 'Psi',
    'Am', 'Cm', 'Gm', 'Um', 'none',
], key=len, reverse=True)


def parse_modification_multilabel(output: str) -> Dict[str, Any]:
    """返回 {labels: List[str]}"""
    if not output:
        return {"labels": []}

    # 优先结构化字段
    struct = extract_structured_field(output)
    search_text = struct if struct else output

    labels = []
    for mod in MODIFICATION_CLASSES:
        if mod == 'none':
            # 单独判：文本明确说"none"或"no modification"或"not in the list"
            if re.search(r"\bnone\b", search_text, re.IGNORECASE) or \
               re.search(r"no\s+(?:modification|modifications)", search_text, re.IGNORECASE):
                labels.append('none')
        else:
            pat = re.compile(rf"\b{re.escape(mod)}\b")
            if pat.search(search_text):
                labels.append(mod)

    # 特殊情况：如果只命中 'none'，且有具体 modification 词，移除 none
    if 'none' in labels and len(labels) > 1:
        labels = [l for l in labels if l != 'none']

    # 如果没命中任何标签：尝试 binary 兜底
    if not labels:
        b = parse_binary_classification(output)
        if b['class'] is not None:
            labels = ['none'] if b['class'] == 0 else []

    return {"labels": labels}


# ============================================================
# 4) Regression (number)
# ============================================================

_NUM_RE = re.compile(r"-?\d+\.?\d*")


def parse_regression_number(output: str,
                            value_range: Optional[Tuple[float, float]] = None,
                            prefer_int: bool = False) -> Dict[str, Any]:
    """返回 {value: float|None, candidates: List[float]}

    优先级：
    1) 结构化字段 "Result: 3.14" / "Score: 0.5" / <ans>3.14</ans>
    2) 文本中第一个合理的数字
    3) 范围限定内取最后一个（很多模型先说"the score is..." 后说数字）
    """
    if not output:
        return {"value": None, "candidates": []}

    candidates = []

    # 1) 结构化字段
    struct = extract_structured_field(output)
    if struct:
        # 字段里取第一个数字
        m = _NUM_RE.search(struct)
        if m:
            try:
                candidates.append(float(m.group(0)))
            except ValueError:
                pass

    # 2) 全文所有数字
    all_nums = [float(m.group(0)) for m in _NUM_RE.finditer(output)]
    candidates.extend(all_nums)

    if not candidates:
        return {"value": None, "candidates": []}

    # 3) 过滤范围
    if value_range:
        lo, hi = value_range
        in_range = [n for n in candidates if lo <= n <= hi]
        if in_range:
            # 优先取结构化字段中的（在范围里）；否则取最后一个（通常"score is X"中 X 在尾部）
            if struct and candidates and value_range[0] <= candidates[0] <= value_range[1]:
                return {"value": candidates[0], "candidates": candidates}
            return {"value": in_range[-1], "candidates": candidates}

    # 没范围限定：返回结构化字段中的第一个数字，或文本中第一个
    if struct and candidates:
        return {"value": candidates[0], "candidates": candidates}
    return {"value": candidates[0], "candidates": candidates}


# ============================================================
# 5) EC numbers (FunctionEC)
# ============================================================

_EC_RE = re.compile(r"(\d+\.\d+\.\d+\.\-?\d*)")
_EC_PREFIXED_RE = re.compile(r"(?:^|\W)EC\s*(\d+\.\d+\.\d+\.\-?\d*)", re.IGNORECASE)


def parse_ec_numbers(output: str) -> Dict[str, Any]:
    """返回 {ecs: List[str]}"""
    if not output:
        return {"ecs": []}

    # 优先结构化字段
    struct = extract_structured_field(output)
    search = struct if struct else output

    ecs = []
    # 优先匹配带 EC 前缀的（更精确）
    for m in _EC_PREFIXED_RE.finditer(search):
        ecs.append(m.group(1))
    # 然后补充不带前缀的（去重保序）
    seen = set(ecs)
    for m in _EC_RE.finditer(search):
        ec = m.group(1)
        if ec not in seen:
            ecs.append(ec)
            seen.add(ec)

    return {"ecs": ecs}


# ============================================================
# 6) Enhancer Activity: {'hk': ..., 'dev': ...}
# ============================================================

def _extract_two_numbers(output: str) -> Tuple[Optional[float], Optional[float]]:
    """从文本中提取两个数字：优先结构化字段(hk/dev 标签)，否则前两个数字"""
    if not output:
        return None, None

    # 优先找 "hk=...", "dev=...", "HK: ...", "Dev: ..." 这种标签
    hk_pat = re.compile(r"\b(?:hk|housekeeping)\s*[:=]\s*(-?\d+\.?\d*)", re.IGNORECASE)
    dev_pat = re.compile(r"\b(?:dev|developmental)\s*[:=]\s*(-?\d+\.?\d*)", re.IGNORECASE)
    hk_m = hk_pat.search(output)
    dev_m = dev_pat.search(output)
    if hk_m and dev_m:
        try:
            return float(hk_m.group(1)), float(dev_m.group(1))
        except ValueError:
            pass

    # 退到结构化字段
    struct = extract_structured_field(output)
    search = struct if struct else output
    nums = _NUM_RE.findall(search)
    if len(nums) >= 2:
        try:
            return float(nums[0]), float(nums[1])
        except ValueError:
            pass
    return None, None


def parse_enhancer_activity(output: str) -> Dict[str, Any]:
    hk, dev = _extract_two_numbers(output)
    return {"hk": hk, "dev": dev}


# ============================================================
# 7) Programmable Switches: {'ON': ..., 'OFF': ..., 'ON_OFF': ...}
# ============================================================

def parse_programmable_switch(output: str) -> Dict[str, Any]:
    if not output:
        return {"ON": None, "OFF": None, "ON_OFF": None}

    # 优先标签化
    on_pat = re.compile(r"\bON(?:\s*state)?\s*[:=]?\s*(?:is\s+|expected\s+to\s+be\s+|of\s+)?(-?\d+\.?\d*)", re.IGNORECASE)
    off_pat = re.compile(r"\bOFF(?:\s*state)?\s*[:=]?\s*(?:is\s+|expected\s+to\s+be\s+|of\s+)?(-?\d+\.?\d*)", re.IGNORECASE)
    ratio_pat = re.compile(r"\bON[\s_/\\]*OFF(?:\s+ratio)?\s*[:=]?\s*(?:is\s+|of\s+)?(-?\d+\.?\d*)", re.IGNORECASE)

    on_v = off_v = ratio_v = None
    m = on_pat.search(output)
    if m:
        try: on_v = float(m.group(1))
        except: pass
    m = off_pat.search(output)
    if m:
        try: off_v = float(m.group(1))
        except: pass
    m = ratio_pat.search(output)
    if m:
        try: ratio_v = float(m.group(1))
        except: pass

    # 如果标签都没匹配到，从结构化字段或全文取前 3 个数字
    if on_v is None and off_v is None and ratio_v is None:
        struct = extract_structured_field(output)
        search = struct if struct else output
        nums = _NUM_RE.findall(search)
        if len(nums) >= 3:
            try:
                on_v = float(nums[0])
                off_v = float(nums[1])
                ratio_v = float(nums[2])
            except ValueError:
                pass

    return {"ON": on_v, "OFF": off_v, "ON_OFF": ratio_v}


# ============================================================
# 单元自测
# ============================================================

if __name__ == "__main__":
    tests = [
        # (output, expected_class_for_binary_or_label_for_mc)
        # Binary
        ("Yes, the sequence contains a core promoter region.", 1),
        ("A core promoter region has not been located within the given DNA.", 0),
        ("The RNA sequence contains motifs known to facilitate protein binding.", 1),
        ("The data does not point towards a specific type of RNA-protein interaction.", 0),
        ("Answer: positive", 1),
        ("Classification: negative.", 0),
        ("My answer is yes.", 1),
        ("Certainly not, the DNA segment does not contain motifs.", 0),
        ("We cannot confirm promoter-enhancer interaction based on the provided sequence data.", 0),
        ("Upon examination, I can confirm that the RNA piece contains interaction with the protein.", 1),
        # 新增 Reason + Answer 格式测试
        ("<reason>\nThe RNA contains AU-rich elements matching the protein RRM domain.\n</reason>\n<ans>\npositive\n</ans>", 1),
        ("<reason>\nNo interaction is predicted based on the sequence composition.\n</reason>\n<ans>\nnegative\n</ans>", 0),
        # Multi-class
        ("This RNA sequence represents a component that likely plays a crucial role in gene regulation, particularly through its classification as a 'leader' RNA.", "leader"),
        ("The sequence is most consistent with the IRES RNA family classification.", "IRES"),
        # Reason+Answer 多分类
        ("<reason>\nThe RNA folds into a structured ribosome-binding motif typical of IRES.\n</reason>\n<ans>\nIRES\n</ans>", "IRES"),
        # Modification
        ("The sequence is linked to the following RNA modifications: m6A.", ["m6A"]),
        ("The sequence is linked to no modifications.", ["none"]),
        # Reason+Answer Modification
        ("<reason>\nStandard modification m6A is detected.\n</reason>\n<ans>\nm6A\n</ans>", ["m6A"]),
        # Regression
        ("The heat stability rating for this amino acid sequence is 51.09.", 51.09),
        ("Result: 3.14", 3.14),
        ("This protein has a thermal resistance score of 47.5.", 47.5),
        # Reason+Answer Regression
        ("<reason>\nBased on the AA composition, melting temp is around 51.\n</reason>\n<ans>\n51.09\n</ans>", 51.09),
        # EC
        ("The reactions it facilitates are identified by EC number EC2.4.1.-.", ["2.4.1.-"]),
        # Reason+Answer EC
        ("<reason>\nBelongs to glycosyltransferase family.\n</reason>\n<ans>\nEC2.4.1.-\n</ans>", ["2.4.1.-"]),
        # enhancer
        ("The enhancer activity prediction tool returns: HK = -0.61, Dev = -0.43", (-0.61, -0.43)),
        # Reason+Answer enhancer (hk=..., dev=...)
        ("<reason>\nBoth HK and dev activities computed.\n</reason>\n<ans>\nhk=-0.61, dev=-0.43\n</ans>", (-0.61, -0.43)),
    ]

    n_ok = 0
    for output, expected in tests:
        if expected in (1, 0):  # binary
            got = parse_binary_classification(output)['class']
            ok = got == expected
            print(f"[binary] got={got} expected={expected} {'OK' if ok else 'FAIL'} | {output[:60]!r}")
        elif expected in ["leader", "IRES"]:  # multiclass
            got = parse_multiclass_classification(output, RNA_CLASSES)['class']
            ok = got == expected
            print(f"[mc] got={got} expected={expected} {'OK' if ok else 'FAIL'} | {output[:60]!r}")
        elif isinstance(expected, list):  # mod or ec
            if "none" in expected:
                got = parse_modification_multilabel(output)['labels']
            else:
                # EC or mod
                if "." in expected[0]:
                    got = parse_ec_numbers(output)['ecs']
                else:
                    got = parse_modification_multilabel(output)['labels']
            ok = got == expected
            print(f"[list] got={got} expected={expected} {'OK' if ok else 'FAIL'} | {output[:60]!r}")
        elif isinstance(expected, tuple):  # enhancer
            got = parse_enhancer_activity(output)
            ok = got['hk'] == expected[0] and got['dev'] == expected[1]
            print(f"[enh] got={got} expected={expected} {'OK' if ok else 'FAIL'} | {output[:60]!r}")
        else:  # number
            got = parse_regression_number(output)['value']
            ok = got is not None and abs(got - expected) < 1e-3
            print(f"[num] got={got} expected={expected} {'OK' if ok else 'FAIL'} | {output[:60]!r}")
        n_ok += int(ok)

    print(f"\n=== {n_ok}/{len(tests)} passed ===")
