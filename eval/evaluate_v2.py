# -*- coding: utf-8 -*-
"""
evaluate_v2.py —— 替换 evaluate.py 的鲁棒评测脚本
====================================================
- 复用 evaluate.py 的指标计算（MCC / spearman / R2 / mixed_score / Acc / Fmax / AUC）
- 用 parser_v2 替换脆弱的关键词/正则提取
- 输出与 evaluate.py 完全兼容的 metrics_result_{model}_{omics}.json
"""
import sys, os, json, re, argparse, logging, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, r2_score, roc_auc_score,
    precision_score, recall_score, mean_absolute_error,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parser_v2 import (
    parse_binary_classification,
    parse_multiclass_classification,
    parse_modification_multilabel,
    parse_regression_number,
    parse_ec_numbers,
    parse_enhancer_activity,
    parse_programmable_switch,
    RNA_CLASSES,
    MODIFICATION_CLASSES,
)

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', required=True)
parser.add_argument('--OMICS', required=True)
parser.add_argument('--input_file_path', required=True)
parser.add_argument('--use_old_parser', action='store_true',
                    help='回退到 evaluate.py 的旧 parser（用于对比）')
parser.add_argument('--save_processed_dir', default=None,
                    help='保存每个任务的 processed data 路径（默认 None 不保存）')
parser.add_argument('--out_suffix', default='',
                    help='输出文件后缀（默认空，例如 _v2/_old 用于对比）')
args = parser.parse_args()

# ============================================================
# 日志
# ============================================================
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
log_dir = Path("logging")
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"metrics_{args.model_name}_{args.OMICS}_{timestamp}.log"
logging.basicConfig(
    filename=str(log_path), level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w',
)
logger = logging.getLogger(__name__)

# ============================================================
# 注册表 / EC labels
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
with open(_SCRIPT_DIR / "ec_labels.json") as f:
    ec_labels = json.load(f)
with open(_SCRIPT_DIR / "register_tasks.json") as f:
    task_type_data = json.load(f)


def save_processed_data(task_name, task_processed_data):
    if not args.save_processed_dir:
        return
    dir_path = Path(args.save_processed_dir) / args.model_name
    dir_path.mkdir(parents=True, exist_ok=True)
    fp = dir_path / f"{task_name}_processed_data.json"
    with open(fp, "w") as f:
        json.dump(task_processed_data, f, indent=4)
    logger.info(f"Task {task_name} processed data saved in {fp}")


# ============================================================
# 旧 parser（可选回退）
# ============================================================
def old_extract_numeric(text):
    return [float(m) for m in re.findall(r'(-?\d+\.?\d*)', str(text))]

def old_classify_by_keywords(text):
    pos_kw = ['yes']
    neg_kw = ['no','absence','not found','not detected','not associated','not inferred',
              'not linked','does not indicate','no evidence','not predicted','absent']
    unk_kw = ["don't know",'unknown','unsure','uncertain','not applicable']
    t = text.lower()
    if any(k in t for k in pos_kw): return 1
    if any(k in t for k in neg_kw): return 0
    if any(k in t for k in unk_kw): return 'dont_know'
    return None  # fallback returns (0, 0.0) in original

def old_extract_rna_family(text):
    for c in RNA_CLASSES:
        if c in text: return c
    return None

def old_extract_modifications(text):
    found = []
    for m in MODIFICATION_CLASSES:
        if re.search(rf'\b{m}\b', text):
            found.append(m)
    return found


# ============================================================
# 数值解析（旧/新）
# ============================================================
def parse_regression_value(out, value_range=None):
    if args.use_old_parser:
        nums = old_extract_numeric(out)
        return nums[0] if nums else None
    r = parse_regression_number(out, value_range=value_range)
    return r['value']


# ============================================================
# 任务处理函数（保留 evaluate.py 同名接口 + 内部用 v2 parser）
# ============================================================
def process_regression_task(task_name, task_entries, value_range=None):
    result_values, label_values = [], []
    task_processed_data = []
    for entry in task_entries:
        try:
            label = float(entry["label"])
        except (TypeError, ValueError):
            logger.warning(f"[{task_name}] label not numeric: {entry['label']!r}, skipping")
            continue
        val = parse_regression_value(entry["model_output"], value_range=value_range)
        result_values.append(val if val is not None else np.inf)
        label_values.append(label)
        task_processed_data.append({
            "input": entry["input"],
            "label": label,
            "processed_model_output": val if val is not None else np.inf,
            "original_model_output": entry["model_output"],
        })
    save_processed_data(task_name, task_processed_data)
    return task_processed_data, label_values, result_values


def process_binary_classification_task(task_name, task_entries):
    label_classes, result_classes = [], []
    task_processed_data = []
    for entry in task_entries:
        lab = entry["label"]
        if isinstance(lab, str) and lab.lower() in ("positive","negative"):
            label_class = 1 if lab.lower() == "positive" else 0
        else:
            # 某些任务（如 Solubility）的 label 也可能是 positive/negative
            try:
                label_class = int(lab)
            except:
                logger.warning(f"[{task_name}] unexpected label: {lab!r}")
                label_class = None

        if args.use_old_parser:
            if entry["model_output"] is None:
                result_class = 1 - (label_class or 0)
                score = 0
            else:
                c = old_classify_by_keywords(entry["model_output"])
                score = 0
                if c == "dont_know":
                    result_class = 1 - (label_class or 0)
                elif c is None:
                    result_class = 0  # fallback
                    score = 0.0
                else:
                    result_class = int(c)
        else:
            r = parse_binary_classification(entry["model_output"] or "")
            result_class = r['class']
            score = r['confidence']
            if result_class is None or result_class == 'dont_know':
                # 不算预测，记为错（按 1 - label_class）
                result_class = 1 - (label_class or 0)
                score = 0.0

        label_classes.append(label_class)
        result_classes.append(result_class)
        task_processed_data.append({
            "input": entry["input"],
            "original_label": entry["label"],
            "processed_label": label_class,
            "original_model_output": entry["model_output"],
            "processed_model_output": result_class,
            "score": f"{score:.2f}" if score else "N/A",
        })
    save_processed_data(task_name, task_processed_data)
    return task_processed_data, label_classes, result_classes


def process_R2_for_ProgrammableRNASwitches_task(task_name, task_entries):
    on_v, off_v, ratio_v = [], [], []
    on_l, off_l, ratio_l = [], [], []
    proc = []
    for e in task_entries:
        lab = e["label"]
        try:
            ol = float(lab["ON"]); fl = float(lab["OFF"]); rl = float(lab["ON_OFF"])
        except (TypeError, KeyError):
            continue
        if args.use_old_parser:
            nums = old_extract_numeric(e["model_output"])
            if len(nums) >= 3:
                ov, fv, rv = nums[0], nums[1], nums[2]
            else:
                ov = fv = rv = np.nan
        else:
            r = parse_programmable_switch(e["model_output"] or "")
            ov = r['ON'] if r['ON'] is not None else np.nan
            fv = r['OFF'] if r['OFF'] is not None else np.nan
            rv = r['ON_OFF'] if r['ON_OFF'] is not None else np.nan
        on_v.append(ov); off_v.append(fv); ratio_v.append(rv)
        on_l.append(ol); off_l.append(fl); ratio_l.append(rl)
        proc.append({
            "input": e["input"], "label": e["label"],
            "processed_model_output": {"ON": ov, "OFF": fv, "ON_Off": rv},
            "original_model_output": e["model_output"],
        })
    save_processed_data(task_name, proc)
    return on_v, off_v, ratio_v, on_l, off_l, ratio_l


def process_PCC_for_enhancer_activity_task(task_name, task_entries):
    hk_v, dev_v = [], []
    hk_l, dev_l = [], []
    proc = []
    for e in task_entries:
        lab = e["label"]
        try:
            hl = float(lab["hk"]); dl = float(lab["dev"])
        except (TypeError, KeyError):
            continue
        if args.use_old_parser:
            nums = old_extract_numeric(e["model_output"])
            hv = nums[0] if len(nums) >= 1 else np.nan
            dv = nums[1] if len(nums) >= 2 else np.nan
        else:
            r = parse_enhancer_activity(e["model_output"] or "")
            hv = r['hk'] if r['hk'] is not None else np.nan
            dv = r['dev'] if r['dev'] is not None else np.nan
        hk_v.append(hv); dev_v.append(dv)
        hk_l.append(hl); dev_l.append(dl)
        proc.append({
            "input": e["input"], "label": e["label"],
            "processed_model_output": {"hk": hv, "dev": dv},
            "original_model_output": e["model_output"],
        })
    save_processed_data(task_name, proc)
    return hk_v, dev_v, hk_l, dev_l


def process_Acc_for_NoncodingRNAFamily_task(task_name, task_entries):
    correct, total = 0, 0
    proc = []
    for e in task_entries:
        lab = e["label"]
        if args.use_old_parser:
            pred = old_extract_rna_family(e["model_output"] or "")
        else:
            r = parse_multiclass_classification(e["model_output"] or "", RNA_CLASSES)
            pred = r['class']
        if pred is None:
            logger.warning(f"No RNA family extracted: {(e['model_output'] or '')[:100]!r}")
        if pred == lab:
            correct += 1
        total += 1
        proc.append({
            "input": e["input"], "label": lab,
            "processed_model_output": pred,
            "original_model_output": e["model_output"],
        })
    save_processed_data(task_name, proc)
    return correct, total


def process_AUC_for_Modification_task(task_name, task_entries, classes):
    y_true, y_pred = [], []
    proc = []
    for e in task_entries:
        if args.use_old_parser:
            pred_mods = old_extract_modifications(e["model_output"] or "")
            true_mods = str(e["label"]).split(',')
            if pred_mods == [] and true_mods == ['none']:
                c = old_classify_by_keywords(e["model_output"] or "")
                if c == 0: pred_mods = ['none']
                elif c == 1: pred_mods = []
                else: pred_mods = []
        else:
            r = parse_modification_multilabel(e["model_output"] or "")
            pred_mods = r['labels']
            true_mods = str(e["label"]).split(',')
        y_true.append(_to_binary(true_mods, classes))
        y_pred.append(_to_binary(pred_mods, classes))
        proc.append({
            "input": e["input"], "label": e["label"],
            "processed_model_output": pred_mods,
            "original_model_output": e["model_output"],
        })
    save_processed_data(task_name, proc)
    return y_true, y_pred


def _to_binary(mods, classes):
    return [1 if c in (mods or []) else 0 for c in classes]


def process_Fmax_for_FunctionEC_task(task_name, task_entries, ec_labels):
    import torch
    preds, labels = [], []
    proc = []
    for e in task_entries:
        label_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', str(e['label']))
        if args.use_old_parser:
            result_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', str(e['model_output']))
        else:
            r = parse_ec_numbers(e['model_output'] or "")
            result_ec = r['ecs']
        if not result_ec:
            logger.warning(f"[{task_name}] EC not found in output: {(e['model_output'] or '')[:100]!r}")
        if not label_ec:
            logger.warning(f"[{task_name}] EC not found in label: {e['label']!r}")
        preds.append(_ec_to_multihot(result_ec, ec_labels))
        labels.append(_ec_to_multihot(label_ec, ec_labels))
        proc.append({
            "input": e["input"], "label": e["label"],
            "processed_label": label_ec,
            "original_model_output": e["model_output"],
            "processed_model_output": result_ec,
        })
    save_processed_data(task_name, proc)
    if preds and labels:
        return torch.stack(preds), torch.stack(labels)
    return None, None


def _ec_to_multihot(ec_list, ec_labels):
    import torch
    v = torch.zeros(len(ec_labels))
    if not ec_list: return v
    for ec in ec_list:
        if ec in ec_labels:
            v[ec_labels.index(ec)] = 1
    return v


def _count_f1_max(pred, target):
    """SaProt Fmax 实现，保留 evaluate.py 同款逻辑"""
    if pred.numel() == 0 or target.numel() == 0:
        return 0.0
    order = pred.argsort(descending=True, dim=1, stable=True)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    all_order = pred.flatten().argsort(descending=True, stable=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    if torch.isnan(all_f1).any():
        return 0.0
    return all_f1.max().item()


# ============================================================
# 指标函数（沿用 evaluate.py）
# ============================================================
def compute_spearman(label_values, result_values):
    a = np.array(result_values, dtype=float)
    b = np.array(label_values, dtype=float)
    inf_mask = np.isinf(a) | ~np.isfinite(a)
    valid = ~inf_mask & np.isfinite(b)
    if valid.sum() == 0: return {"spearman": 0.0}
    sp, _ = spearmanr(a[valid], b[valid])
    total = len(a)
    nv = inf_mask.sum()
    if sp != sp: sp = 0.0
    return {"spearman": sp if nv == 0 else sp * valid.sum() / total}


def compute_R2(label_values, result_values):
    a = np.array(result_values, dtype=float)
    b = np.array(label_values, dtype=float)
    inf_mask = np.isinf(a) | ~np.isfinite(a)
    valid = ~inf_mask & np.isfinite(b)
    if valid.sum() == 0: return {"R2": 0.0}
    try:
        p, _ = pearsonr(a[valid], b[valid]); r2 = p**2
    except Exception:
        r2 = 0.0
    if r2 != r2: r2 = 0.0
    total = len(a); nv = inf_mask.sum()
    return {"R2": r2 if nv == 0 else r2 * valid.sum() / total}


def compute_mixed_score(label_values, result_values, threshold=30, max_value=1e3):
    a = pd.to_numeric(result_values, errors='coerce').astype(float).values
    b = pd.to_numeric(label_values, errors='coerce').astype(float).values
    inf_mask = np.abs(a) > max_value
    valid = ~inf_mask & np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return {"mixed_score": 0.0}
    av = a[valid]; bv = b[valid]
    lb = (bv < threshold).astype(int)
    rb = (av < threshold).astype(int)
    try:
        pr = precision_score(lb, rb, zero_division=0)
        rc = recall_score(lb, rb, zero_division=0)
        f1 = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0.0
    except Exception:
        f1 = 0.0
    try:
        mae = mean_absolute_error(bv, av); mae = min(mae, 100)
    except Exception:
        mae = 100
    mask = (av >= 0) & (av <= threshold)
    if mask.sum() > 0:
        rmae = min(mean_absolute_error(bv[mask], av[mask]), 100)
    else:
        rmae = 100
    ms = (1 - mae/100)*0.5 + (1 - rmae/100)*f1*0.5
    total = len(a); nv = inf_mask.sum()
    return {"mixed_score": ms if nv == 0 else ms * valid.sum() / total}


def compute_MCC(label_classes, result_classes):
    p = []; g = []
    for a,b in zip(result_classes, label_classes):
        if a is None or b is None: continue
        try:
            p.append(int(a)); g.append(int(b))
        except (TypeError, ValueError):
            continue
    if not p: return {"MCC": 0.0}
    if len(set(p)) < 2 or len(set(g)) < 2:
        return {"MCC": 0.0}
    m = matthews_corrcoef(g, p)
    if m != m: m = 0.0
    return {"MCC": m}


def compute_Acc(label_classes, result_classes):
    p = []; g = []
    for a,b in zip(result_classes, label_classes):
        if a is None or b is None: continue
        try:
            p.append(int(a)); g.append(int(b))
        except (TypeError, ValueError):
            continue
    if not p: return {"Acc": 0.0}
    return {"Acc": accuracy_score(g, p)}


# ============================================================
# 任务范围（按 value range 给提示）
# ============================================================
VALUE_RANGES = {
    "Fluorescence": (0, 5),
    "Thermostability": (30, 80),
    "Stability": (-3, 3),
    "sirnaEfficiency": (0, 150),
    "Isoform": (0, 1),
    "MeanRibosomeLoading": (0, 10),
    "CRISPROnTarget": (0, 1),
}


# ============================================================
# Main
# ============================================================
def main():
    # 读 jsonl（兼容 evaluate.py 字段：input/output/label/task；以及 result->model_output）
    valid_lines = []
    with open(args.input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict): continue
            if "result" in d and "model_output" not in d:
                d["model_output"] = d["result"]
            valid_lines.append(d)

    df = pd.DataFrame(valid_lines)
    df.rename(columns={"result": "model_output"}, inplace=True)

    df['task'] = df['task'].replace('rna_protein_interaction', 'ncRNAProteinInter')
    df['task'] = df['task'].replace('antibody_antigen', 'AntibodyAntigen')
    df = df[~df['task'].str.endswith('_all')]
    df['task'] = df['task'].str.replace('tf-h', 'tf_h')
    df['task'] = df['task'].str.replace('tf-m', 'tf_m')
    df = df[df['label'].notna()].reset_index(drop=True)

    data = df.to_dict(orient='records')
    grouped = defaultdict(list)
    for e in data:
        grouped[e['task'].split('-')[0]].append(e)

    logger.info(f"Grouped tasks: {list(grouped.keys())}")
    print(f"Grouped tasks: {list(grouped.keys())}")

    metrics = {}
    for task_name, entries in grouped.items():
        if task_name not in task_type_data:
            logger.warning(f"Task {task_name} not registered; skipping")
            continue
        ttype = task_type_data[task_name]["type"]
        tmet = task_type_data[task_name]["metrics"]
        print(f"Processing {task_name} ({ttype}/{tmet})...")
        try:
            if ttype == "regression":
                rng = VALUE_RANGES.get(task_name)
                _, labels, preds = process_regression_task(task_name, entries, value_range=rng)
                if tmet == "spearman": metrics[task_name] = compute_spearman(labels, preds)
                elif tmet == "R2": metrics[task_name] = compute_R2(labels, preds)
                elif tmet == "mixed_score":
                    metrics[task_name] = compute_mixed_score(labels, preds, threshold=30)
            elif ttype == "binary classification":
                _, labels, preds = process_binary_classification_task(task_name, entries)
                if tmet == "MCC": metrics[task_name] = compute_MCC(labels, preds)
                elif tmet == "Acc": metrics[task_name] = compute_Acc(labels, preds)
            elif ttype == "multilabel regression":
                if task_name == "ProgrammableRNASwitches":
                    on_v, off_v, ratio_v, on_l, off_l, ratio_l = process_R2_for_ProgrammableRNASwitches_task(task_name, entries)
                    def _r2(xs, ys):
                        a = np.array(xs, dtype=float); b = np.array(ys, dtype=float)
                        m = np.isfinite(a) & np.isfinite(b)
                        if m.sum()==0: return 0.0
                        try:
                            p,_=pearsonr(a[m],b[m]); return (p**2) if p==p else 0.0
                        except: return 0.0
                    metrics[task_name] = {"R2": (_r2(on_v,on_l)+_r2(off_v,off_l)+_r2(ratio_v,ratio_l))/3}
                elif task_name == "enhancer_activity":
                    hk_v, dev_v, hk_l, dev_l = process_PCC_for_enhancer_activity_task(task_name, entries)
                    def _pcc(xs, ys):
                        a = np.array(xs, dtype=float); b = np.array(ys, dtype=float)
                        m = np.isfinite(a) & np.isfinite(b)
                        if m.sum()==0: return 0.0
                        try:
                            p,_=pearsonr(a[m],b[m]); return p if p==p else 0.0
                        except: return 0.0
                    metrics[task_name] = {"PCC": {"hk_PCC": _pcc(hk_v,hk_l), "dev_PCC": _pcc(dev_v,dev_l)}}
            elif ttype == "multiclass classification":
                if task_name == "NoncodingRNAFamily":
                    correct, total = process_Acc_for_NoncodingRNAFamily_task(task_name, entries)
                    metrics[task_name] = {"Acc": correct/total if total>0 else 0.0}
            elif ttype == "multilabel classification":
                if task_name == "FunctionEC":
                    p, l = process_Fmax_for_FunctionEC_task(task_name, entries, ec_labels)
                    if p is not None:
                        try: metrics[task_name] = {"Fmax": _count_f1_max(p, l)}
                        except Exception as e:
                            logger.error(f"Fmax error: {e}")
                            metrics[task_name] = {"Fmax": 0.0}
                    else:
                        metrics[task_name] = {"Fmax": 0.0}
                elif task_name == "Modification":
                    y_true, y_pred = process_AUC_for_Modification_task(task_name, entries, MODIFICATION_CLASSES)
                    try:
                        metrics[task_name] = {"AUC": roc_auc_score(y_true, y_pred, average='macro', zero_division=0)}
                    except Exception as e:
                        logger.error(f"AUC error: {e}")
                        metrics[task_name] = {"AUC": 0.0}
        except Exception as e:
            logger.exception(f"Task {task_name} failed")
            print(f"Task {task_name} FAILED: {e}")
            metrics[task_name] = {tmet: 0.0}

        v = metrics[task_name][tmet]
        print(f"  -> {tmet} = {v}")

    # 不缩放，直接输出原始值（与 evaluate.py 兼容：evaluate.py 默认 ×100，
    # 这里通过 --scale_factor 控制，默认 1.0 即不缩放）
    SCALE_FACTOR = float(os.environ.get("SCALE_FACTOR", "1.0"))
    def _maybe_scale(d, dp=4, sf=SCALE_FACTOR):
        for k,v in list(d.items()):
            if isinstance(v, dict):
                _maybe_scale(v, dp, sf)
            elif isinstance(v, (int,float,np.floating,np.integer)):
                try:
                    d[k] = float(round(float(v)*sf, dp))
                except Exception:
                    pass
    out = defaultdict(dict)
    for t, m in metrics.items():
        omics = task_type_data[t]["omics"]
        m_copy = json.loads(json.dumps(m, default=lambda x: float(x) if isinstance(x,(np.floating,np.integer)) else str(x)))
        _maybe_scale(m_copy)
        out[omics][t] = m_copy

    out_dir = Path("metrics_result")
    out_dir.mkdir(exist_ok=True)
    suffix = args.out_suffix or ("_v2" if not args.use_old_parser else "_old")
    out_path = out_dir / f"metrics_result_{args.model_name}_{args.OMICS}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)
    print(f"Metrics saved to {out_path} (scale_factor={SCALE_FACTOR}, parser={'OLD' if args.use_old_parser else 'V2'})")


if __name__ == "__main__":
    main()
