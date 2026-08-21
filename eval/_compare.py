# -*- coding: utf-8 -*-
"""对比 OLD (evaluate.py) vs NEW (evaluate_v2.py) 在 4 个模型上的 metrics"""
import json, collections

# 旧 metrics（metrics_result_old）
OLD = {}
for m in ['base','s2_only','s1_s2','stage3']:
    OLD[m] = json.load(open(f'metrics_result/metrics_result_{m}_all_omics_old.json'))

# 新 metrics（parser_v2）
NEW = {}
for m in ['base','s2_only','s1_s2','stage3']:
    NEW[m] = json.load(open(f'metrics_result/metrics_result_{m}_all_omics_v2.json'))

# 任务 -> (omics, task_name_in_new)
TASK_MAP = {
    # omics = 'Multi'
    'Multi_ncRNAProteinInter': ('Multi','ncRNAProteinInter','MCC'),
    'Multi_promoter_enhancer_interaction': ('Multi','promoter_enhancer_interaction','MCC'),
    'Multi_AntibodyAntigen': ('Multi','AntibodyAntigen','MCC'),
    'Multi_sirnaEfficiency': ('Multi','sirnaEfficiency','mixed_score'),
    # 'Protein'
    'Protein_FunctionEC': ('Protein','FunctionEC','Fmax'),
    'Protein_Solubility': ('Protein','Solubility','Acc'),
    'Protein_Fluorescence': ('Protein','Fluorescence','spearman'),
    'Protein_Thermostability': ('Protein','Thermostability','spearman'),
    'Protein_Stability': ('Protein','Stability','spearman'),
    # 'DNA'
    'DNA_enhancer_activity_hk': ('DNA','enhancer_activity','PCC_hk'),
    'DNA_enhancer_activity_dev': ('DNA','enhancer_activity','PCC_dev'),
    'DNA_tf_h': ('DNA','tf_h','MCC'),
    'DNA_tf_m': ('DNA','tf_m','MCC'),
    'DNA_pd': ('DNA','pd','MCC'),
    'DNA_cpd': ('DNA','cpd','MCC'),
    'DNA_emp': ('DNA','emp','MCC'),
    # 'RNA'
    'RNA_ProgrammableRNASwitches': ('RNA','ProgrammableRNASwitches','R2'),
    'RNA_NoncodingRNAFamily': ('RNA','NoncodingRNAFamily','Acc'),
    'RNA CRISPROnTarget'.strip(): ('RNA','CRISPROnTarget','spearman'),
    'RNA_Isoform': ('RNA','Isoform','R2'),
    'RNA_Modification': ('RNA','Modification','AUC'),
    'RNA_MeanRibosomeLoading': ('RNA','MeanRibosomeLoading','R2'),
}

def get_metric(d, omics, task, metric):
    """d 是 metrics_result json 结构"""
    o = d.get(omics, {})
    t = o.get(task, {})
    if metric.startswith('PCC_'):
        sub = metric.split('_')[1]
        return t.get('PCC', {}).get(f'{sub}_PCC')
    return t.get(metric)

print(f"{'task':<40} {'metric':<12}", end='')
for m in ['base','s2_only','s1_s2','stage3']:
    print(f" {m+'_OLD':>10} {m+'_NEW':>10}", end='')
print()
print('-'*120)
for label, (omics, task, metric) in TASK_MAP.items():
    print(f"{label:<40} {metric:<12}", end='')
    for m in ['base','s2_only','s1_s2','stage3']:
        ov = get_metric(OLD[m], omics, task, metric)
        nv = get_metric(NEW[m], omics, task, metric)
        ov_s = f"{ov:>10.4f}" if isinstance(ov,(int,float)) else f"{'-':>10}"
        nv_s = f"{nv:>10.4f}" if isinstance(nv,(int,float)) else f"{'-':>10}"
        print(f" {ov_s} {nv_s}", end='')
    print()
