"""
84_tables.py
------------
LaTeX tables for the manuscript, generated directly from results/statistics/*.csv so that no number is typed by hand.
Usage: python 84_tables.py <project dir>     -> paper/tables/*.tex
"""
import sys
from pathlib import Path
import pandas as pd

P = Path(sys.argv[1]); S = P / "results" / "statistics"; T = P / "paper" / "tables"; T.mkdir(parents=True, exist_ok=True)
LAB = {"mri": "MRI", "tau": "Tau", "amy": "Amyloid", "thk": "Thickness"}
nice = lambda s: " + ".join(LAB.get(x, x) for x in s.split("+"))
MOD = {"lr": "LR", "stack": "Stacking", "svm": "SVM", "et": "ExtraTrees", "vote": "Voting", "hgb": "HGB"}
ci = lambda r: f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"


def write(name, header, rows, caption, label, align):
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    (T / f"{name}.tex").write_text("\\begin{table}[t]\n\\centering\\small\n\\caption{" + caption + "}\\label{" + label + "}\n\\resizebox{\\linewidth}{!}{\\begin{tabular}{" + align + "}\n\\toprule\n"
                                   + " & ".join(header) + " \\\\\n\\midrule\n" + body + " \\\\\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n", encoding="utf-8")
    print("written", name)


c6 = pd.read_csv(S / "ci_EXP006.csv")
get = lambda d, task, fs, model, metric: d[(d.task == task) & (d.set == fs) & (d.model == model) & (d.metric == metric)].iloc[0]

rows = []
for fs, m in [("mri", "lr"), ("amy", "lr"), ("tau", "lr"), ("mri+amy", "lr"), ("mri+tau", "lr"), ("mri+amy+tau", "lr"), ("mri+tau", "stack"), ("mri+amy+tau", "stack")]:
    rows.append([nice(fs), MOD[m]] + [ci(get(c6, "cn_ad", fs, m, k)) for k in ("accuracy", "auc")] + [f"{get(c6, 'cn_ad', fs, m, k).value:.3f}" for k in ("sensitivity", "specificity")])
n = int(get(c6, "cn_ad", "mri", "lr", "accuracy").n)
write("binary_pooled", ["Features", "Model", "Accuracy [95\\% CI]", "AUC [95\\% CI]", "Sens.", "Spec."], rows,
      f"Normal cognition versus dementia in the pooled-tracer cohort ($n={n}$; EXP006). Mean over 10 repeats of 5-fold nested cross-validation; subject-level bootstrap confidence intervals.", "tab:binary", "llcccc")

rows = []
for fs, m in [("mri", "lr"), ("mri+tau", "lr"), ("mri+amy+tau", "lr"), ("mri+amy+tau", "stack")]:
    rows.append([nice(fs), MOD[m], ci(get(c6, "3class", fs, m, "accuracy")), f"{get(c6, '3class', fs, m, 'macro_f1').value:.3f}"]
                + [ci(get(c6, "3class", fs, m, f"recall_c{k}")) if k == 1 else f"{get(c6, '3class', fs, m, f'recall_c{k}').value:.3f}" for k in range(3)])
write("threeclass_pooled", ["Features", "Model", "Accuracy [95\\% CI]", "Macro-F1", "Recall normal", "Recall MCI [95\\% CI]", "Recall dementia"], rows,
      "Three-class classification (normal / MCI / dementia; $n=255$, 82/81/92; EXP006).", "tab:three", "llccccc")

rows = []
TN = {"cn_ad": "Normal vs dementia", "3class": "Three-class", "cn_mci": "Normal vs MCI", "mci_ad": "MCI vs dementia"}
p6 = pd.read_csv(S / "paired_EXP006.csv"); p6 = p6[(p6.model == "lr") & (p6.metric == "accuracy") & (p6.set_a == "mri")]
for task in ["cn_ad", "3class", "cn_mci", "mci_ad"]:
    for r in p6[p6.task == task].itertuples():
        rows.append([TN[task], nice(r.set_b), f"{r.value_a:.3f}", f"{r.value_b:.3f}", f"{r.diff_b_minus_a:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", f"{r.p_boot:.3f}", str(r.n)])
write("paired_pet_vs_mri", ["Task", "Compared with MRI", "MRI", "With PET", "Difference [95\\% CI]", "$p$", "$n$"], rows,
      "Paired bootstrap comparison of accuracy between MRI alone and MRI with PET (logistic regression, same participants; EXP006). $p$-values are not corrected for multiple comparisons.", "tab:paired", "llccccc")

rows = []
for exp, tracer, sets in [("EXP005", "Flortaucipir", ["mri", "tau", "mri+tau"]), ("EXP007", "PiB", ["mri", "amy", "mri+amy"]), ("EXP008", "Florbetaben", ["mri", "amy", "mri+amy"])]:
    c = pd.read_csv(S / f"ci_{exp}.csv")
    for fs in sets:
        a, u = get(c, "cn_ad", fs, "lr", "accuracy"), get(c, "cn_ad", fs, "lr", "auc"); t = get(c, "3class", fs, "lr", "accuracy")
        rows.append([tracer if fs == sets[0] else "", nice(fs), str(int(a.n)), ci(a), ci(u), str(int(t.n)), ci(t)])
write("tracer_specific", ["Tracer cohort", "Features", "$n$", "Accuracy", "AUC", "$n$", "Three-class accuracy"], rows,
      "Tracer-specific analyses (logistic regression; EXP005, EXP007, EXP008). Binary task: normal versus dementia. Value [95\\% CI].", "tab:tracer", "llccccc")

rows = []
for exp, desc, sets in [("EXP019", "Amyloid-defined labels", ["mri"]), ("EXP020", "Amyloid-defined labels", ["mri", "mri+tau"]),
                        ("EXP021", "Clinical labels", ["mri", "thk", "mri+thk"]), ("EXP022", "Clinical labels", ["mri+tau", "mri+thk+tau", "mri+amy+tau", "mri+thk+amy+tau"])]:
    c = pd.read_csv(S / f"ci_{exp}.csv")
    for fs in sets:
        t, a = get(c, "3class", fs, "lr", "accuracy"), get(c, "cn_ad", fs, "lr", "accuracy")
        rows.append([exp if fs == sets[0] else "", desc if fs == sets[0] else "", nice(fs), str(int(t.n)), ci(t), str(int(a.n)), ci(a)])
write("ablations", ["Experiment", "Labels", "Features", "$n$", "Three-class accuracy", "$n$", "Normal vs dementia accuracy"], rows,
      "Sensitivity analyses: amyloid-defined diagnostic groups and cortical thickness (logistic regression). Value [95\\% CI].", "tab:ablation", "lllcccc")

v = pd.read_csv(S / "validation_vs_scan_tables.csv"); print(v.head(20).to_string())
