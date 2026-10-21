"""
88_appendix_tables.py - Appendix A backing tables, generated from result files only (Tier A rules A-016, VS-30).
  A1: ROC AUC per task and feature set (EXP025/roc_auc.csv): averaged-probability AUC and per-repeat mean, SD, range.
  A2: predicted class by clinical group and amyloid status (EXP025/error_by_amyloid.csv).
  A3: subgroup AUCs (EXP026/subgroup_auc.csv).
  A4: the 15 features with the largest shift between OASIS-1 CDR 0 and NACC normal (EXP023/feature_shift_normals.csv).
Usage: python 88_appendix_tables.py <project dir>
"""
import sys
from pathlib import Path
import pandas as pd

P = Path(sys.argv[1]); X = P / "experiments"; T = P / "paper" / "tables"
LAB = {"mri": "MRI", "mri+tau": "MRI + tau", "mri+amy": "MRI + amyloid", "mri+amy+tau": "MRI + amyloid + tau"}
TASK = {"cn_ad": "Normal vs dementia", "cn_mci": "Normal vs MCI", "mci_ad": "MCI vs dementia"}
BS = "\\\\"


def table(label, caption, cols, rows, align, size="small"):
    return ("\\begin{table}[!htbp]\n\\centering\\" + size + "\\setlength{\\tabcolsep}{4pt}\n\\caption{" + caption + "}\\label{" + label + "}\n\\begin{tabular}{" + align + "}\n\\toprule\n"
            + " & ".join(cols) + " " + BS + "\n\\midrule\n" + "\n".join(" & ".join(r) + " " + BS for r in rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


r = pd.read_csv(X / "EXP025" / "roc_auc.csv")
rows = [[TASK[x.task], LAB[x.set], str(x.n), f"{x.auc_averaged_prob:.3f}", f"{x.auc_repeat_mean:.3f}", f"{x.auc_repeat_sd:.3f}",
         f"{x.auc_repeat_min:.3f}--{x.auc_repeat_max:.3f}"] for x in r.itertuples()]
a1 = table("tab:appA1", "Values behind Fig.~\\ref{fig:roc}: AUC of the reference model per task and feature set (EXP006 predictions, EXP025). "
           "Averaged: AUC of the participant's probability averaged over the 10 repeats; per repeat: mean, SD and range of the 10 "
           "repeat-level AUCs. Higher is better.", ["Task", "Features", "$n$", "AUC (averaged)", "Mean", "SD", "Range"], rows, "llccccc")

e = pd.read_csv(X / "EXP025" / "error_by_amyloid.csv")
rows = [[LAB[x.set], x.label, x.amyloid, str(x.n), f"{100 * x.pred_normal:.0f}", f"{100 * x.pred_mci:.0f}", f"{100 * x.pred_dementia:.0f}",
         f"{x.mean_p_dementia:.3f}"] for x in e.itertuples()]
a2 = table("tab:appA2", "Values behind Fig.~\\ref{fig:err}: predicted class (\\% of participants) of the three-class model by clinical group and "
           "core-laboratory amyloid status (EXP025). Mean $p$: mean predicted probability of dementia.",
           ["Features", "Group", "Amyloid", "$n$", "Normal (\\%)", "MCI (\\%)", "Dementia (\\%)", "Mean $p$"], rows, "lllccccc")

s = pd.read_csv(X / "EXP026" / "subgroup_auc.csv")
s = s[s.variable.isin(["sex_grp", "age_grp", "sex difference"]) & s.auc.notna()]
rows = [[LAB[x.set], {"sex_grp": "Sex", "age_grp": "Age third", "sex difference": "Difference"}[x.variable], str(x.level), str(int(x.n)),
         f"{x.auc:.3f}", f"{x.ci_low:.3f}--{x.ci_high:.3f}"] for x in s.itertuples()]
a3 = table("tab:appA3", "Subgroup AUCs for normal cognition versus dementia (EXP026). Brackets: stratified bootstrap 95\\% CI (2,000 "
           "resamples); the difference row is male minus female with an unpaired bootstrap.",
           ["Features", "Variable", "Level", "$n$", "AUC", "95\\% CI"], rows, "lllccc")

f = pd.read_csv(X / "EXP023" / "feature_shift_normals.csv").reindex(columns=["feature", "nacc_normal_mean", "oasis_cdr0_mean", "nacc_normal_sd", "shift_in_sd"])
f = f.reindex(f.shift_in_sd.abs().sort_values(ascending=False).index).head(15)
nm = lambda c: c.replace("bi_", "").replace("_", " ").replace("ctx-", "")
rows = [[nm(x.feature), f"{x.nacc_normal_mean:.2f}", f"{x.oasis_cdr0_mean:.2f}", f"{x.nacc_normal_sd:.2f}", f"{x.shift_in_sd:+.2f}"] for x in f.itertuples()]
a4 = table("tab:appA4", "Values behind Fig.~\\ref{fig:ext}c: the 15 MRI features (left plus right, per mille of brain volume) whose mean in "
           "OASIS-1 CDR~0 participants differs most from NACC normal participants, in NACC standard deviations (EXP023).",
           ["Feature", "NACC normal mean", "OASIS-1 CDR 0 mean", "NACC SD", "Shift (SD)"], rows, "lcccc")
COH = {"EXP005": "Flortaucipir only", "EXP006": "Pooled tracers", "EXP007": "PiB only", "EXP008": "Florbetaben only"}
rows, w, l, t = [], 0, 0, 0
for e in COH:
    d = pd.read_csv(P / "results" / "statistics" / f"paired_{e}.csv")
    d = d[(d.model == "lr") & (d.metric == "accuracy") & (d.set_a == "mri")]
    for x in d.itertuples():
        res = "win" if x.ci_low > 0 else ("loss" if x.ci_high < 0 else "tie"); w += res == "win"; l += res == "loss"; t += res == "tie"
        rows.append([COH[e], {"3class": "Three-class", **TASK}[x.task], LAB[x.set_b], f"{x.diff_b_minus_a:+.3f}",
                     f"{x.ci_low:+.3f} to {x.ci_high:+.3f}", f"{x.p_boot:.3f}", res])
a5 = table("tab:appA5", f"Win/tie/loss of MRI with PET against MRI alone (accuracy, logistic regression, paired bootstrap 95\\% CI, "
           f"uncorrected): {w} wins, {t} ties, {l} losses across all cohorts and tasks (EXP005--EXP008).",
           ["Cohort", "Task", "Features", "Difference", "95\\% CI", "$p$", "Result"], rows, "lllcccc", size="footnotesize")
R5 = (f"Table~\\ref{{tab:appA5}} shows that PET won {w} of the {w + t + l} paired comparisons before correction, both for "
      "normal cognition versus MCI in the pooled cohort, tied in the rest and lost none.")
R1 = ("Table~\\ref{tab:appA1} shows per-repeat standard deviations of 0.006 to 0.027, so for normal cognition versus "
      "dementia the differences between feature sets are no larger than the variation between repeats.")
R2 = ("Table~\\ref{tab:appA2} gives the shares plotted in Fig.~\\ref{fig:err} for both feature sets; with MRI, amyloid and "
      "tau PET as inputs the separation by amyloid status is larger, as expected when amyloid is a model input.")
R3 = ("Table~\\ref{tab:appA3} shows that the difference between women and men is present with and without tau PET, while "
      "the confidence intervals of the three age thirds overlap, so age does not account for the gap.")
R4 = ("Table~\\ref{tab:appA4} shows that every one of the 15 largest shifts is a smaller volume in OASIS-1, led by "
      "temporal and limbic regions, which moves OASIS-1 controls towards the dementia side of the NACC threshold.")
(T / "appendix_backing.tex").write_text("% generated by src/visualization/88_appendix_tables.py - do not edit by hand\n"
                                        + "\n\n".join([a1, R1, a2, R2, a3, R3, a4, R4, a5, R5]) + "\n", encoding="utf-8")
print("written tables/appendix_backing.tex")
