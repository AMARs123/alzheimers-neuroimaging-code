"""
17_compact_vs_full_errors.py  (EXP311; added in the improvement pass of 15 Oct 2026, after the main analyses)
--------------------------------------------------------------------------------------------------------------
Uses stored predictions only (EXP303 nested-CV out-of-fold predictions in NACC; EXP307 frozen-model predictions on
OASIS-1); nothing is refitted.

A. ROC curves of the full (51 features) and compact medial temporal (5 features) models: NACC normal vs dementia and
   normal vs MCI (probabilities averaged over the 10 repeats per participant) and OASIS-1 CDR 0 vs >= 1 and 0 vs 0.5.
B. Do the two models make the same decisions? Spearman correlation of the participants' probabilities and the share of
   participants with the same class at threshold 0.5 (NACC); Brier scores.
C. Error analysis by age (NACC, normal vs dementia, compact model): are false positives older normal participants and
   false negatives younger people with dementia? Mann-Whitney tests of label-visit age (NACCAGE) between correct and
   wrong predictions within each group.

Usage: python 17_compact_vs_full_errors.py <project dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(sys.argv[1]); X = P / "experiments"; E = X / "EXP311"; E.mkdir(exist_ok=True); FIG = P / "figures" / "final"
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
pr = pd.read_csv(X / "EXP303" / "predictions.csv.gz")
meta = pd.read_csv(P / "dataset" / "metadata" / "meta_v2.csv", usecols=["NACCID", "NACCAGE"])
ox = pd.read_csv(X / "EXP307" / "predictions.csv")
SETS = {"all": "All 51 features", "only: medial temporal": "Medial temporal (5)"}
COL = {"all": "#1f5f8b", "only: medial temporal": "#d9534f"}

agg, rows = {}, []
for task in ["cn_ad", "cn_mci"]:
    for s in SETS:
        d = pr[(pr.task == task) & (pr.set == s)]
        g = d.groupby("NACCID").agg(y=("y", "first"), p=("p", "mean")).reset_index()
        g["y"] = (g.y == g.y.max()).astype(int); agg[(task, s)] = g
        rows.append(dict(cohort="NACC", task=task, set=s, n=len(g), auc=roc_auc_score(g.y, g.p), brier=brier_score_loss(g.y, g.p)))
OASIS = {"CDR 0 vs >=1": ("normal_vs_dementia", ox.cdr >= 1), "CDR 0 vs 0.5": ("normal_vs_MCI", ox.cdr == 0.5)}
for name, (tag, pos) in OASIS.items():
    keep = (ox.cdr == 0) | pos; y = pos[keep].astype(int)
    for s, short in [("all", "all"), ("only: medial temporal", "medial_temporal")]:
        p = ox.loc[keep, f"p_{tag}_{short}"]
        rows.append(dict(cohort="OASIS-1", task=name, set=s, n=int(keep.sum()), auc=roc_auc_score(y, p), brier=brier_score_loss(y, p)))
R = pd.DataFrame(rows); R.to_csv(E / "roc_brier.csv", index=False); print(R.round(3).to_string(index=False))

# B. agreement of decisions (NACC)
agree = []
for task in ["cn_ad", "cn_mci"]:
    a, m = agg[(task, "all")], agg[(task, "only: medial temporal")]
    j = a.merge(m, on=["NACCID", "y"], suffixes=("_all", "_mtl"))
    same = ((j.p_all >= .5) == (j.p_mtl >= .5)).mean()
    agree.append(dict(task=task, n=len(j), spearman=spearmanr(j.p_all, j.p_mtl).statistic, same_class=same,
                      acc_all=((j.p_all >= .5) == j.y).mean(), acc_mtl=((j.p_mtl >= .5) == j.y).mean()))
A = pd.DataFrame(agree); A.to_csv(E / "agreement.csv", index=False); print(A.round(3).to_string(index=False))

# C. errors by age (NACC normal vs dementia, compact model)
g = agg[("cn_ad", "only: medial temporal")].merge(meta, on="NACCID", how="left")
g["pred"] = (g.p >= .5).astype(int); err = []
for y, lab in [(0, "normal"), (1, "dementia")]:
    h = g[g.y == y]; wrong, right = h[h.pred != y], h[h.pred == y]
    err.append(dict(group=lab, n=len(h), n_wrong=len(wrong), age_wrong_median=wrong.NACCAGE.median(), age_right_median=right.NACCAGE.median(),
                    p_mannwhitney=mannwhitneyu(wrong.NACCAGE, right.NACCAGE).pvalue))
ER = pd.DataFrame(err); ER.to_csv(E / "errors_by_age.csv", index=False); print(ER.round(4).to_string(index=False))

# figure: ROC (NACC two tasks, OASIS-1 two contrasts)
fig, axs = plt.subplots(1, 4, figsize=(12.6, 3.4))
panels = [("NACC: normal vs dementia", "cn_ad"), ("NACC: normal vs MCI", "cn_mci"), ("OASIS-1: CDR 0 vs >= 1", "CDR 0 vs >=1"), ("OASIS-1: CDR 0 vs 0.5", "CDR 0 vs 0.5")]
for ax, (ttl, key) in zip(axs, panels):
    for s, lab in SETS.items():
        if key in OASIS:
            tag, pos = OASIS[key]; keep = (ox.cdr == 0) | pos; y = pos[keep].astype(int)
            p = ox.loc[keep, f"p_{tag}_{'all' if s == 'all' else 'medial_temporal'}"]
        else:
            gg = agg[(key, s)]; y, p = gg.y, gg.p
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, color=COL[s], lw=1.6, ls="-" if s == "all" else "--", label=f"{lab} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="0.75", lw=0.8); ax.set_title(ttl, fontsize=9); ax.set_xlabel("1 - specificity"); ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right", frameon=False, title="AUC", title_fontsize=7)
axs[0].set_ylabel("Sensitivity")
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_roc_compact.{ext}")
plt.close(fig)
(E / "command.txt").write_text("python src/analysis/17_compact_vs_full_errors.py .\n")
(E / "config.json").write_text(json.dumps(dict(inputs=["EXP303 predictions", "EXP307 predictions", "meta_v2.csv (NACCAGE)"],
                                               aggregation="mean probability over 10 repeats", threshold=0.5), indent=1))
