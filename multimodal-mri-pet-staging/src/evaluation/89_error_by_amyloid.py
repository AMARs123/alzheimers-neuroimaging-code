"""
89_error_by_amyloid.py (EXP025)
-------------------------------
Error analysis from stored out-of-fold predictions (EXP006, logistic regression); nothing is refitted.

Question: are the three-class errors related to biology? For each participant we average the predicted class
probabilities over the 10 CV repeats and take the most probable class. We then cross-tabulate true label x
core-laboratory amyloid status (UC Berkeley, dataset/metadata/amyloid_status_ucb.csv; never a model input in the MRI
set) x predicted class. Primary set: MRI only (amyloid is not a feature, so the test is not circular). Secondary:
MRI + amyloid + tau (descriptive only, because amyloid is then a feature).

Tests (MRI set): within MCI and within normal cognition, Fisher's exact test of predicted class (dementia vs other;
normal vs other) between amyloid-positive and amyloid-negative participants; Mann-Whitney test of the mean predicted
probability of dementia. Also the ROC figure of the three binary tasks.

Usage: python 89_error_by_amyloid.py <project dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(sys.argv[1]); E = P / "experiments" / "EXP025"; E.mkdir(parents=True, exist_ok=True)
FIG = P / "figures" / "final"
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})
pred = pd.read_csv(P / "experiments" / "EXP006" / "predictions.csv.gz")
pred = pred[pred.model == "lr"]
amy = pd.read_csv(P / "dataset" / "metadata" / "amyloid_status_ucb.csv").dropna(subset=["bio_amy_pos"])
LAB = {0: "Normal", 1: "MCI", 2: "Dementia"}

rows, tests = [], []
for fs in ["mri", "mri+amy+tau"]:
    d = pred[(pred.task == "3class") & (pred.set == fs)]
    g = d.groupby("NACCID").agg(y=("y", "first"), p0=("p0", "mean"), p1=("p1", "mean"), p2=("p2", "mean")).reset_index()
    g["pred"] = g[["p0", "p1", "p2"]].to_numpy().argmax(1)
    g = g.merge(amy, on="NACCID", how="inner")
    for (y, a), h in g.groupby(["y", "bio_amy_pos"]):
        rows.append(dict(set=fs, label=LAB[y], amyloid="A+" if a == 1 else "A-", n=len(h),
                         pred_normal=(h.pred == 0).mean(), pred_mci=(h.pred == 1).mean(), pred_dementia=(h.pred == 2).mean(),
                         mean_p_dementia=h.p2.mean(), mean_p_normal=h.p0.mean()))
    for y, target, col in [(1, 2, "p2"), (1, 0, "p0"), (0, 0, "p0"), (2, 2, "p2")]:
        h = g[g.y == y]; pos, neg = h[h.bio_amy_pos == 1], h[h.bio_amy_pos == 0]
        tab = [[(pos.pred == target).sum(), (pos.pred != target).sum()], [(neg.pred == target).sum(), (neg.pred != target).sum()]]
        OR, pf = fisher_exact(tab)
        _, pm = mannwhitneyu(pos[col], neg[col])
        tests.append(dict(set=fs, label=LAB[y], outcome=f"predicted {LAB[target].lower()}", n_pos=len(pos), n_neg=len(neg),
                          share_pos=(pos.pred == target).mean(), share_neg=(neg.pred == target).mean(), odds_ratio=OR,
                          p_fisher=pf, median_prob_pos=pos[col].median(), median_prob_neg=neg[col].median(), p_mannwhitney=pm))
    g.to_csv(E / f"participant_predictions_{fs.replace('+', '_')}.csv", index=False)
R = pd.DataFrame(rows); T = pd.DataFrame(tests)
R.to_csv(E / "error_by_amyloid.csv", index=False); T.to_csv(E / "tests.csv", index=False)
print(R.round(3).to_string()); print(T.round(4).to_string())

# ---- figure: predicted class by label x amyloid status (MRI only)
r = R[R.set == "mri"].copy()
r["grp"] = r.label + "\n" + r.amyloid + " (n=" + r.n.astype(str) + ")"
fig, ax = plt.subplots(figsize=(6.2, 3.0))
cols = {"pred_normal": ("#2a78d6", "Predicted normal"), "pred_mci": ("#eda100", "Predicted MCI"),
        "pred_dementia": ("#d9534f", "Predicted dementia")}
left = np.zeros(len(r)); yy = np.arange(len(r))[::-1]
for c, (colr, lab) in cols.items():
    ax.barh(yy, r[c], left=left, color=colr, label=lab, edgecolor="white", linewidth=1.5, height=0.7)
    for yi, l, v in zip(yy, left, r[c]):
        if v >= 0.08:
            ax.text(l + v / 2, yi, f"{100 * v:.0f}%", ha="center", va="center", color="white", fontsize=8)
    left += r[c].to_numpy()
ax.set_yticks(yy, r.grp); ax.set_xlim(0, 1); ax.set_xlabel("Share of participants")
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"error_by_amyloid.{ext}")
plt.close(fig)

# ---- figure: ROC curves of the three binary tasks (probabilities averaged over repeats)
SETS = [("mri", "MRI", "#1f5f8b", "-"), ("mri+tau", "MRI + tau", "#eb6834", "-"), ("mri+amy", "MRI + amyloid", "#1baf7a", "--"),
        ("mri+amy+tau", "MRI + amyloid + tau", "#7a4fb3", ":")]
TASKS = [("cn_ad", "Normal vs dementia"), ("cn_mci", "Normal vs MCI"), ("mci_ad", "MCI vs dementia")]
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.5)); roc_rows = []
for ax, (task, ttl) in zip(axs, TASKS):
    for fs, lab, colr, ls in SETS:
        d = pred[(pred.task == task) & (pred.set == fs)]
        if d.empty:
            continue
        g = d.groupby("NACCID").agg(y=("y", "first"), p=("p1", "mean"))
        yb = (g.y == g.y.max()).astype(int)
        auc = roc_auc_score(yb, g.p); fpr, tpr, _ = roc_curve(yb, g.p)
        per_rep = d.groupby("repeat").apply(lambda x: roc_auc_score((x.y == x.y.max()).astype(int), x.p1), include_groups=False)
        roc_rows.append(dict(task=task, set=fs, auc_averaged_prob=auc, auc_repeat_mean=per_rep.mean(), auc_repeat_sd=per_rep.std(ddof=1),
                             auc_repeat_min=per_rep.min(), auc_repeat_max=per_rep.max(), n=len(g)))
        ax.plot(fpr, tpr, color=colr, ls=ls, lw=1.6, label=f"{lab} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.75", lw=0.8)
    ax.set_title(ttl); ax.set_xlabel("1 - specificity"); ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right", frameon=False, title="AUC", title_fontsize=7)
axs[0].set_ylabel("Sensitivity")
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"roc_binary.{ext}")
plt.close(fig)
pd.DataFrame(roc_rows).to_csv(E / "roc_auc.csv", index=False)
print(pd.DataFrame(roc_rows).round(3).to_string())
(E / "command.txt").write_text("python src/evaluation/89_error_by_amyloid.py .\n")
(E / "config.json").write_text(json.dumps(dict(source="EXP006 predictions (lr)", amyloid="dataset/metadata/amyloid_status_ucb.csv",
                                               aggregation="mean probability over 10 repeats; argmax class", tests=["Fisher exact", "Mann-Whitney U"]), indent=1))
