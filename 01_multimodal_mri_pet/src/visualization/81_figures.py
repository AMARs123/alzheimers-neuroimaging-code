"""
81_figures.py
-------------
Figures from stored out-of-fold predictions (predictions.csv.gz of an experiment). Nothing is refitted.

  * binary tasks: ROC curves of every feature set (probabilities averaged over the CV repeats per subject; the AUC in the
    legend is the AUC of that averaged prediction, so it can differ slightly from the mean per-repeat AUC in the tables)
  * three-class: row-normalised confusion matrix per feature set (counts pooled over all repeats)

Usage: python 81_figures.py <experiment dir> <out dir> [--model lr] [--sets mri,mri+tau]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("exp"); ap.add_argument("out"); ap.add_argument("--model", default="lr"); ap.add_argument("--sets", default="")
a = ap.parse_args()
exp = Path(a.exp); out = Path(a.out) / exp.name; out.mkdir(parents=True, exist_ok=True)
P = pd.read_csv(exp / "predictions.csv.gz"); P = P[P.model == a.model]
if a.sets:
    P = P[P.set.isin(a.sets.split(","))]
assert len(P), f"no predictions for model {a.model}"
NAMES = {"cn_ad": ("Normal", "Dementia"), "cn_mci": ("Normal", "MCI"), "mci_ad": ("MCI", "Dementia"), "3class": ("Normal", "MCI", "Dementia")}
LAB = {"mri": "MRI", "tau": "Tau PET", "amy": "Amyloid PET", "thk": "Thickness"}
nice = lambda s: " + ".join(LAB.get(x, x) for x in s.split("+"))
plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight"})

for task, dt in P.groupby("task"):
    names = NAMES.get(task)
    if task != "3class":
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        for fs, d in dt.groupby("set", sort=False):
            classes = np.sort(d.y.unique())
            g = d.groupby("NACCID").agg(y=("y", "first"), p=("p1", "mean"))
            yb = (g.y == classes[1]).astype(int)
            fpr, tpr, _ = roc_curve(yb, g.p)
            ax.plot(fpr, tpr, lw=1.6, label=f"{nice(fs)} (AUC {roc_auc_score(yb, g.p):.3f})")
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls="--")
        ax.set(xlabel="1 - specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01),
               title=f"{names[0]} vs {names[1]} (n = {dt.NACCID.nunique()}, {a.model})")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"roc_{task}_{a.model}.{ext}")
        plt.close(fig)
    else:
        sets = list(dt.set.unique())
        fig, axes = plt.subplots(1, len(sets), figsize=(3.3 * len(sets), 3.5), squeeze=False)
        for ax, fs in zip(axes[0], sets):
            d = dt[dt.set == fs]; classes = np.sort(d.y.unique())
            yi = np.searchsorted(classes, d.y.to_numpy()); pred = d[["p0", "p1", "p2"]].to_numpy().argmax(1)
            cm = np.zeros((3, 3)); np.add.at(cm, (yi, pred), 1); cm = cm / cm.sum(1, keepdims=True)
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
            ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names if ax is axes[0][0] else [], xlabel="Predicted",
                   title=f"{nice(fs)}\naccuracy {(yi == pred).mean():.3f}")
            ax.tick_params(axis="x", labelsize=8); ax.tick_params(axis="y", labelsize=8)
        axes[0][0].set_ylabel("Clinical label")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"confusion_3class_{a.model}.{ext}")
        plt.close(fig)
    print(task, "figure written", flush=True)
