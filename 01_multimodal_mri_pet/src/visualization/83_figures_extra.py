"""
83_figures_extra.py
-------------------
Additional paper figures from stored results only (nothing is refitted):
  validation_scatter      own SUVR vs SCAN core-lab SUVR (amyloid, tau), by tracer
  forest_<task>           accuracy and AUC with bootstrap 95% CI per feature set (ci_<EXP>.csv)
  recall_3class           per-class recall with 95% CI per feature set
  calibration_pr_cn_ad    reliability diagram and precision-recall curves (out-of-fold probabilities, averaged over repeats)
  plateau_3class          three-class accuracy with 95% CI across all analysis variants (label definitions, thickness)
  amyloid_by_label        core-lab amyloid positivity by clinical label with Wilson 95% CI

Usage: python 83_figures_extra.py <project dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

P = Path(sys.argv[1]); S = P / "results" / "statistics"; OUT = P / "figures" / "final"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
LAB = {"mri": "MRI", "tau": "Tau PET", "amy": "Amyloid PET", "thk": "Thickness"}
nice = lambda s: " + ".join(LAB.get(x, x) for x in s.split("+"))
TASK = {"cn_ad": "Normal vs dementia", "cn_mci": "Normal vs MCI", "mci_ad": "MCI vs dementia", "3class": "Three-class"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig); print("written", name, flush=True)


# ---- validation scatter
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.9))
for a, f, x, y, tr, ttl in [(ax[0], "validation_amyloid_pairs.csv", "NPDKA_SUMMARY_SUVR", "amy_comp_global", "amy_tracer", "Amyloid PET: global cortical SUVR"),
                            (ax[1], "validation_tau_pairs.csv", "META_TEMPORAL_SUVR", "tau_comp_meta_temporal", "tau_tracer", "Tau PET: meta-temporal SUVR")]:
    d = pd.read_csv(S / f).dropna(subset=[x, y])
    for t, g in d.groupby(tr):
        a.scatter(g[x], g[y], s=9, alpha=0.7, label=f"{t} (n = {len(g)}, r = {np.corrcoef(g[x], g[y])[0, 1]:.3f})")
    lim = [min(d[x].min(), d[y].min()) * 0.95, max(d[x].max(), d[y].max()) * 1.03]
    a.plot(lim, lim, color="0.6", lw=0.8, ls="--"); a.set(xlim=lim, ylim=lim, xlabel="SCAN core laboratory (UC Berkeley)", ylabel="This study",
                                                       title=f"{ttl}\nall tracers: n = {len(d)}, r = {np.corrcoef(d[x], d[y])[0, 1]:.3f}")
    a.legend(fontsize=7, frameon=False, loc="upper left")
save(fig, "validation_scatter")

# ---- forest plots and per-class recall (EXP006)
ci = pd.read_csv(S / "ci_EXP006.csv")
order = ["amy", "tau", "amy+tau", "mri", "mri+amy", "mri+tau", "mri+amy+tau"]
for task, mets in [("cn_ad", ["accuracy", "auc"]), ("3class", ["accuracy", "macro_f1"]), ("cn_mci", ["accuracy", "auc"]), ("mci_ad", ["accuracy", "auc"])]:
    d = ci[(ci.task == task) & (ci.model == "lr")]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    for a, m in zip(ax, mets):
        g = d[d.metric == m].set_index("set").reindex(order).dropna(subset=["value"])
        yy = np.arange(len(g))[::-1]
        a.errorbar(g.value, yy, xerr=[g.value - g.ci_low, g.ci_high - g.value], fmt="o", color="#1f5f8b", ms=4, lw=1.1, capsize=2)
        for v, lo, hi, y0 in zip(g.value, g.ci_low, g.ci_high, yy):
            a.text(hi + 0.004, y0, f"{v:.3f}", va="center", fontsize=7)
        a.set_yticks(yy); a.set_yticklabels([nice(s) for s in g.index]); a.set_xlabel(m.replace("_", " ").replace("auc", "AUC")); a.grid(axis="x", lw=0.3, alpha=0.5)
    fig.suptitle(f"{TASK[task]} (n = {int(d.n.iloc[0])}, logistic regression, bootstrap 95% CI)", fontsize=9)
    save(fig, f"forest_{task}")

d = ci[(ci.task == "3class") & (ci.model == "lr") & ci.metric.isin(["recall_c0", "recall_c1", "recall_c2"]) & ci.set.isin(["mri", "mri+tau", "mri+amy+tau"])]
fig, ax = plt.subplots(figsize=(5.4, 3.1)); w = 0.26
for k, (m, name, c) in enumerate([("recall_c0", "Normal", "#5a9e6f"), ("recall_c1", "MCI", "#d9a441"), ("recall_c2", "Dementia", "#c8553d")]):
    g = d[d.metric == m].set_index("set").reindex(["mri", "mri+tau", "mri+amy+tau"]); x = np.arange(len(g)) + (k - 1) * w
    ax.bar(x, g.value, w, color=c, label=name, yerr=[g.value - g.ci_low, g.ci_high - g.value], error_kw=dict(lw=0.8, capsize=2))
ax.set_xticks(range(3)); ax.set_xticklabels([nice(s) for s in ["mri", "mri+tau", "mri+amy+tau"]]); ax.set(ylabel="Recall", ylim=(0, 1))
ax.axhline(1 / 3, color="0.6", lw=0.7, ls=":"); ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
ax.set_title("Three-class recall by class (n = 255, logistic regression, 95% CI)", fontsize=9)
save(fig, "recall_3class")

# ---- calibration and precision-recall, normal vs dementia
pr = pd.read_csv(P / "experiments" / "EXP006" / "predictions.csv.gz")
pr = pr[(pr.task == "cn_ad") & (pr.model == "lr") & pr.set.isin(["mri", "mri+tau", "mri+amy+tau"])]
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.7))
STY = {"mri": ("o", "-"), "mri+tau": ("s", "--"), "mri+amy+tau": ("^", ":")}  # VS-03: not colour alone
for fs, g in pr.groupby("set", sort=False):
    cl = np.sort(g.y.unique()); a = g.groupby("NACCID").agg(y=("y", "first"), p=("p1", "mean")); yb = (a.y == cl[1]).astype(int).to_numpy(); p = a.p.to_numpy()
    bins = np.quantile(p, np.linspace(0, 1, 7)); idx = np.clip(np.digitize(p, bins[1:-1]), 0, 5)
    ax[0].plot([p[idx == b].mean() for b in range(6)], [yb[idx == b].mean() for b in range(6)], marker=STY[fs][0], ls=STY[fs][1], ms=4, lw=1.2,
               label=f"{nice(fs)} (Brier {np.mean((p - yb) ** 2):.3f})")
    prec, rec, _ = precision_recall_curve(yb, p); ax[1].plot(rec, prec, lw=1.3, ls=STY[fs][1], label=f"{nice(fs)} (AP {average_precision_score(yb, p):.3f})")
ax[0].plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--"); ax[0].set(xlabel="Mean predicted probability of dementia (sextile bins)", ylabel="Observed fraction with dementia", title="Calibration")
ax[1].axhline(yb.mean(), color="0.6", lw=0.8, ls="--"); ax[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02), title="Precision-recall")
for a in ax:
    a.legend(fontsize=7, frameon=False, loc="upper left" if a is ax[0] else "lower left")
save(fig, "calibration_pr_cn_ad")

# ---- three-class plateau across analysis variants
rows = []
for exp, sets, note in [("EXP006", ["mri", "mri+tau", "mri+amy+tau"], "clinical labels, n = 255"), ("EXP005", ["mri", "mri+tau"], "flortaucipir only, n = 180"),
                        ("EXP007", ["mri", "mri+amy"], "PiB only, n = 196"), ("EXP008", ["mri", "mri+amy"], "florbetaben only, n = 155"),
                        ("EXP019", ["mri"], "amyloid-defined labels, n = 193"), ("EXP020", ["mri", "mri+tau"], "amyloid-defined labels, n = 107"),
                        ("EXP021", ["mri", "mri+thk"], "with thickness, n = 403"), ("EXP022", ["mri+thk+tau", "mri+thk+amy+tau"], "with thickness, n = 255")]:
    f = S / f"ci_{exp}.csv"
    if not f.exists():
        continue
    c = pd.read_csv(f); c = c[(c.task == "3class") & (c.metric == "accuracy") & (c.model == "lr") & c.set.isin(sets)]
    for r in c.itertuples():
        rows.append((f"{nice(r.set)}  [{exp}: {note}]", r.value, r.ci_low, r.ci_high))
if rows:
    fig, ax = plt.subplots(figsize=(7.4, 0.27 * len(rows) + 1)); yy = np.arange(len(rows))[::-1]
    v, lo, hi = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    ax.errorbar(v, yy, xerr=[v - lo, hi - v], fmt="o", color="#1f5f8b", ms=4, lw=1.1, capsize=2)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5); ax.axvline(1 / 3, color="0.6", lw=0.7, ls=":")
    ax.set(xlabel="Three-class accuracy (logistic regression, bootstrap 95% CI); dotted line = chance", xlim=(0.3, 0.8)); ax.grid(axis="x", lw=0.3, alpha=0.5)
    save(fig, "plateau_3class")

# ---- amyloid positivity by clinical label
a = pd.read_csv(S / "amyloid_status_by_label.csv"); n = a.status_known.to_numpy(float); k = a.amyloid_positive.to_numpy(float); p = k / n; z = 1.96
cen = (p + z * z / (2 * n)) / (1 + z * z / n); half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
fig, ax = plt.subplots(figsize=(3.8, 3.1))
ax.bar(["Normal", "MCI", "Dementia"], 100 * p, color=["#5a9e6f", "#d9a441", "#c8553d"], yerr=[100 * (p - (cen - half)), 100 * ((cen + half) - p)], error_kw=dict(lw=0.8, capsize=3))
for i in range(3):
    ax.text(i, 3, f"{int(k[i])}/{int(n[i])}", ha="center", color="white", fontsize=8)
ax.set(ylabel="Amyloid-positive (%)", ylim=(0, 100), title="Core-lab amyloid status by clinical label\n(Wilson 95% CI)")
save(fig, "amyloid_by_label")
