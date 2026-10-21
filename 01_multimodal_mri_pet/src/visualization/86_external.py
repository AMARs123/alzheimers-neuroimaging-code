"""
86_external.py
--------------
External validation (EXP023, OASIS-1) table and figure from stored outputs only (nothing is refitted).
  paper/tables/external.tex        metrics: NACC internal CV, frozen model, label-free aligned model, OASIS-internal CV, age alone
  paper/tables/oasis_demo.tex      OASIS-1 participants by CDR group
  figures/final/external_oasis.{pdf,png}  (a) ROC of the frozen model per OASIS contrast; (b) predicted probability by CDR;
                                          (c) shift of OASIS CDR-0 feature means from NACC normals (NACC SD units)
Usage: python 86_external.py <project dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

P = Path(sys.argv[1]); E = P / "experiments" / "EXP023"; OUT = P / "figures" / "final"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
M = pd.read_csv(E / "metrics.csv"); pr = pd.read_csv(E / "predictions.csv"); sh = pd.read_csv(E / "feature_shift_normals.csv")
o = pd.read_csv(P / "results" / "raw" / "features" / "features_oasis1.csv")

# ---- metrics table
AN = [("internal nested CV (NACC)", "NACC, repeated nested CV"), ("external frozen", "OASIS-1, frozen NACC model"),
      ("external aligned (label-free)", "OASIS-1, cohort-wise $z$-scoring$^{a}$"), ("OASIS-internal nested CV", "OASIS-1, internal nested CV$^{b}$"),
      ("age alone", "OASIS-1, age alone")]
MET = ["auc", "balanced_accuracy", "sensitivity", "specificity", "brier"]
def cell(r):
    if r is None: return "--"
    if pd.notna(r.ci_low): return f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"
    if pd.notna(getattr(r, "sd_over_repeats", np.nan)): return f"{r.value:.3f} $\\pm$ {r.sd_over_repeats:.3f}"
    return f"{r.value:.3f}"
lines = []
for coh in ["normal vs dementia", "CDR 0 vs CDR >= 1", "CDR 0 vs CDR 0.5", "CDR 0 vs CDR > 0"]:
    sub = M[M.cohort == coh]
    if sub.empty: continue
    n, npos = int(sub.n.iloc[0]), int(sub.n_pos.iloc[0])
    lab = coh.replace(">=", "$\\geq$").replace("> 0", "$>$ 0")
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{lab}}} ($n$={n}, {npos} positive)}} \\\\")
    for a, name in AN:
        s = sub[sub.analysis == a]
        if s.empty: continue
        lines.append(name + " & " + " & ".join(cell(next((r for r in s.itertuples() if r.metric == m), None)) for m in MET) + " \\\\")
(P / "paper" / "tables" / "external.tex").write_text(
    "\\begin{table}[t]\n\\centering\\small\n\\caption{External validation on OASIS-1 (EXP023). The MRI model (bilateral regional "
    "volumes, logistic regression) was trained on all NACC normal and dementia participants with MRI and applied unchanged. "
    "Brackets: 95\\% stratified bootstrap CI (2,000 resamples); $\\pm$: SD over 10 repeats of 5-fold nested CV. Threshold 0.5. "
    "$^{a}$Sensitivity analysis: each cohort's features $z$-scored with that cohort's own mean and SD (no labels used) before "
    "training and prediction. $^{b}$Same pipeline trained and tested within OASIS-1 only.}\\label{tab:external}\n"
    "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccc}\n\\toprule\n & AUC & Balanced accuracy & Sensitivity & Specificity & Brier \\\\\n"
    "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n", encoding="utf-8")

# ---- OASIS demographics
o["g"] = pd.cut(pd.to_numeric(o.cdr), [-1, 0, 0.5, 9], labels=["CDR 0", "CDR 0.5", "CDR $\\geq$1"])
G = [o[o.g == k] for k in o.g.cat.categories]
msd = lambda x: f"{x.mean():.1f} ({x.std():.1f})"
rows = [["$n$"] + [str(len(x)) for x in G],
        ["Age, y, mean (SD)"] + [msd(x.age) for x in G],
        ["Female"] + [f"{(x.sex == 'Female').sum()} ({100 * (x.sex == 'Female').mean():.0f}\\%)" for x in G],
        ["MMSE, mean (SD)"] + [msd(x.mmse) for x in G],
        ["nWBV (OASIS), mean (SD)"] + [f"{x.nwbv.mean():.3f} ({x.nwbv.std():.3f})" for x in G]]
(P / "paper" / "tables" / "oasis_demo.tex").write_text(
    "\\begin{table}[t]\n\\centering\\small\n\\caption{OASIS-1 participants used for external validation (age $\\geq$60, CDR rated, "
    "first session). nWBV: normalised whole-brain volume distributed with OASIS-1, shown for description only (not a model input).}"
    "\\label{tab:oasisdemo}\n\\begin{tabular}{lccc}\n\\toprule\n & " + " & ".join(o.g.cat.categories) + " \\\\\n\\midrule\n"
    + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")

# ---- figure
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
cdr = pr.cdr.to_numpy(); p = pr.p_frozen.to_numpy()
for name, m, c in [("CDR 0 vs $\\geq$1", (cdr == 0) | (cdr >= 1), "C3"), ("CDR 0 vs 0.5", cdr <= 0.5, "C1"), ("CDR 0 vs $>$0", np.ones(len(cdr), bool), "C0")]:
    y = (cdr[m] > 0).astype(int); f, t, _ = roc_curve(y, p[m])
    ax[0].plot(f, t, c=c, label=f"{name} (AUC {roc_auc_score(y, p[m]):.3f})")
ax[0].plot([0, 1], [0, 1], ":", c="grey"); ax[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="(a) Frozen NACC model on OASIS-1")
ax[0].legend(fontsize=7, loc="lower right", frameon=False)
grp = [("CDR 0", cdr == 0), ("CDR 0.5", cdr == 0.5), ("CDR $\\geq$1", cdr >= 1)]
for i, (lab, m) in enumerate(grp):
    for j, (col, mk) in enumerate([("p_frozen", "C0"), ("p_aligned", "C2")]):
        v = pr[col][m]; x = i + (j - .5) * .36
        ax[1].boxplot(v, positions=[x], widths=.3, patch_artist=True, showfliers=False, boxprops=dict(facecolor=mk, alpha=.35), medianprops=dict(color="k"))
        ax[1].scatter(x + np.random.default_rng(i).uniform(-.08, .08, len(v)), v, s=4, c=mk, alpha=.6)
ax[1].axhline(.5, ls=":", c="grey"); ax[1].set_xticks(range(3), [g[0] for g in grp])
ax[1].set(ylabel="Predicted probability of dementia", title="(b) Frozen (blue) vs aligned (green)")
s = sh.assign(a=sh.shift_in_sd.abs()).sort_values("a", ascending=False).head(15).iloc[::-1]
ax[2].barh(s.feature.str.replace("bi_", "").str.replace("mid_", ""), s.shift_in_sd, color=np.where(s.shift_in_sd < 0, "C3", "C0"))
ax[2].axvline(0, c="k", lw=.6); ax[2].set(xlabel="OASIS CDR 0 minus NACC normal (NACC SD)", title="(c) Cohort shift in normals"); ax[2].tick_params(axis="y", labelsize=7)
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(OUT / f"external_oasis.{ext}")
print(open(P / "paper" / "tables" / "external.tex").read()); print(open(P / "paper" / "tables" / "oasis_demo.tex").read())
print("median |shift| =", round(sh.shift_in_sd.abs().median(), 2), "; share negative =", round((sh.shift_in_sd < 0).mean(), 2))
