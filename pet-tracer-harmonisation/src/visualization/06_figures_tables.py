"""
06_figures_tables.py
--------------------
All pet-tracer-harmonisation figures (figures/final/*.pdf + .png) and LaTeX tables (paper/tables/*.tex) from stored experiment outputs.
Usage: python 06_figures_tables.py <project dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

P = Path(sys.argv[1]); X = P / "experiments"; FG = P / "figures" / "final"; TB = P / "paper" / "tables"
FG.mkdir(parents=True, exist_ok=True); TB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated reference categorical slots 1-4 (dataviz palette.md)
G = {0: "Normal", 1: "MCI", 2: "Dementia"}
A = pd.read_csv(X / "EXP201" / "centiloid_oof.csv"); AG = pd.read_csv(X / "EXP201" / "agreement.csv"); MAP = pd.read_csv(X / "EXP201" / "maps.csv")
T = pd.read_csv(X / "EXP202" / "tau_z.csv"); TV = pd.read_csv(X / "EXP202" / "validation.csv")
AB = pd.read_csv(X / "EXP203" / "amyloid_by_group.csv"); AT = pd.read_csv(X / "EXP203" / "at_by_group.csv")
AU = pd.read_csv(X / "EXP204" / "auc.csv"); PR = pd.read_csv(X / "EXP204" / "paired.csv")
save = lambda f, n: (f.savefig(FG / f"{n}.pdf"), f.savefig(FG / f"{n}.png"), plt.close(f))

# ---- Fig 1 Centiloid agreement
tracers = ["PiB", "florbetaben", "florbetapir", "NAV4694"]
f, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for i, tr in enumerate(tracers):
    x = A[A.amy_tracer == tr]
    ax[0].scatter(x.CENTILOIDS, x.cl_oof, s=14, color=COL[i], label=f"{tr} (n={len(x)})", edgecolor="white", linewidth=.5)
    ax[1].scatter((x.CENTILOIDS + x.cl_oof) / 2, x.cl_oof - x.CENTILOIDS, s=14, color=COL[i], edgecolor="white", linewidth=.5)
lim = [-60, 200]; ax[0].plot(lim, lim, ":", color="grey"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("Core-laboratory Centiloid"); ax[0].set_ylabel("Our Centiloid (out-of-fold)"); ax[0].legend(frameon=False, fontsize=8)
a0 = AG[AG.tracer == "ALL"].iloc[0]; ax[0].set_title(f"(a) Agreement: r = {a0.pearson_r:.3f}, CCC = {a0.ccc:.3f}", fontsize=9)
for v, ls in ((a0.bias, "-"), (a0.loa_low, "--"), (a0.loa_high, "--")): ax[1].axhline(v, color="grey", ls=ls, lw=1)
ax[1].set_xlabel("Mean of the two Centiloids"); ax[1].set_ylabel("Ours minus core laboratory")
ax[1].set_title(f"(b) Bland-Altman: bias {a0.bias:+.1f}, limits {a0.loa_low:.1f} to {a0.loa_high:+.1f}", fontsize=9)
save(f, "fig_centiloid")

# ---- Fig 2 tau raw vs harmonised
order = ["flortaucipir", "MK-6240", "PI-2620"]
f, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, (col, lab) in enumerate((("tau_comp_meta_temporal", "Meta-temporal SUVR"), ("z_meta_temporal", "Harmonised meta-temporal z"))):
    for gi, g in enumerate((0, 1, 2)):
        for ti, tr in enumerate(order):
            v = T[(T.label == g) & (T.tau_tracer == tr)][col].dropna().to_numpy(); xpos = gi * 4 + ti
            if len(v): ax[k].boxplot(v, positions=[xpos], widths=.7, showfliers=False, patch_artist=True,
                                     boxprops=dict(facecolor=COL[ti], alpha=.35, color=COL[ti]), medianprops=dict(color=COL[ti], lw=2),
                                     whiskerprops=dict(color=COL[ti]), capprops=dict(color=COL[ti]))
            ax[k].scatter(np.full(len(v), xpos) + np.random.default_rng(ti + 3 * gi).uniform(-.18, .18, len(v)), v, s=6, color=COL[ti], zorder=3)
    ax[k].set_xticks([1, 5, 9]); ax[k].set_xticklabels(["Normal", "MCI", "Dementia"]); ax[k].set_ylabel(lab)
    if k == 1: ax[k].set_yscale("symlog", linthresh=5); ax[k].axhline(2, color="grey", ls=":", lw=1)
for ti, tr in enumerate(order): ax[0].scatter([], [], color=COL[ti], label=f"{tr} (n={(T.tau_tracer == tr).sum()})")
ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("(a) Before harmonisation", fontsize=9)
ax[1].set_title("(b) z relative to amyloid-negative normal, same tracer (dotted: z = 2)", fontsize=9)
save(f, "fig_tau")

# ---- Fig 3 A/T profiles
cats = ["A-T-", "A-T+", "A+T-", "A+T+"]; at = AT.set_index("group").loc[["normal", "MCI", "dementia"]]
f, ax = plt.subplots(figsize=(6.5, 2.8)); left = np.zeros(3)
for i, c in enumerate(cats):
    w = at[f"pct_{c}"].to_numpy(); ax.barh(range(3), w, left=left, color=COL[i], edgecolor="white", linewidth=2, label=c)
    for j in range(3):
        if w[j] >= 8: ax.text(left[j] + w[j] / 2, j, f"{w[j]:.0f}%", ha="center", va="center", fontsize=8, color="black")
    left += w
ax.set_yticks(range(3)); ax.set_yticklabels([f"{g} (n={int(n)})" for g, n in zip(["Normal", "MCI", "Dementia"], at.n)]); ax.invert_yaxis()
ax.set_xlabel("Participants (%)"); ax.set_xlim(0, 100); ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.25))
save(f, "fig_at")

# ---- Fig 4 AUC forest
f, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
names = ["amyloid Centiloid", "tau meta_temporal", "tau braak1", "tau braak34", "tau braak56"]
nice = {"amyloid Centiloid": "Amyloid (Centiloid)", "tau meta_temporal": "Tau meta-temporal", "tau braak1": "Tau Braak I",
        "tau braak34": "Tau Braak III/IV", "tau braak56": "Tau Braak V/VI"}
for k, c in enumerate(["normal vs dementia", "normal vs MCI", "MCI vs dementia"]):
    x = AU[AU.contrast == c].set_index("summary").loc[names]
    for i, s in enumerate(names):
        ax[k].errorbar(x.loc[s, "auc"], i, xerr=[[x.loc[s, "auc"] - x.loc[s, "ci_low"]], [x.loc[s, "ci_high"] - x.loc[s, "auc"]]],
                       fmt="s" if s.startswith("amyloid") else "o", color=COL[0] if s.startswith("amyloid") else COL[1], ms=6, capsize=0, lw=2)
    ax[k].axvline(.5, color="grey", ls=":", lw=1); ax[k].set_xlim(.4, 1); ax[k].set_title(c[0].upper() + c[1:], fontsize=9); ax[k].set_xlabel("AUC (95% CI)")
ax[0].set_yticks(range(5)); ax[0].set_yticklabels([nice[s] for s in names]); ax[0].invert_yaxis()
save(f, "fig_auc")

# ---- tables
fp = lambda p: "$<$0.001" if p < .001 else f"{p:.3f}"
# demographics
F = pd.read_csv(P / "dataset" / "metadata" / "features_v2.csv", usecols=["NACCID", "label"])
M = pd.read_csv(P / "dataset" / "metadata" / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex", "EDUC", "amy_qc_ok", "tau_qc_ok"])
D = F.merge(M, on="NACCID")
def block(sel, title):
    x = D[sel]; gs = [x[x.label == g] for g in (0, 1, 2)]
    sx = [f"{(g.sex == 'F').sum()}/{g.sex.notna().sum()} ({100 * (g.sex == 'F').sum() / g.sex.notna().sum():.0f}\\%)" for g in gs]
    pc = stats.chi2_contingency([[(g.sex == "F").sum() for g in gs], [(g.sex == "M").sum() for g in gs]])[1]
    ed = [g.EDUC.where(g.EDUC.between(0, 30)) for g in gs]
    return [f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\",
            f"$n$ & {' & '.join(str(len(g)) for g in gs)} & \\\\",
            f"Age at label visit, y & {' & '.join(f'{g.NACCAGE.mean():.1f} ({g.NACCAGE.std():.1f})' for g in gs)} & {fp(stats.kruskal(*[g.NACCAGE for g in gs]).pvalue)} \\\\",
            f"Female (known sex) & {' & '.join(sx)} & {fp(pc)} \\\\",
            f"Education, y & {' & '.join(f'{e.mean():.1f} ({e.std():.1f})' for e in ed)} & {fp(stats.kruskal(*[e.dropna() for e in ed]).pvalue)} \\\\"]
calib = D.NACCID.isin(A.NACCID)
rows = block(D.amy_qc_ok == 1, "Amyloid PET passing quality control") + ["\\midrule"] + block(calib, "Amyloid PET with core-laboratory Centiloid (calibration cohort)") \
     + ["\\midrule"] + block(D.tau_qc_ok == 1, "Tau PET passing quality control")
tr_amy = D[D.amy_qc_ok == 1].merge(pd.read_csv(P / "dataset" / "metadata" / "meta_v2.csv", usecols=["NACCID", "amy_tracer", "tau_tracer"]), on="NACCID")
(TB / "demographics.tex").write_text("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Participants with amyloid or tau PET. Diagnosis at the baseline UDS visit; $p$: Kruskal--Wallis or chi-square test across groups.}\\label{tab:demo}\n"
    "\\begin{tabular}{lcccc}\n\\toprule\n & Normal & MCI & Dementia & $p$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")
# tracers
tt = pd.crosstab(D.merge(pd.read_csv(P / "dataset" / "metadata" / "meta_v2.csv", usecols=["NACCID", "amy_tracer", "amy_window", "tau_tracer", "tau_window"]), on="NACCID")
                 .pipe(lambda x: x[x.amy_qc_ok == 1]).pipe(lambda x: x.amy_tracer + " (" + x.amy_window.str.extract(r"(d+-d+)")[0] + " min)"), D[D.amy_qc_ok == 1].label)
tt2 = pd.crosstab(D[D.tau_qc_ok == 1].merge(pd.read_csv(P / "dataset" / "metadata" / "meta_v2.csv", usecols=["NACCID", "tau_tracer"]), on="NACCID").set_index(D[D.tau_qc_ok == 1].index).tau_tracer, D[D.tau_qc_ok == 1].label)
lines = ["\\multicolumn{4}{l}{\\textit{Amyloid PET}} \\\\"] + [f"{i} & {' & '.join(str(v) for v in r)} \\\\" for i, r in tt.iterrows()] + \
        ["\\midrule", "\\multicolumn{4}{l}{\\textit{Tau PET}} \\\\"] + [f"{i} & {' & '.join(str(v) for v in r)} \\\\" for i, r in tt2.iterrows()]
(TB / "tracers.tex").write_text("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{PET tracers and acquisition windows (minutes after injection) by diagnostic group, scans passing quality control. Window variants (A/B) are pooled.}\\label{tab:tracers}\n"
    "\\begin{tabular}{lccc}\n\\toprule\nTracer (window) & Normal & MCI & Dementia \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")
# centiloid agreement
L = []
for _, r in AG.iterrows():
    mp = MAP[MAP.tracer == r.tracer]; eq = f"{mp.slope.iloc[0]:.1f}$\\times$SUVR $-$ {-mp.intercept.iloc[0]:.1f}" if len(mp) else "--"
    L.append(f"{'All tracers' if r.tracer == 'ALL' else r.tracer} & {int(r.n)} ({int(r.n_pos_ref)}) & {eq} & {r.pearson_r:.3f} & {r.ccc:.3f} & {r.bias:+.1f} [{r.loa_low:.1f}, {r.loa_high:+.1f}] & "
             f"{r.kappa:.3f} {r.kappa_ci.replace('[', '[').replace(', ', ', ')} & {r.accuracy:.3f} & {r.sensitivity:.3f} / {r.specificity:.3f} \\\\")
(TB / "centiloid.tex").write_text("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Agreement of our out-of-fold Centiloids with the core laboratory (10-fold cross-validation within tracer). $n$ (positive by core laboratory). Equation: map fitted on all scans of the tracer (description only). Bias [95\\% limits of agreement] in Centiloid units. Kappa [95\\% bootstrap CI], accuracy, sensitivity/specificity: amyloid status with an in-fold Centiloid cut-off.}\\label{tab:cl}\n"
    "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccccccc}\n\\toprule\nTracer & $n$ (A+) & Centiloid equation & $r$ & CCC & Bias [LoA] & $\\kappa$ & Accuracy & Sens / Spec \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n", encoding="utf-8")
# tau validation
L = [f"{r.tracer} & {r.measure.replace('_', '-')} & {int(r.n)} & {r.pearson_r:.3f} & {r.spearman_r:.3f} & {r.ccc:.3f} & {r.mean_ours:.2f} / {r.mean_ref:.2f} & {r.slope:.2f} \\\\" for _, r in TV.iterrows()]
(TB / "tau_validation.tex").write_text("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Agreement of our tau SUVRs with the core laboratory (UC Berkeley, MRI-free pipeline). No core-laboratory values were available for PI-2620. Slope: our SUVR regressed on the core-laboratory SUVR.}\\label{tab:tauval}\n"
    "\\begin{tabular}{llcccccc}\n\\toprule\nTracer & Region & $n$ & $r$ & Spearman & CCC & Mean ours / core lab & Slope \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")
# A/T
L = [f"{G[{'normal': 0, 'MCI': 1, 'dementia': 2}[g]]} & {int(r.n)} & " + " & ".join(f"{int(r[c])} ({r['pct_' + c]:.0f}\\%)" for c in cats) + " \\\\" for g, r in at.iterrows()]
L2 = [f"{r.group.capitalize() if r.group != 'MCI' else 'MCI'} & {int(r.n)} & {int(r.a_pos)} ({r.pct_a_pos:.0f}\\%; {r.ci_low:.0f}--{r.ci_high:.0f}) & {r.pct_a_pos_corelab:.0f}\\% & {r.median_cl:.1f} \\\\" for _, r in AB.iterrows()]
(TB / "at.tex").write_text("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Amyloid status by clinical group (calibration cohort; our Centiloid status, with Wilson 95\\% CI, and the core-laboratory status for the same people) and A/T profiles in participants with both PET scans (T+: harmonised meta-temporal $z>2$).}\\label{tab:at}\n"
    "\\begin{tabular}{lcccc}\n\\toprule\nGroup & $n$ & A+ (ours) & A+ (core lab) & Median Centiloid \\\\\n\\midrule\n" + "\n".join(L2) +
    "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{2mm}\n\\begin{tabular}{lccccc}\n\\toprule\nGroup & $n$ & A$-$T$-$ & A$-$T+ & A+T$-$ & A+T+ \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")
# AUC
L = []
# age- and sex-adjusted AUC from EXP205 (added in the improvement pass), matched by contrast and summary
ADJ = pd.read_csv(X / "EXP205" / "adjusted_auc.csv").set_index(["contrast", "summary"]) if (X / "EXP205" / "adjusted_auc.csv").exists() else None
for c in ["normal vs dementia", "normal vs MCI", "MCI vs dementia"]:
    L.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{c[0].upper() + c[1:]}}}}} \\\\")
    for _, r in AU[AU.contrast == c].iterrows():
        w = "--" if pd.isna(r.get("auc_within_tracer")) else f"{r.auc_within_tracer:.3f}"
        a = ADJ.loc[(c, r.summary)] if ADJ is not None else None
        aj = "--" if a is None else f"{a.auc_adjusted:.3f} [{a.adj_ci_low:.3f}, {a.adj_ci_high:.3f}]"
        L.append(f"{nice[r.summary]} & {int(r.n)} ({int(r.n_pos)}) & {r.auc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] & {w} & {aj} \\\\")
L.append("\\midrule"); L.append("\\multicolumn{5}{l}{\\textit{Paired: tau meta-temporal minus amyloid, participants with both}} \\\\")
for _, r in PR.iterrows():
    L.append(f"{r.contrast[0].upper() + r.contrast[1:]} & {int(r.n)} & {r.auc_tau:.3f} vs {r.auc_amyloid:.3f}: {r.diff_tau_minus_amyloid:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] & $p_{{Holm}}$ {fp(r.p_holm)} & \\\\")
(TB / "auc.tex").write_text("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{Separation of clinical groups by single PET summaries (no model fitted). AUC [95\\% bootstrap CI]; within-tracer AUC compares only participants scanned with the same tau tracer; adjusted AUC removes the effect of age and sex estimated in the less impaired group (EXP205). Bottom: paired difference in participants with both PET scans and known amyloid status.}\\label{tab:auc}\n"
    "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccc}\n\\toprule\nSummary & $n$ (positive) & AUC & Within-tracer AUC & Age/sex-adjusted AUC \\\\\n\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n", encoding="utf-8")
print("written", sorted(p.name for p in FG.glob("*.png")), sorted(p.name for p in TB.glob("*.tex")))
