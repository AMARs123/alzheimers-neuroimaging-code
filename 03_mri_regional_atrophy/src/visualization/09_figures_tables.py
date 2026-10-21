"""
09_figures_tables.py
--------------------
Paper 3 figures and LaTeX tables from stored experiment outputs only (nothing refitted).
Figures (figures/final): fig_effects (regional d, NACC vs OASIS-1), fig_replication (scatter of d across cohorts),
fig_groups (AUC by region group, alone and removed), fig_asymmetry (absolute asymmetry, volume-adjusted).
Tables (paper/tables): demographics, top_effects, groups, external, asymmetry.
Per-repeat mean AUCs come from EXP303/metrics_by_repeat.csv; CIs and paired differences from EXP306 (repeat-averaged
probabilities).
Usage: python 09_figures_tables.py <project dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = Path(sys.argv[1]); X = P / "experiments"; F = P / "figures" / "final"; T = P / "paper" / "tables"; M = P / "dataset" / "metadata"
F.mkdir(parents=True, exist_ok=True); T.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
save = lambda fig, name: [fig.savefig(F / f"{name}.{e}") for e in ("pdf", "png")]
eff = pd.read_csv(X / "EXP302" / "regional_effects.csv"); rep = pd.read_csv(X / "EXP302" / "replication.csv")
grp_rep = pd.read_csv(X / "EXP303" / "metrics_by_repeat.csv"); gst = pd.read_csv(X / "EXP306" / "group_auc_stats.csv")
ext = pd.read_csv(X / "EXP307" / "external_groups.csv"); asy = pd.read_csv(X / "EXP304" / "asymmetry.csv")
avc = pd.read_csv(X / "EXP305" / "asymmetry_volume_check.csv")
nice = lambda s: s.replace("Inf-Lat-Vent", "Inferior lateral ventricle").replace("Hippocampus/ILV ratio", "Hippocampus/ILV ratio")


def tex_table(name, caption, label, header, rows, align):
    (T / f"{name}.tex").write_text("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{" + caption + "}\\label{" + label + "}\n"
                                   "\\resizebox{\\linewidth}{!}{\\begin{tabular}{" + align + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
                                   + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n", encoding="utf-8")


# ---------- Fig 1: regional effects NACC vs OASIS
fig, ax = plt.subplots(1, 2, figsize=(10, 9), sharey=True)
order = eff[(eff.cohort == "NACC") & eff.contrast.str.startswith("dementia")].sort_values("d").feature.tolist()
for a, con, title in ((ax[0], "MCI", "(a) MCI vs normal / CDR 0.5 vs CDR 0"), (ax[1], "dementia", "(b) Dementia vs normal / CDR $\\geq$1 vs CDR 0")):
    for k, (coh, col, off) in enumerate((("NACC", "C0", -.18), ("OASIS-1", "C3", .18))):
        e = eff[(eff.cohort == coh) & eff.contrast.str.startswith(con)].set_index("feature").loc[order]
        yy = np.arange(len(order)) + off; sig = e.q_fdr < .05
        a.errorbar(e.d, yy, xerr=[e.d - e.ci_low, e.ci_high - e.d], fmt="none", ecolor=col, alpha=.5, lw=.8)
        mk = "o" if coh == "NACC" else "^"
        a.scatter(e.d[sig], yy[sig.to_numpy()], c=col, s=14, marker=mk, label=f"{coh} (FDR q<0.05)"); a.scatter(e.d[~sig], yy[~sig.to_numpy()], facecolors="white", edgecolors=col, s=14, marker=mk, label=f"{coh} (n.s.)")
    a.axvline(0, c="k", lw=.6); a.set_title(title, fontsize=9); a.set_xlabel("Cohen's d (age- and sex-adjusted)")
ax[0].set_yticks(range(len(order)), [nice(o) for o in order], fontsize=7); ax[0].legend(fontsize=7, loc="lower right", frameon=False)
fig.tight_layout(); save(fig, "fig_effects"); plt.close(fig)

# ---------- Fig 2: replication scatter
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, con in zip(ax, ("MCI", "dementia")):
    A = eff[(eff.cohort == "NACC") & eff.contrast.str.startswith(con)].set_index("feature").d
    Bv = eff[(eff.cohort == "OASIS-1") & eff.contrast.str.startswith(con)].set_index("feature").d.loc[A.index]
    mtl = A.index.isin(["Hippocampus", "Amygdala", "entorhinal", "parahippocampal", "Hippocampus/ILV ratio"])
    vent = A.index.isin(["Lateral-Ventricle", "Inf-Lat-Vent", "3rd-Ventricle", "4th-Ventricle", "CSF", "choroid-plexus"])
    a.scatter(A[~mtl & ~vent], Bv[~mtl & ~vent], s=12, c="0.55", label="other regions")
    a.scatter(A[mtl], Bv[mtl], s=22, c="C3", marker="^", label="medial temporal"); a.scatter(A[vent], Bv[vent], s=18, c="C0", marker="s", label="ventricles and CSF")
    lim = [min(A.min(), Bv.min()) - .1, max(A.max(), Bv.max()) + .1]; a.plot(lim, lim, ":", c="grey"); a.legend(fontsize=7, frameon=False, loc="upper left")
    r = rep[rep.contrast.str.startswith(con)].iloc[0]
    a.set(xlabel="NACC effect size (d)", ylabel="OASIS-1 effect size (d)", title=f"{'MCI / CDR 0.5' if con == 'MCI' else 'Dementia / CDR $\\geq$1'}: Spearman $\\rho$ = {r.spearman_rho:.2f}")
    a.axhline(0, c="k", lw=.4); a.axvline(0, c="k", lw=.4)
fig.tight_layout(); save(fig, "fig_replication"); plt.close(fig)

# ---------- Fig 3: region groups
groups = [s.split(": ", 1)[1] for s in grp_rep.set.unique() if s.startswith("only: ")]
TN = {"cn_ad": "Normal vs dementia", "cn_mci": "Normal vs MCI", "mci_ad": "MCI vs dementia"}
mm = grp_rep.groupby(["task", "set"]).auc.agg(["mean", "std"])
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
gorder = mm.loc["cn_ad"].loc[[f"only: {g}" for g in groups]].sort_values("mean").index.str.replace("only: ", "").tolist()
for a, t in zip(ax, TN):
    allm = mm.loc[(t, "all"), "mean"]; yy = np.arange(len(gorder))
    a.barh(yy - .2, [mm.loc[(t, f"only: {g}"), "mean"] for g in gorder], .38, xerr=[mm.loc[(t, f"only: {g}"), "std"] for g in gorder], color="C0", label="group alone")
    a.barh(yy + .2, [mm.loc[(t, f"all minus: {g}"), "mean"] for g in gorder], .38, xerr=[mm.loc[(t, f"all minus: {g}"), "std"] for g in gorder], color="C1", hatch="///", edgecolor="white", label="all minus group")
    a.axvline(allm, c="k", ls="--", lw=.8, label="all regions"); a.axvline(.5, c="grey", ls=":", lw=.6)
    a.set_xlim(.4, 1); a.set_title(f"{TN[t]} (all regions: {allm:.3f})", fontsize=9); a.set_xlabel("AUC (mean $\\pm$ SD over 10 repeats)")
ax[0].set_yticks(range(len(gorder)), gorder, fontsize=8)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, ["all regions", "group alone", "all regions except group"], loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, .06, 1, 1)); save(fig, "fig_groups"); plt.close(fig)

# ---------- Fig 4: asymmetry
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
ab = asy[(asy.measure == "absolute AI") & asy.contrast.str.startswith("dementia")].set_index("region")
v = avc.set_index("region").loc[ab.index]
top = ab.d.sort_values(ascending=False).head(15).index[::-1]
yy = np.arange(len(top))
ax[0].errorbar(ab.loc[top, "d"], yy, xerr=[ab.loc[top, "d"] - ab.loc[top, "ci_low"], ab.loc[top, "ci_high"] - ab.loc[top, "d"]], fmt="o", ms=3, c="C0")
ax[0].set_yticks(yy, [nice(t) for t in top], fontsize=7); ax[0].axvline(0, c="k", lw=.6)
ax[0].set(xlabel="Cohen's d of |asymmetry index| (age, sex adjusted)", title="(a) Dementia vs normal: absolute asymmetry")
ax[1].scatter(ab.d, v.dementia_adj_diff_in_ref_sd, s=12, c=np.where(v.dementia_q < .05, "C3", "0.6"))
for r_ in v[v.dementia_q < .05].reindex(v[v.dementia_q < .05].dementia_adj_diff_in_ref_sd.abs().sort_values(ascending=False).index).index[:4]: ax[1].annotate(nice(r_), (ab.d[r_], v.dementia_adj_diff_in_ref_sd[r_]), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
ax[1].axhline(0, c="k", lw=.5); ax[1].axvline(0, c="k", lw=.5)
ax[1].set(xlabel="effect without volume adjustment (d)", ylabel="effect adjusted for region volume (normal-group SD)", title="(b) After adjusting for region volume (red: FDR q<0.05)")
fig.tight_layout(); save(fig, "fig_asymmetry"); plt.close(fig)

# ---------- Tables
f = pd.read_csv(M / "features_v2.csv"); m = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex", "EDUC", "CDRGLOB"])
n = f[f.has_mri == 1][["NACCID", "label"]].merge(m, on="NACCID", how="left"); o = pd.read_csv(M / "features_oasis1.csv")
G = [n[n.label == k] for k in (0, 1, 2)]; cd = pd.to_numeric(o.cdr); H = [o[cd == 0], o[cd == .5], o[cd >= 1]]
msd = lambda x: f"{x.mean():.1f} ({x.std():.1f})"
fem = lambda x, col, val: f"{(x[col] == val).sum()}/{x[col].notna().sum()} ({100 * (x[col] == val).sum() / max(1, x[col].notna().sum()):.0f}\\%)"
pk = lambda xs: (lambda p: "$<$0.001" if p < .001 else f"{p:.3f}")(stats.kruskal(*xs).pvalue)
pc = lambda gs, col, val: f"{stats.chi2_contingency([[(g[col] == val).sum() for g in gs], [g[col].notna().sum() - (g[col] == val).sum() for g in gs]])[1]:.3f}"
rows = ["\\multicolumn{5}{l}{\\textit{NACC SCAN (development)}} \\\\",
        "Group & Normal & MCI & Dementia & $p$ \\\\", "$n$ & " + " & ".join(str(len(g)) for g in G) + " & \\\\",
        "Age at label visit, y & " + " & ".join(msd(g.NACCAGE) for g in G) + f" & {pk([g.NACCAGE for g in G])} \\\\",
        "Female (known sex) & " + " & ".join(fem(g, "sex", "F") for g in G) + f" & {pc(G, 'sex', 'F')} \\\\",
        "Education, y & " + " & ".join(msd(g.EDUC[g.EDUC >= 0]) for g in G) + f" & {pk([g.EDUC[g.EDUC >= 0] for g in G])} \\\\",
        "\\midrule", "\\multicolumn{5}{l}{\\textit{OASIS-1 (external)}} \\\\",
        "Group & CDR 0 & CDR 0.5 & CDR $\\geq$1 & $p$ \\\\", "$n$ & " + " & ".join(str(len(h)) for h in H) + " & \\\\",
        "Age, y & " + " & ".join(msd(h.age) for h in H) + f" & {pk([h.age for h in H])} \\\\",
        "Female & " + " & ".join(fem(h, "sex", "Female") for h in H) + f" & {pc(H, 'sex', 'Female')} \\\\",
        "MMSE & " + " & ".join(msd(h.mmse) for h in H) + f" & {pk([h.mmse.dropna() for h in H])} \\\\"]
tex_table("demographics", "Participants. NACC: clinical diagnosis at the baseline UDS visit; age, sex and education from the label "
          "file and image headers. OASIS-1: Clinical Dementia Rating (CDR). $p$: Kruskal--Wallis or chi-square test across the three groups.",
          "tab:demo", "Characteristic & \\multicolumn{3}{c}{Group} & $p$", rows, "lcccc")

top = eff[(eff.cohort == "NACC") & eff.contrast.str.startswith("dementia")].reindex(
    eff[(eff.cohort == "NACC") & eff.contrast.str.startswith("dementia")].d.abs().sort_values(ascending=False).index).head(12).feature
rows = []
fmt = lambda r: f"{r.d:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" + ("" if r.q_fdr < .05 else "$^{\\dagger}$")
for ft in top:
    cells = [fmt(eff[(eff.cohort == c) & eff.contrast.str.startswith(k) & (eff.feature == ft)].iloc[0])
             for c in ("NACC", "OASIS-1") for k in ("MCI", "dementia")]
    rows.append(nice(ft) + " & " + " & ".join(cells) + " \\\\")
tex_table("top_effects", "The twelve regions with the largest volume differences between dementia and normal cognition in NACC, "
          "with the matching OASIS-1 contrasts. Cohen's $d$ of age- and sex-adjusted volumes [95\\% bootstrap CI]; negative = "
          "smaller than in the reference group. $^{\\dagger}$Not significant after Benjamini--Hochberg correction (q$\\geq$0.05).",
          "tab:top", "Region & NACC MCI & NACC dementia & OASIS CDR 0.5 & OASIS CDR $\\geq$1", rows, "lcccc")

rows = []
for t in TN:
    rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{TN[t]}}}}} \\\\")
    for s in ["all", "only: medial temporal", "all minus: medial temporal", "only: lateral temporal", "only: parietal", "only: ventricles and CSF"]:
        r = gst[(gst.task == t) & (gst.set == s)].iloc[0]; mean_, sd_ = mm.loc[(t, s)]
        dtxt = "--" if s == "all" else f"{r.diff_vs_all:+.3f} [{r.diff_ci_low:+.3f}, {r.diff_ci_high:+.3f}]"
        fp = lambda x: "$<$0.001" if x < .001 else f"{x:.3f}"
        ptxt = "--" if s == "all" else (fp(r.p_holm_primary) + "$^{H}$" if pd.notna(r.p_holm_primary) else fp(r.q_bh_within_task))
        rows.append(f"{s.replace('only: ', '').replace('all minus: ', 'all except ').replace('all', 'All regions', 1) if s == 'all' else s.replace('only: ', '').replace('all minus: ', 'all except ')} ({len(pd.read_json(X / 'EXP303' / 'groups.json').sets[s]) if False else ''}) & {mean_:.3f} $\\pm$ {sd_:.3f} & {dtxt} & {ptxt} \\\\".replace(" () ", " "))
tex_table("groups", "Classification by anatomical region group (NACC, logistic regression, 10$\\times$5-fold nested cross-validation). "
          "AUC: mean $\\pm$ SD over repeats. Difference to all regions: AUC of repeat-averaged probabilities with paired bootstrap "
          "95\\% CI. $p$: Holm-adjusted for the six pre-specified comparisons ($^{H}$) or Benjamini--Hochberg q within task. "
          "All groups are listed in Fig.~\\ref{fig:groups}.", "tab:groups", "Feature set & AUC & Difference to all regions & $p$ / $q$", rows, "lccc")

rows = [f"{r.trained_on} & {r.oasis_contrast.replace('>=', '$\\geq$').replace('> 0', '$>$ 0')} & {r.n} ({r.n_pos}) & {r.auc_all:.3f} {r.auc_all_ci} & "
        f"{r.auc_mtl:.3f} {r.auc_mtl_ci} & {r.diff_mtl_minus_all:+.3f} [{r.diff_ci_low:+.3f}, {r.diff_ci_high:+.3f}] \\\\" for r in ext.itertuples()]
tex_table("external", "External test on OASIS-1 of models trained on NACC and applied unchanged: all 51 regional volumes versus the five "
          "medial temporal features. AUC [95\\% bootstrap CI]; difference = medial temporal minus all.", "tab:external",
          "Trained on (NACC) & OASIS-1 contrast & $n$ (positive) & All regions & Medial temporal & Difference", rows, "llcccc")

rows = []
fq = lambda x: "$<$0.001" if x < .001 else f"{x:.3f}"
sel = v[v.dementia_q < .05].index.union(v[v.MCI_q < .05].index)
for r_ in sel:
    a_d = ab.loc[r_]; a_m = asy[(asy.measure == "absolute AI") & asy.contrast.str.startswith("MCI") & (asy.region == r_)].iloc[0]
    rows.append(f"{nice(r_)} & {a_d.mean_ai_ref:.3f} & {a_m.mean_ai_group:.3f} & {a_d.mean_ai_group:.3f} & {a_d.d:.2f} & "
                f"{v.loc[r_, 'dementia_adj_diff_in_ref_sd']:+.2f} ({fq(v.loc[r_, 'dementia_q'])}) & {v.loc[r_, 'MCI_adj_diff_in_ref_sd']:+.2f} ({fq(v.loc[r_, 'MCI_q'])}) \\\\")
tex_table("asymmetry", "Regions whose absolute asymmetry differed from normal cognition after adjustment for age, sex and region volume "
          "(NACC; FDR q$<$0.05 in either contrast). Mean $|$AI$|$: mean absolute asymmetry index $|L-R|/((L+R)/2)$. Adjusted "
          "difference in normal-group SD (q).", "tab:asym",
          "Region & Mean $|$AI$|$ normal & MCI & Dementia & $d$ dementia (unadjusted) & Adjusted dementia & Adjusted MCI", rows, "lcccccc")
print("written:", sorted(p.name for p in F.glob("fig_*.png")), sorted(p.name for p in T.glob("*.tex")))
