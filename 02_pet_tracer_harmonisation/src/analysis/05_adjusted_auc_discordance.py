"""
05_adjusted_auc_discordance.py  (EXP205; added in the improvement pass of 15 Oct 2026, after the main analyses)
-----------------------------------------------------------------------------------------------------------------
Uses stored outputs only (EXP201 centiloid_oof.csv, EXP202 tau_z.csv) plus age and sex from meta_v2.csv.

A. Age- and sex-adjusted AUC. The clinical groups differ in age and sex, and amyloid rises with age. For each contrast
   and PET summary we fit a linear regression of the PET value on age (label visit) and sex (M / F / unknown) in
   the less impaired group only, subtract the prediction from everyone, and compute the AUC of the residual
   (covariate-adjusted ROC in the sense of Janes & Pepe). Bootstrap 2,000 resamples within groups, refitting the
   adjustment in every resample. The AUC of age alone is given for context.
B. Amyloid-status discordance. Which scans do our pipeline and the core laboratory classify differently? We compare
   the core-laboratory Centiloid of discordant and concordant scans and how close they lie to the in-fold cut-off.
C. ROC curves of single PET values (figure).

Usage: python 05_adjusted_auc_discordance.py <project dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(sys.argv[1]); X = P / "experiments"; E = X / "EXP205"; E.mkdir(exist_ok=True)
FIG = P / "figures" / "final"
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
M = pd.read_csv(P / "dataset" / "metadata" / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex"])
A = pd.read_csv(X / "EXP201" / "centiloid_oof.csv").merge(M, on="NACCID", how="left")
T = pd.read_csv(X / "EXP202" / "tau_z.csv").merge(M, on="NACCID", how="left")
rng = np.random.default_rng(7); B = 2000
C = {"normal vs dementia": (0, 2), "normal vs MCI": (0, 1), "MCI vs dementia": (1, 2)}
SUM = [("age", A, "NACCAGE"), ("amyloid Centiloid", A, "cl_oof"), ("tau meta_temporal", T, "z_meta_temporal"), ("tau braak1", T, "z_braak1"),
       ("tau braak34", T, "z_braak34"), ("tau braak56", T, "z_braak56")]


def design(d):
    sex = d.sex.astype(str).str.lower()
    return np.column_stack([np.ones(len(d)), d.NACCAGE.to_numpy(float), (sex == "m").astype(float), (sex == "f").astype(float)])


def adj_auc(d, col, b):
    y = (d.label == b).astype(int).to_numpy(); s = d[col].to_numpy(float)
    ref = y == 0; Xd = design(d)
    beta, *_ = np.linalg.lstsq(Xd[ref], s[ref], rcond=None)
    return roc_auc_score(y, s - Xd @ beta)


rows = []
for cname, (a, b) in C.items():
    for sname, df, col in SUM:
        d = df[df.label.isin([a, b]) & df[col].notna() & df.NACCAGE.notna()].reset_index(drop=True)
        y = (d.label == b).astype(int).to_numpy()
        raw = roc_auc_score(y, d[col]); adj = adj_auc(d, col, b) if sname != "age" else np.nan
        bs_raw, bs_adj = [], []
        for _ in range(B):
            i = np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum()) for k in (0, 1)]); dd = d.iloc[i].reset_index(drop=True)
            bs_raw.append(roc_auc_score(y[i], dd[col]))
            if sname != "age": bs_adj.append(adj_auc(dd, col, b))
        rows.append(dict(contrast=cname, summary=sname, n=len(d), auc=raw, ci_low=np.percentile(bs_raw, 2.5), ci_high=np.percentile(bs_raw, 97.5),
                         auc_adjusted=adj, adj_ci_low=np.percentile(bs_adj, 2.5) if bs_adj else np.nan,
                         adj_ci_high=np.percentile(bs_adj, 97.5) if bs_adj else np.nan))
R = pd.DataFrame(rows); R.to_csv(E / "adjusted_auc.csv", index=False)
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))

# ---- B. discordance
D = A[A.CENTILOIDS.notna() & A.pos_oof.notna() & A.AMYLOID_STATUS.notna()].copy()
D["discordant"] = D.pos_oof != D.AMYLOID_STATUS
D["dist_core_to_cut"] = (D.CENTILOIDS - D.cut_fold).abs()
D["dist_own_to_cut"] = (D.cl_oof - D.cut_fold).abs()
disc, conc = D[D.discordant], D[~D.discordant]
summ = dict(n=len(D), n_discordant=int(D.discordant.sum()), share_discordant=float(D.discordant.mean()),
            discordant_own_pos_core_neg=int(((D.pos_oof == 1) & (D.AMYLOID_STATUS == 0)).sum()),
            discordant_own_neg_core_pos=int(((D.pos_oof == 0) & (D.AMYLOID_STATUS == 1)).sum()),
            median_core_cl_discordant=float(disc.CENTILOIDS.median()), iqr_core_cl_discordant=[float(disc.CENTILOIDS.quantile(.25)), float(disc.CENTILOIDS.quantile(.75))],
            median_dist_core_to_cut_discordant=float(disc.dist_core_to_cut.median()), median_dist_core_to_cut_concordant=float(conc.dist_core_to_cut.median()),
            p_mannwhitney_dist=float(mannwhitneyu(disc.dist_core_to_cut, conc.dist_core_to_cut).pvalue),
            discordant_within_15cl_core=int((disc.dist_core_to_cut <= 15).sum()),
            concordant_within_15cl_core=int((conc.dist_core_to_cut <= 15).sum()),
            share_within_15cl_discordant_if_near=float((D[D.dist_core_to_cut <= 15].discordant).mean()),
            share_discordant_if_far=float((D[D.dist_core_to_cut > 15].discordant).mean()),
            n_near=int((D.dist_core_to_cut <= 15).sum()), n_far=int((D.dist_core_to_cut > 15).sum()))
by_tr = D.groupby("amy_tracer").agg(n=("discordant", "size"), n_discordant=("discordant", "sum")).reset_index()
by_tr.to_csv(E / "discordance_by_tracer.csv", index=False)
(E / "discordance.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1)); print(by_tr)

# ---- figures: discordance (core-lab CL vs own CL, discordant highlighted) and ROC curves
fig, ax = plt.subplots(figsize=(4.4, 4.0))
q1, q3 = D.cut_fold.quantile(.25), D.cut_fold.quantile(.75)  # interquartile range of the in-fold cut-offs (all scans)
ax.axvspan(q1, q3, color="0.88", lw=0)
ax.axhspan(q1, q3, color="0.88", lw=0)
summ["cut_fold_iqr"] = [float(q1), float(q3)]; (E / "discordance.json").write_text(json.dumps(summ, indent=1))
ax.scatter(conc.CENTILOIDS, conc.cl_oof, s=10, color="#9bb7d4", label=f"Same status (n={len(conc)})")
ax.scatter(disc.CENTILOIDS, disc.cl_oof, s=22, color="#d9534f", marker="D", edgecolor="white", lw=0.5, label=f"Different status (n={len(disc)})")
ax.set_xlim(-40, 80); ax.set_ylim(-40, 80); ax.plot([-40, 80], [-40, 80], color="0.6", lw=0.8, ls=":")
ax.set_xlabel("Core-laboratory Centiloid"); ax.set_ylabel("Our out-of-fold Centiloid")
ax.legend(frameon=False, fontsize=8, loc="upper left")
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_discordance.{ext}")
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.5))
SETS = [("amyloid Centiloid", A, "cl_oof", "#2a78d6", "-"), ("tau meta-temporal", T, "z_meta_temporal", "#eb6834", "-"),
        ("tau Braak I", T, "z_braak1", "#eda100", "--"), ("tau Braak V/VI", T, "z_braak56", "#7a4fb3", ":")]
for ax, (cname, (a, b)) in zip(axs, C.items()):
    for lab, df, col, colr, ls in SETS:
        d = df[df.label.isin([a, b]) & df[col].notna()]; y = (d.label == b).astype(int)
        fpr, tpr, _ = roc_curve(y, d[col]); ax.plot(fpr, tpr, color=colr, ls=ls, lw=1.6, label=f"{lab} ({roc_auc_score(y, d[col]):.3f})")
    ax.plot([0, 1], [0, 1], color="0.75", lw=0.8); ax.set_title(cname[0].upper() + cname[1:]); ax.set_xlabel("1 - specificity"); ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right", frameon=False, title="AUC", title_fontsize=7)
axs[0].set_ylabel("Sensitivity")
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_roc.{ext}")
plt.close(fig)
(E / "command.txt").write_text("python src/analysis/05_adjusted_auc_discordance.py .\n")
(E / "config.json").write_text(json.dumps(dict(inputs=["EXP201", "EXP202", "dataset/metadata/meta_v2.csv"], bootstrap=B, seed=7,
                                               adjustment="OLS on age + sex (male/female/unknown) in the less impaired group; AUC of residual",
                                               near_cutoff_band_cl=15), indent=1))
