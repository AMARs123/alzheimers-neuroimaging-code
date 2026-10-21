"""
01_centiloid_calibration.py  (RQ1, EXP201)
-----------------------------------------
Can our own MRI-based amyloid quantification (Paper 1 pipeline: FastSurfer regions, rigid PET-MRI registration, whole
cerebellum reference; feature amy_comp_global) be placed on the Centiloid scale, and how well does it agree with the
SCAN core laboratory (UC Berkeley, MRI-free GAAIN pipeline)?

Participants: amyloid scans that passed our QC (389) and were matched to a core-lab row on NACCID + scan date with a
Centiloid value (286). PiB scans acquired 40-60 min (102) have no core-lab Centiloid and SUVRs depend on the window,
so they cannot be calibrated and are excluded here (reported separately).
Method: per tracer, a linear map Centiloid = a + b * SUVR is fitted with 10-fold CV (out-of-fold predictions only).
Positivity: in each training fold, the cut-off on our Centiloid that maximises Cohen's kappa against the core-lab
amyloid status is chosen and applied to the test fold. The diagnosis is never used.
Reported per tracer and pooled: Pearson r, Lin's concordance correlation (CCC), mean difference (bias) and 95% limits
of agreement (Bland-Altman), kappa / accuracy / sensitivity / specificity for positivity, with participant bootstrap
CIs (2,000). Also the final per-tracer maps fitted on all data (for description only).
Output: experiments/<EXP id>/{centiloid_oof.csv, agreement.csv, maps.csv, command.txt}
Usage: python 01_centiloid_calibration.py <project dir> <EXP id>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
M = P / "dataset" / "metadata"
f = pd.read_csv(M / "features_v2.csv", usecols=["NACCID", "label", "amy_comp_global"])
m = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "amy_tracer", "amy_window", "amy_qc_ok", "mri_site"])
ref = pd.read_csv(M / "validation_amyloid_pairs.csv", usecols=["NACCID", "AMYLOID_STATUS", "CENTILOIDS"])
d = m[m.amy_qc_ok == 1].merge(f, on="NACCID").merge(ref, on="NACCID", how="left")
print(f"amyloid scans passing QC: {len(d)}; with core-lab Centiloid: {d.CENTILOIDS.notna().sum()}")
print("without Centiloid:", d[d.CENTILOIDS.isna()].groupby(["amy_tracer", "amy_window"]).size().to_dict())
d = d[d.CENTILOIDS.notna() & d.AMYLOID_STATUS.notna()].reset_index(drop=True)
rng = np.random.default_rng(7); B = 2000


def best_cut(cl, st):
    grid = np.unique(np.round(cl, 1))
    k = [cohen_kappa_score(st, (cl >= c).astype(int)) for c in grid]
    return grid[int(np.argmax(k))]


d["cl_oof"] = np.nan; d["pos_oof"] = np.nan; d["cut_fold"] = np.nan
for tr_name, g in d.groupby("amy_tracer"):
    idx = g.index.to_numpy()
    for tr, te in KFold(10, shuffle=True, random_state=7).split(idx):
        a, b = idx[tr], idx[te]
        coef = np.polyfit(d.loc[a, "amy_comp_global"], d.loc[a, "CENTILOIDS"], 1)
        d.loc[b, "cl_oof"] = np.polyval(coef, d.loc[b, "amy_comp_global"])
        cut = best_cut(np.asarray(np.polyval(coef, d.loc[a, "amy_comp_global"])), d.loc[a, "AMYLOID_STATUS"].astype(int).to_numpy())
        d.loc[b, "pos_oof"] = (d.loc[b, "cl_oof"] >= cut).astype(int); d.loc[b, "cut_fold"] = cut


def agree(x):
    a, r, s, p = x.cl_oof.to_numpy(), x.CENTILOIDS.to_numpy(float), x.AMYLOID_STATUS.astype(int).to_numpy(), x.pos_oof.astype(int).to_numpy()
    diff = a - r; ccc = 2 * np.cov(a, r, ddof=0)[0, 1] / (a.var() + r.var() + (a.mean() - r.mean()) ** 2)
    return dict(n=len(x), n_pos_ref=int(s.sum()), pearson_r=pearsonr(a, r)[0], ccc=ccc, bias=diff.mean(),
                loa_low=diff.mean() - 1.96 * diff.std(ddof=1), loa_high=diff.mean() + 1.96 * diff.std(ddof=1),
                kappa=cohen_kappa_score(s, p), accuracy=(s == p).mean(), sensitivity=p[s == 1].mean(), specificity=1 - p[s == 0].mean())


rows = []
for name, x in [("ALL", d)] + list(d.groupby("amy_tracer")):
    est = agree(x); est["tracer"] = name
    bs = [agree(x.iloc[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    for k in ("pearson_r", "ccc", "bias", "kappa", "accuracy"):
        v = np.array([b_[k] for b_ in bs]); est[k + "_ci"] = f"[{np.nanpercentile(v, 2.5):.3f}, {np.nanpercentile(v, 97.5):.3f}]"
    rows.append(est)
A = pd.DataFrame(rows)
maps = []
for tr_name, g in d.groupby("amy_tracer"):
    b_, a_ = np.polyfit(g.amy_comp_global, g.CENTILOIDS, 1)
    maps.append(dict(tracer=tr_name, n=len(g), windows=";".join(sorted(g.amy_window.unique())), slope=b_, intercept=a_,
                     median_fold_cut=g.cut_fold.median(), suvr_min=g.amy_comp_global.min(), suvr_max=g.amy_comp_global.max()))
E.mkdir(parents=True)
d.to_csv(E / "centiloid_oof.csv", index=False); A.to_csv(E / "agreement.csv", index=False); pd.DataFrame(maps).to_csv(E / "maps.csv", index=False)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
pd.set_option("display.width", 250); print(A.round(3).to_string(index=False)); print(pd.DataFrame(maps).round(3).to_string(index=False))
