"""
15_scan_age_sensitivity.py  (RQ1/RQ2 sensitivity, EXP310)
---------------------------------------------------------
The main analysis (EXP302) adjusts NACC volumes for age at the label (baseline UDS) visit. Scans were taken later, and
the gap differs by group (median 4 / 2 / 1 years in normal / MCI / dementia), so label-visit age is a biased proxy for
age at scan. Here we repeat the NACC effect sizes on the participants whose age at scan is known from the DICOM header
(dataset/metadata/label_gap.csv) and plausible (label-to-scan gap 0-20 years; negative gaps and ages of 110-111, which
are header artefacts, are excluded), adjusting once for label-visit age and once for age at scan, on the same people.
Reported: number significant (BH-FDR), Spearman correlation of the two sets of d, the largest changes, and the
replication correlation with OASIS-1 (EXP302 OASIS effects are recomputed with the same code).
Statistics functions are taken unchanged from 02_regional_effects.py. No bootstrap (point estimates only).
Output: experiments/<EXP id>/{effects_by_age_definition.csv, summary.csv, command.txt}
Usage: python 15_scan_age_sensitivity.py <project dir> <EXP id>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
src = (P / "src" / "analysis" / "02_regional_effects.py").read_text(encoding="utf-8")
ns = {"np": np, "pd": pd, "stats": stats}
for fn in ("bilateral", "design", "effect", "pvals", "bh"):
    i = src.index(f"def {fn}("); j = src.index("\n\n\n", i); exec(src[i:j], ns)

M = P / "dataset" / "metadata"
f = pd.read_csv(M / "features_v2.csv"); m = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex"])
gap = pd.read_csv(M / "label_gap.csv")
n = f[f.has_mri == 1].merge(m, on="NACCID", how="left").merge(gap, on="NACCID", how="left").reset_index(drop=True)
o = pd.read_csv(M / "features_oasis1.csv")
Xn, Xo = ns["bilateral"](n), ns["bilateral"](o)
feats = [c for c in Xn.columns if c in Xo.columns and Xn[c].notna().all() and Xo[c].notna().all() and Xn[c].nunique() > 1 and Xo[c].nunique() > 1]
keep = n.scan_age.notna() & n.label_gap_y.between(0, 20)
print(f"NACC with plausible age at scan: {keep.sum()} of {len(n)}; by group {n[keep].label.value_counts().sort_index().to_dict()}")
n, Xn = n[keep].reset_index(drop=True), Xn[keep].reset_index(drop=True)
g = n.label.map({0: "ref", 1: "MCI", 2: "dementia"}).to_numpy()
sex = n.sex.map({"F": "F", "M": "M"}).fillna("U").to_numpy()
cov = lambda age: pd.DataFrame(dict(age=age.astype(float), F=(sex == "F") * 1., U=(sex == "U") * 1.))
cdr = pd.to_numeric(o.cdr); go = np.where(cdr == 0, "ref", np.where(cdr == 0.5, "MCI", "dementia"))
so = o.sex.map({"Female": "F", "Male": "M"}).fillna("U").to_numpy()
co = pd.DataFrame(dict(age=o.age.astype(float), F=(so == "F") * 1., U=(so == "U") * 1.))
A, Ao = Xn[feats].to_numpy(float), Xo[feats].to_numpy(float)

rows, summ = [], []
for contrast in ("MCI", "dementia"):
    d_o = ns["effect"](Ao, co, go, contrast)
    res = {}
    for label, age in (("label-visit age", n.NACCAGE), ("age at scan", n.scan_age)):
        C = cov(age); d = ns["effect"](A, C, g, contrast); q = ns["bh"](ns["pvals"](A, C, g, contrast)); res[label] = d
        summ.append(dict(contrast=contrast, age_definition=label, n_ref=int((g == "ref").sum()), n_group=int((g == contrast).sum()),
                         n_fdr_sig=int((q < .05).sum()), rho_with_oasis=stats.spearmanr(d, d_o).statistic))
        rows += [dict(contrast=contrast, age_definition=label, feature=ft, d=d[k], q_fdr=q[k]) for k, ft in enumerate(feats)]
    a, b = res["label-visit age"], res["age at scan"]
    summ.append(dict(contrast=contrast, age_definition="agreement", rho_label_vs_scan=stats.spearmanr(a, b).statistic,
                     median_abs_change=np.median(np.abs(b - a)), max_abs_change=np.abs(b - a).max(),
                     feature_max_change=feats[int(np.argmax(np.abs(b - a)))], mean_change=np.mean(b - a)))
E.mkdir(parents=True)
pd.DataFrame(rows).to_csv(E / "effects_by_age_definition.csv", index=False)
S = pd.DataFrame(summ); S.to_csv(E / "summary.csv", index=False); (E / "command.txt").write_text(" ".join(sys.argv) + "\n")
pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
R = pd.DataFrame(rows).pivot_table(index=["contrast", "feature"], columns="age_definition", values="d")
for c in ("MCI", "dementia"):
    print(f"\n{c}: largest effects"); print(R.loc[c].sort_values("age at scan").head(6).round(2).to_string())
