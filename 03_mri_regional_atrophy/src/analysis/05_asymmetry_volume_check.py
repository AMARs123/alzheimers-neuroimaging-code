"""
05_asymmetry_volume_check.py  (sensitivity for EXP304)
------------------------------------------------------
Is the larger absolute asymmetry in dementia explained by smaller regional volume (relative segmentation noise grows
as a structure shrinks)? For every paired region:
  (1) within normal cognition only: Spearman correlation between |AI| and bilateral volume (L+R). A negative value means
      smaller structures look more asymmetric even without disease.
  (2) group effect on |AI| (dementia vs normal; MCI vs normal) from OLS with age, sex AND log bilateral volume as
      covariates; t-test of the group term, BH-FDR. Adjusting for volume is conservative, because volume loss is itself
      part of the disease.
Usage: python 05_asymmetry_volume_check.py <project dir> <EXP id>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
M = P / "dataset" / "metadata"
f = pd.read_csv(M / "features_v2.csv"); m = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex"])
n = f[f.has_mri == 1].merge(m, on="NACCID", how="left").reset_index(drop=True)
pairs = [c[10:] for c in n.columns if c.startswith("mri_vol_L-") and "mri_vol_R-" + c[10:] in n and n[c].nunique() > 1]
sex = n.sex.map({"F": "F", "M": "M"}).fillna("U")
base = np.column_stack([np.ones(len(n)), n.NACCAGE.astype(float), (sex == "F") * 1., (sex == "U") * 1.])


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


rows = []
for r in pairs:
    L, R = n["mri_vol_L-" + r].to_numpy(float), n["mri_vol_R-" + r].to_numpy(float)
    tot = L + R; ai = np.abs(L - R) / (tot / 2)
    ok = tot > 0
    ref = (n.label == 0).to_numpy() & ok
    rho, prho = stats.spearmanr(ai[ref], tot[ref])
    row = dict(region=r, rho_absAI_volume_in_normals=rho, p_rho=prho)
    for lab, name in ((1, "MCI"), (2, "dementia")):
        s = n.label.isin([0, lab]).to_numpy() & ok
        D = np.column_stack([base[s], np.log(tot[s]), (n.label[s] == lab) * 1.])
        beta, *_ = np.linalg.lstsq(D, ai[s], rcond=None); res = ai[s] - D @ beta; dof = len(D) - D.shape[1]
        se = np.sqrt((res ** 2).sum() / dof * np.linalg.inv(D.T @ D)[-1, -1]); t = beta[-1] / se
        sd_ref = ai[s & (n.label == 0).to_numpy()].std(ddof=1)
        row[f"{name}_adj_diff_in_ref_sd"] = beta[-1] / sd_ref; row[f"{name}_p"] = 2 * stats.t.sf(abs(t), dof)
    rows.append(row)
R = pd.DataFrame(rows)
for name in ("MCI", "dementia"): R[f"{name}_q"] = bh(R[f"{name}_p"])
R["q_rho"] = bh(R.p_rho)
E.mkdir(parents=True); R.to_csv(E / "asymmetry_volume_check.csv", index=False)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
json.dump(dict(script="05_asymmetry_volume_check.py", covariates="age, sex, log(L+R volume)"), open(E / "config.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(f"normals: |AI| negatively related to volume (q<0.05) in {((R.q_rho < .05) & (R.rho_absAI_volume_in_normals < 0)).sum()}/{len(R)} regions; "
      f"median rho {R.rho_absAI_volume_in_normals.median():.2f}")
for name in ("MCI", "dementia"):
    print(f"{name}: |AI| larger after volume adjustment, q<0.05, in {((R[f'{name}_q'] < .05) & (R[f'{name}_adj_diff_in_ref_sd'] > 0)).sum()}/{len(R)}")
print(R.sort_values("dementia_p").head(10)[["region", "rho_absAI_volume_in_normals", "dementia_adj_diff_in_ref_sd", "dementia_q", "MCI_adj_diff_in_ref_sd", "MCI_q"]].round(3).to_string(index=False))
