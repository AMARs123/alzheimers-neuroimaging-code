"""
04_asymmetry.py  (RQ4: left-right asymmetry; NACC only)
-------------------------------------------------------
NACC images were converted from DICOM with dcm2niix, so left and right are reliable (not so for OASIS-1).
For every paired region: asymmetry index AI = (L - R) / ((L + R) / 2) (positive = left larger).
Two questions per region and contrast (MCI vs normal, dementia vs normal):
  (a) signed AI: does the group shift towards one side?  -> age/sex-adjusted Cohen d, OLS p, BH-FDR
  (b) |AI|: is atrophy more asymmetric in the patient group? -> same statistics on |AI|
Covariate handling and bootstrap as in 02_regional_effects.py (reference = normal cognition; 2,000 resamples).
Usage: python 04_asymmetry.py <project dir> <EXP id>
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
spec = importlib.util.spec_from_file_location("re2", Path(__file__).with_name("02_regional_effects.py"))
M = P / "dataset" / "metadata"
f = pd.read_csv(M / "features_v2.csv"); m = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex"])
n = f[f.has_mri == 1].merge(m, on="NACCID", how="left").reset_index(drop=True)
pairs = [c[10:] for c in n.columns if c.startswith("mri_vol_L-") and "mri_vol_R-" + c[10:] in n and n[c].nunique() > 1]
AI = pd.DataFrame({r: (n["mri_vol_L-" + r] - n["mri_vol_R-" + r]) / ((n["mri_vol_L-" + r] + n["mri_vol_R-" + r]) / 2) for r in pairs})
g = n.label.map({0: "ref", 1: "MCI", 2: "dementia"}).to_numpy()
sex = n.sex.map({"F": "F", "M": "M"}).fillna("U").to_numpy()
C = pd.DataFrame(dict(age=n.NACCAGE.astype(float), F=(sex == "F") * 1., U=(sex == "U") * 1.))

# reuse the exact statistics of EXP302 without running its main body
src = Path(__file__).with_name("02_regional_effects.py").read_text(encoding="utf-8")
ns = {}
for fn in ("design", "effect", "pvals", "bh"):
    i = src.index(f"def {fn}("); j = src.index("\n\n\n", i); exec(src[i:j], {"np": np, "stats": __import__("scipy").stats}, ns)
for k, v in ns.items(): v.__globals__.update(ns)
rng = np.random.default_rng(7); B = 2000
strat = lambda: np.concatenate([rng.choice(np.where(g == k)[0], (g == k).sum()) for k in np.unique(g)])
rows = []
for kind, A in (("signed AI", AI.to_numpy(float)), ("absolute AI", AI.abs().to_numpy(float))):
    for contrast in ("MCI", "dementia"):
        d = ns["effect"](A, C, g, contrast); p = ns["pvals"](A, C, g, contrast); q = ns["bh"](p)
        bs = np.array([ns["effect"](A[i], C.iloc[i].reset_index(drop=True), g[i], contrast) for i in (strat() for _ in range(B))])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for k, r in enumerate(pairs):
            rows.append(dict(measure=kind, contrast=f"{contrast} vs normal", region=r, mean_ai_ref=A[g == "ref", k].mean(),
                             mean_ai_group=A[g == contrast, k].mean(), d=d[k], ci_low=lo[k], ci_high=hi[k], p=p[k], q_fdr=q[k]))
R = pd.DataFrame(rows); E.mkdir(parents=True); R.to_csv(E / "asymmetry.csv", index=False)
json.dump(dict(script="04_asymmetry.py", regions=pairs, bootstrap=B, seed=7), open(E / "config.json", "w"), indent=1)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
pd.set_option("display.width", 200)
for (k, c), x in R.groupby(["measure", "contrast"]):
    print(f"\n{k} | {c}: FDR<0.05 in {(x.q_fdr < .05).sum()}/{len(x)}")
    print(x.reindex(x.d.abs().sort_values(ascending=False).index).head(6)[["region", "mean_ai_ref", "mean_ai_group", "d", "ci_low", "ci_high", "q_fdr"]].round(3).to_string(index=False))
