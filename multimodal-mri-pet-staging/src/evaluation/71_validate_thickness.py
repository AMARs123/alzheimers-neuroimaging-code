"""
71_validate_thickness.py
------------------------
Compare our DiReCT regional cortical thickness (47_thickness.py) with the UC Davis FreeSurfer thickness of the SCAN core lab
(UCDMRISBM, columns lh-<region>-avgth / rh-<region>-avgth), matched on NACCID + MRI date. Validation only - the core-lab
values are never used as features.

Usage: python 71_validate_thickness.py <features_thickness.csv> <meta.csv> <tables dir> <out dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

thk = pd.read_csv(sys.argv[1]); meta = pd.read_csv(sys.argv[2]); tabs = Path(sys.argv[3]); out = Path(sys.argv[4])
ucd = pd.read_csv(sorted(tabs.glob("UCDMRISBM*.csv"))[0], low_memory=False)
ucd.columns = [c.strip() for c in ucd.columns]
low = {c.lower(): c for c in ucd.columns}
ucd = ucd.rename(columns={low["naccid"]: "NACCID"})
date_col = low.get("scandt") or low.get("scandate")
ucd["_d"] = pd.to_datetime(ucd[date_col], errors="coerce").dt.date.astype(str)
meta["_d"] = pd.to_datetime(meta["mri_date"], errors="coerce").dt.date.astype(str)
d = thk.merge(meta[["NACCID", "_d"]], on="NACCID").merge(ucd, on=["NACCID", "_d"], how="inner").drop_duplicates("NACCID")
print(f"{len(thk)} subjects with our thickness, {len(d)} matched to a UC Davis row on NACCID + MRI date")

rows = []
for c in thk.columns:
    if not (c.startswith("thk_L-") or c.startswith("thk_R-")):
        continue
    hemi, reg = ("lh", c[6:]) if c.startswith("thk_L-") else ("rh", c[6:])
    u = low.get(f"{hemi}-{reg}-avgth")
    if u is None:
        continue
    x, y = pd.to_numeric(d[c], errors="coerce"), pd.to_numeric(d[u], errors="coerce")
    ok = x.notna() & y.notna() & (y > 0)
    if ok.sum() >= 30:
        rows.append(dict(hemi=hemi, region=reg, n=int(ok.sum()), pearson_r=float(np.corrcoef(x[ok], y[ok])[0, 1]),
                         spearman_r=float(x[ok].corr(y[ok], method="spearman")), ours_mean_mm=float(x[ok].mean()),
                         ucd_mean_mm=float(y[ok].mean()), bias_mm=float((x[ok] - y[ok]).mean())))
R = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
# whole-cortex mean of the matched parcels (unweighted) per subject
ours = [f"thk_{'L' if r.hemi == 'lh' else 'R'}-{r.region}" for r in R.itertuples()]
theirs = [low[f"{r.hemi}-{r.region}-avgth"] for r in R.itertuples()]
if len(R):
    a, b = d[ours].mean(1), d[theirs].apply(pd.to_numeric, errors="coerce").mean(1)
    ok = a.notna() & b.notna()
    print(f"mean over {len(R)} parcels: r = {np.corrcoef(a[ok], b[ok])[0, 1]:.3f} (n = {ok.sum()}); ours {a[ok].mean():.2f} mm, UC Davis {b[ok].mean():.2f} mm")
    print(f"per-parcel r: median {R.pearson_r.median():.3f}, range {R.pearson_r.min():.3f} to {R.pearson_r.max():.3f}; median bias {R.bias_mm.median():.2f} mm")
    print(R.head(8).to_string(index=False)); print(R.tail(8).to_string(index=False))
R.to_csv(out / "validation_thickness_vs_ucd.csv", index=False)
