"""
12_wm_asymmetry_by_site.py  (RQ4 site check, EXP309)
----------------------------------------------------
Records the check that was first run ad hoc after EXP305: is the larger cerebral white-matter |AI| in dementia a site or
scanner artefact? Within every site with at least 3 normal and 3 dementia participants, and within each scanner
manufacturer, we report the median |AI| per group and a Mann-Whitney test (descriptive; no correction).
Output: experiments/<EXP id>/{by_site.csv, by_manufacturer.csv, command.txt}
Usage: python 12_wm_asymmetry_by_site.py <project dir> <EXP id>
"""
import sys
from pathlib import Path
import pandas as pd
from scipy.stats import mannwhitneyu

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
M = P / "dataset" / "metadata"
f = pd.read_csv(M / "features_v2.csv"); m = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "mri_site", "mri_manufacturer"])
n = f[(f.has_mri == 1) & f.label.isin([0, 2])].merge(m, on="NACCID", how="left")
L, R = n["mri_vol_L-Cerebral-White-Matter"], n["mri_vol_R-Cerebral-White-Matter"]
n["abs_ai"] = ((L - R) / ((L + R) / 2)).abs(); n["group"] = n.label.map({0: "normal", 2: "dementia"})


def table(col, min_n):
    rows = []
    for k, x in n.dropna(subset=[col]).groupby(col):
        a, b = x.abs_ai[x.group == "normal"], x.abs_ai[x.group == "dementia"]
        if len(a) >= min_n and len(b) >= min_n:
            rows.append({col: k, "n_normal": len(a), "n_dementia": len(b), "median_normal": a.median(), "median_dementia": b.median(),
                         "p_mannwhitney": mannwhitneyu(b, a).pvalue})
    return pd.DataFrame(rows)


E.mkdir(parents=True)
S = table("mri_site", 3); S.to_csv(E / "by_site.csv", index=False)
V = table("mri_manufacturer", 3); V.to_csv(E / "by_manufacturer.csv", index=False)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
pd.set_option("display.width", 200); print(S.round(4).to_string(index=False)); print(V.round(4).to_string(index=False))
