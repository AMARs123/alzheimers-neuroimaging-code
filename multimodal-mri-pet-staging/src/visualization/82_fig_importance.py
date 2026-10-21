"""
82_fig_importance.py
--------------------
Bar chart of region-wise importance from importance_regions.csv of a 60_importance.py experiment:
share of total |SHAP| (left) and grouped permutation importance with its SD over folds (right), top N regions.

Usage: python 82_fig_importance.py <experiment dir> <out dir> [N] [title]
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exp = Path(sys.argv[1]); out = Path(sys.argv[2]) / exp.name; out.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
title = sys.argv[4] if len(sys.argv) > 4 else exp.name
d = pd.read_csv(exp / "importance_regions.csv").sort_values("shap_share", ascending=False).head(n).iloc[::-1]
col = d.modality.map({"mri": "#3b6fb6", "tau": "#c8553d", "amy": "#5a9e6f"}).fillna("0.5")
lab = d.modality.map({"mri": "MRI", "tau": "Tau PET", "amy": "Amyloid PET"}).fillna(d.modality) + ": " + d.region
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, ax = plt.subplots(1, 2, figsize=(8.6, 0.3 * n + 1.2), sharey=True)
ax[0].barh(lab, 100 * d.shap_share, color=col); ax[0].set_xlabel("Share of total |SHAP| (%)")
ax[1].barh(lab, d.perm_drop_mean, xerr=d.perm_drop_sd, color=col, error_kw=dict(lw=0.7, ecolor="0.4"))
ax[1].axvline(0, color="0.5", lw=0.6); ax[1].set_xlabel("Grouped permutation importance\n(drop in score; line = SD over folds)")
fig.suptitle(title, fontsize=10)
for ext in ("png", "pdf"):
    fig.savefig(out / f"region_importance.{ext}")
print("written", out)
