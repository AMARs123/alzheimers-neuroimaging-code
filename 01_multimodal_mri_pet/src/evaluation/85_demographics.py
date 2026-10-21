"""
85_demographics.py
------------------
Table 1 (participant characteristics) from the label file fields carried in meta_v2.csv (first UDS visit: NACCAGE,
EDUC, CDRGLOB, NACCMMSE; sex and site/scanner from DICOM headers) and label_gap.csv (age at scan from DICOM headers).
Cohorts: MRI cohort (has_mri, n=405) and complete-modality cohort (MRI + amyloid + tau PET passing QC).
Missing codes: NACC negative values are missing. Scan-to-label interval is reported only when the DICOM age at scan is
plausible (50-100 y) and the interval lies in [-1, 20] y; the number of excluded values is reported.
Tests: Kruskal-Wallis (continuous), chi-square (categorical), across the three labels.
Usage: python 85_demographics.py <project dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

P = Path(sys.argv[1]); F = P / "results" / "raw" / "features"
m = pd.read_csv(F / "meta_v2.csv"); f = pd.read_csv(F / "features_v2.csv", usecols=["NACCID", "has_mri", "has_amy", "has_tau"])
g = pd.read_csv(F / "label_gap.csv"); d = m.merge(f, on="NACCID").merge(g, on="NACCID", how="left")
d["mmse"] = d.NACCMMSE.where(d.NACCMMSE >= 0)
ok = d.scan_age.between(50, 100) & d.label_gap_y.between(-1, 20); d["gap"] = d.label_gap_y.where(ok)
d["gap_excluded"] = d.label_gap_y.notna() & ~ok
LAB = {0: "Normal", 1: "MCI", 2: "Dementia"}
coh = {"MRI cohort": d[d.has_mri == 1], "Complete-modality cohort": d[(d.has_mri == 1) & (d.has_amy == 1) & (d.has_tau == 1)]}

rows, tex = [], []
msd = lambda x: f"{x.mean():.1f} ({x.std():.1f})" if x.notna().sum() else "--"
miqr = lambda x: f"{x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]" if x.notna().sum() else "--"
for cname, c in coh.items():
    G = [c[c.label == k] for k in (0, 1, 2)]
    def cont(name, col, fmt):
        vals = [fmt(x[col]) for x in G]; ns = [int(x[col].notna().sum()) for x in G]
        xs = [x[col].dropna() for x in G]; p = stats.kruskal(*xs).pvalue if all(len(x) > 1 for x in xs) else np.nan
        return [name] + [f"{v} (n={n})" if n < len(x) else v for v, n, x in zip(vals, ns, G)] + [f"{p:.3f}" if p == p else "--"]
    def cat(name, mask_col, val):
        k = [int((x[mask_col] == val).sum()) for x in G]; n = [int(x[mask_col].notna().sum()) for x in G]
        tab = np.array([k, [a - b for a, b in zip(n, k)]]); p = stats.chi2_contingency(tab)[1] if tab.min() >= 0 and tab.sum(0).min() > 0 else np.nan
        return [name] + [f"{a}/{b} ({100 * a / b:.0f}\\%)" if b else "--" for a, b in zip(k, n)] + [f"{p:.3f}"]
    block = [["$n$"] + [str(len(x)) for x in G] + [""],
             cont("Age at label visit, y, mean (SD)", "NACCAGE", msd),
             cat("Female, known sex", "sex", "F"),
             cont("Education, y, mean (SD)", "EDUC", msd),
             cont("MMSE, mean (SD)", "mmse", msd),
             cont("Global CDR, median [IQR]", "CDRGLOB", miqr),
             cont("Scan minus label visit, y, median [IQR]", "gap", miqr),
             ["Sex unknown"] + [str(int(x.sex.isna().sum())) for x in G] + [""],
             ["Interval not usable (no/implausible DICOM age)"] + [str(int((x.gap.isna()).sum())) for x in G] + [""],
             ["MRI sites / manufacturers"] + [f"{x.mri_site.nunique()} / {x.mri_manufacturer.nunique()}" for x in G] + [""]]
    if cname.startswith("Complete"):
        for col, nm in (("amy_tracer", "Amyloid tracer"), ("tau_tracer", "Tau tracer")):
            for t in sorted(c[col].dropna().unique()):
                block.append([f"{nm}: {t}"] + [str(int((x[col] == t).sum())) for x in G] + [""])
    tex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{cname}}}}} \\\\")
    tex += [" & ".join(r) + " \\\\" for r in block]
    rows += [dict(cohort=cname, variable=r[0], normal=r[1], mci=r[2], dementia=r[3], p=r[4]) for r in block]

out = P / "results" / "statistics" / "demographics.csv"; pd.DataFrame(rows).to_csv(out, index=False)
T = P / "paper" / "tables" / "demographics.tex"
T.write_text("\\begin{table}[t]\n\\centering\\small\n\\caption{Participant characteristics. Clinical variables are from the "
             "first UDS visit that defines the label; sex and age at scan from DICOM headers. $p$: Kruskal--Wallis or "
             "chi-square test across the three groups. MMSE is missing for most participants (UDS version 3 uses the MoCA, "
             "which was not available to us).}\\label{tab:demo}\n\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccc}\n\\toprule\n"
             " & Normal & MCI & Dementia & $p$ \\\\\n\\midrule\n" + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n", encoding="utf-8")
print(pd.DataFrame(rows).to_string()); print("excluded implausible intervals:", int(d.gap_excluded.sum()))
