"""
87_confounds.py
---------------
Reviewer-driven checks on stored data (no model of the main analysis is refitted).
  1. Site-label association: chi-square of MRI site x label (MRI cohort and complete-modality cohort); cross-validated AUC
     of a classifier that sees only the one-hot MRI site (normal vs dementia), same 10x5 stratified folds.
  2. Age alone: AUC of age at the label visit for normal vs dementia and normal vs MCI (no fitting; age as score).
  3. Multiplicity: Holm correction of the paired accuracy p-values for MRI vs MRI+PET with logistic regression
     (the reference model) over all tasks and both PET modalities in EXP006.
Output: results/statistics/confounds.csv, results/statistics/paired_EXP006_holm.csv
Usage: python 87_confounds.py <project dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

P = Path(sys.argv[1]); F = P / "results" / "raw" / "features"; S = P / "results" / "statistics"
m = pd.read_csv(F / "meta_v2.csv"); f = pd.read_csv(F / "features_v2.csv", usecols=["NACCID", "has_mri", "has_amy", "has_tau"])
d = m.merge(f, on="NACCID"); rows = []
coh = {"MRI cohort": d[d.has_mri == 1], "complete-modality cohort": d[(d.has_mri == 1) & (d.has_amy == 1) & (d.has_tau == 1)]}
for cname, c in coh.items():
    c = c[c.mri_site.notna()]
    tab = pd.crosstab(c.mri_site, c.label); chi = stats.chi2_contingency(tab)
    rows.append(dict(cohort=cname, check="site x label chi-square (3 labels)", n=len(c), value=chi[0], p=chi[1],
                     note=f"{tab.shape[0]} sites; {int((chi[3] < 5).sum())}/{chi[3].size} expected cells < 5"))
    b = c[c.label.isin([0, 2])]; y = (b.label == 2).astype(int).to_numpy(); X = pd.get_dummies(b.mri_site).to_numpy(float)
    aucs = [roc_auc_score(y, cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), X, y, method="predict_proba",
                                               cv=StratifiedKFold(5, shuffle=True, random_state=100 + r))[:, 1]) for r in range(10)]
    rows.append(dict(cohort=cname, check="site-only classifier AUC, normal vs dementia (10x5 CV)", n=len(b), value=np.mean(aucs),
                     p=np.nan, note=f"SD over repeats {np.std(aucs):.3f}"))
    for task, lab in (("normal vs dementia", 2), ("normal vs MCI", 1)):
        a = c[c.label.isin([0, lab]) & (c.NACCAGE > 0)]
        rows.append(dict(cohort=cname, check=f"age alone AUC, {task}", n=len(a), value=roc_auc_score(a.label == lab, a.NACCAGE),
                         p=stats.mannwhitneyu(a.NACCAGE[a.label == 0], a.NACCAGE[a.label == lab]).pvalue, note="age at label visit"))
R = pd.DataFrame(rows); R.to_csv(S / "confounds.csv", index=False); print(R.to_string())

p = pd.read_csv(S / "paired_EXP006.csv")
h = p[(p.model == "lr") & (p.metric == "accuracy") & (p.set_a == "mri") & p.set_b.isin(["mri+amy", "mri+tau"])].copy()
h = h.sort_values("p_boot").reset_index(drop=True); k = len(h)
h["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * pv for i, pv in enumerate(h.p_boot)]))
h.to_csv(S / "paired_EXP006_holm.csv", index=False)
print(h[["task", "set_b", "diff_b_minus_a", "ci_low", "ci_high", "p_boot", "p_holm"]].to_string())
