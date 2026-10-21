"""
88_leave_site_out.py  (EXP024)
------------------------------
Is the MRI model learning site rather than atrophy? Site alone predicts normal vs dementia with AUC ~0.80
(87_confounds.py), so we compare, on the same participants (MRI cohort, known site, normal vs dementia):
  (a) stratified 5-fold CV (participants of one site can be in training and test), 10 repeats
  (b) site-grouped 5-fold CV (all participants of a site in the same fold, so test sites are unseen), 10 repeats with
      random assignment of sites to folds
Pipeline as in the main analysis for MRI: median imputation, scaling, ANOVA feature selection, class-weighted logistic
regression; inner 3-fold grid search on balanced accuracy within the training part. Features: all 122 MRI features.
Also reported: within-site AUC, i.e. AUC of the grouped-CV out-of-fold scores computed within each site that has
both classes, pooled by a site-stratified AUC (Mann-Whitney comparisons only between participants of the same site).
Output: experiments/EXP024/{metrics.csv, predictions.csv, command.txt}
Usage: python 88_leave_site_out.py <project dir>
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

P = Path(sys.argv[1]); E = P / "experiments" / "EXP024"
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
F = P / "results" / "raw" / "features"
feat = pd.read_csv(F / "features_v2.csv"); meta = pd.read_csv(F / "meta_v2.csv", usecols=["NACCID", "mri_site"])
cols = [c for c in feat.select_dtypes("number").columns if c.startswith("mri_")]
d = feat[["NACCID", "label", "has_mri"] + cols].merge(meta, on="NACCID")
d = d[(d.has_mri == 1) & d.label.isin([0, 2]) & d.mri_site.notna()].reset_index(drop=True)
X = d[cols].to_numpy(float); y = (d.label == 2).astype(int).to_numpy(); site = d.mri_site.astype(str).to_numpy()
print(f"n={len(y)} ({(y == 0).sum()} normal, {y.sum()} dementia), {len(set(site))} sites, {len(cols)} features")

def model():
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif)),
                     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    return GridSearchCV(pipe, {"sel__k": [20, 40, 80, "all"], "clf__C": [0.01, 0.1, 1.0]}, scoring="balanced_accuracy",
                        cv=StratifiedKFold(3, shuffle=True, random_state=0), n_jobs=-1)

def site_auc(y, p, s):
    num = den = 0
    for g in set(s):
        m = s == g; a, b = p[m & (y == 1)], p[m & (y == 0)]
        if len(a) and len(b):
            num += ((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()); den += len(a) * len(b)
    return num / den if den else np.nan, den

rows, preds = [], []
for r in range(10):
    rng = np.random.default_rng(200 + r)
    splits = {"stratified": list(StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y))}
    perm = {g: i for i, g in enumerate(rng.permutation(sorted(set(site))))}; grp = np.array([perm[g] for g in site])
    splits["site-grouped"] = list(GroupKFold(5).split(X, y, grp))
    for name, sp in splits.items():
        p = np.zeros(len(y))
        for tr, te in sp:
            p[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        wa, pairs = site_auc(y, p, site)
        rows.append(dict(repeat=r, scheme=name, auc=roc_auc_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p >= .5),
                         within_site_auc=wa, within_site_pairs=pairs))
        preds.append(pd.DataFrame(dict(NACCID=d.NACCID, site=site, y=y, repeat=r, scheme=name, p=p)))
    print(pd.DataFrame(rows).tail(2).round(3).to_string(header=r == 0), flush=True)

R = pd.DataFrame(rows); E.mkdir(parents=True)
S = R.groupby("scheme")[["auc", "balanced_accuracy", "within_site_auc"]].agg(["mean", "std"]).round(4)
R.to_csv(E / "metrics_by_repeat.csv", index=False); S.to_csv(E / "metrics.csv"); pd.concat(preds).to_csv(E / "predictions.csv", index=False)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n"); print(S.to_string())
