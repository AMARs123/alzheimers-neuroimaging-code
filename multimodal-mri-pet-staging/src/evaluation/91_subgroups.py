"""
91_subgroups.py (EXP026; checklist item 21-04, added 16 Oct 2026)
------------------------------------------------------------------
Does the MRI model work equally well in subgroups? From the stored EXP006 out-of-fold predictions (logistic
regression, probabilities averaged over the 10 repeats; nothing refitted), AUC for normal cognition vs dementia by
sex (from image headers; unknown kept as its own group) and by age tertile at the label visit, with stratified
bootstrap 95% CI (2,000). Sets: MRI and MRI + tau.
Usage: python 91_subgroups.py <project dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

P = Path(sys.argv[1]); E = P / "experiments" / "EXP026"; E.mkdir(exist_ok=True)
pred = pd.read_csv(P / "experiments" / "EXP006" / "predictions.csv.gz")
meta = pd.read_csv(P / "results" / "raw" / "features" / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex"])
rng = np.random.default_rng(7); B = 2000; rows = []
for fs in ["mri", "mri+tau"]:
    g = pred[(pred.task == "cn_ad") & (pred.set == fs) & (pred.model == "lr")].groupby("NACCID").agg(y=("y", "first"), p=("p1", "mean")).reset_index()
    g = g.merge(meta, on="NACCID", how="left"); g["y"] = (g.y == g.y.max()).astype(int)
    g["sex_grp"] = g.sex.map({"F": "female", "M": "male"}).fillna("unknown")
    cuts = g.NACCAGE.quantile([1 / 3, 2 / 3]).to_numpy()
    g["age_grp"] = pd.cut(g.NACCAGE, [-np.inf, cuts[0], cuts[1], np.inf], labels=[f"age <= {cuts[0]:.0f}", f"age {cuts[0]:.0f}-{cuts[1]:.0f}", f"age > {cuts[1]:.0f}"])
    for var in ["sex_grp", "age_grp"]:
        for lev, h in g.groupby(var, observed=True):
            y, p = h.y.to_numpy(), h.p.to_numpy()
            if y.min() == y.max(): continue
            bs = [roc_auc_score(y[i], p[i]) for i in (np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum()) for k in (0, 1)]) for _ in range(B))]
            rows.append(dict(set=fs, variable=var, level=str(lev), n=len(h), n_dementia=int(y.sum()), auc=roc_auc_score(y, p),
                             ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5)))
    # difference male minus female (independent groups; stratified bootstrap within each sex and class)
    f_, m_ = g[g.sex_grp == "female"], g[g.sex_grp == "male"]
    def bs_auc(h):
        y, p = h.y.to_numpy(), h.p.to_numpy(); i = np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum()) for k in (0, 1)])
        return roc_auc_score(y[i], p[i])
    d = np.array([bs_auc(m_) - bs_auc(f_) for _ in range(B)])
    rows.append(dict(set=fs, variable="sex difference", level="male minus female", n=len(f_) + len(m_), n_dementia=int(f_.y.sum() + m_.y.sum()),
                     auc=roc_auc_score(m_.y, m_.p) - roc_auc_score(f_.y, f_.p), ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                     p_boot=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    rows.append(dict(set=fs, variable="sex_grp", level="unknown", n=int((g.sex_grp == "unknown").sum()),
                     n_dementia=int(g[g.sex_grp == "unknown"].y.sum()), auc=np.nan, ci_low=np.nan, ci_high=np.nan))
    # women and men: age and brain-volume-normalised hippocampus not available here; report age by sex and group
    for (sx, y), h in g[g.sex_grp != "unknown"].groupby(["sex_grp", "y"]):
        rows.append(dict(set=fs, variable="median age", level=f"{sx}, {'dementia' if y else 'normal'}", n=len(h), n_dementia=int(h.y.sum()),
                         auc=h.NACCAGE.median(), ci_low=np.nan, ci_high=np.nan))
R = pd.DataFrame(rows); R.to_csv(E / "subgroup_auc.csv", index=False)
(E / "command.txt").write_text("python src/evaluation/91_subgroups.py .\n")
(E / "config.json").write_text(json.dumps(dict(source="EXP006 predictions (lr, cn_ad)", bootstrap=B, seed=7), indent=1))
pd.set_option("display.width", 200); print(R.round(3).to_string(index=False))
