"""
06_subgroups.py (EXP206; checklist item 21-04, added 16 Oct 2026)
------------------------------------------------------------------
AUC of the single PET values for normal cognition vs dementia separately in women and men (sex from image headers),
from stored outputs (EXP201 Centiloid, EXP202 tau z); stratified bootstrap 95% CI and an unpaired bootstrap test of
the difference (2,000 resamples). No model is fitted.
Usage: python 06_subgroups.py <project dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

P = Path(sys.argv[1]); X = P / "experiments"; E = X / "EXP206"; E.mkdir(exist_ok=True)
M = pd.read_csv(P / "dataset" / "metadata" / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex"])
A = pd.read_csv(X / "EXP201" / "centiloid_oof.csv").merge(M, on="NACCID", how="left")
T = pd.read_csv(X / "EXP202" / "tau_z.csv").merge(M, on="NACCID", how="left")
rng = np.random.default_rng(7); B = 2000; rows = []


def boot(y, s):
    i = np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum()) for k in (0, 1)]); return roc_auc_score(y[i], s[i])


for name, df, col in [("amyloid Centiloid", A, "cl_oof"), ("tau meta_temporal", T, "z_meta_temporal")]:
    d = df[df.label.isin([0, 2]) & df[col].notna()]; res = {}
    for sx, lab in [("F", "female"), ("M", "male")]:
        h = d[d.sex == sx]; y = (h.label == 2).astype(int).to_numpy(); s = h[col].to_numpy(float)
        bs = np.array([boot(y, s) for _ in range(B)]); res[lab] = (y, s, bs)
        rows.append(dict(summary=name, group=lab, n=len(h), n_dementia=int(y.sum()), auc=roc_auc_score(y, s),
                         ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5)))
    dd = res["male"][2] - res["female"][2]
    rows.append(dict(summary=name, group="male minus female", n=len(d[d.sex.isin(["F", "M"])]), n_dementia=np.nan,
                     auc=roc_auc_score(*res["male"][:2]) - roc_auc_score(*res["female"][:2]), ci_low=np.percentile(dd, 2.5),
                     ci_high=np.percentile(dd, 97.5), p_boot=min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))))
    rows.append(dict(summary=name, group="unknown sex", n=int(d.sex.isna().sum()), n_dementia=int((d[d.sex.isna()].label == 2).sum())))
R = pd.DataFrame(rows); R.to_csv(E / "subgroup_auc.csv", index=False)
(E / "command.txt").write_text("python src/analysis/06_subgroups.py .\n")
(E / "config.json").write_text(json.dumps(dict(inputs=["EXP201", "EXP202", "meta_v2.csv"], bootstrap=B, seed=7), indent=1))
pd.set_option("display.width", 200); print(R.round(3).to_string(index=False))
