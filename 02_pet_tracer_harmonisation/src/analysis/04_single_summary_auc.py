"""
04_single_summary_auc.py  (RQ4, EXP204)
--------------------------------------
How well does a single PET number separate the clinical groups? No model is trained: each summary is used directly
as a score, so there is nothing to overfit. Summaries: amyloid Centiloid (out-of-fold, EXP201) and harmonised tau z
(cross-fitted, EXP202) for meta-temporal, Braak I (entorhinal), Braak III/IV and Braak V/VI regions.
Contrasts: normal vs dementia, normal vs MCI, MCI vs dementia.
AUC with stratified bootstrap 95% CI (2,000). Tau AUCs are also given as tracer-stratified AUC (only pairs scanned with
the same tracer are compared), which does not depend on the harmonisation.
Paired comparison amyloid vs tau meta-temporal in participants with both (bootstrap of the AUC difference, two-sided
p); Holm across the three contrasts (pre-specified). Braak comparisons are descriptive.
Output: experiments/<EXP id>/{auc.csv, paired.csv, command.txt}
Usage: python 04_single_summary_auc.py <project dir> <EXP id> <EXP centiloid> <EXP tau>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
A = pd.read_csv(P / "experiments" / sys.argv[3] / "centiloid_oof.csv", usecols=["NACCID", "label", "cl_oof"])
T = pd.read_csv(P / "experiments" / sys.argv[4] / "tau_z.csv")
rng = np.random.default_rng(7); B = 2000
C = {"normal vs dementia": (0, 2), "normal vs MCI": (0, 1), "MCI vs dementia": (1, 2)}
SUM = [("amyloid Centiloid", A, "cl_oof")] + [(f"tau {k}", T, f"z_{k}") for k in ("meta_temporal", "braak1", "braak34", "braak56")]


def strat_auc(y, s, g):
    num = den = 0.0
    for k in np.unique(g):
        a, b = s[(g == k) & (y == 1)], s[(g == k) & (y == 0)]
        if len(a) and len(b):
            num += (a[:, None] > b[None, :]).sum() + .5 * (a[:, None] == b[None, :]).sum(); den += len(a) * len(b)
    return num / den if den else np.nan


def boot_idx(y):
    return np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum()) for k in (0, 1)])


rows = []
for cname, (a, b) in C.items():
    for sname, df, col in SUM:
        x = df[df.label.isin([a, b]) & df[col].notna()]; y = (x.label == b).astype(int).to_numpy(); s = x[col].to_numpy(float)
        auc = roc_auc_score(y, s); bs = [roc_auc_score(y[i], s[i]) for i in (boot_idx(y) for _ in range(B))]
        r = dict(contrast=cname, summary=sname, n=len(y), n_pos=int(y.sum()), auc=auc, ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5))
        if "tau_tracer" in x: r["auc_within_tracer"] = strat_auc(y, s, x.tau_tracer.to_numpy())
        rows.append(r)
R = pd.DataFrame(rows)

pr = []
both = T[T.pos_oof.notna() & T.cl_oof.notna()]
for cname, (a, b) in C.items():
    x = both[both.label.isin([a, b])]; y = (x.label == b).astype(int).to_numpy()
    s1, s2 = x.cl_oof.to_numpy(float), x.z_meta_temporal.to_numpy(float)
    obs = roc_auc_score(y, s2) - roc_auc_score(y, s1)
    diffs = np.array([roc_auc_score(y[i], s2[i]) - roc_auc_score(y[i], s1[i]) for i in (boot_idx(y) for _ in range(B))])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    pr.append(dict(contrast=cname, n=len(y), auc_amyloid=roc_auc_score(y, s1), auc_tau=roc_auc_score(y, s2), diff_tau_minus_amyloid=obs,
                   ci_low=np.percentile(diffs, 2.5), ci_high=np.percentile(diffs, 97.5), p=p))
PR = pd.DataFrame(pr)
o = np.argsort(PR.p.to_numpy()); adj = np.maximum.accumulate(np.minimum(1, PR.p.to_numpy()[o] * (len(o) - np.arange(len(o)))))
PR.loc[PR.index[o], "p_holm"] = adj
E.mkdir(parents=True)
R.to_csv(E / "auc.csv", index=False); PR.to_csv(E / "paired.csv", index=False); (E / "command.txt").write_text(" ".join(sys.argv) + "\n")
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False)); print(PR.round(3).to_string(index=False))
