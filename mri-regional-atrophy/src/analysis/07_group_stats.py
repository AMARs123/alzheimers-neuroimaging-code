"""
07_group_stats.py  (statistics for EXP303)
------------------------------------------
For each task and feature set: AUC of the out-of-fold probability averaged over the 10 repeats, with a 95% CI from a
stratified bootstrap of participants (2,000 resamples), and the paired difference to the full model ("all") with a
two-sided bootstrap p-value. Pre-specified primary comparisons (Holm across the 6 = 3 tasks x 2): "only: medial
temporal" vs "all" and "all minus: medial temporal" vs "all". All other differences: BH-FDR within task (descriptive).
Usage: python 07_group_stats.py <project dir> <source EXP id> <EXP id>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

P = Path(sys.argv[1]); S = P / "experiments" / sys.argv[2]; E = P / "experiments" / sys.argv[3]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
pr = pd.read_csv(S / "predictions.csv.gz")
avg = pr.groupby(["task", "set", "NACCID", "y"]).p.mean().reset_index()
rng = np.random.default_rng(7); B = 2000
PRIMARY = ["only: medial temporal", "all minus: medial temporal"]


def holm(p):
    p = np.asarray(p); o = np.argsort(p); k = len(p)
    adj = np.minimum(1, np.maximum.accumulate([(k - i) * p[j] for i, j in enumerate(o)])); out = np.empty(k); out[o] = adj; return out


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


rows = []
for task, g in avg.groupby("task"):
    W = g.pivot(index=["NACCID", "y"], columns="set", values="p"); y = W.index.get_level_values("y").to_numpy(); A = W.to_numpy()
    sets = list(W.columns); ia = sets.index("all")
    idx = [np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum()) for k in (0, 1)]) for _ in range(B)]
    full = np.array([roc_auc_score(y, A[:, j]) for j in range(len(sets))])
    bs = np.array([[roc_auc_score(y[i], A[i, j]) for j in range(len(sets))] for i in idx])
    for j, s in enumerate(sets):
        diff = bs[:, j] - bs[:, ia]; dobs = full[j] - full[ia]
        p = 1.0 if s == "all" else min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
        rows.append(dict(task=task, set=s, n=len(y), auc=full[j], ci_low=np.percentile(bs[:, j], 2.5), ci_high=np.percentile(bs[:, j], 97.5),
                         diff_vs_all=dobs, diff_ci_low=np.percentile(diff, 2.5), diff_ci_high=np.percentile(diff, 97.5), p_boot=p))
R = pd.DataFrame(rows)
R["q_bh_within_task"] = np.nan
for t, g in R[R.set != "all"].groupby("task"): R.loc[g.index, "q_bh_within_task"] = bh(g.p_boot)
pm = R.set.isin(PRIMARY); R["p_holm_primary"] = np.nan; R.loc[pm, "p_holm_primary"] = holm(R.loc[pm, "p_boot"])
E.mkdir(parents=True); R.to_csv(E / "group_auc_stats.csv", index=False)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
json.dump(dict(script="07_group_stats.py", source=sys.argv[2], bootstrap=B, seed=7, primary=PRIMARY), open(E / "config.json", "w"), indent=1)
pd.set_option("display.width", 220)
print(R[R.set.isin(["all"] + PRIMARY) | (R.q_bh_within_task < .05)].round(3).to_string(index=False))
