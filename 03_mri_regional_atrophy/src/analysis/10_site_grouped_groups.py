"""
10_site_grouped_groups.py  (RQ3 site check, EXP308)
---------------------------------------------------
Site is associated with the label in NACC (Paper 1 EXP024). Do the region-group results of EXP303 hold when test
sites are unseen? Participants with known MRI site only. For each task and the three key feature sets (all 51 regions,
medial temporal only, all minus medial temporal) we compare, on the same participants:
  (a) stratified 5-fold CV, 10 repeats (seeds 100..109)
  (b) site-grouped 5-fold CV (sites randomly assigned to folds, seeds 200..209), 10 repeats
Pipeline and inner grid are identical to 03_region_groups.py. Within-site AUC (Mann-Whitney pairs only between
participants of the same site) is reported for the grouped scheme.
Output: experiments/<EXP id>/{metrics_by_repeat.csv, metrics.csv, command.txt}
Usage: python 10_site_grouped_groups.py <project dir> <EXP id>
"""
import sys, warnings, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
M = P / "dataset" / "metadata"

# reuse GROUPS, bilateral() and model() from 03_region_groups.py without running it
src = (P / "src" / "analysis" / "03_region_groups.py").read_text()
ns = {}; exec(src.split("f = pd.read_csv")[0].split("P = Path")[0], ns)
exec("GROUPS = {" + src.split("GROUPS = {")[1].split("\n}\n")[0] + "\n}", ns)
exec("def bilateral" + src.split("def bilateral")[1].split("\n\n\n")[0], ns)
exec("def model" + src.split("def model")[1].split("\n\n\n")[0], ns)
GROUPS, bilateral, model = ns["GROUPS"], ns["bilateral"], ns["model"]

f = pd.read_csv(M / "features_v2.csv"); meta = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "mri_site"])
n = f[f.has_mri == 1].merge(meta, on="NACCID", how="left").reset_index(drop=True)
X = bilateral(n); X = X[[c for c in X.columns if X[c].nunique() > 1]]
ALL = list(X.columns); MTL = GROUPS["medial temporal"]
SETS = {"all": ALL, "only: medial temporal": MTL, "all minus: medial temporal": [c for c in ALL if c not in MTL]}
TASKS = {"cn_ad": (0, 2), "cn_mci": (0, 1), "mci_ad": (1, 2)}


def site_auc(y, p, s):
    num = den = 0
    for g in set(s):
        m = s == g; a, b = p[m & (y == 1)], p[m & (y == 0)]
        if len(a) and len(b):
            num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum(); den += len(a) * len(b)
    return num / den if den else np.nan


def run(task, sname, scheme, r):
    a, b = TASKS[task]; s = (n.label.isin([a, b]) & n.mri_site.notna()).to_numpy()
    y = (n.label[s] == b).astype(int).to_numpy(); A = X.loc[s, SETS[sname]].to_numpy(float)
    site = n.mri_site[s].astype(str).to_numpy()
    if scheme == "stratified":
        sp = StratifiedKFold(5, shuffle=True, random_state=100 + r).split(A, y)
    else:
        perm = {g: i for i, g in enumerate(np.random.default_rng(200 + r).permutation(sorted(set(site))))}
        sp = GroupKFold(5).split(A, y, np.array([perm[g] for g in site]))
    p = np.zeros(len(y))
    for tr, te in sp:
        p[te] = model(A.shape[1]).fit(A[tr], y[tr]).predict_proba(A[te])[:, 1]
    return dict(task=task, set=sname, scheme=scheme, repeat=r, n=len(y), n_sites=len(set(site)),
                auc=roc_auc_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p >= .5), within_site_auc=site_auc(y, p, site))


jobs = [(t, s, sc, r) for t in TASKS for s in SETS for sc in ("stratified", "site-grouped") for r in range(10)]
R = pd.DataFrame(Parallel(n_jobs=-1, verbose=1)(delayed(run)(*j) for j in jobs))
E.mkdir(parents=True); R.to_csv(E / "metrics_by_repeat.csv", index=False)
S = R.groupby(["task", "set", "scheme"])[["n", "auc", "balanced_accuracy", "within_site_auc"]].agg(["mean", "std"]).round(3)
S.to_csv(E / "metrics.csv"); (E / "command.txt").write_text(" ".join(sys.argv) + "\n")
pd.set_option("display.width", 220); print(S.to_string())
