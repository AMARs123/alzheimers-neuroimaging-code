"""
19_subgroups.py  (EXP313; subgroup check for Reviewer_Checkpoint row 21-04, added 16 Oct 2026, after the main analyses)
-------------------------------------------------------------------------------------------------------------------------
Uses stored predictions only; nothing is refitted.
NACC: EXP303 out-of-fold probabilities, averaged over the 10 repeats per participant, for the full (51 features) and
compact medial temporal (5 features) models, tasks normal vs dementia and normal vs MCI. AUC in women and in men
(sex from the image header, meta_v2.sex; unknown sex excluded from this check), and the difference women minus men with
a 95% CI from 2,000 bootstrap resamples drawn within each sex x class cell (seed 7); two-sided bootstrap p.
OASIS-1: EXP307 frozen-model probabilities, CDR 0 vs CDR >= 1, by sex (features_oasis1.sex).
Usage: python 19_subgroups.py <project dir> <EXP id>   -> experiments/<EXP id> (never overwritten)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

P = Path(sys.argv[1]); X = P / "experiments"; E = X / sys.argv[2]; M = P / "dataset" / "metadata"
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
rng = np.random.default_rng(7); B = 2000
pr = pd.read_csv(X / "EXP303" / "predictions.csv.gz")
meta = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "sex"])
ox = pd.read_csv(X / "EXP307" / "predictions.csv"); osx = pd.read_csv(M / "features_oasis1.csv", usecols=["ID", "sex"])
ox = ox.merge(osx, on="ID", how="left"); ox["sex"] = ox.sex.map({"Female": "F", "Male": "M"})


def diff_boot(y, p, s):
    """AUC(F) - AUC(M) with resampling inside every sex x class cell."""
    cells = [np.where((s == a) & (y == b))[0] for a in ("F", "M") for b in (0, 1)]
    out = []
    for _ in range(B):
        i = np.concatenate([rng.choice(c, len(c)) for c in cells]); yy, pp, ss = y[i], p[i], s[i]
        out.append(roc_auc_score(yy[ss == "F"], pp[ss == "F"]) - roc_auc_score(yy[ss == "M"], pp[ss == "M"]))
    return np.array(out)


rows = []
def add(cohort, task, model, y, p, s):
    k = np.isin(s, ["F", "M"]); y, p, s = y[k], p[k], s[k]
    aF, aM = roc_auc_score(y[s == "F"], p[s == "F"]), roc_auc_score(y[s == "M"], p[s == "M"]); d = diff_boot(y, p, s)
    rows.append(dict(cohort=cohort, task=task, model=model, n_F=int((s == "F").sum()), pos_F=int(y[s == "F"].sum()),
                     n_M=int((s == "M").sum()), pos_M=int(y[s == "M"].sum()), auc_F=aF, auc_M=aM, diff_F_minus_M=aF - aM,
                     ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                     p_boot=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


for task in ["cn_ad", "cn_mci"]:
    for s_, name in [("all", "full (51)"), ("only: medial temporal", "compact (5)")]:
        d = pr[(pr.task == task) & (pr.set == s_)]
        g = d.groupby("NACCID").agg(y=("y", "first"), p=("p", "mean")).reset_index()
        g["y"] = (g.y == g.y.max()).astype(int); g = g.merge(meta, on="NACCID", how="left")
        add("NACC", task, name, g.y.to_numpy(), g.p.to_numpy(), g.sex.fillna("U").to_numpy())
keep = (ox.cdr == 0) | (ox.cdr >= 1); y = (ox.cdr[keep] >= 1).astype(int).to_numpy()
for col, name in [("p_normal_vs_dementia_all", "full (51)"), ("p_normal_vs_dementia_medial_temporal", "compact (5)")]:
    add("OASIS-1", "CDR 0 vs >=1", name, y, ox.loc[keep, col].to_numpy(), ox.sex[keep].fillna("U").to_numpy())
R = pd.DataFrame(rows); E.mkdir(parents=True)
R.to_csv(E / "subgroup_auc_by_sex.csv", index=False)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
json.dump(dict(script="19_subgroups.py", inputs=["EXP303/predictions.csv.gz", "EXP307/predictions.csv"], bootstrap=B, seed=7,
               refit=False), open(E / "config.json", "w"), indent=1)
pd.set_option("display.width", 220); print(R.round(3).to_string(index=False))
