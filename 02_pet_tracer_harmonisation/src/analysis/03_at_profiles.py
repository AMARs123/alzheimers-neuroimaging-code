"""
03_at_profiles.py  (RQ3, EXP203)
-------------------------------
How well do clinical diagnoses match the biology seen on PET?
A (amyloid) status: our out-of-fold Centiloid positivity (EXP201; 286 participants with a calibratable amyloid scan).
T (tau) status: harmonised meta-temporal z > 2 (EXP202), with z > 1.5 and z > 2.5 as sensitivity analyses.
Reported: % A+ by clinical group (with Wilson 95% CI), median Centiloid by group; A/T profile (A-T-, A-T+, A+T-, A+T+)
by group for participants with both; age by A status within group (descriptive); chi-square tests across groups.
Output: experiments/<EXP id>/{amyloid_by_group.csv, at_by_group.csv, at_sensitivity.csv, command.txt}
Usage: python 03_at_profiles.py <project dir> <EXP id> <EXP centiloid> <EXP tau>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import norm


def proportion_confint(k, n, method='wilson'):
    z = norm.ppf(.975); c = (k + z * z / 2) / (n + z * z); h = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return c - h, c + h

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
A = pd.read_csv(P / "experiments" / sys.argv[3] / "centiloid_oof.csv")
T = pd.read_csv(P / "experiments" / sys.argv[4] / "tau_z.csv")
meta = pd.read_csv(P / "dataset" / "metadata" / "meta_v2.csv", usecols=["NACCID", "NACCAGE"])
G = {0: "normal", 1: "MCI", 2: "dementia"}
A = A.merge(meta, on="NACCID", how="left"); A["group"] = A.label.map(G)

rows = []
for g, x in A.groupby("label"):
    k, n = int(x.pos_oof.sum()), len(x); lo, hi = proportion_confint(k, n, method="wilson")
    k_r = int(x.AMYLOID_STATUS.sum())
    rows.append(dict(group=G[g], n=n, a_pos=k, pct_a_pos=100 * k / n, ci_low=100 * lo, ci_high=100 * hi, pct_a_pos_corelab=100 * k_r / n,
                     median_cl=x.cl_oof.median(), age_a_pos=x[x.pos_oof == 1].NACCAGE.mean(), age_a_neg=x[x.pos_oof == 0].NACCAGE.mean()))
AB = pd.DataFrame(rows); AB["p_chi2_groups"] = chi2_contingency(pd.crosstab(A.label, A.pos_oof))[1]

D = T[T.pos_oof.notna()].copy()
def profile(t):
    p = np.where(D.pos_oof == 1, "A+", "A-").astype(object) + np.where(D.z_meta_temporal > t, "T+", "T-")
    tab = pd.crosstab(D.label.map(G), p).reindex(["normal", "MCI", "dementia"]).fillna(0).astype(int)
    for c in ("A-T-", "A-T+", "A+T-", "A+T+"):
        if c not in tab: tab[c] = 0
    tab = tab[["A-T-", "A-T+", "A+T-", "A+T+"]]; tab["n"] = tab.sum(1)
    return tab
AT = profile(2.0)
for c in ("A-T-", "A-T+", "A+T-", "A+T+"): AT[f"pct_{c}"] = 100 * AT[c] / AT.n
AT["p_chi2_groups"] = chi2_contingency(AT[["A-T-", "A-T+", "A+T-", "A+T+"]].loc[:, lambda x: x.sum() > 0])[1]
tr = D.groupby(D.label.map(G)).tau_tracer.value_counts().unstack(fill_value=0)
AT = AT.join(tr)
sens = []
for t in (1.5, 2.0, 2.5):
    tab = profile(t)
    for grp, r in tab.iterrows():
        sens.append(dict(threshold_z=t, group=grp, n=r.n, **{f"pct_{c}": 100 * r[c] / r.n for c in ("A-T-", "A-T+", "A+T-", "A+T+")}))
E.mkdir(parents=True)
AB.to_csv(E / "amyloid_by_group.csv", index=False); AT.reset_index().rename(columns={"label": "group"}).to_csv(E / "at_by_group.csv", index=False)
pd.DataFrame(sens).to_csv(E / "at_sensitivity.csv", index=False); (E / "command.txt").write_text(" ".join(sys.argv) + "\n")
pd.set_option("display.width", 250)
print(AB.round(1).to_string(index=False)); print(AT.round(1).to_string()); print(pd.DataFrame(sens).round(1).to_string(index=False))
