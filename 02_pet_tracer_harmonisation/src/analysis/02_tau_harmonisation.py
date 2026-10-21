"""
02_tau_harmonisation.py  (RQ2, EXP202)
-------------------------------------
Tau PET in NACC SCAN uses three tracers (flortaucipir, MK-6240, PI-2620) whose SUVRs are on different scales.
(a) Validation: our SUVRs (Paper 1 pipeline, inferior cerebellar grey reference) against the core lab (UC Berkeley,
    MRI-free) for meta-temporal and entorhinal regions, per tracer (core lab has no PI-2620 values).
(b) Harmonisation: each tau summary (meta-temporal, Braak I, Braak III/IV, Braak V/VI) is converted to a z-score
    relative to amyloid-negative normal participants scanned with the same tracer (amyloid status = our out-of-fold
    Centiloid status, EXP201). Cross-fitted: 5 folds stratified by tracer x diagnosis; the reference mean/SD come from the
    reference participants in the other folds, so nobody is scored against a reference that contains themselves.
    If fewer than 8 reference participants are available for a tracer in a fold, all normal participants of that
    tracer in the other folds are used (flag ref_fallback).
(c) Tau positivity: meta-temporal z > 2 (sensitivity analyses z > 1.5, z > 2.5).
(d) Residual tracer effect: among amyloid-positive dementia, compare harmonised meta-temporal z across tracers.
Output: experiments/<EXP id>/{tau_z.csv, validation.csv, tracer_check.csv, command.txt}
Usage: python 02_tau_harmonisation.py <project dir> <EXP id> <EXP id of centiloid calibration>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, kruskal
from sklearn.model_selection import StratifiedKFold

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]; CL = P / "experiments" / sys.argv[3]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
M = P / "dataset" / "metadata"
S = {"meta_temporal": "tau_comp_meta_temporal", "braak1": "tau_comp_braak1", "braak34": "tau_comp_braak34", "braak56": "tau_comp_braak56"}
f = pd.read_csv(M / "features_v2.csv", usecols=["NACCID", "label", "tau_suvr_L-entorhinal", "tau_suvr_R-entorhinal"] + list(S.values()))
m = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "tau_tracer", "tau_qc_ok"])
a = pd.read_csv(CL / "centiloid_oof.csv", usecols=["NACCID", "cl_oof", "pos_oof"])
ref = pd.read_csv(M / "validation_tau_pairs.csv", usecols=["NACCID", "META_TEMPORAL_SUVR", "CTX_ENTORHINAL_SUVR"])
d = m[m.tau_qc_ok == 1].merge(f, on="NACCID").merge(a, on="NACCID", how="left").merge(ref, on="NACCID", how="left").reset_index(drop=True)
d["tau_ent"] = d[["tau_suvr_L-entorhinal", "tau_suvr_R-entorhinal"]].mean(axis=1)
print(f"tau scans passing QC: {len(d)}; by tracer {d.tau_tracer.value_counts().to_dict()}; amyloid status known {d.pos_oof.notna().sum()}")

# (a) validation
val = []
for tr, g in d[d.META_TEMPORAL_SUVR.notna()].groupby("tau_tracer"):
    for name, ours, theirs in (("meta_temporal", "tau_comp_meta_temporal", "META_TEMPORAL_SUVR"), ("entorhinal", "tau_ent", "CTX_ENTORHINAL_SUVR")):
        x, y = g[ours].to_numpy(float), g[theirs].to_numpy(float)
        ccc = 2 * np.cov(x, y, ddof=0)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
        val.append(dict(tracer=tr, measure=name, n=len(g), pearson_r=pearsonr(x, y)[0], spearman_r=spearmanr(x, y)[0], ccc=ccc,
                        mean_ours=x.mean(), mean_ref=y.mean(), bias=(x - y).mean(), slope=np.polyfit(y, x, 1)[0]))

# (b) cross-fitted z-scores
for k in S: d[f"z_{k}"] = np.nan
d["ref_fallback"] = 0
strata = d.tau_tracer + "_" + d.label.astype(str)
for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=7).split(d, strata):
    tr_d = d.iloc[tr_i]
    for tr in d.tau_tracer.unique():
        r = tr_d[(tr_d.tau_tracer == tr) & (tr_d.label == 0) & (tr_d.pos_oof == 0)]; fb = 0
        if len(r) < 8: r = tr_d[(tr_d.tau_tracer == tr) & (tr_d.label == 0)]; fb = 1
        te = [i for i in te_i if d.at[i, "tau_tracer"] == tr]
        for k, c in S.items():
            d.loc[te, f"z_{k}"] = (d.loc[te, c] - r[c].mean()) / r[c].std(ddof=1)
        d.loc[te, "ref_fallback"] = fb
for t in (1.5, 2.0, 2.5): d[f"tpos_z{t}"] = (d.z_meta_temporal > t).astype(int)

# (d) residual tracer effect: raw SUVR vs z among amyloid-positive dementia and amyloid-negative normal
chk = []
for grp, sel in (("A+ dementia", (d.label == 2) & (d.pos_oof == 1)), ("A- normal", (d.label == 0) & (d.pos_oof == 0))):
    g = d[sel]
    for col in ("tau_comp_meta_temporal", "z_meta_temporal"):
        parts = [x[col].to_numpy() for _, x in g.groupby("tau_tracer")]
        row = dict(group=grp, measure=col, p_kruskal=kruskal(*parts).pvalue if len(parts) > 1 else np.nan)
        for tr, x in g.groupby("tau_tracer"): row[f"{tr}_n"] = len(x); row[f"{tr}_median"] = x[col].median()
        chk.append(row)
E.mkdir(parents=True)
d.to_csv(E / "tau_z.csv", index=False); pd.DataFrame(val).to_csv(E / "validation.csv", index=False)
pd.DataFrame(chk).to_csv(E / "tracer_check.csv", index=False); (E / "command.txt").write_text(" ".join(sys.argv) + "\n")
pd.set_option("display.width", 250)
print(pd.DataFrame(val).round(3).to_string(index=False)); print(pd.DataFrame(chk).round(3).to_string(index=False))
print("fallback used:", d.groupby("tau_tracer").ref_fallback.mean().to_dict())
print(d.groupby(["tau_tracer", "label"]).z_meta_temporal.median().unstack().round(2))
