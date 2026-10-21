"""
02_regional_effects.py  (EXP301; RQ1 regional effects, RQ2 cross-cohort replication)
------------------------------------------------------------------------------------
Features: laterality-free regional volumes shared by NACC and OASIS-1 (left+right totals, midline structures,
hippocampus/inferior-lateral-ventricle ratio), per-mille of brain-mask volume. Same definition as multimodal-mri-pet-staging EXP023.

Covariate adjustment (descriptive analysis, no prediction): within each cohort, every feature is regressed on age and
sex (female / male / unknown) in the reference group (NACC normal cognition; OASIS-1 CDR 0); residuals of all
participants are computed from that fit. Effect size = Cohen's d of the residuals (group minus reference, pooled SD),
so negative d = smaller volume than expected for age and sex.
p-value: OLS of the feature on group + age + sex in the two groups compared (two-sided t-test of the group term);
Benjamini-Hochberg FDR across features within each contrast.
95% CI of d: bootstrap of participants within groups (2,000 resamples; the covariate fit is repeated in each resample).
Replication: Spearman correlation across features between NACC and OASIS-1 effect sizes for matched contrasts
(dementia vs normal ~ CDR>=1 vs CDR 0; MCI vs normal ~ CDR 0.5 vs CDR 0), CI by joint bootstrap; overlap of the
ten largest effects.
Features that are constant in either cohort are dropped (the five corpus callosum labels are always 0 because FastSurfer
--seg_only does not produce them; found in EXP301, which failed for that reason).
Usage: python 02_regional_effects.py <project dir> <EXP id>   -> experiments/<EXP id> (never overwritten)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

P = Path(sys.argv[1]); M = P / "dataset" / "metadata"; E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
rng = np.random.default_rng(7); B = 2000


def bilateral(df):
    out = {}
    for c in df.columns:
        if c.startswith("mri_vol_L-"):
            r = "mri_vol_R-" + c[10:]
            if r in df: out[c[10:]] = df[c] + df[r]
        elif c.startswith("mri_vol_") and not c.startswith("mri_vol_R-"):
            out[c[8:]] = df[c]
    o = pd.DataFrame(out, index=df.index)
    o["Hippocampus/ILV ratio"] = o["Hippocampus"] / (o["Inf-Lat-Vent"] + 1e-6)
    return o


f = pd.read_csv(M / "features_v2.csv"); m = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex"])
n = f[f.has_mri == 1].merge(m, on="NACCID", how="left").reset_index(drop=True)
o = pd.read_csv(M / "features_oasis1.csv")
Xn, Xo = bilateral(n), bilateral(o)
feats = [c for c in Xn.columns if c in Xo.columns and Xn[c].notna().all() and Xo[c].notna().all() and Xn[c].nunique() > 1 and Xo[c].nunique() > 1]
grp_n = n.label.map({0: "ref", 1: "MCI", 2: "dementia"}).to_numpy()
cdr = pd.to_numeric(o.cdr); grp_o = np.where(cdr == 0, "ref", np.where(cdr == 0.5, "MCI", "dementia"))
sex_n = n.sex.map({"F": "F", "M": "M"}).fillna("U").to_numpy(); sex_o = o.sex.map({"Female": "F", "Male": "M"}).fillna("U").to_numpy()
cov_n = pd.DataFrame(dict(age=n.NACCAGE.astype(float), F=(sex_n == "F") * 1., U=(sex_n == "U") * 1.))
cov_o = pd.DataFrame(dict(age=o.age.astype(float), F=(sex_o == "F") * 1., U=(sex_o == "U") * 1.))
print(f"features {len(feats)}; NACC {pd.Series(grp_n).value_counts().to_dict()}; OASIS {pd.Series(grp_o).value_counts().to_dict()}")


def design(C):
    D = np.column_stack([np.ones(len(C)), C.to_numpy()]); keep = D.std(0) > 0; keep[0] = True
    return D[:, keep]


def effect(X, C, g, contrast):
    """d of covariate residuals (fit in reference rows) for contrast group vs ref."""
    ref = g == "ref"; tgt = g == contrast
    D = design(C); beta = np.linalg.lstsq(D[ref], X[ref], rcond=None)[0]; R = X - D @ beta
    a, b = R[tgt], R[ref]
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / sp


def pvals(X, C, g, contrast):
    s = (g == "ref") | (g == contrast); D = np.column_stack([design(C[s]), (g[s] == contrast) * 1.])
    beta, *_ = np.linalg.lstsq(D, X[s], rcond=None); res = X[s] - D @ beta; dof = len(D) - D.shape[1]
    s2 = (res ** 2).sum(0) / dof; se = np.sqrt(s2 * np.linalg.inv(D.T @ D)[-1, -1]); t = beta[-1] / se
    return 2 * stats.t.sf(np.abs(t), dof)


def bh(p):
    p = np.asarray(p); o_ = np.argsort(p); q = p[o_] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o_] = np.minimum(q, 1); return out


def strat_idx(g):
    return np.concatenate([rng.choice(np.where(g == k)[0], (g == k).sum()) for k in np.unique(g)])


An, Ao = Xn[feats].to_numpy(float), Xo[feats].to_numpy(float)
rows, dist = [], {}
for cohort, X, C, g in (("NACC", An, cov_n, grp_n), ("OASIS-1", Ao, cov_o, grp_o)):
    for contrast in ("MCI", "dementia"):
        d = effect(X, C, g, contrast); p = pvals(X, C, g, contrast); q = bh(p)
        bs = np.array([effect(X[i], C.iloc[i].reset_index(drop=True), g[i], contrast) for i in (strat_idx(g) for _ in range(B))])
        dist[(cohort, contrast)] = bs
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for k, ft in enumerate(feats):
            rows.append(dict(cohort=cohort, contrast=f"{contrast} vs reference", feature=ft, n_group=int((g == contrast).sum()),
                             n_ref=int((g == "ref").sum()), d=d[k], ci_low=lo[k], ci_high=hi[k], p=p[k], q_fdr=q[k]))
R = pd.DataFrame(rows)

rep = []
for contrast in ("MCI", "dementia"):
    a = R[(R.cohort == "NACC") & R.contrast.str.startswith(contrast)].set_index("feature").d
    b = R[(R.cohort == "OASIS-1") & R.contrast.str.startswith(contrast)].set_index("feature").d.loc[a.index]
    rho = stats.spearmanr(a, b).statistic
    bsr = [stats.spearmanr(x, y).statistic for x, y in zip(dist[("NACC", contrast)], dist[("OASIS-1", contrast)])]
    top_a, top_b = set(a.abs().nlargest(10).index), set(b.abs().nlargest(10).index)
    sig_a = set(R[(R.cohort == "NACC") & R.contrast.str.startswith(contrast) & (R.q_fdr < .05)].feature)
    sig_b = set(R[(R.cohort == "OASIS-1") & R.contrast.str.startswith(contrast) & (R.q_fdr < .05)].feature)
    same_sign = int(((np.sign(a) == np.sign(b)) & a.index.isin(sig_a)).sum())
    rep.append(dict(contrast=f"{contrast} vs reference", n_features=len(a), spearman_rho=rho, ci_low=np.percentile(bsr, 2.5),
                    ci_high=np.percentile(bsr, 97.5), top10_overlap=len(top_a & top_b), nacc_fdr_sig=len(sig_a), oasis_fdr_sig=len(sig_b),
                    both_fdr_sig=len(sig_a & sig_b), nacc_sig_same_sign_in_oasis=same_sign))
Rep = pd.DataFrame(rep)

E.mkdir(parents=True)
R.to_csv(E / "regional_effects.csv", index=False); Rep.to_csv(E / "replication.csv", index=False)
json.dump(dict(script="02_regional_effects.py", seed=7, bootstrap=B, features=feats, reference={"NACC": "normal cognition", "OASIS-1": "CDR 0"},
               covariates="age, sex (F/M/unknown), fitted in reference group", contrasts={"NACC": "MCI, dementia", "OASIS-1": "CDR 0.5 (MCI), CDR>=1 (dementia)"}),
          open(E / "config.json", "w"), indent=1)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
pd.set_option("display.width", 200)
for (c, k), g in R.groupby(["cohort", "contrast"]):
    print(f"\n{c} {k}: FDR<0.05 in {(g.q_fdr < .05).sum()}/{len(g)}"); print(g.nsmallest(8, "d")[["feature", "d", "ci_low", "ci_high", "q_fdr"]].round(3).to_string(index=False))
    print(g.nlargest(3, "d")[["feature", "d", "ci_low", "ci_high", "q_fdr"]].round(3).to_string(index=False))
print("\n", Rep.round(3).to_string(index=False))
