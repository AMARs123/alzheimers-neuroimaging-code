"""
93_external_validation.py
-------------------------
External validation of the MRI model on OASIS-1 (open access; different scanners, 1.5 T, CDR-based labels).

Training: all NACC participants with MRI, normal (label 0) vs dementia (label 2). Features: bilateral regional volumes
(left + right, per-mille of brain-mask volume) and midline structures; no left/right or asymmetry features, because the
laterality of the OASIS-1 Analyze images cannot be verified. Model: median imputation, scaling, univariate selection and
L2 logistic regression, hyper-parameters chosen by 5-fold CV on NACC only. The fitted model is frozen and applied to OASIS-1.

Reported:
  * internal reference: repeated nested CV on NACC with the same bilateral features
  * external, frozen model: CDR 0 vs CDR >= 1 (primary), CDR 0 vs CDR 0.5, CDR 0 vs CDR > 0
  * sensitivity analysis "aligned": OASIS features z-scored with OASIS-wide mean/SD and NACC features with NACC-wide mean/SD
    before the model (label-free alignment of scanner/protocol shift); clearly not the frozen-pipeline result
  * reference: age alone, and OASIS-internal 10x5-fold CV of the same pipeline
95% CIs: stratified bootstrap of OASIS subjects (2,000 resamples).

Usage: python 93_external_validation.py <nacc features.csv> <oasis features.csv> <experiment dir>
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

nacc = pd.read_csv(sys.argv[1]); oas = pd.read_csv(sys.argv[2]); exp = Path(sys.argv[3])
if exp.exists():
    sys.exit(f"{exp} already exists - experiments are never overwritten")
exp.mkdir(parents=True); SEED = 7; rng = np.random.default_rng(SEED)


def bilateral(df):
    out = {}
    for c in df.columns:
        if c.startswith("mri_vol_L-"):
            r = "mri_vol_R-" + c[10:]
            if r in df:
                out["bi_" + c[10:]] = df[c] + df[r]
        elif c.startswith("mri_vol_") and not c.startswith("mri_vol_R-"):
            out["mid_" + c[8:]] = df[c]
    o = pd.DataFrame(out, index=df.index)
    if "bi_Hippocampus" in o and "bi_Inf-Lat-Vent" in o:
        o["hippo_over_ilv"] = o["bi_Hippocampus"] / (o["bi_Inf-Lat-Vent"] + 1e-6)
    return o


N = nacc[(nacc.has_mri == 1) & nacc.label.isin([0, 2])].reset_index(drop=True)
Xn_all, Xo_all = bilateral(N), bilateral(oas)
cols = [c for c in Xn_all.columns if c in Xo_all.columns and Xn_all[c].notna().mean() > 0.9 and Xo_all[c].notna().mean() > 0.9]
Xn, yn = Xn_all[cols].to_numpy(float), (N.label == 2).astype(int).to_numpy()
Xo = Xo_all[cols].to_numpy(float); cdr = pd.to_numeric(oas.cdr, errors="coerce").to_numpy(); age = pd.to_numeric(oas.age, errors="coerce").to_numpy()
print(f"NACC train: {len(yn)} ({(yn == 0).sum()} normal, {(yn == 1).sum()} dementia); OASIS-1: {len(cdr)}; {len(cols)} shared bilateral features")


def make():
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif)),
                     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    grid = {"sel__k": [k for k in (15, 30) if k < len(cols)] + ["all"], "clf__C": [0.01, 0.1, 1.0]}
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)


def nested(X, y, repeats=10):
    P = np.zeros((len(y), repeats))
    for r in range(repeats):
        P[:, r] = cross_val_predict(make(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=100 + r), method="predict_proba")[:, 1]
    return P


def met(y, p):
    pred = (p >= 0.5).astype(int); se = pred[y == 1].mean(); sp = 1 - pred[y == 0].mean()
    return dict(auc=roc_auc_score(y, p), accuracy=(pred == y).mean(), balanced_accuracy=(se + sp) / 2, sensitivity=se, specificity=sp, brier=brier_score_loss(y, p))


def boot(y, p, B=2000):
    m = met(y, p); idx = np.arange(len(y)); bs = []
    for _ in range(B):
        i = np.concatenate([rng.choice(idx[y == k], (y == k).sum()) for k in (0, 1)]); bs.append(met(y[i], p[i]))
    return {k: (v, *np.percentile([b[k] for b in bs], [2.5, 97.5])) for k, v in m.items()}


rows = []
def add(analysis, cohort, y, p):
    for k, (v, lo, hi) in boot(y, p).items():
        rows.append(dict(analysis=analysis, cohort=cohort, n=len(y), n_pos=int(y.sum()), metric=k, value=v, ci_low=lo, ci_high=hi))
    a = [r for r in rows[-6:]]; print(f"{analysis:28s} {cohort:22s} n={len(y):3d} " + " ".join(f"{r['metric'][:4]}={r['value']:.3f}" for r in a), flush=True)


# internal reference (nested CV, pooled over repeats by averaging the per-repeat metrics)
Pn = nested(Xn, yn)
mi = pd.DataFrame([met(yn, Pn[:, r]) for r in range(Pn.shape[1])])
for k in mi:
    rows.append(dict(analysis="internal nested CV (NACC)", cohort="normal vs dementia", n=len(yn), n_pos=int(yn.sum()), metric=k,
                     value=mi[k].mean(), ci_low=np.nan, ci_high=np.nan, sd_over_repeats=mi[k].std()))
print("internal nested CV:", mi.mean().round(3).to_dict(), flush=True)

final = make().fit(Xn, yn); best = final.best_params_
p_frozen = final.predict_proba(Xo)[:, 1]
zs = lambda A: (A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9)
aligned = make().fit(zs(Xn), yn); p_aligned = aligned.predict_proba(zs(Xo))[:, 1]
COH = {"CDR 0 vs CDR >= 1": (cdr == 0) | (cdr >= 1), "CDR 0 vs CDR 0.5": cdr <= 0.5, "CDR 0 vs CDR > 0": np.ones(len(cdr), bool)}
for name, m in COH.items():
    y = (cdr[m] > 0).astype(int)
    add("external frozen", name, y, p_frozen[m]); add("external aligned (label-free)", name, y, p_aligned[m])
    a = np.nan_to_num(age[m], nan=np.nanmean(age)); a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    rows.append(dict(analysis="age alone", cohort=name, n=int(m.sum()), n_pos=int(y.sum()), metric="auc", value=roc_auc_score(y, a), ci_low=np.nan, ci_high=np.nan))
    Po = nested(Xo[m], y)
    mo = pd.DataFrame([met(y, Po[:, r]) for r in range(Po.shape[1])])
    for k in mo:
        rows.append(dict(analysis="OASIS-internal nested CV", cohort=name, n=int(m.sum()), n_pos=int(y.sum()), metric=k, value=mo[k].mean(),
                         ci_low=np.nan, ci_high=np.nan, sd_over_repeats=mo[k].std()))

R = pd.DataFrame(rows); R.to_csv(exp / "metrics.csv", index=False)
pd.DataFrame(dict(ID=oas.ID, cdr=cdr, age=age, p_frozen=p_frozen, p_aligned=p_aligned)).to_csv(exp / "predictions.csv", index=False)
coef = final.best_estimator_; sel = coef.named_steps["sel"].get_support()
pd.DataFrame(dict(feature=np.array(cols)[sel], coef=coef.named_steps["clf"].coef_[0])).sort_values("coef").to_csv(exp / "frozen_model_coefficients.csv", index=False)
shift = pd.DataFrame(dict(feature=cols, nacc_normal_mean=np.nanmean(Xn[yn == 0], 0), oasis_cdr0_mean=np.nanmean(Xo[cdr == 0], 0), nacc_normal_sd=np.nanstd(Xn[yn == 0], 0)))
shift["shift_in_sd"] = (shift.oasis_cdr0_mean - shift.nacc_normal_mean) / shift.nacc_normal_sd; shift.to_csv(exp / "feature_shift_normals.csv", index=False)
json.dump(dict(script="93_external_validation.py", seed=SEED, train="NACC normal vs dementia, has_mri", n_train=int(len(yn)), n_external=int(len(cdr)),
               features=cols, best_params={k: (v if not isinstance(v, np.generic) else v.item()) for k, v in best.items()},
               note="frozen = model fitted on NACC only and applied unchanged; aligned = label-free cohort-wise z-scoring (sensitivity analysis)"),
          open(exp / "config.json", "w"), indent=1)
open(exp / "command.txt", "w").write(" ".join(sys.argv) + "\n")
print("median |shift| of normal-group feature means, in NACC SDs:", round(float(shift.shift_in_sd.abs().median()), 2))
print("written", exp)
