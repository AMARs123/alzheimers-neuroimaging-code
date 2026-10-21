"""
18_age_adjusted_classifiers.py  (EXP312; follow-up of EXP311, added 16 Oct 2026)
---------------------------------------------------------------------------------
EXP311 showed that the compact model misclassifies older normal and younger demented participants, and the
classifier features were not age-adjusted. Here every volume is adjusted for age and sex INSIDE each training fold:
a linear regression of the feature on age and sex is fitted on the training participants of the reference class
(y = 0: normal cognition, or MCI for MCI vs dementia) and the residual replaces the feature in training and test data.
Nothing about the test fold is used. The rest of the pipeline, the grid and the outer splits (seeds 100..109) are
identical to EXP303, so the predictions are paired with EXP303 participant by participant.

Sets: all 51 features and medial temporal (5). Tasks: normal vs dementia, normal vs MCI, MCI vs dementia.
Paired comparison with EXP303 (same set, same task): bootstrap of the AUC difference of the repeat-averaged
probabilities (2,000 resamples within classes).
External: frozen NACC normal-vs-dementia models (raw and age-adjusted; settings by 5-fold CV in NACC) applied to
OASIS-1 CDR 0 vs >= 1: AUC, sensitivity, specificity and balanced accuracy at threshold 0.5. NACC age is age at the
label visit; OASIS-1 age is age at the scan.
Error check: age of misclassified vs correct participants (compact model, normal vs dementia), as in EXP311.

Usage: python 18_age_adjusted_classifiers.py <project dir> <EXP id>
"""
import sys, json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
M = P / "dataset" / "metadata"
src = (P / "src" / "analysis" / "03_region_groups.py").read_text(encoding="utf-8")
ns = {}
exec("import numpy as np, pandas as pd\n" + src[src.index("GROUPS = {"): src.index("f = pd.read_csv")], ns)
GROUPS, bilateral = ns["GROUPS"], ns["bilateral"]


class AgeSexAdjust(BaseEstimator, TransformerMixin):
    """First 3 columns: age, male, female (unknown sex = 0, 0). Fits feature ~ age + sex in the y == 0 training rows."""
    def fit(self, X, y):
        C = np.column_stack([np.ones(len(X)), X[:, :3]]); ref = np.asarray(y) == 0
        self.beta_, *_ = np.linalg.lstsq(C[ref], X[ref, 3:], rcond=None)
        return self

    def transform(self, X):
        return X[:, 3:] - np.column_stack([np.ones(len(X)), X[:, :3]]) @ self.beta_


def model(nf, adjust):
    ks = sorted({k for k in (3, 5, 10, 20) if k < nf} | {"all"}, key=str)
    steps = ([("adj", AgeSexAdjust())] if adjust else []) + [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                                           ("sel", SelectKBest(f_classif)), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]
    return GridSearchCV(Pipeline(steps), {"sel__k": ks, "clf__C": [0.01, 0.1, 1.0]}, scoring="roc_auc",
                        cv=StratifiedKFold(3, shuffle=True, random_state=0), n_jobs=1)


f = pd.read_csv(M / "features_v2.csv"); n = f[f.has_mri == 1].reset_index(drop=True)
meta = pd.read_csv(M / "meta_v2.csv", usecols=["NACCID", "NACCAGE", "sex"])
n = n.merge(meta, on="NACCID", how="left"); assert n.NACCAGE.notna().all()
X = bilateral(n); X = X[[c for c in X.columns if X[c].nunique() > 1]]
COV = np.column_stack([n.NACCAGE.to_numpy(float), (n.sex == "M").astype(float), (n.sex == "F").astype(float)])
SETS = {"all": list(X.columns), "only: medial temporal": GROUPS["medial temporal"]}
TASKS = {"cn_ad": (0, 2), "cn_mci": (0, 1), "mci_ad": (1, 2)}


def run(task, sname, r):
    a, b = TASKS[task]; s = n.label.isin([a, b]).to_numpy(); y = (n.label[s] == b).astype(int).to_numpy()
    A = np.column_stack([COV[s], X.loc[s, SETS[sname]].to_numpy(float)]); p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(A, y):
        p[te] = model(len(SETS[sname]), True).fit(A[tr], y[tr]).predict_proba(A[te])[:, 1]
    return task, sname, r, n.NACCID[s].to_numpy(), y, p


t0 = time.time()
jobs = [(t, s, r) for t in TASKS for s in SETS for r in range(10)]
res = Parallel(n_jobs=-1, verbose=0)(delayed(run)(*j) for j in jobs)
E.mkdir(parents=True)
pred = pd.concat([pd.DataFrame(dict(task=t, set=s, repeat=r, NACCID=ids, y=y, p=p)) for t, s, r, ids, y, p in res])
pred.to_csv(E / "predictions.csv.gz", index=False)
met = pred.groupby(["task", "set", "repeat"]).apply(lambda g: pd.Series(dict(auc=roc_auc_score(g.y, g.p),
      balanced_accuracy=balanced_accuracy_score(g.y, g.p >= .5))), include_groups=False).reset_index()
met.to_csv(E / "metrics_by_repeat.csv", index=False)

# ---- paired comparison with EXP303 (unadjusted), same participants, same outer splits
old = pd.read_csv(P / "experiments" / "EXP303" / "predictions.csv.gz"); oldm = pd.read_csv(P / "experiments" / "EXP303" / "metrics_by_repeat.csv")
rng = np.random.default_rng(7); B = 2000; rows = []
for task in TASKS:
    for s in SETS:
        g1 = pred[(pred.task == task) & (pred.set == s)].groupby("NACCID").agg(y=("y", "first"), p=("p", "mean"))
        g0 = old[(old.task == task) & (old.set == s)].groupby("NACCID").agg(p0=("p", "mean"))
        j = g1.join(g0, how="inner"); y = j.y.to_numpy(); a, b = j.p0.to_numpy(), j.p.to_numpy()
        idx = [np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum()) for k in (0, 1)]) for _ in range(B)]
        d = np.array([roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]) for i in idx])
        mm = met[(met.task == task) & (met.set == s)]; om = oldm[(oldm.task == task) & (oldm.set == s)]
        rows.append(dict(task=task, set=s, n=len(y), auc_adjusted_mean=mm.auc.mean(), auc_adjusted_sd=mm.auc.std(ddof=1),
                         auc_raw_mean=om.auc.mean(), auc_raw_sd=om.auc.std(ddof=1),
                         balacc_adjusted_mean=mm.balanced_accuracy.mean(), balacc_raw_mean=om.balanced_accuracy.mean(),
                         diff_auc_avgprob=roc_auc_score(y, b) - roc_auc_score(y, a), diff_ci_low=np.percentile(d, 2.5),
                         diff_ci_high=np.percentile(d, 97.5), p_boot=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
C = pd.DataFrame(rows); C.to_csv(E / "comparison_vs_EXP303.csv", index=False)

# ---- errors by age (compact model, normal vs dementia)
g = pred[(pred.task == "cn_ad") & (pred.set == "only: medial temporal")].groupby("NACCID").agg(y=("y", "first"), p=("p", "mean")).reset_index().merge(meta, on="NACCID")
err = []
for y, lab in [(0, "normal"), (1, "dementia")]:
    h = g[g.y == y]; wrong = h[(h.p >= .5).astype(int) != y]; right = h[(h.p >= .5).astype(int) == y]
    err.append(dict(group=lab, n=len(h), n_wrong=len(wrong), age_wrong_median=wrong.NACCAGE.median(), age_right_median=right.NACCAGE.median(),
                    p_mannwhitney=mannwhitneyu(wrong.NACCAGE, right.NACCAGE).pvalue if len(wrong) and len(right) else np.nan))
pd.DataFrame(err).to_csv(E / "errors_by_age.csv", index=False)

# ---- external OASIS-1: raw vs adjusted frozen models, normal vs dementia
o = pd.read_csv(M / "features_oasis1.csv"); Xo = bilateral(o); cdr = pd.to_numeric(o.cdr).to_numpy()
COVo = np.column_stack([o.age.to_numpy(float), (o.sex == "Male").astype(float), (o.sex == "Female").astype(float)])
s = n.label.isin([0, 2]).to_numpy(); ytr = (n.label[s] == 2).astype(int).to_numpy(); keep = (cdr == 0) | (cdr >= 1); yo = (cdr[keep] >= 1).astype(int)
ext = []
for sname in SETS:
    cols = [c for c in SETS[sname] if c in Xo]
    for adjust in (False, True):
        mdl = model(len(cols), adjust); mdl.cv = StratifiedKFold(5, shuffle=True, random_state=7)
        Atr = X.loc[s, cols].to_numpy(float); Ao = Xo.loc[keep, cols].to_numpy(float)
        if adjust:
            Atr = np.column_stack([COV[s], Atr]); Ao = np.column_stack([COVo[keep], Ao])
        po = mdl.fit(Atr, ytr).predict_proba(Ao)[:, 1]; yh = (po >= .5).astype(int)
        idx = [np.concatenate([rng.choice(np.where(yo == k)[0], (yo == k).sum()) for k in (0, 1)]) for _ in range(B)]
        ba = np.array([roc_auc_score(yo[i], po[i]) for i in idx])
        ext.append(dict(set=sname, features="age/sex-adjusted" if adjust else "raw", n=len(yo), auc=roc_auc_score(yo, po),
                        auc_ci=f"[{np.percentile(ba, 2.5):.3f}, {np.percentile(ba, 97.5):.3f}]",
                        sensitivity=yh[yo == 1].mean(), specificity=1 - yh[yo == 0].mean(), balanced_accuracy=balanced_accuracy_score(yo, yh)))
X_ = pd.DataFrame(ext); X_.to_csv(E / "external_oasis.csv", index=False)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
json.dump(dict(script="18_age_adjusted_classifiers.py", adjustment="feature ~ age + sex (M/F/unknown) in training reference class, inside each fold",
               seeds_outer="100..109 (as EXP303)", bootstrap=B, seed=7, external_threshold=0.5), open(E / "config.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(C.round(3).to_string(index=False)); print(pd.DataFrame(err).round(4).to_string(index=False)); print(X_.round(3).to_string(index=False))
print(f"minutes: {(time.time() - t0) / 60:.1f}")
