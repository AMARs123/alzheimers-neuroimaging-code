"""
52_eval_plus.py
---------------
Extended leakage-free evaluation: same outer/inner protocol as 50_eval_nested.py, plus

  * W-scores: every imaging feature is adjusted for age and sex with a linear model fitted on the
    cognitively normal subjects of the training fold only (falls back to all training rows)
  * optional covariates (age, sex, education) as inputs
  * larger model set: lr, svm, et, hgb, lgbm, xgb, cat, mlp, ord (ordinal logistic), vote (soft voting), stack
  * extra feature prefix: dl_ (pretrained 3D network embeddings)

Usage:
  python 52_eval_plus.py --features F.csv --exp EXP011 --sets mri,mri+tau --tasks 3class \
        --models lr,ord,vote,cat --wscore 1 --need mri,tau
"""
import argparse, json, os, sys, time, shutil, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
shutil.copyfile(HERE / "50_eval_nested.py", HERE / "eval_nested.py")
sys.path.insert(0, str(HERE))
import eval_nested as ev
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from joblib import Parallel, delayed

P1 = ev.P1
PREFIX = dict(ev.PREFIX); PREFIX["dl"] = ("dl_",); PREFIX["thk"] = ("thk_",)
HAS = ("mri", "amy", "tau", "dl")


class Front(BaseEstimator, TransformerMixin):
    """Per-tracer standardisation, then optional age/sex W-scores, then removal of helper columns.

    tracer_groups: [(value_cols, tracer_col)]   adj_cols: columns to W-score
    cov_cols: (age_col, sex_col) helper columns appended at the end of X    drop: columns removed at the end
    """
    def __init__(self, tracer_groups=None, adj_cols=None, cov_cols=None, wscore=False, drop=None, ref_class=0, min_ref=20):
        self.tracer_groups = tracer_groups; self.adj_cols = adj_cols; self.cov_cols = cov_cols
        self.wscore = wscore; self.drop = drop; self.ref_class = ref_class; self.min_ref = min_ref

    def _design(self, X):
        age = X[:, self.cov_cols[0]].astype(float); sex = X[:, self.cov_cols[1]].astype(float)
        age = np.where(np.isnan(age), self.age_med_, age); sex = np.where(np.isnan(sex), self.sex_med_, sex)
        return np.c_[np.ones(len(X)), (age - self.age_med_) / 10.0, sex]

    def fit(self, X, y=None):
        self.ts_ = ev.TracerStandardizer(self.tracer_groups, ref_class=self.ref_class)
        self.ts_.fit(X, y)
        if self.wscore:
            X1 = self._tracer(X)
            self.age_med_ = np.nanmedian(X[:, self.cov_cols[0]]); self.sex_med_ = np.nanmedian(X[:, self.cov_cols[1]])
            ref = (np.asarray(y) == self.ref_class) if y is not None else np.ones(len(X), bool)
            if ref.sum() < self.min_ref:
                ref = np.ones(len(X), bool)
            D = self._design(X1)[ref]; A = X1[ref][:, self.adj_cols]
            self.col_med_ = np.nanmedian(A, 0); A = np.where(np.isnan(A), self.col_med_, A)
            self.beta_, *_ = np.linalg.lstsq(D, A, rcond=None)
            self.sd_ = (A - D @ self.beta_).std(0) + 1e-6
        return self

    def _tracer(self, X):
        X = X.copy()
        for cols, tcol, st, pooled in self.ts_.stats_:
            for t in np.unique(X[:, tcol][~np.isnan(X[:, tcol])]):
                m = X[:, tcol] == t
                mu, sd = st.get(t, pooled)
                X[np.ix_(m, cols)] = (X[np.ix_(m, cols)] - mu) / sd
        return X

    def transform(self, X):
        X1 = self._tracer(X)
        if self.wscore:
            D = self._design(X1)
            X1[:, self.adj_cols] = (X1[:, self.adj_cols] - D @ self.beta_) / self.sd_      # NaN stays NaN, imputed later
        return np.delete(X1, self.drop, axis=1) if self.drop else X1


class OrdinalLR(BaseEstimator, ClassifierMixin):
    """Ordinal classification by cumulative binary logistic models: P(y > c) for each cut point."""
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = [LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(X, (y > c).astype(int))
                        for c in self.classes_[:-1]]
        return self

    def predict_proba(self, X):
        gt = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        gt = np.minimum.accumulate(gt, axis=1)                    # enforce P(y>c) non-increasing in c
        cum = np.c_[np.ones(len(X)), gt, np.zeros(len(X))]
        p = np.clip(cum[:, :-1] - cum[:, 1:], 1e-6, None)
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def make(name, n_out, seed, front):
    ks = sorted({k for k in (20, 40, 80) if k < n_out} | {"all"}, key=lambda v: 10**9 if v == "all" else v)
    pre = lambda: [("front", clone(front)), ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif))]
    if name == "lr":
        return Pipeline(pre() + [("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]), {"sel__k": ks, "clf__C": [0.01, 0.1, 1.0]}
    if name == "ord":
        return Pipeline(pre() + [("clf", OrdinalLR())]), {"sel__k": ks, "clf__C": [0.01, 0.1, 1.0]}
    if name == "svm":
        return Pipeline(pre() + [("clf", SVC(probability=True, class_weight="balanced", random_state=seed))]), {"sel__k": ks, "clf__C": [0.3, 1.0, 3.0]}
    if name == "et":
        return Pipeline(pre() + [("clf", ExtraTreesClassifier(500, class_weight="balanced", random_state=seed, n_jobs=1))]), {"sel__k": ks, "clf__min_samples_leaf": [1, 3]}
    if name == "mlp":
        return Pipeline(pre() + [("clf", MLPClassifier((64,), max_iter=500, early_stopping=True, random_state=seed))]), {"sel__k": ks, "clf__alpha": [1e-2, 1.0, 10.0]}
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return Pipeline(pre() + [("clf", LGBMClassifier(n_estimators=300, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                                         class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1))]), {"sel__k": ks[-2:], "clf__num_leaves": [4, 8]}
    if name == "xgb":
        from xgboost import XGBClassifier
        return Pipeline(pre() + [("clf", XGBClassifier(n_estimators=300, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                                                        random_state=seed, n_jobs=1, verbosity=0))]), {"sel__k": ks[-2:], "clf__max_depth": [2, 3]}
    if name == "cat":
        from catboost import CatBoostClassifier
        return Pipeline(pre() + [("clf", CatBoostClassifier(iterations=400, learning_rate=0.05, auto_class_weights="Balanced",
                                                             random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False))]), {"sel__k": ks[-2:], "clf__depth": [3, 5]}
    if name == "vote":
        k = 40 if 40 < n_out else "all"
        def p(clf):
            return Pipeline([("front", clone(front)), ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                             ("sel", SelectKBest(f_classif, k=k)), ("clf", clf)])
        v = VotingClassifier([("lr", p(LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))),
                              ("svm", p(SVC(C=1.0, probability=True, class_weight="balanced", random_state=seed))),
                              ("et", p(ExtraTreesClassifier(500, class_weight="balanced", random_state=seed, n_jobs=1)))], voting="soft")
        return v, {"weights": [None, [2, 1, 1], [1, 1, 2]]}
    raise ValueError(name)


def run_fold(model, X, y, tr, te, seed, n_out, front, mod_idx):
    if model == "stack":
        from sklearn.ensemble import StackingClassifier
        est = []
        for m, keep in mod_idx.items():       # one branch per modality: shared front end, then that modality's columns
            est.append((m, Pipeline([("front", clone(front)), ("cols", ev.Cols(keep)), ("imp", SimpleImputer(strategy="median")),
                                     ("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=min(30, len(keep)))),
                                     ("clf", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))])))
        pipe = StackingClassifier(est, final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced"),
                                  cv=StratifiedKFold(5, shuffle=True, random_state=seed), stack_method="predict_proba", n_jobs=1)
        grid = {"final_estimator__C": [0.1, 1.0, 10.0]}
    else:
        pipe, grid = make(model, n_out, seed, front)
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="balanced_accuracy", n_jobs=1)
    gs.fit(X[tr], y[tr])
    return te, gs.predict_proba(X[te]), gs.best_params_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--extra", default="", help="second CSV merged on NACCID (e.g. deep features)")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--sets", required=True); ap.add_argument("--tasks", default="3class,cn_ad,cn_mci,mci_ad")
    ap.add_argument("--models", default="lr,svm,et,ord,vote")
    ap.add_argument("--need", default="mri"); ap.add_argument("--where", default="")
    ap.add_argument("--wscore", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=10); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out_dir = P1 / "experiments" / a.exp
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} already exists - experiments are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n"); json.dump(vars(a), open(out_dir / "config.json", "w"), indent=1)

    df = pd.read_csv(a.features)
    if a.extra:
        df = df.merge(pd.read_csv(a.extra), on="NACCID", how="left")
    for h in ("has_dl", "has_thk"):
        if h not in df:
            df[h] = 0
        df[h] = df[h].fillna(0)
    if a.where:
        df = df.query(a.where).reset_index(drop=True)
    for m in a.need.split(","):
        df = df[df[f"has_{m}"] == 1]
    df = df.reset_index(drop=True)
    nj = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    rows, preds = [], []; t0 = time.time()
    for fset in a.sets.split(","):
        mods = fset.split("+")
        cols = [c for c in df.columns if any(c.startswith(p) for m in mods for p in PREFIX[m])]
        allc = cols + ["cov_age", "cov_sex_m"]                         # helper columns for W-scores (always last two)
        ia, isx = len(cols), len(cols) + 1
        tracer_groups, tcols = [], []
        for m in ("amy", "tau"):
            tc = f"{m}_tracer_code"
            if tc in cols:
                tracer_groups.append(([i for i, c in enumerate(cols) if c.startswith(m + "_") and c != tc], cols.index(tc))); tcols.append(cols.index(tc))
        adj = [i for i, c in enumerate(cols) if not c.startswith("cov_") and i not in tcols]
        drop = sorted(tcols + [ia, isx])
        out_cols = [c for i, c in enumerate(allc) if i not in drop]
        mod_idx = {m: [i for i, c in enumerate(out_cols) if c.startswith(PREFIX[m])] for m in mods}
        for task in a.tasks.split(","):
            classes = ev.TASKS[task]
            dd = df[df.label.isin(classes)].reset_index(drop=True)
            X = dd[allc].values.astype(float)
            y = np.searchsorted(classes, dd.label.values.astype(int))      # class index 0..K-1 (needed by xgboost)
            cidx = list(range(len(classes)))
            front = Front(tracer_groups, adj, (ia, isx), bool(a.wscore), drop, ref_class=classes.index(0) if 0 in classes else -1)
            for model in a.models.split(","):
                if (model == "stack" and len(mods) < 2) or (model == "ord" and len(classes) < 3):
                    continue
                jobs = [(r, tr, te) for r in range(a.repeats) for tr, te in StratifiedKFold(a.folds, shuffle=True, random_state=a.seed + r).split(X, y)]
                res = Parallel(n_jobs=nj)(delayed(run_fold)(model, X, y, tr, te, a.seed + r, len(out_cols), front, mod_idx) for r, tr, te in jobs)
                rep, cm_sum = [], np.zeros((len(classes), len(classes)), int)
                for r in range(a.repeats):
                    proba = np.zeros((len(y), len(classes)))
                    for (rr, _, _), (te, p, _) in zip(jobs, res):
                        if rr == r:
                            proba[te] = p
                    m_, cm = ev.metrics(y, proba, cidx)
                    m_ = {(f"recall_{classes[int(k[7:])]}" if k.startswith("recall_") else k): v for k, v in m_.items()}
                    rep.append(m_); cm_sum += cm
                    for i in range(len(y)):
                        preds.append([fset, task, model, r, dd.NACCID[i], classes[y[i]]] + list(np.round(proba[i], 5)) + [np.nan] * (3 - len(classes)))
                mm = pd.DataFrame(rep)
                row = {"set": fset, "task": task, "model": model, "wscore": a.wscore, "n": len(y), "n_feat": len(out_cols),
                       "class_counts": "/".join(str(int((y == c).sum())) for c in cidx)}
                for k in mm.columns:
                    row[k + "_mean"] = mm[k].mean(); row[k + "_sd"] = mm[k].std(ddof=1) if len(mm) > 1 else 0.0
                row["confusion_sum"] = json.dumps(cm_sum.tolist())
                rows.append(row)
                print(f"[{time.time()-t0:6.0f}s] {fset:16s} {task:7s} {model:5s} n={len(y)} feat={len(out_cols)} acc={row['acc_mean']:.3f}+-{row['acc_sd']:.3f} "
                      f"bal={row['bal_acc_mean']:.3f} f1={row['f1_macro_mean']:.3f} qwk={row['qwk_mean']:.3f} auc={row['auc_mean']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(preds, columns=["set", "task", "model", "repeat", "NACCID", "y", "p0", "p1", "p2"]).to_csv(out_dir / "predictions.csv.gz", index=False)
    json.dump({"rows": rows}, open(out_dir / "results.json", "w"), indent=1)
    print("done", out_dir, flush=True)


if __name__ == "__main__":
    main()
