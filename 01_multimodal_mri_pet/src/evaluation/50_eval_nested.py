"""
50_eval_nested.py
-----------------
Leakage-free evaluation of feature sets for AD staging.

Everything that learns from data (imputation, scaling, per-tracer standardisation, feature
selection, hyper-parameters) is fitted inside the training part of each outer fold only.

  outer: repeated stratified K-fold (subject level, one row per subject)
  inner: stratified 3-fold grid search on the training part

Usage:
  python 50_eval_nested.py --features F.csv --exp EXP001 --sets mri,tau,mri+tau \
        --tasks 3class,cn_ad --models lr,svm,et,hgb --cohort common --repeats 10
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
                             roc_auc_score, confusion_matrix, recall_score)
from joblib import Parallel, delayed

P1 = Path.home() / "alz" / "p1"
TASKS = {"3class": [0, 1, 2], "cn_ad": [0, 2], "cn_mci": [0, 1], "mci_ad": [1, 2]}
PREFIX = {"mri": ("mri_",), "amy": ("amy_",), "tau": ("tau_",), "cov": ("cov_",)}


class TracerStandardizer(BaseEstimator, TransformerMixin):
    """Z-score the columns of one PET modality separately for each tracer, using training rows only.

    groups: list of (value_column_indices, tracer_code_column_index).
    Reference rows are the training subjects of class `ref_class` (cognitively normal) with that tracer;
    if fewer than `min_ref` are available (or the task has no such class) all training rows of the
    tracer are used. The tracer code columns are removed afterwards so the classifier never sees them.
    """
    def __init__(self, groups=None, ref_class=0, min_ref=8):
        self.groups = groups; self.ref_class = ref_class; self.min_ref = min_ref

    @staticmethod
    def _ms(A):
        return np.nanmean(A, 0), np.nanstd(A, 0) + 1e-6

    def fit(self, X, y=None):
        self.stats_ = []
        for cols, tcol in (self.groups or []):
            st = {}
            pooled = self._ms(X[:, cols])
            for t in np.unique(X[:, tcol][~np.isnan(X[:, tcol])]):
                m = X[:, tcol] == t
                mref = m & (np.asarray(y) == self.ref_class) if y is not None else m
                if mref.sum() >= self.min_ref:
                    st[t] = self._ms(X[mref][:, cols])
                elif m.sum() >= self.min_ref:
                    st[t] = self._ms(X[m][:, cols])
                else:
                    st[t] = pooled
            self.stats_.append((cols, tcol, st, pooled))
        return self

    def transform(self, X):
        X = X.copy()
        for cols, tcol, st, pooled in self.stats_:
            for t in np.unique(X[:, tcol][~np.isnan(X[:, tcol])]):
                m = X[:, tcol] == t
                mu, sd = st.get(t, pooled)
                X[np.ix_(m, cols)] = (X[np.ix_(m, cols)] - mu) / sd
        drop = [tcol for _, tcol, _, _ in self.stats_]
        return np.delete(X, drop, axis=1) if drop else X


class Cols(BaseEstimator, TransformerMixin):
    """Keep a fixed list of columns (used to give each modality its own branch in late fusion)."""
    def __init__(self, idx=None):
        self.idx = idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.idx]


def make_stack(seed, tracer_groups, mod_cols):
    """Late fusion: one regularised logistic model per modality, probabilities combined by a meta logistic model."""
    est = []
    for m, idx in mod_cols.items():
        local = {g: i for i, g in enumerate(idx)}
        groups = [([local[v] for v in vals], local[tc]) for vals, tc in tracer_groups if tc in local]
        n_val = len(idx) - len(groups)
        est.append((m, Pipeline([("cols", Cols(idx)), ("tracer", TracerStandardizer(groups)),
                                 ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                 ("sel", SelectKBest(f_classif, k=min(30, n_val))),
                                 ("clf", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))])))
    st = StackingClassifier(est, final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced"),
                            cv=StratifiedKFold(5, shuffle=True, random_state=seed), stack_method="predict_proba", n_jobs=1)
    return st, {"final_estimator__C": [0.1, 1.0, 10.0]}


def make_model(name, n_feat, seed, tracer_groups, mod_cols=None):
    if name == "stack":
        return make_stack(seed, tracer_groups, mod_cols)
    ks = sorted({k for k in (20, 40, 80) if k < n_feat - 2} | {"all"}, key=lambda v: 10**9 if v == "all" else v)
    pre = [("tracer", TracerStandardizer(tracer_groups)), ("imp", SimpleImputer(strategy="median")),
           ("sc", StandardScaler()), ("sel", SelectKBest(f_classif))]
    if name == "lr":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        grid = {"sel__k": ks, "clf__C": [0.01, 0.1, 1.0]}
    elif name == "svm":
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
        grid = {"sel__k": ks, "clf__C": [0.3, 1.0, 3.0]}
    elif name == "et":
        clf = ExtraTreesClassifier(500, class_weight="balanced", random_state=seed, n_jobs=1)
        grid = {"sel__k": ks, "clf__min_samples_leaf": [1, 3]}
    elif name == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=seed)
        grid = {"sel__k": ks, "clf__max_depth": [2, 3]}
    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                             class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1)
        grid = {"sel__k": ks, "clf__num_leaves": [4, 8]}
    else:
        raise ValueError(name)
    return Pipeline(pre + [("clf", clf)]), grid


def run_fold(model, X, y, tr, te, seed, n_feat, tracer_groups, mod_cols=None):
    pipe, grid = make_model(model, n_feat, seed, tracer_groups, mod_cols)
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, cv=inner, scoring="balanced_accuracy", n_jobs=1, refit=True)
    gs.fit(X[tr], y[tr])
    return te, gs.predict_proba(X[te]), gs.best_params_


def metrics(y, proba, classes):
    pred = np.asarray(classes)[proba.argmax(1)]
    out = {"acc": accuracy_score(y, pred), "bal_acc": balanced_accuracy_score(y, pred),
           "f1_macro": f1_score(y, pred, average="macro"),
           "qwk": cohen_kappa_score(y, pred, weights="quadratic")}
    out["auc"] = roc_auc_score(y, proba[:, 1]) if len(classes) == 2 else roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    rec = recall_score(y, pred, average=None, labels=classes)
    for c, r in zip(classes, rec):
        out[f"recall_{c}"] = r
    return out, confusion_matrix(y, pred, labels=classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--sets", default="mri,amy,tau,mri+amy,mri+tau,mri+amy+tau")
    ap.add_argument("--tasks", default="3class,cn_ad,cn_mci,mci_ad")
    ap.add_argument("--models", default="lr,svm,et,hgb")
    ap.add_argument("--cohort", default="common", help="common = subjects having every modality named in --need; own = each set's own subjects")
    ap.add_argument("--need", default="mri,amy,tau")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tracer_std", type=int, default=1)
    ap.add_argument("--where", default="", help="pandas query applied to the feature table first, e.g. tau_tracer_code==0")
    a = ap.parse_args()

    out_dir = P1 / "experiments" / a.exp
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} already exists - experiments are never overwritten")
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    json.dump(vars(a), open(out_dir / "config.json", "w"), indent=1)

    df = pd.read_csv(a.features)
    if a.where:
        df = df.query(a.where).reset_index(drop=True)
    has = {m: df[f"has_{m}"].astype(bool) for m in ("mri", "amy", "tau")}
    nj = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    rows, preds = [], []
    t0 = time.time()
    for fset in a.sets.split(","):
        mods = fset.split("+")
        need = a.need.split(",") if a.cohort == "common" else [m for m in mods if m in has]
        keep = np.ones(len(df), bool)
        for m in need:
            keep &= has[m].values
        d = df[keep].reset_index(drop=True)
        cols = [c for c in d.columns if any(c.startswith(p) for m in mods for p in PREFIX[m])]
        # tracer code columns are named amy_tracer_code / tau_tracer_code
        tracer_groups = []
        if a.tracer_std:
            for m in ("amy", "tau"):
                tc = f"{m}_tracer_code"
                if tc in cols:
                    vals = [i for i, c in enumerate(cols) if c.startswith(m + "_") and c != tc]
                    tracer_groups.append((vals, cols.index(tc)))
        mod_cols = {m: [i for i, c in enumerate(cols) if c.startswith(PREFIX[m])] for m in mods}
        for task in a.tasks.split(","):
            classes = TASKS[task]
            dd = d[d["label"].isin(classes)].reset_index(drop=True)
            X = dd[cols].values.astype(float); y = dd["label"].values.astype(int)
            for model in a.models.split(","):
                if model == "stack" and len(mods) < 2:
                    continue
                rep_metrics = []; cm_sum = np.zeros((len(classes), len(classes)), int)
                jobs = []
                for r in range(a.repeats):
                    skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed + r)
                    for tr, te in skf.split(X, y):
                        jobs.append((r, tr, te))
                res = Parallel(n_jobs=nj)(delayed(run_fold)(model, X, y, tr, te, a.seed + r, len(cols), tracer_groups, mod_cols) for r, tr, te in jobs)
                for r in range(a.repeats):
                    proba = np.zeros((len(y), len(classes)))
                    for (rr, _, _), (te, p, _) in zip(jobs, res):
                        if rr == r:
                            proba[te] = p
                    m, cm = metrics(y, proba, classes)
                    rep_metrics.append(m); cm_sum += cm
                    for i in range(len(y)):
                        preds.append([fset, task, model, r, dd["NACCID"][i], y[i]] + list(np.round(proba[i], 5)) + [np.nan] * (3 - len(classes)))
                mm = pd.DataFrame(rep_metrics)
                row = {"set": fset, "task": task, "model": model, "n": len(y), "n_feat": len(cols),
                       "class_counts": "/".join(str(int((y == c).sum())) for c in classes)}
                for k in mm.columns:
                    row[k + "_mean"] = mm[k].mean(); row[k + "_sd"] = mm[k].std(ddof=1) if len(mm) > 1 else 0.0
                row["confusion_sum"] = json.dumps(cm_sum.tolist())
                ks = [str(p.get("sel__k")) for _, _, p in res]
                row["k_chosen"] = json.dumps(pd.Series(ks).value_counts().to_dict())
                rows.append(row)
                print(f"[{time.time()-t0:6.0f}s] {fset:14s} {task:7s} {model:5s} n={len(y)} feat={len(cols)} "
                      f"acc={row['acc_mean']:.3f}+-{row['acc_sd']:.3f} bal={row['bal_acc_mean']:.3f} "
                      f"f1={row['f1_macro_mean']:.3f} qwk={row['qwk_mean']:.3f} auc={row['auc_mean']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(preds, columns=["set", "task", "model", "repeat", "NACCID", "y", "p0", "p1", "p2"]).to_csv(out_dir / "predictions.csv.gz", index=False)
    json.dump({"rows": rows}, open(out_dir / "results.json", "w"), indent=1)
    print("done", out_dir, flush=True)


if __name__ == "__main__":
    main()
