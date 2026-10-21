"""
60_importance.py
----------------
Region-wise explainability for one feature set / task / model, computed out-of-fold.

For every outer fold (repeated stratified K-fold) the model is tuned and fitted on the training
part exactly as in 50_eval_nested.py. On the held-out part we compute:
  * grouped permutation importance: all features of one brain region within one modality
    (left, right, totals, asymmetry) are permuted together; drop in AUC (binary) or balanced accuracy
  * SHAP values of the fitted classifier (linear or tree explainer), mapped back to input features

Outputs in ~/alz/p1/experiments/<exp>/: importance_features.csv, importance_regions.csv, config.json
"""
import argparse, json, os, sys, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
# 50_eval_nested.py is not an importable module name; keep an identical copy that worker processes can import
import shutil
shutil.copyfile(HERE / "50_eval_nested.py", HERE / "eval_nested.py")
sys.path.insert(0, str(HERE))
import eval_nested as ev
P1 = ev.P1


def region_of(col):
    """feature name -> (modality, region). Left/right, totals and asymmetry of a region share one group."""
    mod, rest = col.split("_", 1)
    rest = re.sub(r"^(vol|suvr|tot|asym|comp)_", "", rest)
    rest = re.sub(r"^[LR]-", "", rest)
    return mod, rest


def score(y, proba, classes):
    if len(classes) == 2:
        return roc_auc_score(y, proba[:, 1])
    return balanced_accuracy_score(y, np.asarray(classes)[proba.argmax(1)])


def one_fold(model, X, y, tr, te, seed, cols, tracer_groups, classes, groups, n_perm):
    import shap
    pipe, grid = ev.make_model(model, len(cols), seed, tracer_groups)
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="balanced_accuracy", n_jobs=1)
    gs.fit(X[tr], y[tr]); best = gs.best_estimator_
    base = score(y[te], best.predict_proba(X[te]), classes)
    rng = np.random.RandomState(seed)
    perm = {}
    for g, idx in groups.items():
        drops = []
        for _ in range(n_perm):
            Xp = X[te].copy(); o = rng.permutation(len(te))
            Xp[:, idx] = Xp[o][:, idx]
            drops.append(base - score(y[te], best.predict_proba(Xp), classes))
        perm[g] = float(np.mean(drops))
    # SHAP on the selected, standardised features, mapped back to input columns
    pre = best[:-1]; clf = best.named_steps["clf"]
    Ztr = pre.transform(X[tr]); Zte = pre.transform(X[te])
    kept = [c for i, c in enumerate(cols) if i not in {tc for _, tc in tracer_groups}]
    kept = [c for c, s in zip(kept, best.named_steps["sel"].get_support()) if s]
    if model == "lr":
        sv = shap.LinearExplainer(clf, Ztr).shap_values(Zte)
    else:
        sv = shap.TreeExplainer(clf).shap_values(Zte, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[0] == len(clf.classes_) and sv.shape[1] == len(te):
        sv = np.moveaxis(sv, 0, 2)                     # -> (n, features, classes)
    if sv.ndim == 2:
        sv = sv[:, :, None]
    mabs = np.abs(sv).mean(0)                          # (features, classes or 1)
    sh = {c: mabs[i].tolist() for i, c in enumerate(kept)}
    # direction: correlation between feature value and SHAP toward the last class (most impaired)
    sign = {}
    for i, c in enumerate(kept):
        v = sv[:, i, -1]
        sign[c] = float(np.corrcoef(Zte[:, i], v)[0, 1]) if np.std(v) > 0 and np.std(Zte[:, i]) > 0 else 0.0
    return base, perm, sh, sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--exp", required=True)
    ap.add_argument("--set", required=True); ap.add_argument("--task", default="cn_ad")
    ap.add_argument("--model", default="lr", choices=["lr", "et"])
    ap.add_argument("--need", default=""); ap.add_argument("--where", default="")
    ap.add_argument("--repeats", type=int, default=10); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=20); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = P1 / "experiments" / a.exp
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already exists - experiments are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n"); json.dump(vars(a), open(out / "config.json", "w"), indent=1)

    df = pd.read_csv(a.features)
    if a.where:
        df = df.query(a.where).reset_index(drop=True)
    mods = a.set.split("+")
    need = a.need.split(",") if a.need else [m for m in mods if m in ("mri", "amy", "tau")]
    for m in need:
        df = df[df[f"has_{m}"] == 1]
    classes = ev.TASKS[a.task]
    df = df[df.label.isin(classes)].reset_index(drop=True)
    cols = [c for c in df.columns if any(c.startswith(p) for m in mods for p in ev.PREFIX[m])]
    tracer_groups = []
    for m in ("amy", "tau"):
        tc = f"{m}_tracer_code"
        if tc in cols:
            tracer_groups.append(([i for i, c in enumerate(cols) if c.startswith(m + "_") and c != tc], cols.index(tc)))
    X = df[cols].values.astype(float); y = df.label.values.astype(int)
    groups = {}
    for i, c in enumerate(cols):
        if c.endswith("_tracer_code"):
            continue
        groups.setdefault("|".join(region_of(c)), []).append(i)
    jobs = []
    for r in range(a.repeats):
        for tr, te in StratifiedKFold(a.folds, shuffle=True, random_state=a.seed + r).split(X, y):
            jobs.append((r, tr, te))
    res = Parallel(n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))(
        delayed(one_fold)(a.model, X, y, tr, te, a.seed + r, cols, tracer_groups, classes, groups, a.n_perm) for r, tr, te in jobs)
    nf = len(res); ncls = len(next(iter(res[0][2].values()))) if res[0][2] else 1
    feat = []
    for c in cols:
        if c.endswith("_tracer_code"):
            continue
        vals = np.array([r[2].get(c, [0.0] * ncls) for r in res])
        sg = [r[3][c] for r in res if c in r[3]]
        row = {"feature": c, "modality": region_of(c)[0], "region": region_of(c)[1],
               "selected_frac": float(np.mean([c in r[2] for r in res])),
               "mean_abs_shap": float(vals.mean()), "direction_toward_impaired": float(np.mean(sg)) if sg else 0.0}
        for k in range(ncls):
            row[f"mean_abs_shap_class{k}"] = float(vals[:, k].mean())
        feat.append(row)
    feat = pd.DataFrame(feat).sort_values("mean_abs_shap", ascending=False)
    reg = feat.groupby(["modality", "region"]).agg(n_features=("feature", "size"), sum_abs_shap=("mean_abs_shap", "sum"),
                                                   selected_frac=("selected_frac", "mean")).reset_index()
    pm = pd.DataFrame([r[1] for r in res])
    reg["perm_drop_mean"] = [pm[f"{m}|{g}"].mean() for m, g in zip(reg.modality, reg.region)]
    reg["perm_drop_sd"] = [pm[f"{m}|{g}"].std(ddof=1) for m, g in zip(reg.modality, reg.region)]
    reg["shap_share"] = reg.sum_abs_shap / reg.sum_abs_shap.sum()
    reg = reg.sort_values("sum_abs_shap", ascending=False)
    feat.to_csv(out / "importance_features.csv", index=False); reg.to_csv(out / "importance_regions.csv", index=False)
    base = np.mean([r[0] for r in res])
    json.dump({"n": int(len(y)), "class_counts": [int((y == c).sum()) for c in classes], "folds": nf, "base_score_mean": float(base),
               "score": "roc_auc" if len(classes) == 2 else "balanced_accuracy",
               "modality_shap_share": reg.groupby("modality").shap_share.sum().round(4).to_dict()}, open(out / "results.json", "w"), indent=1)
    print(f"{a.set} {a.task} {a.model} n={len(y)} base={base:.3f}")
    print("modality SHAP share:", reg.groupby("modality").shap_share.sum().round(3).to_dict())
    print(reg.head(20).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
