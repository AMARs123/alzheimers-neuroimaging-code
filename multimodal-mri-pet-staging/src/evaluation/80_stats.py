"""
80_stats.py
-----------
Confidence intervals and paired comparisons from stored out-of-fold predictions (predictions.csv.gz of an experiment).

For every task / feature set / model:
  * metric = mean over the CV repeats; 95% CI from a subject-level bootstrap (subjects resampled with replacement, all
    repeats of a subject kept together, stratified by class)
  * binary tasks: accuracy, balanced accuracy, AUC, sensitivity (higher-severity class), specificity
  * three-class: accuracy, balanced accuracy, macro-F1, macro one-vs-rest AUC, per-class precision / recall / F1
Paired comparisons (same subjects, same model): bootstrap distribution of the difference between two feature sets,
95% CI and two-sided bootstrap p-value. p-values are NOT corrected for multiple comparisons.

Usage: python 80_stats.py <experiment dir> <out dir> [--pairs mri:mri+tau,mri:mri+amy+tau] [--boot 2000]
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("exp"); ap.add_argument("out")
ap.add_argument("--pairs", default="mri:mri+tau,mri:mri+amy,mri:mri+amy+tau,mri+tau:mri+amy+tau")
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
exp = Path(a.exp); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tag = exp.name
P = pd.read_csv(exp / "predictions.csv.gz")
rng = np.random.default_rng(a.seed)


def cube(d):
    """-> subjects, y (n,), prob (n, repeats, K) aligned on subject."""
    classes = np.sort(d.y.unique()); K = len(classes)
    subs = np.sort(d.NACCID.unique()); reps = np.sort(d.repeat.unique())
    d = d.set_index(["NACCID", "repeat"]).sort_index()
    pr = np.stack([d[f"p{k}"].unstack("repeat").reindex(index=subs, columns=reps).to_numpy() for k in range(K)], axis=2)
    y = d.y.groupby(level=0).first().reindex(subs).to_numpy()
    return subs, np.searchsorted(classes, y), pr, classes


def metrics(y, pr):
    """y (n,), pr (n, R, K) -> dict of metrics averaged over repeats."""
    K = pr.shape[2]; pred = pr.argmax(2); m = {}
    acc = (pred == y[:, None]).mean(0)
    rec = np.stack([(pred[y == k] == k).mean(0) for k in range(K)])                  # (K, R)
    m["accuracy"] = acc.mean(); m["balanced_accuracy"] = rec.mean(0).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.stack([((pred == k) & (y[:, None] == k)).sum(0) / np.maximum((pred == k).sum(0), 1) for k in range(K)])
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    if K == 2:
        m["auc"] = np.mean([roc_auc_score(y, pr[:, r, 1]) for r in range(pr.shape[1])])
        m["sensitivity"] = rec[1].mean(); m["specificity"] = rec[0].mean()
    else:
        m["macro_f1"] = f1.mean(0).mean()
        m["auc_macro_ovr"] = np.mean([roc_auc_score(y, pr[:, r, :] / pr[:, r, :].sum(1, keepdims=True), multi_class="ovr") for r in range(pr.shape[1])])
        for k in range(K):
            m[f"precision_c{k}"] = prec[k].mean(); m[f"recall_c{k}"] = rec[k].mean(); m[f"f1_c{k}"] = f1[k].mean()
    return m


def boot_idx(y):
    idx = np.arange(len(y))
    return np.concatenate([rng.choice(idx[y == k], (y == k).sum(), replace=True) for k in np.unique(y)])


rows, comp = [], []
pairs = [p.split(":") for p in a.pairs.split(",") if p]
for task, dt in P.groupby("task"):
    store = {}
    for (fs, model), d in dt.groupby(["set", "model"]):
        subs, y, pr, classes = cube(d); store[(fs, model)] = (subs, y, pr)
    ref_subs, y, _ = next(iter(store.values()))
    assert all(np.array_equal(v[0], ref_subs) for v in store.values()), "subject sets differ within task"
    B = [boot_idx(y) for _ in range(a.boot)]
    bs = {}
    for key, (_, _, pr) in store.items():
        m = metrics(y, pr); bm = [metrics(y[i], pr[i]) for i in B]; bs[key] = bm
        for k, v in m.items():
            b = np.array([x[k] for x in bm])
            rows.append(dict(experiment=tag, task=task, set=key[0], model=key[1], n=len(y), classes="/".join(map(str, classes)),
                             class_counts="/".join(str(int((y == c).sum())) for c in range(len(classes))),
                             metric=k, value=v, ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5)))
    for s1, s2 in pairs:
        for model in sorted({k[1] for k in store}):
            if (s1, model) in store and (s2, model) in store:
                m1, m2 = metrics(y, store[(s1, model)][2]), metrics(y, store[(s2, model)][2])
                for k in [k for k in ("accuracy", "balanced_accuracy", "auc", "auc_macro_ovr", "macro_f1") if k in m1]:
                    d = np.array([b2[k] - b1[k] for b1, b2 in zip(bs[(s1, model)], bs[(s2, model)])])
                    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                    comp.append(dict(experiment=tag, task=task, model=model, set_a=s1, set_b=s2, metric=k, value_a=m1[k], value_b=m2[k],
                                     diff_b_minus_a=m2[k] - m1[k], ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5), p_boot=p, n=len(y)))
    print(task, "done", flush=True)

R = pd.DataFrame(rows); C = pd.DataFrame(comp)
R.to_csv(out / f"ci_{tag}.csv", index=False); C.to_csv(out / f"paired_{tag}.csv", index=False)
print(f"wrote {len(R)} metric rows, {len(C)} comparisons to {out}")
