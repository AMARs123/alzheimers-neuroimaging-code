"""
03_region_groups.py  (RQ3: contribution of anatomical region groups)
--------------------------------------------------------------------
NACC MRI cohort (405). Features: the 51 non-constant laterality-free volumes of EXP302, grouped into 11 anatomical
groups (Desikan-Killiany lobe assignment for cortex). For each task (normal vs dementia, normal vs MCI, MCI vs dementia)
and each feature set -- all groups, each group alone, all groups minus one group -- we run 10 x stratified 5-fold
nested CV (inner 3-fold grid over number of features and C, scored by AUC) with median imputation, scaling, ANOVA
selection and class-weighted logistic regression, all fitted in the training folds. Outer splits are identical across
feature sets (seeds 100..109), so the out-of-fold probabilities are paired.
Output: experiments/<EXP id>/{predictions.csv.gz, metrics_by_repeat.csv, groups.json, command.txt}
Usage: python 03_region_groups.py <project dir> <EXP id>
"""
import sys, json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
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

GROUPS = {
    "medial temporal": ["Hippocampus", "Amygdala", "entorhinal", "parahippocampal", "Hippocampus/ILV ratio"],
    "lateral temporal": ["superiortemporal", "middletemporal", "inferiortemporal", "transversetemporal", "fusiform"],
    "parietal": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
    "frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis", "parstriangularis",
                "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal", "precentral", "paracentral"],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "cingulate": ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate", "isthmuscingulate"],
    "insula": ["insula"],
    "subcortical grey": ["Thalamus", "Caudate", "Putamen", "Pallidum", "Accumbens-area", "VentralDC"],
    "ventricles and CSF": ["Lateral-Ventricle", "Inf-Lat-Vent", "3rd-Ventricle", "4th-Ventricle", "CSF", "choroid-plexus"],
    "white matter": ["Cerebral-White-Matter", "WM-hypointensities", "Cerebellum-White-Matter"],
    "cerebellum and brainstem": ["Cerebellum-Cortex", "Brain-Stem"],
}


def bilateral(df):
    out = {}
    for c in df.columns:
        if c.startswith("mri_vol_L-"):
            r = "mri_vol_R-" + c[10:]
            if r in df: out[c[10:]] = df[c] + df[r]
        elif c.startswith("mri_vol_") and not c.startswith("mri_vol_R-"):
            out[c[8:]] = df[c]
    o = pd.DataFrame(out, index=df.index); o["Hippocampus/ILV ratio"] = o["Hippocampus"] / (o["Inf-Lat-Vent"] + 1e-6)
    return o


f = pd.read_csv(M / "features_v2.csv"); n = f[f.has_mri == 1].reset_index(drop=True)
X = bilateral(n); X = X[[c for c in X.columns if X[c].nunique() > 1]]
grouped = sum(GROUPS.values(), [])
missing = [c for c in grouped if c not in X.columns]; unassigned = [c for c in X.columns if c not in grouped]
assert not missing and not unassigned, (missing, unassigned)
ALL = list(X.columns)
SETS = {"all": ALL}
for g, cols in GROUPS.items():
    SETS[f"only: {g}"] = cols
    SETS[f"all minus: {g}"] = [c for c in ALL if c not in cols]
TASKS = {"cn_ad": (0, 2), "cn_mci": (0, 1), "mci_ad": (1, 2)}


def model(nf):
    ks = sorted({k for k in (3, 5, 10, 20) if k < nf} | {"all"}, key=str)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif)),
                     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    return GridSearchCV(pipe, {"sel__k": ks, "clf__C": [0.01, 0.1, 1.0]}, scoring="roc_auc",
                        cv=StratifiedKFold(3, shuffle=True, random_state=0), n_jobs=1)


def run(task, sname, r):
    a, b = TASKS[task]; s = n.label.isin([a, b]).to_numpy(); y = (n.label[s] == b).astype(int).to_numpy()
    A = X.loc[s, SETS[sname]].to_numpy(float); p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(A, y):
        p[te] = model(A.shape[1]).fit(A[tr], y[tr]).predict_proba(A[te])[:, 1]
    return task, sname, r, n.NACCID[s].to_numpy(), y, p


t0 = time.time()
jobs = [(t, s, r) for t in TASKS for s in SETS for r in range(10)]
print(f"{len(ALL)} features, {len(SETS)} feature sets, {len(jobs)} CV runs", flush=True)
res = Parallel(n_jobs=-1, verbose=2)(delayed(run)(*j) for j in jobs)
E.mkdir(parents=True)
pred = pd.concat([pd.DataFrame(dict(task=t, set=s, repeat=r, NACCID=ids, y=y, p=p)) for t, s, r, ids, y, p in res])
pred.to_csv(E / "predictions.csv.gz", index=False)
met = pred.groupby(["task", "set", "repeat"]).apply(lambda g: pd.Series(dict(
    auc=roc_auc_score(g.y, g.p), balanced_accuracy=balanced_accuracy_score(g.y, g.p >= .5), n=len(g), n_features=len(SETS[g.name[1]])))).reset_index()
met.to_csv(E / "metrics_by_repeat.csv", index=False)
json.dump(dict(groups=GROUPS, sets={k: v for k, v in SETS.items()}, tasks=TASKS, seeds_outer="100..109", inner="3-fold, AUC"), open(E / "groups.json", "w"), indent=1)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
S = met.groupby(["task", "set"])[["auc", "balanced_accuracy"]].agg(["mean", "std"]).round(3)
pd.set_option("display.width", 200); print(S.to_string()); print(f"minutes: {(time.time() - t0) / 60:.1f}")
