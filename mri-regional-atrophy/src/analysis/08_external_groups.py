"""
08_external_groups.py  (RQ3 external: does the compact medial-temporal model transfer?)
--------------------------------------------------------------------------------------
Two frozen models trained on all NACC normal and dementia participants (138 / 134) with the EXP303 pipeline
(hyper-parameters by 5-fold CV on NACC only, scored by AUC): (a) all 51 bilateral volumes, (b) medial temporal only
(5 features). Both applied unchanged to OASIS-1. A third model trained on NACC normal vs MCI is applied to CDR 0 vs 0.5.
Reported per OASIS contrast: AUC with stratified bootstrap 95% CI (2,000) and the paired AUC difference (b - a).
Calibration is not compared here (multimodal-mri-pet-staging EXP023 showed the threshold does not transfer).
Usage: python 08_external_groups.py <project dir> <EXP id>
"""
import sys, json, warnings, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

P = Path(sys.argv[1]); E = P / "experiments" / sys.argv[2]
if E.exists(): sys.exit(f"{E} exists - experiments are never overwritten")
src = Path(__file__).with_name("03_region_groups.py").read_text(encoding="utf-8")
ns = {"pd": pd, "np": np}
exec("import numpy as np, pandas as pd\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\n"
     "from sklearn.preprocessing import StandardScaler\nfrom sklearn.feature_selection import SelectKBest, f_classif\n"
     "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import GridSearchCV, StratifiedKFold\n"
     + src[src.index("GROUPS = {"): src.index("f = pd.read_csv")] + src[src.index("def model("): src.index("def run(")], ns)
GROUPS, bilateral, model = ns["GROUPS"], ns["bilateral"], ns["model"]
M = P / "dataset" / "metadata"
f = pd.read_csv(M / "features_v2.csv"); n = f[f.has_mri == 1].reset_index(drop=True); o = pd.read_csv(M / "features_oasis1.csv")
Xn, Xo = bilateral(n), bilateral(o); ALL = [c for c in Xn.columns if Xn[c].nunique() > 1 and c in Xo]
SETS = {"all": ALL, "medial temporal": GROUPS["medial temporal"]}
cdr = pd.to_numeric(o.cdr).to_numpy(); rng = np.random.default_rng(7); B = 2000


def fit(sname, labs):
    s = n.label.isin(labs).to_numpy(); y = (n.label[s] == labs[1]).astype(int).to_numpy()
    mdl = model(len(SETS[sname])); mdl.cv = StratifiedKFold(5, shuffle=True, random_state=7)
    return mdl.fit(Xn.loc[s, SETS[sname]].to_numpy(float), y)


rows, preds = [], {"ID": o.ID, "cdr": cdr}
for train, labs, contrasts in (("normal vs dementia", (0, 2), {"CDR 0 vs CDR >= 1": (cdr == 0) | (cdr >= 1), "CDR 0 vs CDR > 0": np.ones(len(cdr), bool)}),
                               ("normal vs MCI", (0, 1), {"CDR 0 vs CDR 0.5": cdr <= 0.5})):
    P_ = {}
    for sname in SETS:
        m_ = fit(sname, labs); P_[sname] = m_.predict_proba(Xo[SETS[sname]].to_numpy(float))[:, 1]
        preds[f"p_{train}_{sname}".replace(" ", "_")] = P_[sname]
    for cname, msk in contrasts.items():
        y = (cdr[msk] > 0).astype(int); a, b = P_["all"][msk], P_["medial temporal"][msk]
        idx = [np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum()) for k in (0, 1)]) for _ in range(B)]
        ba = np.array([roc_auc_score(y[i], a[i]) for i in idx]); bb = np.array([roc_auc_score(y[i], b[i]) for i in idx]); dd = bb - ba
        rows.append(dict(trained_on=train, oasis_contrast=cname, n=len(y), n_pos=int(y.sum()),
                         auc_all=roc_auc_score(y, a), auc_all_ci=f"[{np.percentile(ba, 2.5):.3f}, {np.percentile(ba, 97.5):.3f}]",
                         auc_mtl=roc_auc_score(y, b), auc_mtl_ci=f"[{np.percentile(bb, 2.5):.3f}, {np.percentile(bb, 97.5):.3f}]",
                         diff_mtl_minus_all=roc_auc_score(y, b) - roc_auc_score(y, a), diff_ci_low=np.percentile(dd, 2.5), diff_ci_high=np.percentile(dd, 97.5),
                         p_boot=min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))))
R = pd.DataFrame(rows); E.mkdir(parents=True)
R.to_csv(E / "external_groups.csv", index=False); pd.DataFrame(preds).to_csv(E / "predictions.csv", index=False)
(E / "command.txt").write_text(" ".join(sys.argv) + "\n")
json.dump(dict(script="08_external_groups.py", sets=SETS, bootstrap=B, seed=7), open(E / "config.json", "w"), indent=1)
pd.set_option("display.width", 220); print(R.round(3).to_string(index=False))
