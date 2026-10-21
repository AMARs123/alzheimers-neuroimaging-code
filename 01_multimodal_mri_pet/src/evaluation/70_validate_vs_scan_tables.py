"""
70_validate_vs_scan_tables.py
-----------------------------
External check of our own PET/MRI quantification against the SCAN core-lab tables distributed on LONI
(UC Berkeley MRI-free PET SUVRs, UC Davis MRI volumes). The core-lab values are used ONLY for validation
and for describing amyloid status of the cohort; they are never used as model features.

Usage: python 70_validate_vs_scan_tables.py <features.csv> <meta.csv> <tables dir> <out dir>
"""
import sys, glob, os
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

feat, meta, tdir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
F = pd.read_csv(feat); M = pd.read_csv(meta); D = F.merge(M, on="NACCID", suffixes=("", "_m"))
def T(key): return pd.read_csv(glob.glob(os.path.join(tdir, f"*{key}*.csv"))[0], low_memory=False)
tau, amy, gaain, ucd = T("TAU_MRIFREE_NPDKA"), T("AMYLOID_MRIFREE_NPDKA"), T("AMYLOID_MRIFREE_GAAIN"), T("UCDMRISBM")

def stats(a, b):
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 5: return dict(n=len(a))
    sl, ic = np.polyfit(b, a, 1)
    return dict(n=len(a), pearson_r=pearsonr(a, b)[0], spearman_r=spearmanr(a, b)[0], slope_ours_on_ref=sl, intercept=ic,
                mean_ours=a.mean(), mean_ref=b.mean(), mean_abs_diff=np.abs(a - b).mean())

rows = []
def compare(df, group, pairs, label):
    for g, sub in [("ALL", df)] + list(df.groupby(group)):
        for name, ours, ref in pairs:
            r = stats(sub[ours].to_numpy(float), pd.to_numeric(sub[ref], errors="coerce").to_numpy(float))
            rows.append(dict(modality=label, tracer=g, measure=name, **r))

# ---- tau
d = D[D.has_tau == 1].copy()
d["tau_ent"] = d[["tau_suvr_L-entorhinal", "tau_suvr_R-entorhinal"]].mean(axis=1)
d["tau_it"] = d[["tau_suvr_L-inferiortemporal", "tau_suvr_R-inferiortemporal"]].mean(axis=1)
m = d.merge(tau, left_on=["NACCID", "tau_pet_date"], right_on=["NACCID", "SCANDATE"], how="left")
print(f"tau: {len(d)} of our scans, {m.META_TEMPORAL_SUVR.notna().sum()} matched to a UC Berkeley row on NACCID + scan date")
m = m[m.META_TEMPORAL_SUVR.notna()].drop_duplicates("NACCID")
compare(m, "tau_tracer", [("meta_temporal", "tau_comp_meta_temporal", "META_TEMPORAL_SUVR"), ("entorhinal", "tau_ent", "CTX_ENTORHINAL_SUVR"),
                          ("inferior_temporal", "tau_it", "CTX_INFERIORTEMPORAL_SUVR")], "tau")
m[["NACCID", "tau_tracer", "tau_comp_meta_temporal", "META_TEMPORAL_SUVR", "tau_ent", "CTX_ENTORHINAL_SUVR", "QC_STATUS"]].to_csv(f"{out}/validation_tau_pairs.csv", index=False)

# ---- amyloid
d = D[D.has_amy == 1].copy()
m = d.merge(amy, left_on=["NACCID", "amy_pet_date"], right_on=["NACCID", "SCANDATE"], how="left")
print(f"amyloid: {len(d)} of our scans, {m.NPDKA_SUMMARY_SUVR.notna().sum()} matched")
m = m[m.NPDKA_SUMMARY_SUVR.notna()].drop_duplicates("NACCID")
m["ref_precuneus"] = pd.to_numeric(m.CTX_PRECUNEUS_SUVR, errors="coerce")
m["our_precuneus"] = m[["amy_suvr_L-precuneus", "amy_suvr_R-precuneus"]].mean(axis=1)
compare(m, "amy_tracer", [("global_cortical", "amy_comp_global", "NPDKA_SUMMARY_SUVR"), ("precuneus", "our_precuneus", "ref_precuneus")], "amyloid")
g = gaain[["NACCID", "SCANDATE", "AMYLOID_STATUS", "CENTILOIDS"]]
m = m.merge(g, on=["NACCID", "SCANDATE"], how="left")
m[["NACCID", "label", "amy_tracer", "amy_comp_global", "NPDKA_SUMMARY_SUVR", "AMYLOID_STATUS", "CENTILOIDS"]].to_csv(f"{out}/validation_amyloid_pairs.csv", index=False)
r = stats(m.amy_comp_global.to_numpy(float), pd.to_numeric(m.CENTILOIDS, errors="coerce").to_numpy(float))
rows.append(dict(modality="amyloid", tracer="ALL", measure="our_global_vs_centiloids", **r))
st = m.groupby("label").agg(n=("NACCID", "size"), status_known=("AMYLOID_STATUS", "count"), amyloid_positive=("AMYLOID_STATUS", "sum"),
                            centiloid_median=("CENTILOIDS", "median")).reset_index()
st["pct_positive"] = (100 * st.amyloid_positive / st.status_known).round(1)
st.to_csv(f"{out}/amyloid_status_by_label.csv", index=False); print("\namyloid status by clinical label (0 CN, 1 MCI, 2 dementia):\n", st.to_string(index=False))

# ---- MRI
d = D[D.has_mri == 1].copy(); ucd = ucd.rename(columns={"naccid": "NACCID"})
m = d.merge(ucd, left_on=["NACCID", "mri_date"], right_on=["NACCID", "scandt"], how="left")
print(f"\nMRI: {len(d)} of our scans, {m.hippocampus.notna().sum()} matched to a UC Davis row")
m = m[pd.to_numeric(m.hippocampus, errors="coerce").notna()].drop_duplicates("NACCID")
m["ref_hippo_frac"] = pd.to_numeric(m.hippocampus, errors="coerce") / pd.to_numeric(m.cerebrumtcv, errors="coerce") * 1000
compare(m.assign(all="all"), "all", [("hippocampus_normalised", "mri_tot_Hippocampus", "ref_hippo_frac")], "mri")

R = pd.DataFrame(rows).drop_duplicates(); R.to_csv(f"{out}/validation_vs_scan_tables.csv", index=False)
pd.set_option("display.width", 220); print("\n", R.round(3).to_string(index=False))
