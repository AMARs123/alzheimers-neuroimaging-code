# Study 3: Regional MRI atrophy across clinical stages, with two-cohort replication

**Question.** Which brain regions show atrophy in MCI and dementia compared with normal cognition, does the regional
pattern replicate in an independent cohort, how much does each group of regions contribute to classification, and
is hemispheric asymmetry informative?

**Data.** NACC SCAN T1 MRI (all participants with a usable segmentation) and OASIS-1 (age >= 60; CDR 0 / 0.5 / >= 1).
Features are FastSurfer regional volumes from the Study 1 pipeline, normalised to brain-mask volume.

## Method

| Step | Method | Script |
|---|---|---|
| Dataset report | cohort description, missingness, group sizes | `src/utils/01_dataset_report.py` |
| Regional effect sizes | age + sex regression fitted in the reference group; Cohen's d of residuals; BH-FDR; bootstrap CIs | `src/analysis/02_regional_effects.py` |
| Replication | Spearman correlation of regional effect sizes between NACC and OASIS-1; joint bootstrap; top-region overlap; sign agreement | `src/analysis/02_regional_effects.py` |
| Region-group models | 11 anatomical groups tested as "only", "all" and "all minus group"; logistic regression in 10 x 5 nested CV | `src/analysis/03_region_groups.py` |
| Group statistics | repeat-averaged out-of-fold AUC; paired bootstrap; Holm / BH correction | `src/analysis/07_group_stats.py` |
| Site check | stratified vs site-grouped CV; within-site AUC | `src/analysis/10_site_grouped_groups.py` |
| External test | frozen NACC models (all regions vs compact medial temporal set) applied to OASIS-1 | `src/analysis/08_external_groups.py` |
| Asymmetry | signed and absolute asymmetry index; adjusted effect sizes; volume-adjusted sensitivity; check by site and scanner | `src/analysis/04_asymmetry.py`, `05_asymmetry_volume_check.py`, `12_wm_asymmetry_by_site.py` |
| Sensitivity | age at scan vs age at label visit; compact vs full model errors; age-adjusted classifiers; subgroups | `src/analysis/15_*.py`, `17_*.py`, `18_*.py`, `19_*.py` |

`src/visualization/09_figures_tables.py` builds the figures and tables from the stored outputs.

**Leakage control.** One scan per participant; all splits by participant; every fitted step inside training folds;
covariate models fitted in the reference group only; the external cohort is never used for fitting.

**Run.** Each script takes the project root as its first argument and an experiment ID where needed, for example
`python src/analysis/02_regional_effects.py <project_root> EXP302`.

**Results:** withheld until publication.
