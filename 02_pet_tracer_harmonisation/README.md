# Study 2: Common scales for amyloid and tau PET across seven tracers

**Question.** Can amyloid and tau PET measured with our own MRI-based pipeline be put on common scales across
tracers, how well do these values agree with an independent core laboratory, and how do the resulting biological
(A/T) profiles relate to the clinical diagnosis?

**Data.** NACC SCAN amyloid PET (PiB, florbetaben, florbetapir, NAV4694) and tau PET (flortaucipir, MK-6240,
PI-2620), processed with the Study 1 pipeline. Core-laboratory Centiloid and SUVR values are used only as the
reference for calibration and validation, never as predictors of diagnosis.

## Method

| Step | Method | Script |
|---|---|---|
| Amyloid on the Centiloid scale | per-tracer linear SUVR -> Centiloid map, cross-validated out-of-fold; in-fold cut-off; correlation, concordance, Bland-Altman, kappa; bootstrap CIs | `src/analysis/01_centiloid_calibration.py` |
| Tau harmonisation | cross-fitted z-scores against amyloid-negative normal participants of the same tracer; validation against core lab; sensitivity to the T+ threshold | `src/analysis/02_tau_harmonisation.py` |
| A/T profiles vs diagnosis | amyloid positivity with Wilson CIs; A/T profile frequencies by diagnosis; chi-square tests | `src/analysis/03_at_profiles.py` |
| Single-summary discrimination | AUC of single PET summaries (no trained model), overall and within tracer; paired bootstrap amyloid vs tau with Holm correction | `src/analysis/04_single_summary_auc.py` |
| Adjusted AUC and discordance | age/sex-adjusted discrimination; biology-vs-diagnosis discordance overall and by tracer | `src/analysis/05_adjusted_auc_discordance.py` |
| Subgroups | discrimination by sex, age and other subgroups | `src/analysis/06_subgroups.py` |

`src/visualization/06_figures_tables.py` builds the figures and tables from the stored outputs.

**Leakage control.** Diagnosis is not used in measurement, calibration or harmonisation, except the amyloid-negative
normal reference for tau z-scores, which is cross-fitted.

**Run.** Each script takes the project root as its first argument and an experiment ID where needed, for example
`python src/analysis/01_centiloid_calibration.py <project_root> EXP201`.

**Results:** withheld until publication.
