# Added value of amyloid and tau PET over MRI for Alzheimer's staging

**Question.** When a cohort mixes several amyloid and tau PET tracers, does PET improve MRI-based classification of
normal cognition, MCI and dementia, and does the MRI model hold up in an independent cohort?

**Data.** NACC SCAN, 450 participants sampled equally by baseline clinical diagnosis (normal / MCI / dementia); the
subset with MRI + amyloid PET + tau PET is used for the multimodal comparison. OASIS-1 (age >= 60, CDR-rated) is the
external MRI test set.

**Tasks.** Primary: normal vs dementia. Secondary: three-class; normal vs MCI; MCI vs dementia.

## Method

1. **Image processing (label-free).** T1 DICOM to NIfTI (dcm2niix); FastSurfer segmentation with DKT parcellation;
   SCAN-standardised PET series chosen per tracer; rigid PET-to-T1 registration (ANTs, mutual information); mean
   uptake in each FastSurfer label; SUVR against whole cerebellum (amyloid) or eroded inferior cerebellar grey matter
   (tau); registration and field-of-view QC.
2. **Features.** MRI: regional volumes as per-mille of brain volume, bilateral totals, asymmetry indices,
   hippocampus / inferior-lateral-ventricle ratio. Tau: regional SUVR, Braak and meta-temporal composites, asymmetry.
   Amyloid: regional SUVR, global and regional cortical composites.
3. **Tracer-aware standardisation.** Inside each training fold, PET columns are z-scored against training normal
   participants scanned with the same tracer, so tracer differences do not masquerade as disease signal.
4. **Models.** Logistic regression, SVM, random forest and gradient boosting (LightGBM / XGBoost / CatBoost), compared
   across feature sets (MRI, amyloid, tau, and their combinations).
5. **Evaluation.** 10 x 5-fold repeated nested cross-validation (inner 3-fold search); balanced accuracy, AUC, F1,
   MCC and related metrics; paired bootstrap comparisons with multiple-testing correction.
6. **Robustness.** Leave-site-out validation, confound checks (age, sex, site, tracer), subgroup analysis, errors by
   amyloid status, cortical-thickness and deep-feature ablations, and external validation of the frozen MRI model on
   OASIS-1.
7. **Explainability.** Permutation and SHAP importance at feature and region level.

## Code

| Folder | Content |
|---|---|
| `src/preprocessing` | FastSurfer batch run, PET extraction and SUVR, cortical thickness (ANTs DiReCT) |
| `src/data` | inventory of scans, feature table construction, OASIS-1 preparation |
| `src/models` | deep-feature extraction (pretrained SFCN, ablation only) |
| `src/evaluation` | nested CV, statistics, validation against core-lab tables, confounds, site, subgroup and external validation |
| `src/explainability` | feature and region importance |
| `src/visualization` | scripts that build the paper's figures and tables from stored results |
| `scripts` | SLURM job files that run the steps above in order |

Scripts are numbered in run order (`20_` -> `93_`).

**Results:** withheld until publication.
