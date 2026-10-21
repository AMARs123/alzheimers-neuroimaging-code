# Alzheimer's disease neuroimaging: MRI, amyloid PET and tau PET

Analysis code for three related studies on Alzheimer's disease and cognitive impairment, built on T1 MRI and
amyloid / tau PET from the NACC SCAN cohort, with OASIS-1 as an external MRI cohort.

All image features are extracted by our own pipeline (FastSurfer segmentation, ANTs PET-to-MRI registration,
regional SUVR). No precomputed tabular imaging features are used as predictors.

> **Note:** the articles are currently being submitted to journals. This repository contains **code only**: no paper,
> no manuscript or other documents, no images or figures, no data and no numerical results. These will be linked
> after publication.

## Studies

| Folder | Question | Data |
|---|---|---|
| [`multimodal-mri-pet-staging`](multimodal-mri-pet-staging) | Do amyloid and tau PET add to MRI for normal / MCI / dementia staging when several PET tracers are mixed? | NACC SCAN MRI + amyloid PET + tau PET; OASIS-1 (MRI) |
| [`pet-tracer-harmonisation`](pet-tracer-harmonisation) | Can our own amyloid and tau PET measurements be put on common scales across seven tracers, and how do the resulting A/T profiles match the clinical diagnosis? | NACC SCAN amyloid and tau PET |
| [`mri-regional-atrophy`](mri-regional-atrophy) | Which brain regions show atrophy in MCI and dementia, does the pattern replicate in a second cohort, and how much does each region group add to classification? | NACC SCAN T1 MRI; OASIS-1 |

The three projects share one image-processing pipeline (in `multimodal-mri-pet-staging/src/preprocessing`). The PET and MRI-regional projects
start from the feature tables it produces.

## Pipeline at a glance

```
T1 DICOM ──dcm2niix──> NIfTI ──FastSurfer (seg_only, DKT)──> regional volumes, brain mask
                                                │
amyloid / tau PET (SCAN standardised series) ───┴──ANTs rigid (MI)──> PET in T1 space
                                                         │
                                  reference region (whole cerebellum / inferior cerebellar GM)
                                                         │
                                             regional SUVR + composites
                                                         │
          ┌──────────────────────────────┬───────────────┴───────────────────┐
   Multimodal: nested CV            PET: Centiloid / tau z          MRI regional: effect sizes,
   classifiers (MRI vs MRI+PET,     common scales, A/T profiles     region-group models, asymmetry,
   tracer-aware z, external test)   vs diagnosis                    two-cohort replication
```

## Design principles

- **Participant-level analysis.** One MRI and at most one amyloid and one tau PET per participant; every split is by
  participant, so no scan of the same person appears in both training and test data.
- **No leakage.** Imputation, scaling, feature selection, tracer standardisation and hyper-parameter search are fitted
  inside training folds only (repeated nested cross-validation).
- **No diagnosis-derived inputs.** Cognitive tests, CDR and diagnosis-related variables are never used as features.
- **External validation.** Models are frozen on NACC and applied unchanged to OASIS-1.
- **Traceable experiments.** Every analysis has an experiment ID (EXP0xx / EXP2xx / EXP3xx) and writes its outputs to
  its own folder.

## Data access

The data are **not** included and cannot be redistributed.

- **NACC** (National Alzheimer's Coordinating Center) data, including SCAN MRI/PET, are available to approved
  researchers under the NACC data use agreement: <https://naccdata.org>.
- **OASIS-1** is open access after accepting its data use terms: <https://sites.wustl.edu/oasisbrains/>.

Participant identifiers, feature tables, per-participant predictions and model files are withheld for the same reason.

## Requirements

Python 3.10+ and the packages in [`requirements.txt`](requirements.txt). Image processing additionally needs
[FastSurfer](https://github.com/Deep-MI/FastSurfer), [dcm2niix](https://github.com/rordenlab/dcm2niix) and
[ANTsPy](https://github.com/ANTsX/ANTsPy). The optional deep-feature ablation uses the pretrained SFCN brain-age model
(Peng et al., 2021), which is not included here.

Cluster scripts (`scripts/*.sh`) are SLURM job files; adapt the partition, module and paths to your system.

## Citation

Manuscripts under review. Citation details will be added here after publication. Until then, please contact the
author before reusing this code in a publication.

## Author

Dr. Amar Shukla, IILM University, Gurugram, India.

## Acknowledgement

Data used in these studies were obtained from the National Alzheimer's Coordinating Center (NACC) and the Open Access
Series of Imaging Studies (OASIS-1). The views expressed are those of the authors and not of NACC or OASIS.

## License

MIT, see [`LICENSE`](LICENSE). The license covers the code only, not any data.
