#!/bin/bash
# Unattended follow-up for the cortical-thickness run (light polling loop for the login node).
#   1. keeps the paired one-hour thickness jobs going until every subject has an output
#      (waits one extra cycle before submitting, so it does not double-submit with 47_thickness_chain.sh)
#   2. collects features_thickness.csv
#   3. submits EXP021 (MRI cohort: volumes vs thickness) and EXP022 (multimodal cohort with thickness)
cd ~/alz/p1
TOTAL=${1:-405}; W=$HOME/alz/p1/work
for round in $(seq 1 100); do
  done_n=$(ls work/thickness | wc -l); n=$(squeue -h -u $USER | wc -l)
  echo "$(date) done=$done_n queued=$n"
  [ "$done_n" -ge "$TOTAL" ] && [ "$n" -eq 0 ] && break
  if [ "$n" -eq 0 ]; then
    sleep 330
    if [ "$(squeue -h -u $USER | wc -l)" -eq 0 ] && [ "$(ls work/thickness | wc -l)" -lt "$TOTAL" ]; then
      sbatch --job-name=thkA scripts/47_thickness.sh shard 0 2
      sbatch --job-name=thkB scripts/47_thickness.sh shard 1 2
    fi
  fi
  sleep 240
done
[ "$(ls work/thickness | wc -l)" -ge "$TOTAL" ] || { echo "thickness incomplete - stopping"; exit 1; }
c=$(sbatch --parsable --job-name=thk_collect scripts/47_thickness.sh shard 0 1)
sbatch --job-name=EXP021 --dependency=afterok:$c scripts/52_eval_plus.sh --features $W/features_v2.csv --extra $W/features_thickness.csv \
  --exp EXP021 --sets mri,thk,mri+thk --need mri,thk --models lr,svm,et,vote --tasks 3class,cn_ad,cn_mci,mci_ad
sbatch --job-name=EXP022 --dependency=afterok:$c scripts/52_eval_plus.sh --features $W/features_v2.csv --extra $W/features_thickness.csv \
  --exp EXP022 --sets mri+tau,mri+thk+tau,mri+amy+tau,mri+thk+amy+tau --need mri,amy,tau,thk --models lr,svm,et,vote,stack --tasks 3class,cn_ad,cn_mci,mci_ad
echo "$(date) submitted collect + EXP021 + EXP022"
