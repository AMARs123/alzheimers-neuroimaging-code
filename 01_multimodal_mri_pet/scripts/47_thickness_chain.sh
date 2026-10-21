#!/bin/bash
# Resubmits paired one-hour thickness jobs (two shards) until every subject has an output. Light polling loop for the login node.
cd ~/alz/p1
TOTAL=${1:-405}
for round in $(seq 1 60); do
  done_n=$(ls work/thickness | wc -l)
  [ "$done_n" -ge "$TOTAL" ] && { echo "$(date) all $done_n done"; break; }
  n=$(squeue -h -u $USER | wc -l)
  if [ "$n" -eq 0 ]; then
    sbatch --job-name=thkA scripts/47_thickness.sh shard 0 2
    sbatch --job-name=thkB scripts/47_thickness.sh shard 1 2
    echo "$(date) submitted pair, done=$done_n"
  fi
  sleep 300
done
