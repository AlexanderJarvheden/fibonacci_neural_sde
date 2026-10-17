#!/bin/bash
# Trains SEB's classifier on real data (benchmark) and on each synthetic dataset.
# Each run trains 10 classifiers. Run from the repository root.
set -e
LOGS_FOLDER=experiments/neural_SDE/group9/exp5/logs
mkdir -p "$LOGS_FOLDER"

# share 0 uses only real data, the synthetic name is required by the script but not used
python experiments/neural_SDE/train_classification.py --trainer_cfg US_Stocks \
  --share 0.0 --synthetic-name A_s1 > "$LOGS_FOLDER/clf_real.log" 2>&1

for generator in A_s1 A_s2 A_s3 C_s1 C_s2 C_s3; do
  python experiments/neural_SDE/train_classification.py --trainer_cfg US_Stocks \
    --share 1.0 --synthetic-name $generator > "$LOGS_FOLDER/clf_$generator.log" 2>&1
done
