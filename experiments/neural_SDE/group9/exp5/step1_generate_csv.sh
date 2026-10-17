#!/bin/bash
# Generates the synthetic CSVs the classifier trains on, from the A and C models of Experiment 2.
# Run from the repository root.
set -e
TRAINING_LOGS=workspace/neural_SDE/logs/train_neural_SDE
CHECKPOINT_FOLDER=$TRAINING_LOGS/US_Stocks/checkpoints

for generator in A_s1 A_s2 A_s3 C_s1 C_s2 C_s3; do
  run_name=Exp2_$generator
  checkpoint=$(ls $TRAINING_LOGS/$run_name/checkpoints/*.ckpt)
  temporary_name="${run_name}__$(basename "$checkpoint")"
  cp "$checkpoint" "$CHECKPOINT_FOLDER/$temporary_name"
  python experiments/neural_SDE/generate_samples.py --trainer_cfg $run_name --mode standard \
    --batch-size 3000 --n-steps 100 --seed 1834714493 --checkpoint-name "$temporary_name" \
    --no-save-rollout-plots
  cp Data/Synthetic/synthetic_rollout_paths_MoE.csv Data/Synthetic/synthetic_rollout_paths_$generator.csv
  rm "$CHECKPOINT_FOLDER/$temporary_name"
done
