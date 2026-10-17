#!/bin/bash
# Generates the synthetic paths used in Experiments 3 and 4 from the nine Experiment 2 models.
# Run from the repository root.
set -e
SAMPLES_FOLDER=experiments/neural_SDE/group9/exp3/samples
TRAINING_LOGS=workspace/neural_SDE/logs/train_neural_SDE
CHECKPOINT_FOLDER=$TRAINING_LOGS/US_Stocks/checkpoints
mkdir -p "$SAMPLES_FOLDER"

for run_name in Exp2_{A,B,C}_s{1,2,3}; do
  checkpoint=$(ls $TRAINING_LOGS/$run_name/checkpoints/*.ckpt)
  temporary_name="${run_name}__$(basename "$checkpoint")"
  cp "$checkpoint" "$CHECKPOINT_FOLDER/$temporary_name"
  python experiments/neural_SDE/generate_samples.py --trainer_cfg $run_name --mode mc \
    --batch-size 10000 --seed 1834714493 --checkpoint-name "$temporary_name"
  cp Data/Synthetic/synthetic_rollout_MC_valid_samples.npz "$SAMPLES_FOLDER/$run_name.npz"
  rm "$CHECKPOINT_FOLDER/$temporary_name"
done
