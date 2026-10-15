#!/bin/bash
# KG-only and features-only bridge on StarCoderBase-3B at its main batch size (64). Run from src/.
set -e
for v in kgonly featonly; do
  extra=""; [ "$v" = featonly ] && extra="--use_structural --structural_only"
  python primevul_phase1_auc.py \
    --train ../data/primevul/primevul_train.jsonl \
    --valid ../data/primevul/primevul_valid.jsonl \
    --test  ../data/primevul/primevul_test.jsonl \
    --graphs ../data/primevul_graphs \
    --out ../checkpoints/starcoder_$v --systems G_starcoder_bridge --fixed_len_pad \
    --backbone_mode lora --lr 2e-4 --batch_size 64 --device cuda $extra
  mkdir -p ../experiments/results/starcoder_source_ablation/$v
  cp ../checkpoints/starcoder_$v/G_starcoder_bridge/{results,threshold}.json ../experiments/results/starcoder_source_ablation/$v/
done
