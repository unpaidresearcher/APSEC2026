#!/bin/bash
# Seed variance: BASK, KG with learnable [PAD], and KG with masked padding. Run from src/.
# Seed 123456 of BASK and of the [PAD] variant come from run_encoders.sh and run_ablations.sh.
set -e
TRAIN="--train ../data/primevul/primevul_train.jsonl --valid ../data/primevul/primevul_valid.jsonl --test ../data/primevul/primevul_test.jsonl --graphs ../data/primevul_graphs --device cuda"
ENC="D_codebert_bridge,E_unixcoder_bridge"
for seed in 42 123 456; do
  PRIMEVUL_SEED=$seed python primevul_phase1_auc.py $TRAIN --out ../checkpoints/seeds_bask_seed$seed \
    --systems $ENC --fixed_len_pad --use_structural
  PRIMEVUL_SEED=$seed python primevul_phase1_auc.py $TRAIN --out ../checkpoints/seeds_kg_pad_seed$seed \
    --systems $ENC --fixed_len_pad
done
for seed in 42 123 456 123456; do
  PRIMEVUL_SEED=$seed python primevul_phase1_auc.py $TRAIN --out ../checkpoints/seeds_kg_masked_seed$seed \
    --systems $ENC
done
