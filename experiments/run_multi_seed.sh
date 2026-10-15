#!/bin/bash
# Encoders with and without the bridge, seeds 42, 123 and 456. Seed 123456 comes from
# scripts/run_encoders.sh. Run from src/.
set -e
RES=../experiments/results/multi_seed/runs
for sys in A_codebert B_unixcoder D_codebert_bridge E_unixcoder_bridge; do
  extra=""; [[ $sys == *bridge ]] && extra="--fixed_len_pad --use_structural"
  for seed in 42 123 456; do
    PRIMEVUL_SEED=$seed python primevul_phase1_auc.py \
      --train ../data/primevul/primevul_train.jsonl \
      --valid ../data/primevul/primevul_valid.jsonl \
      --test  ../data/primevul/primevul_test.jsonl \
      --graphs ../data/primevul_graphs \
      --out ../checkpoints/multi_seed/${sys}_seed$seed --systems $sys --device cuda $extra
    mkdir -p $RES/${sys}_seed$seed
    cp ../checkpoints/multi_seed/${sys}_seed$seed/$sys/{results,threshold}.json $RES/${sys}_seed$seed/
  done
done
for p in A_codebert:encoders_base B_unixcoder:encoders_base D_codebert_bridge:encoders_bask E_unixcoder_bridge:encoders_bask; do
  sys=${p%%:*}; mkdir -p $RES/${sys}_seed123456
  cp ../checkpoints/${p#*:}/$sys/{results,threshold}.json $RES/${sys}_seed123456/
done
cd .. && python experiments/summarize_seeds.py --runs experiments/results/multi_seed/runs --out experiments/results/multi_seed/summary.json
